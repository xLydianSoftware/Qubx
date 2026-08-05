# Order Replace Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ctx.update_order` correct for partially-filled orders on every venue by (a) fixing the `Order.quantity` bookkeeping invariant and (b) routing quantity updates of partially-filled orders through a framework-side cancel+replace orchestration instead of venue-native amend.

**Architecture:** The orchestration lives entirely **above the `IConnector` boundary** — in the TradingManager (initiation), the account reducer (terminal-event interception), and the account Reconciler/manager `_execute` action pattern (decision + I/O). Connectors contribute only two things: their existing `cancel_order`/`submit_order` primitives, and an optional enrichment of the terminal cancel event with the venue's final fill figure. The replacement reuses the **same client_order_id** (the same-cid splice HL's native modify already performs), so callers like quantkit's maker-taker slots see one continuous non-terminal order — no API change for strategies.

**Tech Stack:** Python 3.12, qubx core (`src/qubx/core/`), pytest (`tests/qubx/core/account_manager/`).

## Global Constraints

- `Order.quantity` is ABSOLUTE (unsigned); side is carried separately. This plan makes it mean **total including everything ever filled on this client order id** — at all times, on every venue.
- `update_order(price, amount)` contract: **`amount` = desired signed REMAINING to keep working** (sign must match order side). This matches the only existing caller (quantkit passes `signed_remaining`).
- Native venue amend is used **only when `filled_quantity == 0`** — the one case where every venue dialect (HL replace-size, Binance/Gate new-total) coincides.
- No per-venue amend adapters. No changes to the `IConnector` Protocol beyond an optional event field.
- Connectors are stateless adapters (see `src/qubx/core/connector.py` docstring) — no replace state may live in a connector.
- Failure contract: when a replace cannot complete, degrade to **truth** (order reported CANCELED / UPDATE_REJECTED as appropriate), never to fiction ("order alive" when it is dead).

## Background (why — read before implementing)

Incident 2026-08-05 (dev frab bot, pair ACE): repricing a partially-filled HL maker order poisoned the qubx Order record. HL modify = cancel+replace under the same client id; `_handle_updated` (reducer.py:472-473) overwrote `order.quantity` with the amended (remaining) size while `record_fill` (basics.py:865-880) kept `filled_quantity` cumulative → `remaining = quantity − filled` went **negative** (log-verified: −183.08, −580.92, −475.86). Downstream, quantkit's `effective_position` read a fictional maker position, trimmed the wrong leg, and livelocked for 2.5 minutes. On Binance/Gate the same call pattern is wrong differently: their amend-quantity means *new total* with executedQty preserved, so sending "remaining" silently double-shrinks the working order. Design discussion: frab session "refactor", 2026-08-05.

## Event flow (target design)

```
strategy thread                     connector event loop            AM (reducer/reconciler/_execute)
---------------                     --------------------            --------------------------------
update_order(price, amount)
  filled == 0 ──────────────────►   native editOrder ──ack──►       _handle_updated: splice quantity
  filled > 0:
    arm ReplaceIntent(cid)
    order → PENDING_UPDATE
    connector.cancel_order ─────►   REST cancel ──ack/WS──►         _handle_canceled: intent armed →
                                                                      suppress terminal, stash final fills
                                                                    reconciler.on_event → SubmitReplacement
                                                                      or AbandonReplace (residual < min)
                                                                    _execute:
                                                                      SubmitReplacement: splice via routed
                                                                        OrderUpdatedEvent + connector.submit_order
                                                                        (same cid, residual qty)
                                                                      AbandonReplace: clear intent + route
                                                                        OrderCanceledEvent (truth)
```

Existing machinery reused (do NOT reimplement): `PENDING_UPDATE` re-entrancy no-op (trading.py:387), `pre_pending` status capture, `set_venue_id` re-keying, `_is_superseded_oid_cancel` stale-cancel guard (reducer.py:230-246), the `RequestStatus`/`RouteEvent` action pattern (account_manager/reconciler.py:55-90, manager.py:276-312).

---

### Task 1: Reducer splice invariant — `quantity = filled + new_quantity`

The one-line root fix. Ships value even without the rest of the plan (it fixes HL native-modify poisoning immediately).

**Files:**
- Modify: `src/qubx/core/account_manager/reducer.py:463-479` (`_handle_updated`)
- Test: `tests/qubx/core/account_manager/reducer_test.py`

**Interfaces:**
- Produces: `_handle_updated` semantics — `event.new_quantity` is the desired REMAINING; the handler maintains `order.quantity = order.filled_quantity + event.new_quantity`.

- [ ] **Step 1: Write the failing regression test** (mirror existing reducer_test.py fixtures for building a state with one active order — copy the setup pattern from the nearest `_handle_updated` test in that file):

```python
def test_updated_after_partial_fill_keeps_remaining_positive(state_with_order):
    """ACE 2026-08-05 regression: amend of a partially-filled order must not
    produce quantity < filled_quantity (negative remaining)."""
    state, order = state_with_order  # BUY 1344.92, ACCEPTED
    order.record_fill(764.0, 0.0719)

    apply_event(state, OrderUpdatedEvent(
        instrument=None, client_order_id=order.client_order_id,
        venue_order_id="new-vid", new_price=0.072, new_quantity=580.92,
    ))

    assert order.quantity == pytest.approx(764.0 + 580.92)          # total incl. filled
    assert order.quantity - order.filled_quantity == pytest.approx(580.92)  # remaining = what was sent

    # more fills arrive on the replacement — remaining must NEVER go negative
    order.record_fill(292.78, 0.0722)
    assert order.quantity - order.filled_quantity >= 0
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/qubx/core/account_manager/reducer_test.py -k remaining_positive -v` → FAIL (quantity == 580.92, remaining negative after second fill).

- [ ] **Step 3: Implement.** In `_handle_updated`, replace `order.quantity = event.new_quantity` with:

```python
    if event.new_quantity is not None:
        # new_quantity is the desired REMAINING. quantity's invariant is
        # total-including-filled, so splice the cumulative fills back in —
        # otherwise remaining (quantity - filled) goes negative after the
        # first amend of a partially-filled order (ACE incident 2026-08-05).
        order.quantity = order.filled_quantity + event.new_quantity
```

- [ ] **Step 4: Run the reducer suite** — `uv run pytest tests/qubx/core/account_manager/reducer_test.py -v` → all PASS. If an existing test asserts the old overwrite semantics, update that test's expectation to the splice (its old expectation is the bug).

- [ ] **Step 5: Commit** — `git commit -m "fix(account): amend ack splices quantity as filled + remaining, never overwrites"`

---

### Task 2: `update_order` contract — sign validation + routing rule

**Files:**
- Modify: `src/qubx/core/mixins/trading.py:363-411` (`update_order`)
- Test: `tests/qubx/core/` — find the trading-mixin test module via `grep -rn "def test.*update_order" tests/`; add there (create `tests/qubx/core/trading_update_order_test.py` if none exists, reusing whatever context/mock fixtures the nearest trading test uses).

**Interfaces:**
- Consumes: `AccountManagerFacade.arm_replace_intent(exchange, client_order_id, desired_remaining, price, filled_at_request)` — defined in Task 3. Until Task 3 lands, guard the new branch behind the intent-arm call and implement Tasks 2+3 on one branch, committing after both suites pass (they are one reviewable unit with Task 3; split shown here for readability).
- Produces: routing behavior — `filled == 0` → native `connector.update_order` (unchanged); `filled > 0` → replace path (no native amend call).

- [ ] **Step 1: Write failing tests:**

```python
def test_update_order_rejects_sign_mismatch(trading_ctx):
    ctx, order = trading_ctx  # BUY order, ACCEPTED, filled 0
    with pytest.raises(ValueError, match="sign"):
        ctx.update_order(price=1.0, amount=-5.0, client_order_id=order.client_order_id)

def test_update_order_partially_filled_routes_to_cancel(trading_ctx):
    ctx, order = trading_ctx
    order.record_fill(3.0, 1.0)
    ctx.update_order(price=1.01, amount=7.0, client_order_id=order.client_order_id)
    connector = ctx._connector_mock
    connector.update_order.assert_not_called()          # NO native amend
    connector.cancel_order.assert_called_once()          # replace path starts with cancel
    assert order.status is OrderStatus.PENDING_UPDATE

def test_update_order_unfilled_uses_native_amend(trading_ctx):
    ctx, order = trading_ctx  # filled == 0
    ctx.update_order(price=1.01, amount=10.0, client_order_id=order.client_order_id)
    ctx._connector_mock.update_order.assert_called_once()
    ctx._connector_mock.cancel_order.assert_not_called()
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement.** In `update_order`, after the existing resolve/terminal/PENDING_UPDATE checks and `_adjust_size`/`_adjust_price`:

```python
        if amount == 0 or (amount > 0) != (order.side == OrderSide.BUY):
            raise ValueError(
                f"update_order amount {amount} sign does not match order side {order.side} "
                f"for {order.client_order_id} — amount is the desired signed remaining"
            )

        cid = order.client_order_id
        if order.filled_quantity > 0:
            # Venue amend dialects disagree once fills exist (HL: replace-size;
            # Binance/Gate: new-total with executedQty kept). Route through the
            # framework cancel+replace orchestration instead.
            self._account_manager.arm_replace_intent(
                instrument.exchange, cid,
                desired_remaining=abs(amount), price=adjusted_price,
                filled_at_request=order.filled_quantity,
            )
            self._account_manager.transition_order(instrument.exchange, cid, OrderStatus.PENDING_UPDATE)
            self._get_connector(instrument.exchange).cancel_order(order)
            return
```

Keep the existing native path (PENDING_UPDATE transition + `connector.update_order(...)` + synchronous-failure revert) for the `filled == 0` case, unchanged except it now runs only in that case.

- [ ] **Step 4: Run the tests** → PASS. **Step 5: Commit** with Task 3.

---

### Task 3: `ReplaceIntent` table in `AccountState` + manager facade

**Files:**
- Modify: `src/qubx/core/account_manager/state.py` (fields at :48-70, methods nearby)
- Modify: `src/qubx/core/account_manager/manager.py` (facade methods next to `transition_order` — locate via `grep -n "def transition_order" src/qubx/core/account_manager/manager.py`)
- Test: `tests/qubx/core/account_manager/state_test.py`

**Interfaces:**
- Produces:
  - `@dataclass ReplaceIntent: desired_remaining: float; price: float | None; filled_at_request: float; armed_at: np.datetime64; filled_at_cancel: float | None = None`
  - `AccountState.arm_replace_intent(cid, intent)`, `AccountState.get_replace_intent(cid) -> ReplaceIntent | None`, `AccountState.clear_replace_intent(cid) -> ReplaceIntent | None`, `AccountState.replace_intents() -> dict[str, ReplaceIntent]` (read-only view for expiry sweep)
  - `AccountManager.arm_replace_intent(exchange, cid, *, desired_remaining, price, filled_at_request)` — resolves the state and stamps `armed_at` from `self._time.time()`.

- [ ] **Step 1: Failing tests** in `state_test.py`:

```python
def test_replace_intent_arm_get_clear():
    state = AccountState("TEST", "USDT")
    intent = ReplaceIntent(desired_remaining=7.0, price=1.01, filled_at_request=3.0,
                           armed_at=np.datetime64("2026-08-05T16:05:20", "ns"))
    state.arm_replace_intent("cid-1", intent)
    assert state.get_replace_intent("cid-1") is intent
    assert state.clear_replace_intent("cid-1") is intent
    assert state.get_replace_intent("cid-1") is None
    assert state.clear_replace_intent("cid-1") is None  # idempotent
```

- [ ] **Step 2: verify fail. Step 3: implement** — `self._replace_intents: dict[str, ReplaceIntent] = {}` in `__init__`, plus the four trivial methods and the manager facade. **Step 4: run state_test.py + the Task 2 tests → PASS. Step 5: Commit** — `git commit -m "feat(account): replace-intent table + partially-filled update_order routes to cancel+replace"`

---

### Task 4: Reducer interception — suppress the internal cancel, record final fills

**Files:**
- Modify: `src/qubx/core/account_manager/reducer.py:248-255` (`_handle_canceled`), the FILLED terminal handler (`grep -n "_handle_fill" reducer.py`), and `_handle_cancel_rejected` (reducer.py:500-506)
- Test: `tests/qubx/core/account_manager/reducer_test.py`

**Interfaces:**
- Consumes: Task 3's intent table; Task 5's `OrderCanceledEvent.venue_filled_quantity`.
- Produces: an armed intent makes the CANCELED terminal **suppressed** (order stays PENDING_UPDATE, `intent.filled_at_cancel` populated, `ApplyResult()` empty); FILLED clears the intent and flows normally; a cancel-reject with an armed intent clears it and reverts via `pre_pending` with `OrderChange.UPDATE_REJECTED`.

- [ ] **Step 1: Failing tests:**

```python
def test_canceled_with_armed_intent_is_suppressed(state_with_order):
    state, order = state_with_order
    order.record_fill(3.0, 1.0)
    state.arm_replace_intent(order.client_order_id, ReplaceIntent(7.0, 1.01, 3.0, NOW))
    transition(state, order.client_order_id, OrderStatus.PENDING_UPDATE, NOW)

    result = apply_event(state, OrderCanceledEvent(
        instrument=None, client_order_id=order.client_order_id,
        venue_order_id=order.venue_order_id, venue_filled_quantity=3.5))

    assert result.is_empty()                                   # no strategy callback
    assert order.status is OrderStatus.PENDING_UPDATE          # NOT terminal
    assert state.get_replace_intent(order.client_order_id).filled_at_cancel == 3.5

def test_canceled_without_venue_fill_figure_falls_back_to_record(state_with_order):
    ...  # same, venue_filled_quantity=None → filled_at_cancel == order.filled_quantity

def test_filled_terminal_clears_intent_and_flows(state_with_order):
    ...  # armed intent + OrderFilledEvent → intent cleared, order FILLED, result not empty

def test_cancel_rejected_with_intent_reverts_to_truth(state_with_order):
    ...  # armed intent, PENDING_UPDATE + OrderCancelRejectedEvent →
    ...  # intent cleared, status reverted via pre_pending, change == UPDATE_REJECTED
```

- [ ] **Step 2: verify fail. Step 3: implement.** In `_handle_canceled`, after the superseded-oid guard (keep it FIRST — it must still drop stale cancels of replaced venue ids):

```python
    intent = state.get_replace_intent(order.client_order_id)
    if intent is not None:
        # Internal cancel of a replace orchestration: not a terminal for the
        # strategy — the same-cid replacement is about to be submitted. Record
        # the venue's authoritative final fill so the residual is computed from
        # truth (fills that raced the cancel are included).
        intent.filled_at_cancel = (
            event.venue_filled_quantity
            if event.venue_filled_quantity is not None
            else order.filled_quantity
        )
        return ApplyResult()
```

In the FILLED terminal handler add `state.clear_replace_intent(order.client_order_id)` before normal processing. In `_handle_cancel_rejected` add the PENDING_UPDATE + intent branch (clear intent, `_revert_from_pending(..., OrderChange.UPDATE_REJECTED, ...)`).

- [ ] **Step 4: run reducer_test.py → PASS. Step 5: Commit** — `git commit -m "feat(account): replace-intent intercepts internal cancel; filled/reject paths degrade to truth"`

---

### Task 5: `OrderCanceledEvent.venue_filled_quantity` + ccxt enrichment

**Files:**
- Modify: `src/qubx/core/events.py:110-112` (`OrderCanceledEvent`)
- Modify: `src/qubx/connectors/ccxt/connector.py` (`_emit_canceled_from_response` — locate via `grep -n "_emit_canceled_from_response"`)
- Test: `tests/qubx/core/account_manager/reducer_test.py` (event field), connector unit tests (`grep -rn "emit_canceled" tests/` for the module that tests cancel emission; add there)

**Interfaces:**
- Produces: `OrderCanceledEvent(..., venue_filled_quantity: float | None = None)`; the ccxt cancel-ack emitter populates it from the response's unified `filled` field when the venue returns one (Binance futures cancel response carries `executedQty` → ccxt unified `filled`).

- [ ] **Step 1: failing test** — build a fake ccxt cancel response dict with `{"filled": 3.5}`, call `_emit_canceled_from_response`, assert the sent event has `venue_filled_quantity == 3.5`; and a response without `filled` → `None`.
- [ ] **Step 2: verify fail. Step 3: implement** — add the field with default `None` (keeps every existing constructor call site valid); in the emitter: `venue_filled_quantity=response.get("filled") if isinstance(response, dict) else None` (coerce via `float()` when present).
- [ ] **Step 4: run both suites → PASS. Step 5: Commit** — `git commit -m "feat(events): terminal cancel carries the venue's final fill figure when reported"`

---

### Task 6: Reconciler decision — `SubmitReplacement` / `AbandonReplace`

**Files:**
- Modify: `src/qubx/core/account_manager/reconciler.py` (action dataclasses next to `RequestStatus` at :55; decision in `on_event` — locate via `grep -n "def on_event" reconciler.py`)
- Test: `tests/qubx/core/account_manager/reconciler_test.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class SubmitReplacement:
    cid: str

@dataclass(frozen=True)
class AbandonReplace:
    cid: str
    reason: str = ""
```

  `on_event` returns `[SubmitReplacement(cid)]` when a suppressed cancel's residual is executable, `[AbandonReplace(cid, reason)]` when not. Residual formula (all magnitudes): `residual = intent.desired_remaining - max(0.0, (intent.filled_at_cancel or 0.0) - intent.filled_at_request)`.

- [ ] **Step 1: failing tests:**

```python
def test_on_event_canceled_with_intent_submits_replacement(...):
    # intent: desired 7.0, filled_at_request 3.0, filled_at_cancel 3.5 → residual 6.5
    actions = rec.on_event(state, canceled_event, NOW)
    assert actions == [SubmitReplacement(cid)]

def test_on_event_residual_below_min_abandons(...):
    # desired 7.0, filled_at_request 3.0, filled_at_cancel 9.9 → residual 0.1 < min_size
    actions = rec.on_event(state, canceled_event, NOW)
    assert isinstance(actions[0], AbandonReplace)

def test_on_event_canceled_without_intent_unchanged(...):
    # no intent → whatever on_event returned before this task (assert no new action types)
```

- [ ] **Step 2: verify fail. Step 3: implement** — in `on_event`, before existing logic:

```python
        if isinstance(event, OrderCanceledEvent):
            cid = event.client_order_id
            intent = state.get_replace_intent(cid) if cid else None
            if intent is not None and intent.filled_at_cancel is not None:
                order = state.get_order(cid)
                raced = max(0.0, intent.filled_at_cancel - intent.filled_at_request)
                residual = intent.desired_remaining - raced
                min_size = order.instrument.min_size if order and order.instrument else 0.0
                if order is None or residual < max(min_size, 0.0) or residual <= 0.0:
                    return [AbandonReplace(cid, reason=f"residual {residual} below min")]
                return [SubmitReplacement(cid)]
```

- [ ] **Step 4: run reconciler_test.py → PASS. Step 5: Commit** — `git commit -m "feat(account): reconciler decides replacement vs abandon from post-cancel residual"`

---

### Task 7: `_execute` performs the replacement (or the truthful abandon)

**Files:**
- Modify: `src/qubx/core/account_manager/manager.py:276-312` (`_execute` match)
- Test: `tests/qubx/core/account_manager/manager_test.py`

**Interfaces:**
- Consumes: Tasks 3-6. Mirror `TradingManager.trade`'s `OrderRequest` construction (trading.py:158-176) for field names — reduce_only/post_only travel exactly as `trade()` sends them; copy that construction, do not invent field names.
- Produces: two new `_execute` cases.

- [ ] **Step 1: failing tests** (fixture pattern: `manager_test.py` already builds a manager with a mock connector — reuse):

```python
def test_execute_submit_replacement_splices_and_submits(manager_with_order):
    mgr, state, order, connector = manager_with_order   # BUY 10, filled 3.5, PENDING_UPDATE
    state.arm_replace_intent(order.client_order_id, ReplaceIntent(7.0, 1.01, 3.0, NOW, filled_at_cancel=3.5))
    mgr._execute(state, [SubmitReplacement(order.client_order_id)])

    req = connector.submit_order.call_args.args[0]
    assert req.client_id == order.client_order_id            # same-cid splice
    assert req.quantity == pytest.approx(6.5)                # residual 7.0 - 0.5, signed BUY
    assert req.price == 1.01
    assert order.quantity == pytest.approx(3.5 + 6.5)        # invariant restored
    assert state.get_replace_intent(order.client_order_id) is None
    assert order.status is not None and not order.status.is_terminal

def test_execute_abandon_replace_reports_canceled_truth(manager_with_order):
    mgr, state, order, connector = manager_with_order
    state.arm_replace_intent(order.client_order_id, ReplaceIntent(7.0, 3.0, 3.0, NOW, filled_at_cancel=9.9))
    mgr._execute(state, [AbandonReplace(order.client_order_id, "residual below min")])
    assert state.get_replace_intent(order.client_order_id) is None
    assert order.status is OrderStatus.CANCELED              # truth surfaced via routed event
```

- [ ] **Step 2: verify fail. Step 3: implement** — new match arms in `_execute`:

```python
                    case SubmitReplacement(cid=cid):
                        order = state.get_order(cid)
                        intent = state.clear_replace_intent(cid)
                        if order is None or intent is None or connector is None:
                            logger.warning(f"[{state.exchange}] SubmitReplacement dropped: {cid}")
                            continue
                        raced = max(0.0, (intent.filled_at_cancel or 0.0) - intent.filled_at_request)
                        residual = intent.desired_remaining - raced
                        signed = residual if order.side == OrderSide.BUY else -residual
                        request = ...  # build exactly as TradingManager.trade (trading.py:158-176),
                                       # with client_id=cid, quantity=signed, price=intent.price,
                                       # order_type/side/time_in_force/reduce_only/post_only from `order`
                        connector.submit_order(request)
                        # Splice + strategy notification via the normal reducer path (idempotent
                        # with Task 1's rule: quantity = filled + residual).
                        if self._pm is not None:
                            self._pm.process_event(OrderUpdatedEvent(
                                instrument=None, client_order_id=cid,
                                venue_order_id=None, new_price=intent.price,
                                new_quantity=residual))

                    case AbandonReplace(cid=cid, reason=reason):
                        order = state.get_order(cid)
                        state.clear_replace_intent(cid)
                        logger.warning(f"[{state.exchange}] replace abandoned for {cid}: {reason}")
                        if self._pm is not None and order is not None:
                            self._pm.process_event(OrderCanceledEvent(
                                instrument=None, client_order_id=cid,
                                venue_order_id=order.venue_order_id))
```

Note the `RouteEvent` reentrancy comment at manager.py:296-301 — `process_event` is the same synchronous re-entry pattern; keep the ordering **submit first, then route** so the strategy's UPDATED callback observes the intent already cleared.

- [ ] **Step 4: run manager_test.py → PASS. Step 5: verify end-to-end at unit level** — add one test driving the full chain: `apply(OrderCanceledEvent with intent)` → assert connector.submit_order called and order non-terminal with spliced quantity (this exercises `_apply_to_state` → reducer suppress → `rec.on_event` → `_execute`). **Step 6: Commit** — `git commit -m "feat(account): _execute submits same-cid replacement or surfaces the truthful cancel"`

---

### Task 8: Stale-intent expiry + superseded-oid regression

**Files:**
- Modify: `src/qubx/core/account_manager/reconciler.py` (periodic tick — locate the reconcile-tick entry via `grep -n "def on_tick\|reconcile_tick" reconciler.py manager.py`)
- Test: `tests/qubx/core/account_manager/reconciler_test.py`

- [ ] **Step 1: failing tests** — (a) an intent with `armed_at` 30s+ ago and `filled_at_cancel is None` (cancel outcome unknown) → tick returns `[AbandonReplace(cid, "intent expired"), RequestStatus(cid)]`; (b) regression: after a replacement is live (order venue id "B"), a late `OrderCanceledEvent(venue_order_id="A")` is dropped by `_is_superseded_oid_cancel` — order stays non-terminal.
- [ ] **Step 2: verify (b) already passes** (the guard exists — this pins it for the replace flow); (a) fails.
- [ ] **Step 3: implement** the expiry sweep in the periodic tick: `REPLACE_INTENT_MAX_AGE = np.timedelta64(30, "s")` module constant; iterate `state.replace_intents()`, emit the two actions per expired entry.
- [ ] **Step 4: run → PASS. Step 5: Commit** — `git commit -m "feat(account): stale replace intents expire to truth + status refetch"`

---

### Task 9: Sim parity — integration test through the backtester OME

**Files:**
- Test: locate the sim execution-integration tests via `grep -rln "update_order" tests/qubx/backtester/` and add alongside; if none exists, create `tests/qubx/backtester/update_partially_filled_test.py` reusing the simulator setup from the nearest OME/limit-order test.

- [ ] **Step 1: write the scenario test** (this is the ACE replay, in sim):

```python
def test_reprice_partially_filled_limit_keeps_remaining_correct(sim_ctx):
    # 1. place BUY limit 1000 below touch; 2. move market so ~300 fills;
    # 3. ctx.update_order(price=new, amount=700); 4. fill the rest.
    order = ...  # after step 3, read via ctx.find_order_by_client_id
    assert order.quantity - order.filled_quantity == pytest.approx(700)   # never negative
    # after step 4:
    assert ctx.get_position(instrument).quantity == pytest.approx(300 + 700)
```

- [ ] **Step 2: run — if the sim connector's `update_order` (backtester/connector.py:108) semantics diverge (e.g., it resets filled), fix the SIM to mirror the live contract (amount = desired remaining), not vice versa.** The sim goes through the same AM path, so Tasks 1-7 apply automatically; this test pins it.
- [ ] **Step 3: PASS → Commit** — `git commit -m "test(backtester): partially-filled reprice keeps remaining truthful through the sim"`

---

### Task 10: Docs

- [ ] Update `docs/account-management/design.md`: new subsection "Order updates and the replace orchestration" — the contract (amount = desired remaining), the invariant (quantity = total incl. filled), the routing rule, the intent lifecycle, the failure contract table. One page, follow the doc's existing voice.
- [ ] Commit — `git commit -m "docs(account): order-replace orchestration design"`

---

## Self-review checklist (run after writing code)

- Spec coverage: Tasks 1-9 map to: splice invariant ✓, sign validation ✓, routing ✓, intent table ✓, suppress/truth paths ✓, venue fill figure ✓, decision ✓, I/O ✓, expiry ✓, superseded-oid ✓, sim parity ✓.
- The Binance/Gate double-shrink is fixed by *never native-amending at filled > 0* — verify no code path can reach `connector.update_order` with `order.filled_quantity > 0`.
- Grep the final diff for `order.quantity =` — the ONLY assignments must be initial construction and the Task 1 splice.

## Non-goals (explicitly out of scope)

- Per-venue amend translation (eliminated by the routing rule).
- The BINANCE.PM missing order-stream / `AwaitOrderConfirm LOST` re-fire defect (KAITO over-fill facet) — separate connector work.
- Conformance/e2e coverage — see `exchanges:docs/superpowers/plans/2026-08-05-cancel-fill-reporting-conformance.md`.
- quantkit consumer hardening — see `quantkit:docs/superpowers/plans/2026-08-05-maker-taker-venue-truth.md`.
