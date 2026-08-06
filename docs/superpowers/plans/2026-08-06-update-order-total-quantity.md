# update_order Total-Quantity Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ctx.update_order` correct for partially-filled orders on every venue by switching the quantity contract to "new TOTAL including filled" and making each connector own its venue's amend-dialect translation — no framework orchestration.

**Architecture:** The trading mixin validates (`quantity > filled`, at least one of price/quantity) and passes the total through. The ccxt connector resolves `None` fields from the Order, translates the total to the venue dialect (passthrough for Binance/Gate whose amend quantity *is* the new total; `total − filled` for replacement-dialect venues, declared via a class attribute on the qubx exchange subclass), detects the documented Binance/Gate silent-cancel response, and echoes the *requested* total on the ack event. The reducer applies the ack total with a never-negative-remaining clamp. Spec: `docs/superpowers/specs/2026-08-06-update-order-total-quantity-design.md` (committed on this branch).

**Tech Stack:** Python 3.12, pytest (+ pytest-asyncio for connector tests), ccxt 4.5.x, ruff.

## Global Constraints

- Worktree: `~/devs/Qubx/.worktrees/update-order-total`, branch `feat/update-order-total-quantity` (based on `origin/main` @ `e5c9a002`). ALL commands run from this worktree root.
- Use `uv run pytest ...` for tests; final full-suite check is `just test`.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`); no co-authored-by lines.
- **NEVER run `git stash` in this worktree** (the stash list is shared across all Qubx worktrees).
- ruff, 120-char lines; modern typing (`float | None`, no `typing.Optional`).
- The parameter rename `amount` → `quantity` is deliberate (loud break for old callers). Do not add a compatibility alias.
- `docs/superpowers/` is gitignored in this repo — commit files under it with `git add -f`.

---

### Task 1: Trading mixin contract — optional price/quantity, total semantics, shrink-below-filled guard

**Files:**
- Modify: `src/qubx/core/mixins/trading.py:363-410` (`update_order`)
- Modify: `src/qubx/core/interfaces.py:917-930` (`ITradingManager.update_order` declaration)
- Modify: `src/qubx/core/context.py:829-843` (`StrategyContext.update_order` delegation)
- Modify: `tests/qubx/core/mixins/trading_test.py` (existing `amount=` call sites at lines 674, 692, 702 + new test class)
- Modify: `tests/qubx/core/test_order_identifier_routing.py` (existing `amount=` call sites at lines 92, 104, 106-107 and the connector-call assertion near line 97)

**Interfaces:**
- Consumes: existing `_resolve_order`, `_adjust_size(instrument, amount)`, `_adjust_price(instrument, price, amount)` (sign of `amount` = rounding-direction hint), `OrderAlreadyTerminal`, `OrderNotFound`, `OrderUpdateRejectedEvent` — all already imported in `trading.py`.
- Produces: `update_order(price: float | None = None, quantity: float | None = None, order_id: str | None = None, client_order_id: str | None = None, exchange: str | None = None) -> None`. Calls `IConnector.update_order(order, price=<adjusted or None>, quantity=<adjusted total or None>)`. **`quantity=None` means "unchanged" and is passed through as `None`** — Tasks 3/5 rely on that.

- [ ] **Step 1: Write the failing tests**

Append to `tests/qubx/core/mixins/trading_test.py` (reuse the module's existing `trading_manager` / `mock_connector` / `mock_account` fixtures and `_live_order()` helper, same as `TestTradingManagerUpdateOrderGuards` at line 683):

```python
class TestTradingManagerUpdateOrderContract:
    """Total-quantity contract: quantity is the order's new TOTAL including filled,
    unsigned; at least one of price/quantity is required; a total at or below the
    already-filled quantity raises (Binance/Gate silently CANCEL in that case)."""

    def test_both_none_raises_connector_not_called(self, trading_manager, mock_connector, mock_account):
        order = _live_order()
        mock_account.find_order_by_id.return_value = order

        with pytest.raises(ValueError, match="price and/or quantity"):
            trading_manager.update_order(order_id="test_order_123")

        mock_connector.update_order.assert_not_called()
        mock_account.transition_order.assert_not_called()

    def test_non_positive_quantity_raises(self, trading_manager, mock_connector, mock_account):
        order = _live_order()
        mock_account.find_order_by_id.return_value = order

        with pytest.raises(ValueError, match="positive"):
            trading_manager.update_order(quantity=0.0, order_id="test_order_123")

        mock_connector.update_order.assert_not_called()

    def test_total_at_or_below_filled_raises(self, trading_manager, mock_connector, mock_account):
        # Order for 0.1, already filled 0.06: a new total of 0.05 (< filled) is a
        # silent venue cancel in disguise — must raise, never reach the connector.
        order = _live_order()
        order.filled_quantity = 0.06
        mock_account.find_order_by_id.return_value = order

        with pytest.raises(ValueError, match="filled"):
            trading_manager.update_order(quantity=0.05, order_id="test_order_123")
        with pytest.raises(ValueError, match="filled"):
            trading_manager.update_order(quantity=0.06, order_id="test_order_123")  # == filled: that's a cancel

        mock_connector.update_order.assert_not_called()

    def test_price_only_passes_none_quantity(self, trading_manager, mock_connector, mock_account):
        order = _live_order()
        mock_account.find_order_by_id.return_value = order

        trading_manager.update_order(price=51_000.0, order_id="test_order_123")

        mock_connector.update_order.assert_called_once()
        _, kwargs = mock_connector.update_order.call_args
        assert kwargs["quantity"] is None
        assert kwargs["price"] is not None

    def test_quantity_only_passes_none_price(self, trading_manager, mock_connector, mock_account):
        order = _live_order()
        mock_account.find_order_by_id.return_value = order

        trading_manager.update_order(quantity=0.2, order_id="test_order_123")

        mock_connector.update_order.assert_called_once()
        _, kwargs = mock_connector.update_order.call_args
        assert kwargs["price"] is None
        assert kwargs["quantity"] == pytest.approx(0.2)

    def test_old_amount_kwarg_is_a_loud_typeerror(self, trading_manager, mock_account):
        # Deployment-coupling tripwire: an old caller using amount= must fail loudly,
        # never silently reinterpret (spec: "Deployment coupling").
        order = _live_order()
        mock_account.find_order_by_id.return_value = order

        with pytest.raises(TypeError):
            trading_manager.update_order(price=51_000.0, amount=0.1, order_id="test_order_123")
```

Also update the three existing call sites in this file from `amount=0.1` to `quantity=0.1` (lines 674, 692, 702), and in `tests/qubx/core/test_order_identifier_routing.py` change `amount=1.0` to `quantity=1.0` (lines 92, 104, 106-107). At `test_order_identifier_routing.py:97`, the assertion on the connector call becomes:

```python
    trading_manager._exchange_to_connector["BINANCE.UM"].update_order.assert_called_once_with(
        order, price=123.0, quantity=1.0
    )
```

(keep whatever exact price value the current assertion uses after `_adjust_price` — run the test and read the failure if unsure).

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/qubx/core/mixins/trading_test.py::TestTradingManagerUpdateOrderContract -x -q`
Expected: FAIL — `TypeError: update_order() got an unexpected keyword argument 'quantity'` (old signature still has `amount`).

- [ ] **Step 3: Implement the new mixin signature**

Replace `update_order` in `src/qubx/core/mixins/trading.py:363-410` with:

```python
    def update_order(
        self,
        price: float | None = None,
        quantity: float | None = None,
        order_id: str | None = None,
        client_order_id: str | None = None,
        exchange: str | None = None,
    ) -> None:
        """Update a live limit order's price and/or quantity.

        ``quantity`` is the order's new TOTAL, including everything already filled
        (unsigned). ``None`` means "leave unchanged" — at least one of ``price`` /
        ``quantity`` must be given. ``quantity <= filled_quantity`` raises: Binance and
        Gate silently CANCEL an order amended to a total at/below the executed quantity,
        so a shrink below filled must be requested as a cancel, never an update.

        Raises OrderAlreadyTerminal on a settled order (updating a settled order is
        meaningful misuse); a no-op while a previous update is still in flight.
        A synchronous connector failure reverts the order to its pre-pending status
        (via a synthetic OrderUpdateRejectedEvent) and re-raises to the caller.
        """
        if price is None and quantity is None:
            raise ValueError("update_order requires price and/or quantity")
        self._ensure_writable()
        order_id, client_order_id = self._normalize_order_ids(order_id, client_order_id)
        order = self._resolve_order(order_id, client_order_id)
        if order is None:
            raise OrderNotFound(client_order_id or order_id or "")

        if order.status.is_terminal:
            raise OrderAlreadyTerminal(order.client_order_id, order.status)
        if order.status is OrderStatus.PENDING_UPDATE:
            return

        instrument = order.instrument
        if quantity is not None:
            if quantity <= 0:
                raise ValueError(f"update_order quantity must be positive, got {quantity}")
            quantity = abs(self._adjust_size(instrument, quantity))
            if quantity <= order.filled_quantity:
                raise ValueError(
                    f"update_order total {quantity} <= filled {order.filled_quantity} for "
                    f"{order.client_order_id}: a total at/below filled is a silent venue "
                    f"cancel — use cancel_order instead"
                )

        adjusted_price: float | None = None
        if price is not None:
            # _adjust_price uses the amount's sign as the rounding-direction hint;
            # quantity is unsigned now, so derive the sign from the order's side.
            side_sign = 1.0 if order.side == "BUY" else -1.0
            adjusted_price = self._adjust_price(instrument, price, side_sign * (quantity or order.quantity))
            if adjusted_price is None:
                raise ValueError(f"Price adjustment failed for {instrument.symbol}")

        cid = order.client_order_id
        self._account_manager.transition_order(instrument.exchange, cid, OrderStatus.PENDING_UPDATE)
        try:
            self._get_connector(instrument.exchange).update_order(order, price=adjusted_price, quantity=quantity)
        except Exception as e:
            # Same contract as cancel_order: synthetic reject through the PM reverts
            # PENDING_UPDATE via pre_pending, then the original exception re-raises.
            self._context.process_event(
                OrderUpdateRejectedEvent(
                    instrument=instrument,
                    client_order_id=cid,
                    venue_order_id=order.venue_order_id,
                    reason=f"update request failed before reaching venue: {e}",
                )
            )
            raise
```

Update `ITradingManager.update_order` in `src/qubx/core/interfaces.py:917-930` to the same signature, docstring first line: `"""Update a live limit order's price and/or total quantity (total includes filled; None = unchanged; at least one required)."""` Keep the identifier docs.

Update `src/qubx/core/context.py:829-843`:

```python
    def update_order(
        self,
        price: float | None = None,
        quantity: float | None = None,
        order_id: str | None = None,
        client_order_id: str | None = None,
        exchange: str | None = None,
    ) -> None:
        """
        Update a live limit order's price and/or total quantity (total includes filled).
        """
        self._assert_not_fit_thread("update_order")
        self._trading_manager.update_order(
            order_id=order_id, client_order_id=client_order_id, price=price, quantity=quantity, exchange=exchange
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/qubx/core/mixins/trading_test.py tests/qubx/core/test_order_identifier_routing.py -q`
Expected: PASS (including the pre-existing guard/failure classes — they now use `quantity=`).

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/mixins/trading.py src/qubx/core/interfaces.py src/qubx/core/context.py \
        tests/qubx/core/mixins/trading_test.py tests/qubx/core/test_order_identifier_routing.py
git commit -m "feat(core)!: update_order takes optional price/quantity; quantity is the new total incl. filled"
```

---

### Task 2: Reducer — never-negative-remaining clamp on the update ack

**Files:**
- Modify: `src/qubx/core/account_manager/reducer.py:463-478` (`_handle_updated`)
- Modify: `tests/qubx/core/account_manager/reducer_test.py` (append tests; helpers `_state()`, `_order()`, `_fill()`, `apply`, `T0`, `T1` already exist at lines 65-89)

**Interfaces:**
- Consumes: `OrderUpdatedEvent.new_quantity` — under the new contract this is the requested **total** (or `None` = unchanged). Connectors (Task 3) guarantee they echo the requested total, never a venue-dialect figure.
- Produces: `Order.quantity` invariant after any update ack: `quantity >= filled_quantity` (remaining never negative — the ACE 2026-08-05 poison class).

- [ ] **Step 1: Write the failing tests**

Append to `tests/qubx/core/account_manager/reducer_test.py` (section 4.4):

```python
def test_update_ack_total_below_filled_clamps_remaining_at_zero():
    # ACE 2026-08-05 regression class: an ack whose total is below cumulative fills
    # must never make remaining (= quantity - filled) negative. Clamp to filled and
    # let the snapshot/status refetch reconcile.
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)  # quantity=1.0
    apply(state, OrderFilledEvent(instrument=None, client_order_id="c1", fill=_fill(amount=0.6)), T0)
    state.transition_order("c1", OrderStatus.PENDING_UPDATE, T0)

    r = apply(state, OrderUpdatedEvent(instrument=None, client_order_id="c1", new_price=None, new_quantity=0.4), T1)

    o = r.order
    assert o is not None
    assert o.filled_quantity == 0.6
    assert o.quantity == 0.6                       # clamped to filled, not 0.4
    assert o.quantity - o.filled_quantity == 0.0   # remaining never negative


def test_update_ack_total_above_filled_applies_verbatim():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)  # quantity=1.0
    apply(state, OrderFilledEvent(instrument=None, client_order_id="c1", fill=_fill(amount=0.6)), T0)
    state.transition_order("c1", OrderStatus.PENDING_UPDATE, T0)

    r = apply(state, OrderUpdatedEvent(instrument=None, client_order_id="c1", new_price=None, new_quantity=2.0), T1)

    o = r.order
    assert o is not None
    assert o.quantity == 2.0
    assert o.quantity - o.filled_quantity == pytest.approx(1.4)


def test_canceled_during_pending_update_terminalizes_and_clears_pre_pending():
    # Binance/Gate amend with total <= executedQty silently cancels the order; the
    # connector surfaces that as OrderCanceledEvent while we sit in PENDING_UPDATE.
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    state.transition_order("c1", OrderStatus.PENDING_UPDATE, T0)

    r = apply(state, OrderCanceledEvent(instrument=None, client_order_id="c1"), T1)

    assert r.order is not None
    assert r.order.status is OrderStatus.CANCELED
    assert state.get_pre_pending("c1") is None
```

(`OrderFilledEvent`, `OrderCanceledEvent` are already imported at the top of this test module; verify and add if missing.)

- [ ] **Step 2: Run to verify the clamp test fails**

Run: `uv run pytest tests/qubx/core/account_manager/reducer_test.py -k "clamps or above_filled or pending_update" -q`
Expected: `test_update_ack_total_below_filled_clamps_remaining_at_zero` FAILS (`o.quantity == 0.4`); the other two may already pass (that's fine — they pin current behavior against regression).

- [ ] **Step 3: Implement the clamp**

In `src/qubx/core/account_manager/reducer.py`, `_handle_updated`, replace lines 470-473:

```python
    if event.new_price is not None:
        order.price = event.new_price
    if event.new_quantity is not None:
        if event.new_quantity < order.filled_quantity:
            # Ack total below cumulative fills would make remaining negative (the
            # ACE 2026-08-05 poison). Clamp to filled (remaining 0); the periodic
            # snapshot / status refetch reconciles the true figure.
            logger.error(
                f"[{order.client_order_id}] update-ack total {event.new_quantity} < "
                f"filled {order.filled_quantity}; clamping quantity to filled"
            )
            order.quantity = order.filled_quantity
        else:
            order.quantity = event.new_quantity
```

- [ ] **Step 4: Run the reducer suite**

Run: `uv run pytest tests/qubx/core/account_manager/reducer_test.py -q`
Expected: PASS (all existing update tests still green — mechanics unchanged for `new_quantity >= filled`).

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/account_manager/reducer.py tests/qubx/core/account_manager/reducer_test.py
git commit -m "fix(account): update-ack clamps quantity at filled — remaining can never go negative"
```

---

### Task 3: ccxt connector — None-resolution, dialect translation, requested-total echo

**Files:**
- Modify: `src/qubx/connectors/ccxt/connector.py:523-619` (`update_order`, `_update_async`, `_edit_order_direct`)
- Modify: `src/qubx/connectors/ccxt/exchanges/hyperliquid/hyperliquid.py` (dialect attribute + pre-ack guard on `HyperliquidEnhanced`)
- Modify: `tests/qubx/connectors/ccxt/test_ccxt_connector_writes.py` (section "(6) update_order", lines 522-592)

**Interfaces:**
- Consumes: `IConnector.update_order(order, *, price: float | None, quantity: float | None)` — `quantity` is the new TOTAL or `None` (unchanged), from Task 1.
- Produces: venue request always carries a concrete price and amount (resolved from the Order when `None`); `OrderUpdatedEvent(new_price=<requested price or None>, new_quantity=<requested total or None>)` — the reducer (Task 2) relies on `new_quantity` being total-or-None. Dialect declared by the exchange subclass class attribute `AMEND_QUANTITY_DIALECT: str = "replacement"` (absent ⇒ `"total"` passthrough).

- [ ] **Step 1: Write the failing tests**

In `tests/qubx/connectors/ccxt/test_ccxt_connector_writes.py`, section (6). Check the module's `_order()` helper: it builds an Order with `quantity` and `filled_quantity` defaults — if it has no `filled_quantity` parameter, set the attribute on the returned order (Order fields are mutable; the reducer itself assigns them directly).

```python
@pytest.mark.asyncio
async def test_update_replacement_dialect_sends_total_minus_filled() -> None:
    # Hyperliquid-style modify is a venue-side cancel+replace: the amend amount is the
    # REPLACEMENT order's size (remaining), while the framework speaks totals. The
    # connector must translate on the wire and still echo the requested TOTAL on the ack.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.AMEND_QUANTITY_DIALECT = "replacement"
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    order = _order(venue_order_id="VENUE123")
    order.quantity = 2.0
    order.filled_quantity = 0.4
    conn.update_order(order, price=102.0, quantity=2.0)
    await _drive(conn)

    _, kwargs = exchange.edit_order.await_args
    assert kwargs["amount"] == pytest.approx(1.6)   # total - filled on the wire
    ev = sent[0]
    assert isinstance(ev, OrderUpdatedEvent)
    assert ev.new_quantity == 2.0                   # requested TOTAL on the event


@pytest.mark.asyncio
async def test_update_total_dialect_passes_total_verbatim() -> None:
    # Binance/Gate amend quantity IS the new total (executedQty preserved): passthrough.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    del exchange.AMEND_QUANTITY_DIALECT  # plain Mock auto-creates attrs; ensure absent
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    order = _order(venue_order_id="VENUE123")
    order.quantity = 2.0
    order.filled_quantity = 0.4
    conn.update_order(order, price=102.0, quantity=2.0)
    await _drive(conn)

    _, kwargs = exchange.edit_order.await_args
    assert kwargs["amount"] == pytest.approx(2.0)
    assert sent[0].new_quantity == 2.0


@pytest.mark.asyncio
async def test_update_price_only_resolves_quantity_from_order_and_echoes_none() -> None:
    # Binance requires both quantity and price on modify — a price-only update sends the
    # order's current total on the wire but the ack event says "quantity unchanged".
    exchange = Mock()
    exchange.has = {"editOrder": True}
    del exchange.AMEND_QUANTITY_DIALECT
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    order = _order(venue_order_id="VENUE123")
    order.quantity = 2.0
    conn.update_order(order, price=102.0)
    await _drive(conn)

    _, kwargs = exchange.edit_order.await_args
    assert kwargs["amount"] == pytest.approx(2.0)   # resolved from the order
    assert kwargs["price"] == 102.0
    ev = sent[0]
    assert isinstance(ev, OrderUpdatedEvent)
    assert ev.new_price == 102.0
    assert ev.new_quantity is None                  # unchanged — reducer keeps its total


@pytest.mark.asyncio
async def test_update_quantity_only_resolves_price_from_order() -> None:
    exchange = Mock()
    exchange.has = {"editOrder": True}
    del exchange.AMEND_QUANTITY_DIALECT
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    order = _order(venue_order_id="VENUE123")
    order.quantity = 2.0
    conn.update_order(order, quantity=3.0)
    await _drive(conn)

    _, kwargs = exchange.edit_order.await_args
    assert kwargs["amount"] == pytest.approx(3.0)
    assert kwargs["price"] == order.price           # resolved from the order
    ev = sent[0]
    assert ev.new_price is None
    assert ev.new_quantity == 3.0
```

Note on the existing tests in this section: `test_update_direct_edit_emits_updated` (line 526) and `test_update_by_cloid_uses_cloid_edit_endpoint` (line 548) assert positional/kwarg shapes of the ccxt call — update their expected `amount`/event fields to the new behavior if they fail (the wire amount for a total-dialect exchange is unchanged, so most should pass as-is; `test_update_network_error_leaves_inflight_no_reject` at line 510 calls `update_order(..., price=123.0)` with no quantity, which now resolves from the order instead of sending `None` — its assertion, `sent == []`, is unaffected).

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/qubx/connectors/ccxt/test_ccxt_connector_writes.py -k "dialect or resolves" -q`
Expected: FAIL — replacement-dialect test sends `amount=2.0` (no translation yet); price-only test sends `amount=None`.

- [ ] **Step 3: Implement translation + resolution + echo**

In `src/qubx/connectors/ccxt/connector.py`, replace `update_order` (line 523) and `_update_async`/`_edit_order_direct` plumbing:

```python
    def update_order(self, order: Order, *, price: float | None = None, quantity: float | None = None) -> None:
        # Read venue-call fields off the Order SYNCHRONOUSLY (see cancel_order); editOrder
        # needs side/type too, so pass them straight through.
        # The framework speaks TOTAL quantity (incl. filled); translate to the venue's
        # amend dialect here. "replacement" (declared on the exchange subclass): the venue
        # modify is a cancel+replace and its amount is the replacement's size = remaining.
        wire_price = price if price is not None else order.price
        total = quantity if quantity is not None else order.quantity
        if getattr(self._em.exchange, "AMEND_QUANTITY_DIALECT", "total") == "replacement":
            wire_amount = total - order.filled_quantity
        else:
            wire_amount = total
        if wire_amount <= 0:
            raise ValueError(
                f"update_order for {order.client_order_id}: effective amend amount "
                f"{wire_amount} <= 0 (total {total}, filled {order.filled_quantity})"
            )
        self._spawn(
            self._update_async(
                order.client_order_id,
                order.venue_order_id,
                instrument_to_ccxt_symbol(order.instrument),
                order.side.lower(),
                order.type.lower(),
                wire_price,
                wire_amount,
                price,
                quantity,
            )
        )
```

`_update_async` gains the two trailing parameters and emits the *requested* values (the try/except structure, venue-verdict handling, and vid recovery stay byte-identical):

```python
    async def _update_async(
        self,
        client_order_id: str | None,
        venue_order_id: str | None,
        symbol: str,
        side: str,
        order_type: str,
        wire_price: float | None,
        wire_amount: float | None,
        requested_price: float | None,
        requested_total: float | None,
    ) -> None:
        ...
        # (inside the success path, replacing the current OrderUpdatedEvent emission)
        self.send(
            OrderUpdatedEvent(
                instrument=None,
                client_order_id=client_order_id,
                venue_order_id=vid,  # str | None — never coerce to "" (AM would index a bogus id)
                new_price=requested_price,
                new_quantity=requested_total,
            )
        )
```

`_edit_order_direct` is called with `wire_price` / `wire_amount` (its body is unchanged — it already sends `abs(quantity)` as `amount`).

In `src/qubx/connectors/ccxt/exchanges/hyperliquid/hyperliquid.py`, on `HyperliquidEnhanced` (line 18):

```python
class HyperliquidEnhanced(CcxtFuturePatchMixin, cxp.hyperliquid):
    # HL modify is a cancel+replace at the matching engine: the amend amount is the
    # replacement order's size (= desired remaining), NOT the new total. The qubx ccxt
    # connector subtracts filled before hitting the wire when it sees this marker.
    AMEND_QUANTITY_DIALECT = "replacement"
```

and add a pre-ack guard override (upstream `edit_order('')` crashes in `parse_to_int('')` before any HTTP call — turn that into a clean venue-verdict reject):

```python
    async def edit_order_with_client_order_id(self, clientOrderId, symbol, type, side, amount=None, price=None, params={}):
        # HL modify addresses the order by venue oid; there is no cloid-amend endpoint.
        # Upstream's base fallback would call edit_order('') and crash in parse_to_int('').
        raise ccxt.NotSupported("hyperliquid: modify requires the venue order id (order not acked yet)")
```

(`ccxt.NotSupported` subclasses `ExchangeError`, which is in `_VENUE_VERDICT_ERRORS` — the connector emits a clean `OrderUpdateRejectedEvent`. Add a test for this in `tests/qubx/connectors/ccxt/test_ccxt_connector_writes.py` only if an HL-exchange-class test fixture already exists; otherwise cover it in the exchange-class test file used in Task 4.)

- [ ] **Step 4: Run the connector writes suite**

Run: `uv run pytest tests/qubx/connectors/ccxt/test_ccxt_connector_writes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qubx/connectors/ccxt/connector.py src/qubx/connectors/ccxt/exchanges/hyperliquid/hyperliquid.py \
        tests/qubx/connectors/ccxt/test_ccxt_connector_writes.py
git commit -m "feat(ccxt): connector owns amend-dialect translation; ack echoes the requested total"
```

---

### Task 4: Silent-cancel detection + Gate cid-only edit fix

**Files:**
- Modify: `src/qubx/connectors/ccxt/connector.py` (`_update_async` success path)
- Modify: `src/qubx/connectors/ccxt/exchanges/gateio/gateio.py` (`GateioFutures`)
- Modify: `tests/qubx/connectors/ccxt/test_ccxt_connector_writes.py`
- Create or modify: `tests/qubx/connectors/ccxt/test_gateio_exchange.py` (check `ls tests/qubx/connectors/ccxt/` first — if a gateio exchange-class test module exists, append there; follow the fixture pattern of `tests/qubx/connectors/ccxt/test_binance_exchange.py:215-277`, which tests `BinanceQV.edit_contract_order_request` at the request-dict level)

**Interfaces:**
- Consumes: `_emit_canceled_from_response(client_order_id, venue_order_id, response)` — already exists in `connector.py` (used by the cancel path).
- Produces: an edit response with ccxt-unified `status == "canceled"` emits `OrderCanceledEvent` instead of `OrderUpdatedEvent`. Gate `edit_order_request` accepts `id=''` + `params={'clientOrderId': cid}` and produces `order_id = 't-' + cid` in the path (matching how qubx-placed Gate orders carry `text = 't-<cid>'`).

- [ ] **Step 1: Write the failing silent-cancel test**

```python
@pytest.mark.asyncio
async def test_update_silent_cancel_response_emits_canceled_not_updated() -> None:
    # Binance/Gate documented behavior: an amend whose total lands at/below executedQty
    # CANCELS the order (no reject). Racing fills can trigger this despite the mixin
    # pre-check — the connector must surface the truth as a cancel, not an update-ack.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    del exchange.AMEND_QUANTITY_DIALECT
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123", "status": "canceled"})
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.update_order(_order(venue_order_id="VENUE123"), price=102.0, quantity=2.0)
    await _drive(conn)

    assert len(sent) == 1
    assert isinstance(sent[0], OrderCanceledEvent)
    assert sent[0].venue_order_id == "VENUE123"
```

(`OrderCanceledEvent` import: check the module's import block; add if missing.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/qubx/connectors/ccxt/test_ccxt_connector_writes.py -k silent_cancel -q`
Expected: FAIL — an `OrderUpdatedEvent` is emitted.

- [ ] **Step 3: Implement silent-cancel detection**

In `_update_async`, immediately after the vid-recovery line (`vid = vid or str(r.get("id"))`) and before the `OrderUpdatedEvent` emission:

```python
        if isinstance(r, dict) and r.get("status") == "canceled":
            # Binance/Gate amend to a total at/below executedQty CANCELS the order
            # (documented, no error). Surface the truth: this is a cancel, not an update.
            logger.warning(
                f"[{self.exchange_name}] edit of {client_order_id} was a silent venue cancel "
                f"(amend total at/below executed) — emitting cancel"
            )
            self._emit_canceled_from_response(client_order_id, vid, r)
            return
```

Run: `uv run pytest tests/qubx/connectors/ccxt/test_ccxt_connector_writes.py -q` — expected PASS.

- [ ] **Step 4: Write the failing Gate request-level test**

Append to the existing `tests/qubx/connectors/ccxt/test_gateio_exchange.py` (it has registration tests only, no offline fixture yet — add one modeled on `offline_binance_usdm` in `test_binance_exchange.py:73-84`; reuse this file's existing `GateioFutures` import and add `pytest`/`run` imports as the binance file does):

```python
def _gate_swap_market(base: str) -> dict:
    return {
        "id": f"{base}_USDT",
        "symbol": f"{base}/USDT:USDT",
        "base": base, "quote": "USDT", "settle": "USDT",
        "baseId": base, "quoteId": "USDT", "settleId": "usdt",
        "type": "swap", "spot": False, "swap": True, "future": False, "option": False,
        "contract": True, "linear": True, "inverse": False,
        "contractSize": 1.0, "active": True,
        "precision": {"amount": 1, "price": 0.1},
        "limits": {"amount": {"min": 1}, "price": {}, "cost": {}},
        "info": {},
    }


@pytest.fixture
def offline_gateio_futures():
    """GateioFutures with preseeded swap markets — no network calls."""
    ex = GateioFutures({"options": {"defaultType": "swap"}})
    ex.set_markets([_gate_swap_market("BTC")])
    yield ex
    run(ex.close())


class TestGateioCidOnlyEdit:
    """ccxt's base edit_order_with_client_order_id routes to edit_order_request with
    id='' and the cid in params; upstream gate has no cid substitution for amends
    (unlike its fetch_order/cancel_order builders): order_id='' lands in the URL path
    and a stray ``clientOrderId`` body key rides along. The override must produce the
    documented custom-id path form: order_id = 't-<cid>' (qubx-placed gate orders carry
    text='t-<cid>', which the venue resolves while the order is live)."""

    def test_cid_only_edit_request_substitutes_t_prefixed_order_id(self, offline_gateio_futures):
        request = offline_gateio_futures.edit_order_request(
            "", "BTC/USDT:USDT", "limit", "buy", 1.0, 100.0, {"clientOrderId": "myCid123"}
        )
        assert request["order_id"] == "t-myCid123"
        assert "clientOrderId" not in request
        assert request["settle"] == "usdt"

    def test_venue_id_edit_request_unchanged(self, offline_gateio_futures):
        request = offline_gateio_futures.edit_order_request(
            "123456789", "BTC/USDT:USDT", "limit", "buy", 1.0, 100.0, {}
        )
        assert request["order_id"] == "123456789"
```

Run: `uv run pytest tests/qubx/connectors/ccxt/test_gateio_exchange.py -q` — expected: the cid test FAILS (`order_id == ""` and the stray key present), the venue-id test passes.

- [ ] **Step 5: Implement the Gate override**

In `src/qubx/connectors/ccxt/exchanges/gateio/gateio.py`, on `GateioFutures`:

```python
    def edit_order_request(self, id, symbol, type, side, amount=None, price=None, params={}):
        """ccxt's base edit_order_with_client_order_id routes here with id='' and the
        cid in params, but upstream gate has no cid substitution for amends (unlike its
        fetch_order/cancel_order builders): it sends order_id='' in the URL path plus a
        stray ``clientOrderId`` body key the amend endpoint doesn't define. Mirror
        fetch_order_request's documented 't-' substitution.
        """
        clientOrderId = self.safe_string_2(params, "clientOrderId", "text")
        if not id and clientOrderId is not None:
            params = self.omit(params, ["clientOrderId", "text"])
            if clientOrderId[0] != "t":
                clientOrderId = "t-" + clientOrderId
            id = clientOrderId
        return super().edit_order_request(id, symbol, type, side, amount, price, params)
```

Run the Gate test — expected PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qubx/connectors/ccxt/connector.py src/qubx/connectors/ccxt/exchanges/gateio/gateio.py \
        tests/qubx/connectors/ccxt/
git commit -m "fix(ccxt): surface silent venue cancel on amend; gate cid-only edit builds a valid order_id path"
```

---

### Task 5: Backtester sim tests — reprice through the real OME (salvaged from PR #367, adapted)

**Files:**
- Create: `tests/qubx/backtester/reprice_unfilled_limit_test.py` — start from the PR #367 version at `~/devs/Qubx/.worktrees/order-replace/tests/qubx/backtester/reprice_unfilled_limit_test.py` (142 lines; fixture builds SimulatedConnector + real OME + SimulatedAccountManager + TradingManager on one clock), with the adaptations below.
- No production-code changes: `src/qubx/backtester/connector.py:108-135` already resolves `None` price/quantity from the old order and re-places with the given quantity as the new order's full size — since the OME has no partial fills, a resting order always has `filled == 0` and "new total" ≡ "replacement size".

**Interfaces:**
- Consumes: `TradingManager.update_order(price=..., quantity=...)` from Task 1.
- Produces: sim-level pins that (a) a price+quantity amend moves the actual OME book level and keeps `remaining == total - filled` truthful, (b) a price-only amend preserves quantity.

- [ ] **Step 1: Copy and adapt the salvaged test**

Changes to the copied file:
1. Module docstring: replace the "amount = desired remaining" sentence with: `TradingManager.update_order's total-quantity semantics (quantity = new total incl. filled; None = unchanged) through the REAL backtester OME`. Keep the OME-has-no-partial-fills caveat sentence.
2. Line 104: `tm.update_order(price=30800.0, amount=0.3, client_order_id=order.client_order_id)` → `tm.update_order(price=30800.0, quantity=0.3, client_order_id=order.client_order_id)` (with `filled == 0`, total 0.3 ⇒ remaining 0.3 — same book assertions hold).
3. Append a price-only test to the same file:

```python
def test_price_only_reprice_preserves_quantity_through_sim(sim):
    tm, am, conn, instr, time = sim

    order = tm.trade(instr, amount=0.5, price=31000.0)
    assert order is not None

    # Reprice only — quantity omitted must mean "unchanged", venue-side and locally.
    tm.update_order(price=30800.0, client_order_id=order.client_order_id)

    live = am.find_order_by_client_id(order.client_order_id)
    assert live is not None
    assert live.price == pytest.approx(30800.0)
    assert live.quantity == pytest.approx(0.5)      # untouched
    assert live.filled_quantity == 0.0

    book = conn._ome._ome[instr].bids
    assert 30800.0 in book, book
    assert 31000.0 not in book, book

    # Fill through the new level: the full 0.5 executes.
    conn.process_market_data(instr, time.feed(Q("2020-01-01 10:01", 30700.0, 30701.0)))
    live = am.find_order_by_client_id(order.client_order_id)
    assert live is not None
    assert live.filled_quantity == pytest.approx(0.5)
```

- [ ] **Step 2: Run the file**

Run: `uv run pytest tests/qubx/backtester/reprice_unfilled_limit_test.py -q`
Expected: PASS (Task 1 already shipped the signature; if the price-only test fails, read `src/qubx/backtester/connector.py:108-135` — the fix belongs there, keeping `quantity if quantity is not None else old_order.quantity`).

- [ ] **Step 3: Commit**

```bash
git add tests/qubx/backtester/reprice_unfilled_limit_test.py
git commit -m "test(backtester): repricing keeps totals truthful through the real OME; price-only preserves quantity"
```

---

### Task 6: Repo-wide sweep + full canonical suite

**Files:**
- Modify: whatever the sweep finds (expected: none beyond already-updated tests).

- [ ] **Step 1: Sweep for stale call sites and semantics**

```bash
grep -rn "update_order(" src/ tests/ --include="*.py" | grep -v "def update_order" | grep "amount"
grep -rn "amount=" src/qubx/core/mixins/trading.py src/qubx/core/context.py src/qubx/core/interfaces.py | grep -i update
```

Expected: zero hits. Any hit = a missed caller; convert it to `quantity=` with total semantics (if it was passing a remaining, convert the value to `filled + remaining` — but per the ecosystem sweep there are no such callers inside qubx).

- [ ] **Step 2: Run the full canonical suite**

Run: `just test`
Expected: green (baseline caveat: `tests/qubx/connectors/ccxt/test_ohlc_pagination.py` has 5 pre-existing ordering-dependent failures ONLY in single-process runs; under `just test`'s xdist they don't appear — if they do, compare against `main` before assuming branch damage).

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/qubx/core src/qubx/connectors/ccxt tests/qubx && uv run ruff format --check src/qubx/core/mixins/trading.py src/qubx/core/account_manager/reducer.py src/qubx/connectors/ccxt/connector.py`
Expected: clean. Fix and amend if not.

- [ ] **Step 4: Commit (only if the sweep changed anything)**

```bash
git add -A && git commit -m "chore: align remaining update_order call sites with the quantity contract"
```

---

### Task 7: Docs + PR

**Files:**
- Modify: `docs/account-management/design.md` (new section)
- PR via `gh`

- [ ] **Step 1: Write the design-doc section**

Append a section `## Order updates: total-quantity contract and amend dialects` to `docs/account-management/design.md` covering, in this order (condense from the spec `docs/superpowers/specs/2026-08-06-update-order-total-quantity-design.md`, keep it ~60 lines):
1. The contract (total incl. filled; None = unchanged; `<= filled` raises and why — the documented Binance/Gate silent cancel).
2. The dialect table (Binance/Gate = total; HL = replacement/remaining; declared via `AMEND_QUANTITY_DIALECT` on the exchange subclass) and the both-directions translation rule (wire out, requested-total echoed back).
3. The reducer clamp invariant (`quantity >= filled` always; ACE reference).
4. Accepted races verbatim from the spec's "Accepted races" section (total-dialect under-work/silent-cancel; replacement-dialect δ-oversize; pre-existing HL fills-from-zero snapshot caveat).

- [ ] **Step 2: Commit docs (plan file included)**

```bash
git add docs/account-management/design.md
git add -f docs/superpowers/plans/2026-08-06-update-order-total-quantity.md
git commit -m "docs(account): document the update_order total-quantity contract and amend dialects"
```

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin feat/update-order-total-quantity
gh pr create --repo xLydianSoftware/Qubx --base main --title "feat!: update_order total-quantity contract; connectors own amend dialects" --body-file <body written to scratchpad>
```

PR body must cover: What (contract + dialect ownership + clamp + silent-cancel + Gate cid fix), Why (ACE 2026-08-05; dialect table with the doc citations; supersedes #367 and why — link the spec file), Testing (suites run + the salvaged sim tests), Breaking (the `amount`→`quantity` rename + deployment coupling paragraph from the spec), Follow-ups (exchanges-repo HL translation PR gated on semantics verification; qubx-lighter dialect verification; quantkit price-only PR).

- [ ] **Step 4: Close PR #367 with a pointer**

```bash
gh pr close 367 --repo xLydianSoftware/Qubx --comment "Superseded by <new PR url>: review concluded the replace orchestration was over-engineered relative to real usage (the only production caller repriced only). The replacement switches update_order to a total-quantity contract with connector-owned amend dialects — see docs/superpowers/specs/2026-08-06-update-order-total-quantity-design.md on the new branch. Salvaged from this branch: the reducer never-negative-remaining regression, both sim reprice tests, and the design-doc analysis."
```

- [ ] **Step 5: Report** — PR URL, test counts, and the two follow-up repos' next steps.

---

## Deferred to follow-up repos (NOT in this plan)

- **exchanges (`qubx-hyperliquid`)**: `_build_modify_action` translation (`size = total − filled`) + **verify the HL remaining-semantics inference first** — per review discussion: read the official `hyperliquid-python-sdk` source (clone it) and HL docs for how their own client builds `modify` sizes after partial fills; then a testnet probe. Gated on this PR shipping.
- **qubx-lighter**: verify Lighter `base_amount` dialect on `sign_modify_order`; fix the stale `update_order=False` conformance capability flag in the exchanges-monorepo copy.
- **quantkit**: `executor.py:461` → price-only amend; update the three remaining-semantics tests; pin new qubx version (deploy coupled — see spec "Deployment coupling").
