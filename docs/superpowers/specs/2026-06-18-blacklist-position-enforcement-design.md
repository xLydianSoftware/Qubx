# Blacklist Position Enforcement — Design

**Goal:** Guarantee the invariant *"a bot never holds an open position in a blacklisted instrument."* Today the blacklist only filters the selected universe and force-closes the *delta* of newly-added entries on Apply; positions can survive or be re-opened. This adds continuous, idempotent enforcement.

**Status:** Approved (brainstorm 2026-06-18). Targets Qubx, branch `fix/blacklist-position-enforcement` off `main` (post-1.9.1).

---

## Background — two observed failures

Tested on dev by blacklisting `ONDOUSDT` while the bot held shorts; afterwards `ONDOUSDT` (−891.4) and `WLFIUSDT` (−34767) were still open (only `ICPUSDT` was clean).

**Bug A — re-open by a stale signal (ONDO).** `InstrumentServiceManager.run_cycle` force-closes only `diff.blacklisted_added ∩ held` (mixins/instrument_service.py). ONDO *was* newly-added, so it was force-closed (`buy 887.6`). But strategies emit signals asynchronously (factors uses a 1-minute `ctx.delay`), and `__process_signals` (mixins/processing.py:552) does **not** gate on universe membership or blacklist — it processes any signal that has a quote, routing it through trackers → position gathering → `ctx.trade()` (mixins/trading.py:95). A pre-blacklist signal still in the queue re-shorted ONDO *after* the close. Nothing in the signal/deal/order path re-adds the instrument to the universe, so the result is an **orphan position**: held by the account, absent from `_instruments`, unsubscribed — invisible to both `set_universe` and the delta force-close.

**Bug B — already-blacklisted holding never closed (WLFI).** WLFI was blacklisted in an earlier Apply; a redeploy restored its short. The new Apply (adding ONDO) has `diff.blacklisted_added = [ONDO]`, so WLFI is never force-closed. `set_universe` only closes instruments transitioning *out* of the universe (`prev_set − new_set`); a blacklisted instrument already outside the universe with a lingering position is invisible to it.

Common root cause: **enforcement is event-shaped (on-change delta + universe-exit), not invariant-shaped.** And order submission is never gated, so async signals can reopen exposure.

---

## Design — two complementary mechanisms

Neither alone suffices: #1 makes the invariant *true* (closes holdings); #2 keeps it *true* (blocks re-open).

### #1 — Proactive close (full held set, on Apply **and** at fit)

Replace the delta-based force-close with a full-set sweep, and run it at both triggers.

**File:** `src/qubx/core/mixins/instrument_service.py`

Shared helper:

```python
def _force_close_held_blacklisted(self) -> list[Instrument]:
    """Close every currently-held position whose instrument is blacklisted (idempotent).
    Reduce-only by construction (target 0), so it passes the #2 trade gate."""
    positions = self._context.get_positions()
    held = [i for i, p in positions.items() if p.quantity != 0 and self._service.is_blacklisted(i)]
    if held:
        self._context.remove_instruments(held, if_has_position_then="close")
    return held
```

- `run_cycle()` (Apply / `refresh_instrument_service` action): unchanged ordering — refresh → fire change callbacks (re-fit) → **`_force_close_held_blacklisted()`** (was: delta `still_held`). Return dict reports `force_closed` from the full set.
- New `enforce_at_fit()`: refresh cache → `_force_close_held_blacklisted()`, **no callbacks** (the fit is already running; firing re-fit callbacks here would loop). Replaces the cache-only `refresh_only()` call site.

**File:** `src/qubx/core/mixins/processing.py` — in `__invoke_on_fit` (~line 621), call `self._context._instrument_service_manager.enforce_at_fit()` instead of `refresh_only()`. Sequence becomes: refresh + close blacklisted holdings → `on_fit` (whose `get_universe`/`filter_blacklisted` already exclude blacklisted, so the rebalance covers only survivors).

`refresh_only()` is **replaced by** `enforce_at_fit()`: its only production caller is `__invoke_on_fit`, and its prior assumption that "set_universe handles closing" is the bug being fixed. This requires updating the `IInstrumentServiceManager` interface (interfaces.py:1364, swap `refresh_only` → `enforce_at_fit`) and retargeting its tests (`instrument_service_manager_test.py::test_refresh_only_*`, and `processing_fit_refresh_test.py` which asserts `refresh_only` runs before `on_fit`).

### #2 — Preventive gate (reduce-only clamp at the order choke)

`ctx.trade()` / `ctx.trade_async()` are the single universal choke — position gathering (`gathering/simplest.py:64`) and any manual call funnel through them. Gate there.

**File:** `src/qubx/core/mixins/trading.py`

```python
def _blacklist_clamp(self, instrument: Instrument, amount: float) -> float:
    """Reduce-only for blacklisted instruments: an order may only move the position
    toward zero, never open or increase exposure. No-op when not blacklisted (the
    common case, and always so without an instrument service). Clamps rather than
    rejects: the reducing portion is kept; a flip through zero is clamped to an exact
    close. Returns the (possibly adjusted) amount; 0.0 means 'do not send'."""
    if not self._context.is_blacklisted(instrument):
        return amount
    current = self._account.get_position(instrument).quantity
    new = current + amount
    if current == 0 or abs(new) > abs(current):
        logger.warning(f"[Blacklist] :: blocked order increasing exposure on {instrument.symbol} "
                       f"(current={current}, amount={amount})")
        return 0.0
    if (current > 0) != (new > 0) and new != 0:  # would flip through zero
        logger.warning(f"[Blacklist] :: clamping {instrument.symbol} order to close "
                       f"(current={current}, requested new={new})")
        return -current
    return amount  # reduces |position| toward zero -> allowed
```

Both `trade()` and `trade_async()` call it first thing:

```python
amount = self._blacklist_clamp(instrument, amount)
if amount == 0.0:
    return None
size_adj = self._adjust_size(instrument, amount)
...
```

This stops the stale ONDO signal (opens from ~0 → `current == 0` → blocked) while allowing the force-close (a reduction). It also blocks restore-reconciliation re-opens once the cache is warm.

---

## Interaction & data flow

- **Apply:** `run_cycle` → re-fit callbacks → full-set force-close. Survivors rebalanced by the re-fit; all blacklisted holdings closed. #2 prevents any same-tick stale signal from reopening.
- **Fit (no Apply):** `enforce_at_fit` closes blacklisted holdings before `on_fit`; the rebalance covers survivors only. This is what makes "the blacklist is picked up at the next fit" actually close positions (the poll was removed).
- **Between events:** #2 guarantees no order can open/increase blacklisted exposure, from any source (async signals, manual `ctx.trade`, restore).

## Edge cases

- **NullInstrumentService:** `is_blacklisted` is always False → #2 is a no-op; `_force_close_held_blacklisted` finds nothing. Zero behavior change for non-blacklist users.
- **Force-close vs the gate:** closing reduces `|position|` → #2 allows it. No deadlock between #1 and #2.
- **Flip through zero:** a target that would cross from short to long (or vice-versa) on a blacklisted instrument is clamped to an exact close (`-current`), never to the opposite side.
- **Startup transient (accepted):** state-restore can re-establish a blacklisted position in the ~1s window before the first blacklist fetch (`start()` → `delay("1s", run_cycle)`). The startup `run_cycle` then force-closes it. Brief, self-healing; not blocked on a warm cache (explicitly out of scope).
- **Rounding/min-size:** clamp operates on the raw `amount` before `_adjust_size`; if the clamped close is below min size, existing min-size handling applies as today.

## Testing

- **#1 (`instrument_service` tests):**
  - Held blacklisted instrument **not** in `diff.blacklisted_added` is force-closed by `run_cycle` (WLFI regression).
  - `enforce_at_fit` refreshes the cache **and** force-closes held blacklisted **without** firing change callbacks (replaces the `refresh_only` tests).
  - No holdings / NullInstrumentService → no `remove_instruments` call.
  - `processing_fit_refresh_test.py` updated: assert `enforce_at_fit` runs before `on_fit`.
- **#2 (`trading` tests, mocked account/context):**
  - Blacklisted + flat → `trade` returns None, no order sent (ONDO stale-signal regression).
  - Blacklisted + reducing order → passes through unchanged.
  - Blacklisted + flip-through-zero → clamped to `-current` (exact close).
  - Not blacklisted → unchanged (no-op).
  - Same coverage for `trade_async`.

## Out of scope

- Blocking restore on a warm blacklist cache (the startup transient above).
- Order-gateway / exchange-side reduce-only flags (this is an in-process guard).
- Any change to how blacklist entries are authored or distributed (platform side unchanged).
