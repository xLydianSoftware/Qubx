import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx import logger
from qubx.core.account_manager.reconcile import ReconcileDiff
from qubx.core.account_manager.reducer import ApplyResult
from qubx.core.basics import DataType, Deal, FundingPayment, MarketEvent, OrderChange
from qubx.core.events import (
    AccountSnapshot,
    AccountSnapshotEvent,
    BalanceUpdateEvent,
    FundingPaymentEvent,
    OrderAcceptedEvent,
    OrderCancelRejectedEvent,
    OrderFilledEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.interfaces import IStrategy
from qubx.core.mixins.processing import ProcessingManager, validate_account_callback_signatures
from tests.qubx.core.conftest import make_pm


def _fill() -> Deal:
    return Deal(trade_id="t1", order_id="V1", time=np.datetime64("now"), amount=1.0, price=50_000.0, aggressive=True)


def test_filled_event_routes_to_all_three_callbacks():
    # Full field->callback mapping for a fill: order -> on_order(order, change),
    # deal -> on_execution(instrument, deal) + downstream consumers, position -> on_position_change.
    pm = make_pm()
    fill = _fill()
    order, position, instrument = MagicMock(), MagicMock(), MagicMock()
    pm._account_manager.apply.return_value = ApplyResult(
        order=order, order_change=OrderChange.FILLED, deal=fill, position=position
    )
    pm.process_event(OrderFilledEvent(instrument=instrument, client_order_id="cid", venue_order_id="V1", fill=fill))
    pm._account_manager.apply.assert_called_once()
    pm._strategy.on_order.assert_called_once()
    assert pm._strategy.on_order.call_args.args[1] is order
    assert pm._strategy.on_order.call_args.args[2] is OrderChange.FILLED
    pm._strategy.on_execution.assert_called_once()
    assert pm._strategy.on_execution.call_args.args[1] is instrument
    assert pm._strategy.on_execution.call_args.args[2] is fill
    pm._strategy.on_position_change.assert_called_once()
    assert pm._strategy.on_position_change.call_args.args[1] is position
    # downstream per-fill notification ran with the AM-confirmed deal
    pm._position_gathering.on_execution_report.assert_called_once()
    pm._logging.save_deals.assert_called_once()


def test_accepted_event_fires_on_order_with_change():
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult(order=MagicMock(), order_change=OrderChange.ACCEPTED)
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    pm._strategy.on_order.assert_called_once()
    assert pm._strategy.on_order.call_args.args[0] is pm._context
    assert pm._strategy.on_order.call_args.args[2] is OrderChange.ACCEPTED
    pm._strategy.on_execution.assert_not_called()
    pm._strategy.on_position_change.assert_not_called()


def test_suppressed_result_fires_no_callbacks():
    # None-as-suppress: an empty ApplyResult (late/duplicate/terminal/unknown event)
    # must fire no strategy callback and no downstream fill consumer.
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult()
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    pm._account_manager.apply.assert_called_once()
    assert pm._strategy.mock_calls == []
    pm._position_gathering.on_execution_report.assert_not_called()
    pm._logging.save_deals.assert_not_called()


def test_deduped_fill_skips_execution_and_downstream():
    # A venue re-sends an order report whose embedded deal the AM already applied: it reports
    # the status change (result.order set) but suppresses the deal (result.deal None), so
    # neither on_execution nor save_deals/gatherer/trackers may see the same Deal twice.
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult(order=MagicMock(), order_change=OrderChange.FILLED)
    pm.process_event(OrderFilledEvent(instrument=MagicMock(), client_order_id="cid", venue_order_id="V1", fill=_fill()))
    pm._strategy.on_order.assert_called_once()
    pm._strategy.on_execution.assert_not_called()
    pm._position_gathering.on_execution_report.assert_not_called()
    pm._logging.save_deals.assert_not_called()


def test_execution_only_result_fires_execution_without_on_order():
    # A fill with no status change (subsequent partial, fill while PENDING_*) carries a
    # deal but no order: on_execution + downstream + on_position_change run, on_order does not.
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult(deal=_fill(), position=MagicMock())
    pm.process_event(OrderFilledEvent(instrument=MagicMock(), client_order_id="cid", venue_order_id="V1", fill=_fill()))
    pm._strategy.on_order.assert_not_called()
    pm._strategy.on_execution.assert_called_once()
    pm._strategy.on_position_change.assert_called_once()
    pm._position_gathering.on_execution_report.assert_called_once()
    pm._logging.save_deals.assert_called_once()


def test_fill_for_unknown_instrument_skips_on_execution():
    # No instrument on the event -> on_execution cannot be delivered; dispatch must not crash
    # (downstream skip is handled inside _notify_downstream_fill).
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult(deal=_fill())
    pm.process_event(OrderFilledEvent(instrument=None, client_order_id="cid", venue_order_id="V1", fill=_fill()))
    pm._strategy.on_execution.assert_not_called()
    pm._logging.save_deals.assert_not_called()


def test_callback_exception_does_not_halt_dispatch():
    # _safe_call isolation: a raising on_order must not stop on_execution/on_position_change.
    pm = make_pm()
    pm._strategy.on_order.side_effect = RuntimeError("boom")
    pm._account_manager.apply.return_value = ApplyResult(
        order=MagicMock(), order_change=OrderChange.FILLED, deal=_fill(), position=MagicMock()
    )
    pm.process_event(OrderFilledEvent(instrument=MagicMock(), client_order_id="cid", venue_order_id="V1", fill=_fill()))
    pm._strategy.on_order.assert_called_once()
    pm._strategy.on_execution.assert_called_once()
    pm._strategy.on_position_change.assert_called_once()


class _FakeEmitter:
    """Minimal IMetricEmitter stand-in that records emit() calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, dict | None]] = []

    def emit(self, name, value, tags=None, timestamp=None, instrument=None) -> None:
        self.calls.append((name, value, tags))

    def emit_deals(self, time, instrument, deals, account) -> None: ...


def test_am_apply_error_swallowed_and_emits_error_metric():
    # AM.apply raising is logged and swallowed: no callback fires on a failed apply,
    # and the error counter is emitted with the event type as a tag.
    pm = make_pm()
    emitter = _FakeEmitter()
    pm._context.emitter = emitter
    pm._account_manager.apply.side_effect = RuntimeError("kaboom")
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    pm._account_manager.apply.assert_called_once()
    assert pm._strategy.mock_calls == []
    matches = [c for c in emitter.calls if c[0] == "account_manager_apply_errors"]
    assert len(matches) == 1
    name, value, tags = matches[0]
    assert value == 1.0
    assert tags == {"event": "OrderAcceptedEvent"}


def test_strategy_callback_exception_emits_error_metric():
    pm = make_pm()
    emitter = _FakeEmitter()
    pm._context.emitter = emitter
    pm._strategy.on_order.side_effect = RuntimeError("boom")
    pm._strategy.on_order.__name__ = "on_order"
    pm._account_manager.apply.return_value = ApplyResult(order=MagicMock(), order_change=OrderChange.ACCEPTED)
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    matches = [c for c in emitter.calls if c[0] == "strategy_callback_errors"]
    assert len(matches) == 1
    name, value, tags = matches[0]
    assert value == 1.0
    assert tags == {"callback": "on_order"}


def test_cancel_rejected_logs_warning_in_dispatch():
    # The venue-rejection warning lives in the dispatch (keyed off result.order_change),
    # so it fires regardless of whether the strategy reacts in on_order.
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult(
        order=MagicMock(client_order_id="C-1"), order_change=OrderChange.CANCEL_REJECTED
    )
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        pm.process_event(OrderCancelRejectedEvent(instrument=MagicMock(), client_order_id="C-1", reason="nope"))
    finally:
        logger.remove(sink)
    assert any(m.record["level"].name == "WARNING" and "C-1" in m for m in messages)
    pm._strategy.on_order.assert_called_once()
    assert pm._strategy.on_order.call_args.args[2] is OrderChange.CANCEL_REJECTED


def test_update_rejected_logs_warning_in_dispatch():
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult(
        order=MagicMock(client_order_id="C-2"), order_change=OrderChange.UPDATE_REJECTED
    )
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        pm.process_event(OrderUpdateRejectedEvent(instrument=MagicMock(), client_order_id="C-2", reason="nope"))
    finally:
        logger.remove(sink)
    assert any(m.record["level"].name == "WARNING" and "C-2" in m for m in messages)
    pm._strategy.on_order.assert_called_once()
    assert pm._strategy.on_order.call_args.args[2] is OrderChange.UPDATE_REJECTED


def test_suppressed_reject_logs_no_warning():
    # A cancel-reject the AM suppressed (unknown order / not PENDING_CANCEL) must not
    # produce the still-alive warning — there is no live order to warn about.
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult()
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        pm.process_event(OrderCancelRejectedEvent(instrument=MagicMock(), client_order_id="C-3", reason="nope"))
    finally:
        logger.remove(sink)
    assert not any("C-3" in m for m in messages)
    pm._strategy.on_order.assert_not_called()


def _funding_event() -> FundingPaymentEvent:
    payment = FundingPayment(time=1736294400_000_000_000, funding_rate=0.0001, funding_interval_hours=8)
    return FundingPaymentEvent(instrument=MagicMock(), payment=payment)


def _snapshot_event() -> AccountSnapshotEvent:
    return AccountSnapshotEvent(
        instrument=None, snapshot=AccountSnapshot(exchange="BINANCE.UM", as_of=np.datetime64("now"))
    )


def test_applied_funding_fires_on_position_change_once():
    # No dedicated funding callback: funding routes through on_position_change.
    pm = make_pm()
    position = MagicMock()
    pm._account_manager.apply.return_value = ApplyResult(position=position)
    pm.process_event(_funding_event())
    pm._strategy.on_position_change.assert_called_once()
    assert pm._strategy.on_position_change.call_args.args[1] is position
    pm._strategy.on_order.assert_not_called()
    pm._strategy.on_execution.assert_not_called()


def test_suppressed_funding_fires_no_callback():
    # Bucket-deduped (or skipped: no position / NaN mark) funding returns an empty
    # ApplyResult — the strategy must not see the payment twice.
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult()
    pm.process_event(_funding_event())
    pm._account_manager.apply.assert_called_once()
    assert pm._strategy.mock_calls == []


def test_balance_update_fires_no_strategy_callback():
    # Deliberate contract: there is NO balance callback — balances are read via ctx.
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult()
    pm.process_event(BalanceUpdateEvent(instrument=None, balance=MagicMock(), as_of=np.datetime64("now")))
    pm._account_manager.apply.assert_called_once()
    assert pm._strategy.mock_calls == []


def test_applied_balance_push_fires_no_strategy_callback():
    # The balance-silent contract holds for APPLIED pushes too (ApplyResult.balance set),
    # not just suppressed ones — the field is internal/diff visibility only.
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult(balance=MagicMock())
    pm.process_event(BalanceUpdateEvent(instrument=None, balance=MagicMock(), as_of=np.datetime64("now")))
    pm._account_manager.apply.assert_called_once()
    assert pm._strategy.mock_calls == []


def test_stale_snapshot_is_suppressed():
    # A snapshot rejected by the as_of ratchet returns an empty ApplyResult: nothing
    # was applied, so no callback fires.
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult()
    pm.process_event(_snapshot_event())
    pm._account_manager.apply.assert_called_once()
    assert pm._strategy.mock_calls == []


def test_applied_snapshot_fires_on_position_change_per_corrected_position():
    pm = make_pm()
    p1, p2 = MagicMock(), MagicMock()
    pm._account_manager.apply.return_value = ApplyResult(reconcile_diff=ReconcileDiff(positions=[p1, p2]))
    pm.process_event(_snapshot_event())
    assert pm._strategy.on_position_change.call_count == 2
    assert [c.args[1] for c in pm._strategy.on_position_change.call_args_list] == [p1, p2]
    pm._strategy.on_order.assert_not_called()
    pm._strategy.on_execution.assert_not_called()


def test_snapshot_with_order_only_corrections_is_callback_silent():
    # Reconcile order corrections carry no strategy callback (pending on_reconcile_complete).
    pm = make_pm()
    diff = ReconcileDiff(terminated=[MagicMock()], materialized=[MagicMock()], updated=[MagicMock()])
    pm._account_manager.apply.return_value = ApplyResult(reconcile_diff=diff)
    pm.process_event(_snapshot_event())
    assert pm._strategy.mock_calls == []


def test_reconcile_terminations_emit_counters():
    # F8 ops counters: the PM emits reconcile_orders_terminated/materialized off the
    # ApplyResult diff (the AM holds no emitter — the metric seam lives on the PM).
    pm = make_pm()
    emitter = _FakeEmitter()
    pm._context.emitter = emitter
    diff = ReconcileDiff(
        terminated=[MagicMock(client_order_id="c1"), MagicMock(client_order_id="c2")],
        materialized=[MagicMock(client_order_id="e1")],
    )
    pm._account_manager.apply.return_value = ApplyResult(reconcile_diff=diff)
    pm.process_event(_snapshot_event())
    by_name = {name: value for name, value, _ in emitter.calls}
    assert by_name["reconcile_orders_terminated"] == 2.0
    assert by_name["reconcile_orders_materialized"] == 1.0


def test_clean_reconcile_emits_no_counters():
    pm = make_pm()
    emitter = _FakeEmitter()
    pm._context.emitter = emitter
    pm._account_manager.apply.return_value = ApplyResult(reconcile_diff=ReconcileDiff())
    pm.process_event(_snapshot_event())
    assert not any(name.startswith("reconcile_orders") for name, _, _ in emitter.calls)


class _RemovedOrderCallbackStrategy(IStrategy):
    def on_order_update(self, ctx, order, event) -> None: ...


class _RemovedDealsCallbackStrategy(IStrategy):
    def on_deals(self, ctx, instrument, deals) -> None: ...


class _RemovedAccountCallbackStrategy(IStrategy):
    def on_account_update(self, ctx, event) -> None: ...


class _CurrentCallbacksStrategy(IStrategy):
    def on_order(self, ctx, order, change) -> None: ...

    def on_execution(self, ctx, instrument, deal) -> None: ...

    def on_position_change(self, ctx, position) -> None: ...


def test_removed_on_order_update_raises_with_migration_message():
    with pytest.raises(TypeError, match=r"on_order_update.*replaced by on_order/on_execution/on_position_change"):
        validate_account_callback_signatures(_RemovedOrderCallbackStrategy())


def test_removed_on_deals_raises_with_migration_message():
    # R12: on_deals existed on IStrategy at merge-base — a migrating strategy overriding it
    # would otherwise construct cleanly and silently never receive fills.
    with pytest.raises(TypeError, match=r"on_deals.*replaced by"):
        validate_account_callback_signatures(_RemovedDealsCallbackStrategy())


def test_removed_on_account_update_raises_with_migration_message():
    with pytest.raises(TypeError, match=r"on_account_update.*replaced by"):
        validate_account_callback_signatures(_RemovedAccountCallbackStrategy())


def test_current_callback_signatures_pass():
    validate_account_callback_signatures(_CurrentCallbacksStrategy())


def test_stale_arity_on_order_raises():
    class _TwoArg(IStrategy):
        def on_order(self, ctx, order) -> None: ...

    with pytest.raises(TypeError, match=r"on_order.*on_order\(self, ctx, order, change\)"):
        validate_account_callback_signatures(_TwoArg())


def test_stale_arity_on_position_change_raises():
    class _Extra(IStrategy):
        def on_position_change(self, ctx, position, extra) -> None: ...

    with pytest.raises(TypeError, match=r"on_position_change\(self, ctx, position\)"):
        validate_account_callback_signatures(_Extra())


def test_non_overriding_strategy_passes():
    class _Plain(IStrategy): ...

    validate_account_callback_signatures(_Plain())


def test_var_positional_override_passes():
    class _Wildcard(IStrategy):
        def on_order(self, ctx, *args) -> None: ...

    validate_account_callback_signatures(_Wildcard())


def test_extra_defaulted_param_passes():
    class _Extra(IStrategy):
        def on_order(self, ctx, order, change, extra=None) -> None: ...

    validate_account_callback_signatures(_Extra())


@pytest.mark.parametrize(
    "strategy_cls, removed_name",
    [(_RemovedOrderCallbackStrategy, "on_order_update"), (_RemovedDealsCallbackStrategy, "on_deals")],
)
def test_removed_callback_fails_loudly_at_pm_construction(strategy_cls, removed_name):
    # the guard runs first in __init__, so all-mock collaborators never get touched
    with pytest.raises(TypeError, match=removed_name):
        ProcessingManager(
            context=MagicMock(),
            strategy=strategy_cls(),
            logging=MagicMock(),
            market_data=MagicMock(),
            subscription_manager=MagicMock(),
            time_provider=MagicMock(),
            account_manager=MagicMock(),
            connectors={},
            position_tracker=MagicMock(),
            position_gathering=MagicMock(),
            universe_manager=MagicMock(),
            scheduler=MagicMock(),
            is_simulation=True,
            health_monitor=MagicMock(),
            delisting_detector=MagicMock(),
        )


def test_on_warmup_finished_runs_synchronously_on_processing_thread():
    # F12/D3 regression: on_warmup_finished executes inline on the processing thread in live
    # mode (no thread pool), so ctx.trade()/account access inside it cannot race the reducer,
    # and account events arriving during the callback are dispatched only after it returns.
    pm = make_pm()
    pm._is_simulation = False
    pm._health_monitor = MagicMock()
    pm._subscription_manager = MagicMock()
    pm._strategy_name = "TestStrategy"
    pm._context.instruments = []

    calls: list[str] = []
    callback_thread: list[int] = []

    def _warmup(ctx):
        ctx.trade(MagicMock(), 1.0)
        callback_thread.append(threading.get_ident())
        calls.append("on_warmup_finished")

    pm._strategy.on_warmup_finished.side_effect = _warmup
    pm._strategy.on_order.side_effect = lambda *a: calls.append("on_order")
    pm._account_manager.apply.return_value = ApplyResult(order=MagicMock(), order_change=OrderChange.ACCEPTED)

    pm._handle_warmup_finished()

    assert callback_thread == [threading.get_ident()]
    assert pm._context._strategy_state.is_on_warmup_finished_called is True
    pm._context.trade.assert_called_once()

    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    assert calls == ["on_warmup_finished", "on_order"]


def test_raising_save_deals_does_not_suppress_position_callback():
    # R40: a persistent deals-writer failure (I/O-backed) must not break callback delivery —
    # the remaining downstream consumers and on_position_change still fire, and the error
    # is logged + counted.
    pm = make_pm()
    emitter = _FakeEmitter()
    pm._context.emitter = emitter
    pm._logging.save_deals.side_effect = RuntimeError("disk full")
    pm._account_manager.apply.return_value = ApplyResult(deal=_fill(), position=MagicMock())
    pm.process_event(OrderFilledEvent(instrument=MagicMock(), client_order_id="cid", venue_order_id="V1", fill=_fill()))
    pm._strategy.on_execution.assert_called_once()
    pm._strategy.on_position_change.assert_called_once()
    pm._position_gathering.on_execution_report.assert_called_once()
    pm._universe_manager.on_alter_position.assert_called_once()
    assert ("downstream_fill_errors", 1.0, {"consumer": "save_deals"}) in emitter.calls


def test_raising_gatherer_does_not_suppress_tracker_notification():
    # R40 isolation between fill consumers: a persistently raising gatherer must not
    # suppress the tracker — risk-controller stop cancellation rides on its
    # on_execution_report — and the failure is counted per consumer.
    pm = make_pm()
    emitter = _FakeEmitter()
    pm._context.emitter = emitter
    pm._position_gathering.on_execution_report.side_effect = RuntimeError("gatherer boom")
    pm._account_manager.apply.return_value = ApplyResult(deal=_fill(), position=MagicMock())
    pm.process_event(OrderFilledEvent(instrument=MagicMock(), client_order_id="cid", venue_order_id="V1", fill=_fill()))
    pm._position_tracker.on_execution_report.assert_called_once()
    pm._strategy.on_position_change.assert_called_once()
    assert ("downstream_fill_errors", 1.0, {"consumer": "gatherer"}) in emitter.calls


def test_raising_on_alter_position_does_not_suppress_position_callback():
    pm = make_pm()
    emitter = _FakeEmitter()
    pm._context.emitter = emitter
    pm._universe_manager.on_alter_position.side_effect = RuntimeError("boom")
    pm._account_manager.apply.return_value = ApplyResult(deal=_fill(), position=MagicMock())
    pm.process_event(OrderFilledEvent(instrument=MagicMock(), client_order_id="cid", venue_order_id="V1", fill=_fill()))
    pm._strategy.on_position_change.assert_called_once()
    assert ("downstream_fill_errors", 1.0, {"consumer": "on_alter_position"}) in emitter.calls


def test_fit_flag_resets_when_finalize_raises_and_next_tick_retries():
    # R14: finalize_ohlc_for_instruments raising after _fit_is_running is set must not strand
    # the flag True (which would permanently disable fit/on_event) — the next tick retries.
    pm = make_pm()
    pm._time_provider = MagicMock()
    pm._cache = MagicMock()
    pm._health_monitor = MagicMock()
    pm._subscription_manager = MagicMock()
    pm._strategy_name = "FitRetry"
    pm._emitted_signals = []
    pm._context.instruments = []
    pm._context._strategy_state.is_on_start_called = True
    pm._context._strategy_state.is_on_warmup_finished_called = True
    pm._context._strategy_state.is_on_fit_called = False
    pm._cache.finalize_ohlc_for_instruments.side_effect = RuntimeError("finalize boom")

    with pytest.raises(RuntimeError, match="finalize boom"):
        pm._run_strategy_pipeline(None)

    assert pm._fit_is_running is False
    assert pm._context._strategy_state.is_on_fit_called is False
    pm._strategy.on_fit.assert_not_called()

    pm._cache.finalize_ohlc_for_instruments.side_effect = None
    pm._run_strategy_pipeline(None)
    assert pm._strategy.on_fit.call_count == 1
    assert pm._context._strategy_state.is_on_fit_called is True


def test_warmup_flag_resets_when_invoke_raises_and_next_tick_retries():
    # R14 (warmup shape): a raise outside __invoke_on_warmup_finished's internal except —
    # e.g. the health monitor context — must not strand _warmup_finished_is_running True
    # (which would permanently disable the strategy) — the next tick retries the warmup
    # and on_warmup_finished eventually fires.
    pm = make_pm()
    pm._time_provider = MagicMock()
    pm._health_monitor = MagicMock(side_effect=RuntimeError("monitor boom"))
    pm._subscription_manager = MagicMock()
    pm._strategy_name = "WarmupRetry"
    pm._emitted_signals = []
    pm._context.instruments = []
    pm._context.get_warmup_positions.return_value = []
    pm._context.get_warmup_orders.return_value = []
    pm._context.get_restored_state.return_value = None
    pm._context._strategy_state.is_on_start_called = True
    pm._context._strategy_state.is_warmup_in_progress = False
    pm._context._strategy_state.is_on_warmup_finished_called = False
    pm._context._strategy_state.is_on_fit_called = True

    with pytest.raises(RuntimeError, match="monitor boom"):
        pm._run_strategy_pipeline(None)

    assert pm._warmup_finished_is_running is False
    assert pm._context._strategy_state.is_on_warmup_finished_called is False
    pm._strategy.on_warmup_finished.assert_not_called()

    pm._health_monitor = MagicMock()
    pm._run_strategy_pipeline(None)
    assert pm._strategy.on_warmup_finished.call_count == 1
    assert pm._context._strategy_state.is_on_warmup_finished_called is True
    assert pm._warmup_finished_is_running is False


def test_funding_payment_tuple_reaches_on_market_data():
    # Dual-emit restoration (backtester): the funding-payment TUPLE (the second of the runner's
    # two sends) rides the tuple path with no registered handler -> _process_custom_event returns
    # a MarketEvent that reaches the strategy's on_market_data. Booking is the typed event's job
    # (process_event); this tuple path does NOT book — it only restores the strategy reaction.
    pm = make_pm()
    pm._time_provider = MagicMock()
    pm._health_monitor = MagicMock()
    pm._subscription_manager = MagicMock()
    pm._cache = MagicMock()
    pm._strategy_name = "FundingReact"
    pm._emitted_signals = []
    pm._data_throttler = None
    # the real handler registry is built from _handle_* methods on the class; there is no
    # _handle_funding_payment, so a funding tuple correctly falls through to _process_custom_event.
    pm._handlers = {
        n.split("_handle_")[1]: f for n, f in ProcessingManager.__dict__.items() if n.startswith("_handle_")
    }
    pm._custom_scheduled_methods = {}
    pm._pending_no_quote_signals = {}
    pm._fit_is_running = False
    pm._warmup_finished_is_running = False
    # funding is not the base subscription -> tracker produces no targets on this tick
    pm._position_tracker.update.return_value = []
    pm._context.instruments = []
    pm._context._strategy_state.is_on_start_called = True
    pm._context._strategy_state.is_warmup_in_progress = False
    pm._context._strategy_state.is_on_warmup_finished_called = True
    pm._context._strategy_state.is_on_fit_called = True
    pm._strategy.on_market_data.return_value = None

    payment = FundingPayment(time=0, funding_rate=0.0001, funding_interval_hours=8)
    instrument = MagicMock()

    pm.process_data(instrument, DataType.FUNDING_PAYMENT, payment, is_historical=False)

    pm._strategy.on_market_data.assert_called_once()
    event = pm._strategy.on_market_data.call_args.args[1]
    assert isinstance(event, MarketEvent)
    assert event.type == "funding_payment"
    assert event.data is payment
    assert event.instrument is instrument
    # the tuple path must NOT book funding — booking is the typed FundingPaymentEvent's job
    pm._account_manager.apply.assert_not_called()


def test_on_fit_pumping_events_does_not_retrigger_fit():
    # Hang regression: _fit_is_running is a same-thread re-entrancy guard. on_fit runs inline
    # on the processing thread, and a strategy that pumps events from on_fit (e.g.
    # ctx.set_universe feeding data back through the tuple tail) re-enters
    # _run_strategy_pipeline BEFORE is_on_fit_called is set — without the guard that
    # re-triggers _handle_fit and recurses into on_fit again.
    pm = make_pm()
    pm._time_provider = MagicMock()
    pm._cache = MagicMock()
    pm._health_monitor = MagicMock()
    pm._subscription_manager = MagicMock()
    pm._strategy_name = "ReentrantFit"
    pm._emitted_signals = []
    pm._context.instruments = []  # empty universe: _is_data_ready() short-circuits True
    pm._context._strategy_state.is_on_start_called = True
    pm._context._strategy_state.is_on_warmup_finished_called = True
    pm._context._strategy_state.is_on_fit_called = False

    # an event pumped from inside on_fit lands back in the shared tuple-path tail
    pm._strategy.on_fit.side_effect = lambda ctx: pm._run_strategy_pipeline(None)

    pm._run_strategy_pipeline(None)

    assert pm._strategy.on_fit.call_count == 1
    assert pm._context._strategy_state.is_on_fit_called is True
    assert pm._fit_is_running is False


# --- gatherer order/position callbacks (framework-level propagation to every gatherer) ----- #


def test_filled_event_fires_gatherer_on_order_and_position_change():
    # The gatherer gets the same ApplyResult-driven callbacks as the strategy: a fill fires
    # gatherer.on_order(order, change), on_position_change(position), AND on_execution_report.
    pm = make_pm()
    fill = _fill()
    order, position, instrument = MagicMock(), MagicMock(), MagicMock()
    pm._account_manager.apply.return_value = ApplyResult(
        order=order, order_change=OrderChange.FILLED, deal=fill, position=position
    )
    pm.process_event(OrderFilledEvent(instrument=instrument, client_order_id="cid", venue_order_id="V1", fill=fill))
    pm._position_gathering.on_order.assert_called_once()
    assert pm._position_gathering.on_order.call_args.args[1] is order
    assert pm._position_gathering.on_order.call_args.args[2] is OrderChange.FILLED
    pm._position_gathering.on_position_change.assert_called_once()
    assert pm._position_gathering.on_position_change.call_args.args[1] is position
    pm._position_gathering.on_execution_report.assert_called_once()


def test_accepted_fires_gatherer_on_order_only():
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult(order=MagicMock(), order_change=OrderChange.ACCEPTED)
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    pm._position_gathering.on_order.assert_called_once()
    assert pm._position_gathering.on_order.call_args.args[2] is OrderChange.ACCEPTED
    pm._position_gathering.on_position_change.assert_not_called()


def test_cancel_rejected_reaches_gatherer_on_order():
    # The dangerous-but-recoverable case the gatherer must see to retry its cancel.
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult(
        order=MagicMock(client_order_id="C-1"), order_change=OrderChange.CANCEL_REJECTED
    )
    pm.process_event(OrderCancelRejectedEvent(instrument=MagicMock(), client_order_id="C-1", reason="nope"))
    pm._position_gathering.on_order.assert_called_once()
    assert pm._position_gathering.on_order.call_args.args[2] is OrderChange.CANCEL_REJECTED


def test_snapshot_corrections_fire_gatherer_on_position_change_per_position():
    pm = make_pm()
    p1, p2 = MagicMock(), MagicMock()
    pm._account_manager.apply.return_value = ApplyResult(reconcile_diff=ReconcileDiff(positions=[p1, p2]))
    pm.process_event(_snapshot_event())
    assert pm._position_gathering.on_position_change.call_count == 2
    assert [c.args[1] for c in pm._position_gathering.on_position_change.call_args_list] == [p1, p2]


def test_suppressed_result_fires_no_gatherer_callback():
    pm = make_pm()
    pm._account_manager.apply.return_value = ApplyResult()
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    pm._position_gathering.on_order.assert_not_called()
    pm._position_gathering.on_position_change.assert_not_called()


def test_raising_gatherer_on_order_does_not_halt_dispatch():
    # gatherer error-isolation: a raising gatherer.on_order must not stop the strategy
    # callbacks or the gatherer's position fire, and is counted under strategy_callback_errors.
    pm = make_pm()
    emitter = _FakeEmitter()
    pm._context.emitter = emitter
    pm._position_gathering.on_order.side_effect = RuntimeError("gatherer on_order boom")
    pm._account_manager.apply.return_value = ApplyResult(
        order=MagicMock(), order_change=OrderChange.FILLED, deal=_fill(), position=MagicMock()
    )
    pm.process_event(OrderFilledEvent(instrument=MagicMock(), client_order_id="cid", venue_order_id="V1", fill=_fill()))
    pm._strategy.on_order.assert_called_once()
    pm._strategy.on_position_change.assert_called_once()
    pm._position_gathering.on_position_change.assert_called_once()
    assert any(name == "strategy_callback_errors" for name, _, _ in emitter.calls)
