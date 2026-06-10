import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx import logger
from qubx.core.account_manager.reconcile import ReconcileDiff
from qubx.core.account_manager.reducer import ApplyResult
from qubx.core.basics import Deal, FundingPayment, OrderChange
from qubx.core.events import (
    AccountSnapshot,
    AccountSnapshotEvent,
    BalanceUpdateEvent,
    FundingPaymentEvent,
    OrderAcceptedEvent,
    OrderCancelRejectedEvent,
    OrderFilledEvent,
    OrderUpdateRejectedEvent,
    PositionUpdateEvent,
)
from qubx.core.interfaces import IStrategy
from qubx.core.mixins.processing import ProcessingManager, validate_account_callback_signatures


def _pm() -> ProcessingManager:
    pm = ProcessingManager.__new__(ProcessingManager)
    pm._is_simulation = True  # not paper: keeps _feed_simulated_connector a no-op here
    pm._strategy = MagicMock()
    pm._account_manager = MagicMock()
    pm._context = MagicMock()
    pm._context.emitter = None
    pm._position_gathering = MagicMock()
    pm._exporter = None
    pm._universe_manager = MagicMock()
    pm._logging = MagicMock()
    pm._market_data = MagicMock()
    pm._position_tracker = MagicMock()
    pm._instruments_in_init_stage = set()
    pm._init_stage_position_tracker = MagicMock()
    pm._active_targets = {}
    return pm


def _fill() -> Deal:
    return Deal(trade_id="t1", order_id="V1", time=np.datetime64("now"), amount=1.0, price=50_000.0, aggressive=True)


def test_filled_event_routes_through_am_then_strategy():
    pm = _pm()
    fill = _fill()
    order = MagicMock()
    pm._account_manager.apply.return_value = ApplyResult(
        order=order, order_change=OrderChange.FILLED, deal=fill, position=MagicMock()
    )
    pm.process_event(OrderFilledEvent(instrument=MagicMock(), client_order_id="cid", venue_order_id="V1", fill=fill))
    pm._account_manager.apply.assert_called_once()
    # unified callback receives (ctx, order, event)
    pm._strategy.on_order_update.assert_called_once()
    assert pm._strategy.on_order_update.call_args.args[1] is order
    assert isinstance(pm._strategy.on_order_update.call_args.args[2], OrderFilledEvent)
    # downstream per-fill notification ran with the AM-confirmed deal
    pm._position_gathering.on_execution_report.assert_called_once()
    pm._logging.save_deals.assert_called_once()


def test_accepted_event_routes_through_am_then_strategy():
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult(order=MagicMock(), order_change=OrderChange.ACCEPTED)
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    pm._strategy.on_order_update.assert_called_once()


def test_suppressed_result_fires_no_callbacks():
    # None-as-suppress: an empty ApplyResult (late/duplicate/terminal/unknown event)
    # must fire neither the strategy callback nor any downstream fill consumer.
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult()
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    pm._account_manager.apply.assert_called_once()
    pm._strategy.on_order_update.assert_not_called()
    pm._position_gathering.on_execution_report.assert_not_called()
    pm._logging.save_deals.assert_not_called()


def test_deduped_fill_skips_downstream_delivery():
    # A venue re-sends an order report whose embedded deal the AM already applied: it reports
    # the status change (result.order set) but suppresses the deal (result.deal None), so
    # save_deals/gatherer/trackers must NOT receive the same Deal twice.
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult(order=MagicMock(), order_change=OrderChange.FILLED)
    pm.process_event(OrderFilledEvent(instrument=MagicMock(), client_order_id="cid", venue_order_id="V1", fill=_fill()))
    pm._strategy.on_order_update.assert_called_once()
    pm._position_gathering.on_execution_report.assert_not_called()
    pm._logging.save_deals.assert_not_called()


def test_execution_only_result_notifies_downstream_without_strategy_callback():
    # A fill with no status change (subsequent partial, fill while PENDING_*) carries a
    # deal but no order: downstream consumers run, on_order_update does not fire.
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult(deal=_fill(), position=MagicMock())
    pm.process_event(OrderFilledEvent(instrument=MagicMock(), client_order_id="cid", venue_order_id="V1", fill=_fill()))
    pm._strategy.on_order_update.assert_not_called()
    pm._position_gathering.on_execution_report.assert_called_once()
    pm._logging.save_deals.assert_called_once()


def test_callback_exception_does_not_halt_dispatch():
    pm = _pm()
    pm._strategy.on_order_update.side_effect = RuntimeError("boom")
    pm._account_manager.apply.return_value = ApplyResult(order=MagicMock(), order_change=OrderChange.ACCEPTED)
    # the bad callback is swallowed (logged, not raised); reaching here proves dispatch survived
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    pm._strategy.on_order_update.assert_called_once()


def test_am_apply_error_is_swallowed_and_does_not_raise():
    pm = _pm()
    pm._account_manager.apply.side_effect = RuntimeError("kaboom")
    # AM.apply raising is logged and swallowed; the callback must not fire on a failed apply
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    pm._account_manager.apply.assert_called_once()
    pm._strategy.on_order_update.assert_not_called()


class _FakeEmitter:
    """Minimal IMetricEmitter stand-in that records emit() calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, dict | None]] = []

    def emit(self, name, value, tags=None, timestamp=None, instrument=None) -> None:
        self.calls.append((name, value, tags))


def test_am_apply_error_emits_error_metric():
    pm = _pm()
    emitter = _FakeEmitter()
    pm._context.emitter = emitter
    pm._account_manager.apply.side_effect = RuntimeError("kaboom")
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    matches = [c for c in emitter.calls if c[0] == "account_manager_apply_errors"]
    assert len(matches) == 1
    name, value, tags = matches[0]
    assert value == 1.0
    assert tags == {"event": "OrderAcceptedEvent"}


def test_strategy_callback_exception_emits_error_metric():
    pm = _pm()
    emitter = _FakeEmitter()
    pm._context.emitter = emitter
    pm._strategy.on_order_update.side_effect = RuntimeError("boom")
    pm._strategy.on_order_update.__name__ = "on_order_update"
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
    assert tags == {"callback": "on_order_update"}


def test_cancel_rejected_logs_warning_in_dispatch():
    # The venue-rejection warning lives in the dispatch (keyed off result.order_change),
    # so it fires regardless of whether the strategy reacts in on_order_update.
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult(
        order=MagicMock(client_order_id="C-1"), order_change=OrderChange.CANCEL_REJECTED
    )
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        pm.process_event(OrderCancelRejectedEvent(instrument=MagicMock(), client_order_id="C-1", reason="nope"))
    finally:
        logger.remove(sink)
    assert any("STILL ALIVE at the venue" in m for m in messages)
    pm._strategy.on_order_update.assert_called_once()
    assert isinstance(pm._strategy.on_order_update.call_args.args[2], OrderCancelRejectedEvent)


def test_update_rejected_logs_warning_in_dispatch():
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult(
        order=MagicMock(client_order_id="C-2"), order_change=OrderChange.UPDATE_REJECTED
    )
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        pm.process_event(OrderUpdateRejectedEvent(instrument=MagicMock(), client_order_id="C-2", reason="nope"))
    finally:
        logger.remove(sink)
    assert any("STILL ALIVE with prior parameters" in m for m in messages)
    pm._strategy.on_order_update.assert_called_once()
    assert isinstance(pm._strategy.on_order_update.call_args.args[2], OrderUpdateRejectedEvent)


def test_suppressed_reject_logs_no_warning():
    # A cancel-reject the AM suppressed (unknown order / not PENDING_CANCEL) must not
    # produce the STILL-ALIVE warning — there is no live order to warn about.
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult()
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        pm.process_event(OrderCancelRejectedEvent(instrument=MagicMock(), client_order_id="C-3", reason="nope"))
    finally:
        logger.remove(sink)
    assert not any("STILL ALIVE" in m for m in messages)
    pm._strategy.on_order_update.assert_not_called()


def _funding_event() -> FundingPaymentEvent:
    payment = FundingPayment(time=1736294400_000_000_000, funding_rate=0.0001, funding_interval_hours=8)
    return FundingPaymentEvent(instrument=MagicMock(), payment=payment)


def _snapshot_event() -> AccountSnapshotEvent:
    return AccountSnapshotEvent(
        instrument=None, snapshot=AccountSnapshot(exchange="BINANCE.UM", as_of=np.datetime64("now"))
    )


def test_applied_funding_fires_on_account_update_once():
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult(position=MagicMock())
    event = _funding_event()
    pm.process_event(event)
    pm._strategy.on_account_update.assert_called_once()
    assert pm._strategy.on_account_update.call_args.args[1] is event
    pm._strategy.on_order_update.assert_not_called()


def test_suppressed_funding_fires_no_callback():
    # Bucket-deduped (or skipped: no position / NaN mark) funding returns an empty
    # ApplyResult — the strategy must not see the payment twice.
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult()
    pm.process_event(_funding_event())
    pm._account_manager.apply.assert_called_once()
    pm._strategy.on_account_update.assert_not_called()


@pytest.mark.parametrize(
    "event",
    [
        PositionUpdateEvent(instrument=MagicMock(), position=MagicMock()),
        BalanceUpdateEvent(instrument=None, balance=MagicMock()),
    ],
    ids=["position", "balance"],
)
def test_position_balance_updates_fire_through_on_empty_result(event):
    # The documented fire-through pair: the reducer is a no-op for these until WS
    # application lands, but the venue push still reaches the strategy.
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult()
    pm.process_event(event)
    pm._strategy.on_account_update.assert_called_once()
    assert pm._strategy.on_account_update.call_args.args[1] is event


def test_stale_snapshot_is_suppressed():
    # A snapshot rejected by the as_of ratchet returns an empty ApplyResult: nothing
    # was applied, so on_account_update must not fire.
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult()
    pm.process_event(_snapshot_event())
    pm._account_manager.apply.assert_called_once()
    pm._strategy.on_account_update.assert_not_called()


def test_applied_snapshot_fires_on_account_update():
    pm = _pm()
    pm._account_manager.apply.return_value = ApplyResult(reconcile_diff=ReconcileDiff())
    event = _snapshot_event()
    pm.process_event(event)
    pm._strategy.on_account_update.assert_called_once()
    assert pm._strategy.on_account_update.call_args.args[1] is event
    pm._strategy.on_order_update.assert_not_called()


def test_reconcile_terminations_emit_counters():
    # F8 ops counters: the PM emits reconcile_orders_terminated/materialized off the
    # ApplyResult diff (the AM holds no emitter — the metric seam lives on the PM).
    pm = _pm()
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
    pm = _pm()
    emitter = _FakeEmitter()
    pm._context.emitter = emitter
    pm._account_manager.apply.return_value = ApplyResult(reconcile_diff=ReconcileDiff())
    pm.process_event(_snapshot_event())
    assert not any(name.startswith("reconcile_orders") for name, _, _ in emitter.calls)


class _LegacyOrderCallbackStrategy(IStrategy):
    def on_order_update(self, ctx, order) -> None: ...


class _LegacyAccountCallbackStrategy(IStrategy):
    def on_account_update(self, ctx) -> None: ...


class _CurrentCallbacksStrategy(IStrategy):
    def on_order_update(self, ctx, order, event) -> None: ...

    def on_account_update(self, ctx, event) -> None: ...


def test_legacy_two_arg_on_order_update_raises():
    with pytest.raises(TypeError, match=r"on_order_update.*on_order_update\(self, ctx, order, event\)"):
        validate_account_callback_signatures(_LegacyOrderCallbackStrategy())


def test_legacy_on_account_update_raises():
    with pytest.raises(TypeError, match=r"on_account_update\(self, ctx, event\)"):
        validate_account_callback_signatures(_LegacyAccountCallbackStrategy())


def test_current_callback_signatures_pass():
    validate_account_callback_signatures(_CurrentCallbacksStrategy())


def test_non_overriding_strategy_passes():
    class _Plain(IStrategy): ...

    validate_account_callback_signatures(_Plain())


def test_var_positional_override_passes():
    class _Wildcard(IStrategy):
        def on_order_update(self, ctx, *args) -> None: ...

    validate_account_callback_signatures(_Wildcard())


def test_extra_defaulted_param_passes():
    class _Extra(IStrategy):
        def on_order_update(self, ctx, order, event, extra=None) -> None: ...

    validate_account_callback_signatures(_Extra())


def test_legacy_override_fails_loudly_at_pm_construction():
    # the guard runs first in __init__, so all-mock collaborators never get touched
    with pytest.raises(TypeError, match="on_order_update"):
        ProcessingManager(
            context=MagicMock(),
            strategy=_LegacyOrderCallbackStrategy(),
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
    pm = _pm()
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
    pm._strategy.on_order_update.side_effect = lambda *a: calls.append("on_order_update")
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
    assert calls == ["on_warmup_finished", "on_order_update"]
