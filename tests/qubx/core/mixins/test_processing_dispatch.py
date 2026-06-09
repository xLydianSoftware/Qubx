from unittest.mock import MagicMock

import numpy as np

from qubx import logger
from qubx.core.account_manager.reducer import ApplyResult
from qubx.core.basics import Deal, OrderChange
from qubx.core.events import (
    OrderAcceptedEvent,
    OrderCancelRejectedEvent,
    OrderFilledEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.mixins.processing import ProcessingManager


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
