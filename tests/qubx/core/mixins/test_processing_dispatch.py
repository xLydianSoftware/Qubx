from unittest.mock import MagicMock

import numpy as np

from qubx.core.basics import Deal
from qubx.core.events import OrderAcceptedEvent, OrderFilledEvent, QuoteEvent
from qubx.core.mixins.processing import ProcessingManager, _CounterSink


def _pm() -> ProcessingManager:
    pm = ProcessingManager.__new__(ProcessingManager)
    pm._strategy = MagicMock()
    pm._account_manager = MagicMock()
    pm._metrics = _CounterSink()
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


def test_quote_event_marks_to_market_only():
    # A typed QuoteEvent feeds AM mark-to-market and does NOT touch AM.apply or
    # fire any strategy market-data callback (those stay on on_market_data).
    pm = _pm()
    pm.process_event(QuoteEvent(instrument=MagicMock(), quote=MagicMock()))
    pm._account_manager.apply.assert_not_called()
    pm._account_manager.on_market_quote.assert_called_once()


def test_filled_event_routes_through_am_then_strategy():
    pm = _pm()
    updated = MagicMock()
    pm._account_manager.apply.return_value = updated
    fill = Deal(
        trade_id="t1", order_id="V1", time=np.datetime64("now"), amount=1.0, price=50_000.0, aggressive=True
    )
    pm.process_event(
        OrderFilledEvent(instrument=MagicMock(), client_order_id="cid", venue_order_id="V1", fill=fill)
    )
    pm._account_manager.apply.assert_called_once()
    pm._strategy.on_order_filled.assert_called_once()
    # downstream per-fill notification ran
    pm._position_gathering.on_execution_report.assert_called_once()


def test_accepted_event_routes_through_am_then_strategy():
    pm = _pm()
    pm._account_manager.apply.return_value = MagicMock()
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    pm._strategy.on_order_accepted.assert_called_once()


def test_callback_exception_does_not_halt_dispatch():
    pm = _pm()
    pm._strategy.on_order_accepted.side_effect = RuntimeError("boom")
    pm._account_manager.apply.return_value = MagicMock()
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    # the bad callback is swallowed and recorded as a metric, dispatch survives
    assert pm._metrics.counts.get("strategy_callback_errors", 0) >= 1


def test_am_apply_error_is_recorded_and_does_not_raise():
    pm = _pm()
    pm._account_manager.apply.side_effect = RuntimeError("kaboom")
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(), client_order_id="c", venue_order_id="V", accepted_at=np.datetime64("now")
        )
    )
    assert pm._metrics.counts.get("account_manager_apply_errors", 0) >= 1
    pm._strategy.on_order_accepted.assert_not_called()
