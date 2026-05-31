from unittest.mock import MagicMock

import numpy as np

from qubx.core.basics import Deal
from qubx.core.events import (
    OrderAcceptedEvent,
    OrderFilledEvent,
    QuoteEvent,
)


def _pm():
    from qubx.core.mixins.processing import ProcessingManager

    pm = ProcessingManager.__new__(ProcessingManager)
    pm._strategy = MagicMock()
    pm._account_manager = MagicMock()
    pm._metrics = MagicMock()
    pm._context = MagicMock()
    pm._position_gathering = MagicMock()
    pm._exporter = None
    pm._universe_manager = MagicMock()
    pm._market_data = MagicMock()
    pm._position_tracker = MagicMock()
    pm._init_stage_position_tracker = MagicMock()
    pm._instruments_in_init_stage = set()
    pm._time_provider = MagicMock()
    return pm


def test_quote_event_calls_on_market_quote_then_strategy():
    pm = _pm()
    pm.process_event(QuoteEvent(instrument=None, quote=MagicMock()))
    pm._account_manager.apply.assert_not_called()
    pm._account_manager.on_market_quote.assert_called_once()
    pm._strategy.on_quote.assert_called_once()


def test_filled_event_routes_through_am_then_strategy():
    pm = _pm()
    updated = MagicMock()
    pm._account_manager.apply.return_value = updated
    pm._account_manager.get_position.return_value = MagicMock()
    fill = Deal(
        trade_id="t1",
        order_id="V1",
        time=np.datetime64("now"),
        amount=1.0,
        price=50_000.0,
        aggressive=True,
    )
    pm.process_event(
        OrderFilledEvent(
            instrument=MagicMock(),
            client_order_id="cid",
            venue_order_id="V1",
            fill=fill,
        )
    )
    pm._account_manager.apply.assert_called_once()
    pm._strategy.on_order_filled.assert_called_once()


def test_callback_exception_does_not_halt_dispatch():
    pm = _pm()
    pm._strategy.on_order_accepted.side_effect = RuntimeError("boom")
    pm._account_manager.apply.return_value = MagicMock()
    pm.process_event(
        OrderAcceptedEvent(
            instrument=MagicMock(),
            client_order_id="c",
            venue_order_id="V",
            accepted_at=np.datetime64("now"),
        )
    )
    pm._metrics.inc.assert_called()
