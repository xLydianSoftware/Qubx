from unittest.mock import MagicMock

from qubx import logger
from qubx.core.interfaces import IStrategy


def test_istrategy_has_new_callbacks():
    required = {
        "on_order_accepted", "on_order_rejected",
        "on_order_partially_filled", "on_order_filled",
        "on_order_canceled", "on_order_expired", "on_order_updated",
        "on_order_cancel_rejected", "on_order_update_rejected",
        "on_position_update", "on_balance_update", "on_funding_payment",
        "on_reconcile_complete",
    }
    actual = {n for n in dir(IStrategy) if n.startswith("on_")}
    assert required <= actual, f"missing: {required - actual}"


def test_old_callbacks_retained():
    assert hasattr(IStrategy, "on_order_update")
    assert hasattr(IStrategy, "on_deals")


def test_cancel_rejected_default_warns():
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        strategy = IStrategy()
        order = MagicMock(client_order_id="C-1")
        strategy.on_order_cancel_rejected(MagicMock(), order, "venue says no")
    finally:
        logger.remove(sink_id)
    assert any("STILL ALIVE at the venue" in m for m in messages)


def test_update_rejected_default_warns():
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        strategy = IStrategy()
        order = MagicMock(client_order_id="C-2")
        strategy.on_order_update_rejected(MagicMock(), order, "venue says no")
    finally:
        logger.remove(sink_id)
    assert any("STILL ALIVE with prior parameters" in m for m in messages)
