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


def test_rejected_callback_defaults_are_silent_noops():
    # The venue-rejection warning is logged by ProcessingManager's dispatch, not by these
    # interface defaults — a default body would be silently lost on override-without-super.
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        strategy = IStrategy()
        strategy.on_order_cancel_rejected(MagicMock(), MagicMock(client_order_id="C-1"), "venue says no")
        strategy.on_order_update_rejected(MagicMock(), MagicMock(client_order_id="C-2"), "venue says no")
    finally:
        logger.remove(sink_id)
    assert messages == []
