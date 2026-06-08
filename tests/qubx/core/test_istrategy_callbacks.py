from unittest.mock import MagicMock

from qubx.core.interfaces import IStrategy


def test_istrategy_has_unified_callbacks():
    # The whole order + account lifecycle collapses into two general callbacks (plus
    # on_market_data); the per-event callbacks (on_order_filled, on_position_update, ...) are gone.
    actual = {n for n in dir(IStrategy) if n.startswith("on_")}
    assert {"on_market_data", "on_order_update", "on_account_update"} <= actual

    removed = {
        "on_quote", "on_trade", "on_orderbook",
        "on_order_accepted", "on_order_rejected", "on_order_partially_filled",
        "on_order_filled", "on_order_canceled", "on_order_expired", "on_order_updated",
        "on_order_cancel_rejected", "on_order_update_rejected",
        "on_position_update", "on_balance_update", "on_funding_payment",
        "on_reconcile_complete",
    }
    assert not (removed & actual), f"stale callbacks still present: {removed & actual}"


def test_unified_callback_defaults_are_silent_noops():
    # Base-class defaults must be no-ops that don't raise (strategies override what they need).
    strategy = IStrategy()
    strategy.on_order_update(MagicMock(), MagicMock(), MagicMock())
    strategy.on_account_update(MagicMock(), MagicMock())
