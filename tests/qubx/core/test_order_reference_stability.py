"""Test that Order object references remain stable during updates."""

import numpy as np
import pytest

from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import AssetType, Instrument, MarketType, Order


@pytest.fixture
def btc_instrument():
    """Create a test BTC instrument."""
    return Instrument(
        symbol="BTCUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="TEST",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        min_notional=10.0,
    )


@pytest.fixture
def account_processor():
    """Create a test account processor."""
    from unittest.mock import Mock

    time_provider = Mock()
    time_provider.time = Mock(return_value=np.datetime64(1000000000, "ns"))
    return BasicAccountProcessor(
        account_id="test",
        time_provider=time_provider,
        base_currency="USDT",
        exchange="TEST",
    )


def test_order_reference_preserved_on_update(account_processor, btc_instrument):
    """Test that external references to Order objects remain valid after updates."""
    # Create initial order
    initial_order = Order(
        id="order_123",
        type="LIMIT",
        instrument=btc_instrument,
        time=np.datetime64(1000000000, "ns"),
        quantity=1.0,
        price=50000.0,
        side="BUY",
        status="NEW",
        time_in_force="GTC",
        client_id="client_123",
    )

    # Process initial order
    account_processor.process_order(initial_order)

    # Get reference to the stored order (simulating strategy holding a reference)
    orders = account_processor.get_orders()
    strategy_order_ref = orders["order_123"]
    original_id = id(strategy_order_ref)  # Get object identity

    # Verify initial state
    assert strategy_order_ref.status == "NEW"
    assert strategy_order_ref.price == 50000.0

    # Create order update with status change
    order_update = Order(
        id="order_123",
        type="LIMIT",
        instrument=btc_instrument,
        time=np.datetime64(1000000000, "ns"),
        quantity=1.0,
        price=50000.0,
        side="BUY",
        status="OPEN",  # Changed status
        time_in_force="GTC",
        client_id="client_123",
    )

    # Process update
    account_processor.process_order(order_update)

    # Verify the reference we hold still points to the same object
    assert id(strategy_order_ref) == original_id  # Same object identity!
    assert strategy_order_ref.status == "OPEN"  # But with updated status

    # Also verify the stored order is the same object
    updated_orders = account_processor.get_orders()
    assert id(updated_orders["order_123"]) == original_id


def test_order_reference_preserved_on_partial_update(account_processor, btc_instrument):
    """Test that partial updates (missing fields) preserve existing values."""
    # Create initial order with full details
    initial_order = Order(
        id="order_456",
        type="LIMIT",
        instrument=btc_instrument,
        time=np.datetime64(1000000000, "ns"),
        quantity=2.0,
        price=51000.0,
        side="SELL",
        status="NEW",
        time_in_force="GTC",
        client_id="client_456",
        cost=102000.0,
        options={"reduce_only": False, "post_only": True},
    )

    # Process initial order
    account_processor.process_order(initial_order)

    # Get reference
    orders = account_processor.get_orders()
    strategy_order_ref = orders["order_456"]
    original_id = id(strategy_order_ref)

    # Create minimal update (only status change, other fields missing/zero)
    minimal_update = Order(
        id="order_456",
        type="UNKNOWN",  # Will be ignored
        instrument=btc_instrument,
        time=np.datetime64(0, "ns"),  # Will be ignored
        quantity=0,  # Will be ignored
        price=0,  # Will be ignored
        side="UNKNOWN",  # Will be ignored
        status="OPEN",  # Only this should update
        time_in_force="",  # Will be ignored
        client_id=None,  # Will be ignored
    )

    # Process minimal update
    account_processor.process_order(minimal_update)

    # Verify reference stability and selective updates
    assert id(strategy_order_ref) == original_id  # Same object!
    assert strategy_order_ref.status == "OPEN"  # Updated
    assert strategy_order_ref.quantity == 2.0  # Preserved
    assert strategy_order_ref.price == 51000.0  # Preserved
    assert strategy_order_ref.type == "LIMIT"  # Preserved
    assert strategy_order_ref.side == "SELL"  # Preserved
    assert strategy_order_ref.client_id == "client_456"  # Preserved
    assert strategy_order_ref.cost == 102000.0  # Preserved
    assert strategy_order_ref.options == {"reduce_only": False, "post_only": True}  # Preserved
