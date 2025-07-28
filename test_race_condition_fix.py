#!/usr/bin/env python3
"""
Test script to demonstrate that the subscription race condition is fixed.
"""

from qubx.connectors.ccxt.subscription_manager import SubscriptionManager
from qubx.core.basics import AssetType, DataType, Instrument, MarketType


def test_race_condition_fix():
    """Test that subscription state consistency is maintained during transitions."""
    print("🔍 Testing subscription race condition fix...")

    # Create test instruments
    btc_instrument = Instrument(
        symbol="BTCUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="BINANCE.UM",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.1,
        lot_size=0.001,
        min_size=0.001,
    )
    eth_instrument = Instrument(
        symbol="ETHUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="BINANCE.UM",
        base="ETH",
        quote="USDT",
        settle="USDT",
        exchange_symbol="ETHUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )

    subscription_manager = SubscriptionManager()
    subscription_type = DataType.OHLC["5m"]

    # Test the race condition scenario
    print("📊 Step 1: Add initial subscription")
    subscription_manager.add_subscription(subscription_type, [btc_instrument], reset=True)
    
    # Simulate the subscription becoming active
    subscription_manager.mark_subscription_active(subscription_type)
    
    print("📊 Step 2: Add second instrument (triggers resubscription)")
    subscription_manager.add_subscription(subscription_type, [btc_instrument, eth_instrument], reset=True)
    
    # At this point we have the race condition scenario:
    # - New instruments are in pending 
    # - Old instruments might still be in active (before cleanup)
    # - Connection is marked as not ready
    
    print("📊 Step 3: Check state consistency BEFORE marking active")
    subscribed = subscription_manager.get_subscribed_instruments(subscription_type)
    print(f"   Subscribed instruments: {[inst.symbol for inst in subscribed]}")
    
    # Test the consistency - this is where the race condition would manifest
    for instrument in [btc_instrument, eth_instrument]:
        has_sub = subscription_manager.has_subscription(instrument, subscription_type)
        has_pending = subscription_manager.has_pending_subscription(instrument, subscription_type)
        in_subscribed = instrument in subscribed
        
        print(f"   {instrument.symbol}: has_sub={has_sub}, has_pending={has_pending}, in_list={in_subscribed}")
        
        # The fix ensures that if an instrument is in the subscribed list,
        # it should have either an active subscription or a pending one
        if in_subscribed:
            assert has_sub or has_pending, (
                f"RACE CONDITION: {instrument.symbol} in subscribed list but "
                f"has_subscription={has_sub}, has_pending={has_pending}"
            )
    
    print("📊 Step 4: Mark subscription active and verify final state")
    subscription_manager.mark_subscription_active(subscription_type)
    
    subscribed_final = subscription_manager.get_subscribed_instruments(subscription_type)
    print(f"   Final subscribed instruments: {[inst.symbol for inst in subscribed_final]}")
    
    # Final verification
    for instrument in [btc_instrument, eth_instrument]:
        has_sub = subscription_manager.has_subscription(instrument, subscription_type)
        has_pending = subscription_manager.has_pending_subscription(instrument, subscription_type)
        in_subscribed = instrument in subscribed_final
        
        print(f"   {instrument.symbol}: has_sub={has_sub}, has_pending={has_pending}, in_list={in_subscribed}")
        
        if in_subscribed:
            assert has_sub or has_pending, (
                f"INCONSISTENCY: {instrument.symbol} in subscribed list but "
                f"has_subscription={has_sub}, has_pending={has_pending}"
            )
    
    print("✅ Race condition fix verified!")
    print("✅ Subscription state remains consistent during transitions")
    return True


if __name__ == "__main__":
    success = test_race_condition_fix()
    if success:
        print("🎉 Race condition fix validation passed!")
    else:
        print("❌ Race condition fix validation failed!")
        exit(1)