#!/usr/bin/env python3
"""
Debug test to see what OHLC data is being sent during subscription.
"""

import time
from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.connectors.ccxt.factory import get_ccxt_exchange
from qubx.core.basics import CtrlChannel, Instrument, AssetType, MarketType, LiveTimeProvider
from qubx.core.series import Bar
from qubx.health import DummyHealthMonitor


def debug_ohlc_subscription():
    """Debug what OHLC data is received during subscription."""
    print("🔍 Debugging OHLC subscription data...")
    
    # Create test instrument
    btc_instrument = Instrument(
        symbol="BTCUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="BINANCE.UM",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTC/USDT:USDT",
        tick_size=0.1,
        lot_size=0.001,
        min_size=0.001,
    )
    
    # Create channel and data provider
    channel = CtrlChannel("debug_channel")
    exchange = get_ccxt_exchange(exchange="binanceusdm", use_testnet=True)
    
    data_provider = CcxtDataProvider(
        exchange=exchange,
        time_provider=LiveTimeProvider(),
        channel=channel,
        max_ws_retries=3,
        warmup_timeout=30,
        health_monitor=DummyHealthMonitor(),
    )
    
    received_data = []
    original_send = channel.send
    
    def capture_send(data):
        received_data.append(data)
        if len(data) >= 3:
            instrument, data_type, payload, is_historical = data
            if isinstance(payload, Bar):
                print(f"📊 OHLC Bar: {data_type}, historical={is_historical}, time={payload.time}")
        return original_send(data)
    
    channel.send = capture_send
    
    try:
        print("📊 Subscribing to OHLC(1m)...")
        data_provider.subscribe("ohlc(1m)", [btc_instrument])
        
        print("📊 Waiting 5 seconds to collect data...")
        time.sleep(5)
        
        # Count bars
        bar_count = 0
        historical_count = 0
        live_count = 0
        
        for data in received_data:
            if len(data) >= 3:
                instrument, data_type, payload, is_historical = data
                if isinstance(payload, Bar):
                    bar_count += 1
                    if is_historical:
                        historical_count += 1
                    else:
                        live_count += 1
        
        print(f"📊 Results:")
        print(f"   Total bars: {bar_count}")
        print(f"   Historical bars: {historical_count}")
        print(f"   Live bars: {live_count}")
        
        # Now clear and test again
        print("\n📊 Clearing data and resubscribing...")
        received_data.clear()
        data_provider.subscribe("ohlc(1m)", [btc_instrument], reset=True)
        
        print("📊 Waiting 3 seconds after reset...")
        time.sleep(3)
        
        # Count bars again
        bar_count_after_reset = 0
        historical_count_after_reset = 0
        live_count_after_reset = 0
        
        for data in received_data:
            if len(data) >= 3:
                instrument, data_type, payload, is_historical = data
                if isinstance(payload, Bar):
                    bar_count_after_reset += 1
                    if is_historical:
                        historical_count_after_reset += 1
                    else:
                        live_count_after_reset += 1
        
        print(f"📊 Results after reset:")
        print(f"   Total bars: {bar_count_after_reset}")
        print(f"   Historical bars: {historical_count_after_reset}")
        print(f"   Live bars: {live_count_after_reset}")
        
    finally:
        data_provider.close()
        channel.stop()


if __name__ == "__main__":
    debug_ohlc_subscription()