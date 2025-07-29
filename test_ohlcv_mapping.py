#!/usr/bin/env python3
"""
Test script to verify OHLCV data mapping against Binance documentation.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from qubx.core.basics import CtrlChannel, Instrument, AssetType, MarketType
from qubx.connectors.ccxt.handlers.ohlc import OhlcDataHandler


def test_ohlcv_mapping_against_binance_docs():
    """Test that our OHLCV mapping matches Binance documentation exactly."""
    
    # Mock data provider and exchange
    mock_data_provider = MagicMock()
    mock_exchange = MagicMock()
    
    # Create handler
    handler = OhlcDataHandler(
        data_provider=mock_data_provider,
        exchange=mock_exchange,
        exchange_id="BINANCE.UM"
    )
    
    # Create test OHLCV data as processed by CCXT (10 fields total)
    # CCXT processes raw Binance data and provides: [open_time, open, high, low, close, volume, quote_volume, trade_count, taker_buy_base, taker_buy_quote]
    binance_ohlcv = [
        1499040000000,          # Index 0: Open time (timestamp)
        0.01634790,            # Index 1: Open price (converted to float by CCXT)
        0.80000000,            # Index 2: High price  
        0.01575800,            # Index 3: Low price
        0.01577100,            # Index 4: Close price
        148976.11427815,       # Index 5: Volume (base asset)
        2434.19055334,         # Index 6: Quote asset volume
        308,                   # Index 7: Number of trades
        1756.87402397,         # Index 8: Taker buy base asset volume
        28.46694368,           # Index 9: Taker buy quote asset volume
    ]
    
    # Convert using our utility method
    bar = handler._convert_ohlcv_to_bar(binance_ohlcv)
    
    # Verify the mapping is correct
    assert bar.time == 1499040000000000000, "Timestamp should be converted to nanoseconds"
    assert bar.open == 0.01634790, "Open price mapping"
    assert bar.high == 0.80000000, "High price mapping"
    assert bar.low == 0.01575800, "Low price mapping"
    assert bar.close == 0.01577100, "Close price mapping"
    assert bar.volume == 148976.11427815, "Volume (base asset) should be from index 5"
    assert bar.volume_quote == 2434.19055334, "Quote asset volume should be from index 6"
    assert bar.trade_count == 308, "Trade count should be from index 7"
    assert bar.bought_volume == 1756.87402397, "Taker buy base volume should be from index 8"
    assert bar.bought_volume_quote == 28.46694368, "Taker buy quote volume should be from index 9"
    
    print("✅ All OHLCV mappings verified against Binance documentation!")
    print(f"   Volume (base): {bar.volume}")
    print(f"   Volume (quote): {bar.volume_quote}")
    print(f"   Bought volume (base): {bar.bought_volume}")
    print(f"   Bought volume (quote): {bar.bought_volume_quote}")
    print(f"   Trade count: {bar.trade_count}")


def test_backwards_compatibility_with_standard_ohlcv():
    """Test that standard 6-field OHLCV still works."""
    
    # Mock data provider and exchange
    mock_data_provider = MagicMock()
    mock_exchange = MagicMock()
    
    # Create handler
    handler = OhlcDataHandler(
        data_provider=mock_data_provider,
        exchange=mock_exchange,
        exchange_id="TEST"
    )
    
    # Standard OHLCV data (6 fields)
    standard_ohlcv = [
        1499040000000,      # Index 0: Open time (timestamp)
        0.01634790,         # Index 1: Open price
        0.80000000,         # Index 2: High price  
        0.01575800,         # Index 3: Low price
        0.01577100,         # Index 4: Close price
        148976.11427815,    # Index 5: Volume
    ]
    
    # Convert using our utility method
    bar = handler._convert_ohlcv_to_bar(standard_ohlcv)
    
    # Verify the mapping
    assert bar.time == 1499040000000000000, "Timestamp should be converted to nanoseconds"
    assert bar.open == 0.01634790, "Open price mapping"
    assert bar.high == 0.80000000, "High price mapping"
    assert bar.low == 0.01575800, "Low price mapping"
    assert bar.close == 0.01577100, "Close price mapping"
    assert bar.volume == 148976.11427815, "Volume should be from index 5"
    
    # New fields should have default values
    assert bar.volume_quote == 0, "Quote volume should default to 0"
    assert bar.bought_volume == 0, "Bought volume should default to 0" 
    assert bar.bought_volume_quote == 0, "Bought quote volume should default to 0"
    assert bar.trade_count == 0, "Trade count should default to 0"
    
    print("✅ Standard OHLCV backwards compatibility verified!")


if __name__ == "__main__":
    test_ohlcv_mapping_against_binance_docs()
    test_backwards_compatibility_with_standard_ohlcv()
    print("✅ All OHLCV mapping tests passed!")