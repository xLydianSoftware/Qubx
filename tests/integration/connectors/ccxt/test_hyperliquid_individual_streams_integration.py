"""
Integration tests for CCXT data provider with real Hyperliquid exchange.

These tests validate the individual instrument streaming functionality,
subscription lifecycle, and dynamic instrument management with a live 
Hyperliquid perpetuals exchange connection.

Key features tested:
- Individual instrument streams (no waiting between instruments)
- Dynamic instrument addition/removal
- Multiple data type subscriptions
- Subscription lifecycle management
- Real-time WebSocket streaming

Requirements:
- Network connectivity to Hyperliquid
- Mark tests with @pytest.mark.integration
- No API credentials required for public data
"""

import asyncio
import os
import time
from typing import Set
from unittest.mock import MagicMock

import pytest

from qubx.connectors.ccxt.data import CcxtDataProvider  
from qubx.connectors.ccxt.factory import get_ccxt_exchange
from qubx.core.basics import AssetType, CtrlChannel, DataType, Instrument, MarketType
from qubx.core.series import Bar, OrderBook, Quote, Trade


@pytest.mark.integration
class TestHyperliquidIndividualStreamsIntegration:
    """Integration tests with real Hyperliquid exchange for individual streams."""

    @pytest.fixture(scope="class")
    def real_exchange(self):
        """Create a real CCXT Hyperliquid exchange for testing."""
        exchange = get_ccxt_exchange(
            exchange="hyperliquid.f",
            use_testnet=False,
            # No API keys needed for public data subscriptions
        )

        yield exchange

        # Cleanup
        try:
            asyncio.create_task(exchange.close())
        except Exception:
            pass

    @pytest.fixture
    def test_instruments(self):
        """Create test instruments for Hyperliquid perpetuals."""
        return [
            Instrument(
                symbol="BTCUSDC",
                asset_type=AssetType.CRYPTO,
                market_type=MarketType.SWAP,
                exchange="HYPERLIQUID.F",
                base="BTC",
                quote="USDC",
                settle="USDC",
                exchange_symbol="BTC/USDC:USDC",
                tick_size=0.01,
                lot_size=0.0001,
                min_size=0.0001,
            ),
            Instrument(
                symbol="ETHUSDC",
                asset_type=AssetType.CRYPTO,
                market_type=MarketType.SWAP,
                exchange="HYPERLIQUID.F",
                base="ETH",
                quote="USDC",
                settle="USDC",
                exchange_symbol="ETH/USDC:USDC",
                tick_size=0.01,
                lot_size=0.001,
                min_size=0.001,
            ),
            Instrument(
                symbol="SOLUSDC", 
                asset_type=AssetType.CRYPTO,
                market_type=MarketType.SWAP,
                exchange="HYPERLIQUID.F",
                base="SOL",
                quote="USDC",
                settle="USDC",
                exchange_symbol="SOL/USDC:USDC",
                tick_size=0.01,
                lot_size=0.01,
                min_size=0.01,
            ),
        ]

    @pytest.fixture
    def test_channel(self):
        """Create a real control channel that captures data."""
        from qubx.core.basics import CtrlChannel

        channel = CtrlChannel("test_hyperliquid_integration_channel")
        channel.received_data = []

        # Store original send method
        original_send = channel.send

        def capture_send(data):
            channel.received_data.append(data)
            return original_send(data)

        channel.send = capture_send

        yield channel

        # Cleanup
        try:
            channel.stop()
        except Exception:
            pass

    @pytest.fixture
    def data_provider(self, real_exchange, test_channel):
        """Create a CcxtDataProvider with real Hyperliquid exchange."""
        from qubx.core.basics import LiveTimeProvider

        provider = CcxtDataProvider(
            exchange=real_exchange,
            time_provider=LiveTimeProvider(),
            channel=test_channel,
            max_ws_retries=3,
            warmup_timeout=30,
        )

        yield provider

        # Cleanup
        try:
            provider.stop()
        except Exception:
            pass

    def test_individual_streams_basic_functionality(self, data_provider, test_instruments, test_channel):
        """Test that individual streams work with multiple instruments."""
        # Verify exchange doesn't support bulk watching (forces individual streams)
        assert not data_provider._exchange.has.get("watchOHLCVForSymbols", True), \
            "Test requires exchange to use individual streams (watchOHLCVForSymbols=False)"
        
        # Subscribe to OHLC data for multiple instruments
        data_provider.subscribe("ohlc(1m)", test_instruments[:2])  # BTC and ETH

        # Wait for data from individual streams
        timeout = 20  # seconds
        start_time = time.time()

        symbols_with_data = set()
        received_bars = []
        
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], Bar):
                    instrument, data_type, bar, is_historical = data
                    symbols_with_data.add(instrument.symbol)
                    received_bars.append(data)
            
            # Wait until we have data from both instruments
            if len(symbols_with_data) >= 2:
                break
            time.sleep(0.5)

        # Verify individual streams are working
        assert len(symbols_with_data) >= 2, f"Should receive data from multiple instruments, got: {symbols_with_data}"
        assert "BTCUSDC" in symbols_with_data, f"Should receive BTC data, got symbols: {symbols_with_data}"
        assert "ETHUSDC" in symbols_with_data, f"Should receive ETH data, got symbols: {symbols_with_data}"
        assert len(received_bars) > 0, "Should receive OHLC bar data"
        
        # Verify OHLC data quality
        btc_bars = [data for data in received_bars if data[0].symbol == "BTCUSDC"]
        assert len(btc_bars) > 0, "Should have BTC bars"
        
        _, _, bar, _ = btc_bars[0]
        assert bar.open > 0, "Bar should have valid open price"
        assert bar.high >= bar.open, "High should be >= open"
        assert bar.low <= bar.open, "Low should be <= open" 
        assert bar.close > 0, "Bar should have valid close price"
        assert bar.volume >= 0, "Bar should have valid volume"

    def test_individual_streams_concurrent_processing(self, data_provider, test_instruments, test_channel):
        """Test that individual streams process concurrently (don't wait for each other)."""
        # Clear any existing data
        test_channel.received_data.clear()
        
        # Subscribe to all three instruments
        data_provider.subscribe("ohlc(1m)", test_instruments)  # BTC, ETH, SOL

        # Track data arrival times per instrument
        data_arrival_times = {}
        timeout = 25  # seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], Bar):
                    instrument, _, _, _ = data
                    symbol = instrument.symbol
                    
                    if symbol not in data_arrival_times:
                        data_arrival_times[symbol] = time.time() - start_time
                        print(f"First data from {symbol} at {data_arrival_times[symbol]:.1f}s")
            
            # Once we have data from at least 2 instruments, we can analyze concurrency
            if len(data_arrival_times) >= 2:
                break
            time.sleep(0.5)

        # Verify concurrent processing
        assert len(data_arrival_times) >= 2, f"Should receive data from multiple instruments, got: {list(data_arrival_times.keys())}"
        
        # Check that data doesn't arrive in perfect sequential order (indicating concurrency)
        arrival_times = list(data_arrival_times.values())
        arrival_times.sort()
        
        # If streams were sequential, we'd expect significant delays between arrivals
        # With concurrent streams, arrivals should be closer together
        if len(arrival_times) >= 2:
            time_diff = arrival_times[1] - arrival_times[0]
            assert time_diff < 10.0, f"Concurrent streams should have closer arrival times, got {time_diff:.1f}s difference"
            print(f"✅ Concurrent processing verified: {time_diff:.1f}s between first two instruments")

    def test_add_instruments_to_individual_streams(self, data_provider, test_instruments, test_channel):
        """Test dynamically adding instruments to existing individual streams."""
        # Start with one instrument
        data_provider.subscribe("ohlc(1m)", test_instruments[:1])  # Just BTC

        # Wait for initial data
        time.sleep(5)
        initial_symbols = set()
        for data in test_channel.received_data:
            if len(data) >= 3 and isinstance(data[2], Bar):
                initial_symbols.add(data[0].symbol)

        assert "BTCUSDC" in initial_symbols, "Should have initial BTC data"
        
        # Add second instrument (should create new individual stream)
        data_provider.subscribe("ohlc(1m)", test_instruments[:2], reset=False)  # Add ETH

        # Wait for data from both instruments  
        timeout = 20
        start_time = time.time()
        symbols_found = set()
        
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], Bar):
                    symbols_found.add(data[0].symbol)
            
            if len(symbols_found) >= 2:
                break
            time.sleep(0.5)

        # Verify both instruments are now streaming
        assert "BTCUSDC" in symbols_found, f"Should still have BTC data, got: {symbols_found}"
        assert "ETHUSDC" in symbols_found, f"Should now have ETH data, got: {symbols_found}"
        print(f"✅ Successfully added instrument: {symbols_found}")

    def test_remove_instruments_from_individual_streams(self, data_provider, test_instruments, test_channel):
        """Test dynamically removing instruments from individual streams."""
        # Start with multiple instruments
        data_provider.subscribe("ohlc(1m)", test_instruments[:2])  # BTC and ETH

        # Wait for data from both
        time.sleep(8)
        
        # Verify both are streaming
        initial_symbols = set()
        for data in test_channel.received_data:
            if len(data) >= 3 and isinstance(data[2], Bar):
                initial_symbols.add(data[0].symbol)
                
        assert len(initial_symbols) >= 2, f"Should have data from multiple instruments initially, got: {initial_symbols}"

        # Remove one instrument (reset to just BTC)
        test_channel.received_data.clear()
        data_provider.subscribe("ohlc(1m)", test_instruments[:1], reset=True)  # Just BTC

        # Wait for new data pattern
        time.sleep(8)
        
        # Analyze new data pattern
        final_symbols = set()
        btc_count = 0
        eth_count = 0
        
        for data in test_channel.received_data:
            if len(data) >= 3 and isinstance(data[2], Bar):
                symbol = data[0].symbol
                final_symbols.add(symbol)
                if symbol == "BTCUSDC":
                    btc_count += 1
                elif symbol == "ETHUSDC":
                    eth_count += 1

        # Should primarily have BTC data now
        assert "BTCUSDC" in final_symbols, f"Should still have BTC data, got: {final_symbols}"
        
        # Should have significantly more BTC data than ETH (ETH stream should be stopped)
        assert btc_count > eth_count, f"Should have more BTC data ({btc_count}) than ETH data ({eth_count}) after removal"
        print(f"✅ Successfully removed instrument: BTC={btc_count}, ETH={eth_count}")

    def test_individual_streams_with_multiple_data_types(self, data_provider, test_instruments, test_channel):
        """Test individual streams with multiple data types simultaneously."""
        instrument = test_instruments[:1]  # Just BTC for simplicity

        # Subscribe to multiple data types
        data_provider.subscribe("ohlc(1m)", instrument)
        data_provider.subscribe("trade", instrument) 
        # Note: Not all data types may be supported by Hyperliquid, so we'll test what works

        # Wait for data from multiple types
        timeout = 25
        start_time = time.time() 

        received_types = set()
        data_by_type = {}
        
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3:
                    instrument_obj, data_type, payload, is_historical = data
                    
                    if isinstance(payload, Bar):
                        received_types.add("ohlc")
                        data_by_type.setdefault("ohlc", []).append(data)
                    elif isinstance(payload, Trade):
                        received_types.add("trade")
                        data_by_type.setdefault("trade", []).append(data)
                    elif isinstance(payload, Quote):
                        received_types.add("quote")
                        data_by_type.setdefault("quote", []).append(data)
                    elif isinstance(payload, OrderBook):
                        received_types.add("orderbook")
                        data_by_type.setdefault("orderbook", []).append(data)

            # We expect at least OHLC data
            if "ohlc" in received_types:
                break
            time.sleep(0.5)

        # Verify we received expected data types
        assert "ohlc" in received_types, f"Should receive OHLC data, got types: {received_types}"
        assert len(data_by_type.get("ohlc", [])) > 0, "Should have OHLC data"
        
        print(f"✅ Received data types: {received_types}")
        for data_type, data_list in data_by_type.items():
            print(f"  {data_type}: {len(data_list)} messages")

    def test_individual_streams_subscription_lifecycle(self, data_provider, test_instruments, test_channel):
        """Test complete subscription lifecycle with individual streams."""
        instrument = test_instruments[:1]  # BTC

        # Phase 1: Subscribe to OHLC
        print("Phase 1: Subscribe to OHLC")
        data_provider.subscribe("ohlc(1m)", instrument)
        
        time.sleep(5)
        phase1_data = [data for data in test_channel.received_data if len(data) >= 3 and isinstance(data[2], Bar)]
        assert len(phase1_data) > 0, "Phase 1: Should receive OHLC data"

        # Phase 2: Change to trade data (different data type)
        print("Phase 2: Change to trade data")
        test_channel.received_data.clear()
        data_provider.subscribe("trade", instrument, reset=True)
        
        time.sleep(8)
        phase2_data = [data for data in test_channel.received_data if len(data) >= 3 and isinstance(data[2], Trade)]
        
        # Phase 3: Back to OHLC (verify can resubscribe)
        print("Phase 3: Back to OHLC")
        test_channel.received_data.clear()
        data_provider.subscribe("ohlc(1m)", instrument, reset=True)
        
        time.sleep(5)
        phase3_data = [data for data in test_channel.received_data if len(data) >= 3 and isinstance(data[2], Bar)]
        assert len(phase3_data) > 0, "Phase 3: Should receive OHLC data after resubscription"

        print(f"✅ Lifecycle test: Phase1={len(phase1_data)} bars, Phase2={len(phase2_data)} trades, Phase3={len(phase3_data)} bars")

    def test_individual_streams_rapid_resubscription(self, data_provider, test_instruments, test_channel):
        """Test rapid resubscription scenarios with individual streams."""
        instrument = test_instruments[:1]  # BTC

        # Rapid subscribe/unsubscribe cycles
        for cycle in range(3):
            print(f"Rapid resubscription cycle {cycle + 1}")
            
            # Subscribe
            data_provider.subscribe("ohlc(1m)", instrument)
            time.sleep(2)

            # Unsubscribe (empty instrument list)
            data_provider.subscribe("ohlc(1m)", [], reset=True)
            time.sleep(1)

        # Final subscription should still work
        print("Final subscription test")
        test_channel.received_data.clear()
        data_provider.subscribe("ohlc(1m)", instrument)

        # Should still work after rapid changes
        timeout = 15
        start_time = time.time()

        received_bars = []
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], Bar):
                    received_bars.append(data)
            
            if received_bars:
                break
            time.sleep(0.5)

        assert len(received_bars) > 0, "Should still receive data after rapid resubscription"
        print(f"✅ Survived rapid resubscription: {len(received_bars)} bars received")

    def test_individual_streams_error_isolation(self, data_provider, test_instruments, test_channel):
        """Test that errors in one individual stream don't affect others."""
        # Use valid instruments plus one potentially problematic one
        valid_instruments = test_instruments[:2]  # BTC, ETH
        
        # Create a potentially invalid instrument to test error handling
        invalid_instrument = Instrument(
            symbol="NOTEXIST",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            exchange="HYPERLIQUID.F",
            base="NOTEXIST",
            quote="USDC",
            settle="USDC",
            exchange_symbol="NOTEXIST/USDC:USDC",
            tick_size=0.01,
            lot_size=0.01,
            min_size=0.01,
        )
        
        all_instruments = valid_instruments + [invalid_instrument]

        # Subscribe to all instruments (including invalid one)
        try:
            data_provider.subscribe("ohlc(1m)", all_instruments)
        except Exception as e:
            print(f"Expected potential error for invalid instrument: {e}")

        # Wait and check that valid instruments still work
        time.sleep(10)
        
        symbols_with_data = set()
        for data in test_channel.received_data:
            if len(data) >= 3 and isinstance(data[2], Bar):
                symbols_with_data.add(data[0].symbol)

        # Should still receive data from valid instruments
        valid_symbols_found = symbols_with_data.intersection({"BTCUSDC", "ETHUSDC"})
        assert len(valid_symbols_found) > 0, f"Should receive data from valid instruments despite invalid one, got: {symbols_with_data}"
        
        print(f"✅ Error isolation working: {valid_symbols_found} instruments still streaming")

    def test_individual_streams_performance_characteristics(self, data_provider, test_instruments, test_channel):
        """Test performance characteristics specific to individual streams."""
        # Test with multiple instruments to verify no blocking behavior
        instruments = test_instruments  # All 3 instruments
        
        # Clear data and subscribe
        test_channel.received_data.clear()
        start_time = time.time()
        data_provider.subscribe("ohlc(1m)", instruments)

        # Track data arrival pattern 
        data_counts_by_symbol = {}
        data_timestamps = {}
        
        # Monitor for 15 seconds
        monitoring_duration = 15
        while time.time() - start_time < monitoring_duration:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], Bar):
                    symbol = data[0].symbol
                    timestamp = time.time()
                    
                    data_counts_by_symbol[symbol] = data_counts_by_symbol.get(symbol, 0) + 1
                    
                    if symbol not in data_timestamps:
                        data_timestamps[symbol] = timestamp
                        
            time.sleep(0.1)

        # Analyze performance characteristics
        total_instruments = len(data_counts_by_symbol)
        total_messages = sum(data_counts_by_symbol.values())
        
        print(f"Performance test results:")
        print(f"  Instruments with data: {total_instruments}")
        print(f"  Total messages: {total_messages}")
        print(f"  Data per instrument: {data_counts_by_symbol}")
        
        # Verify good performance characteristics
        assert total_instruments >= 2, f"Should have data from multiple instruments, got {total_instruments}"
        assert total_messages > 0, "Should receive some data"
        
        # Check that all instruments with data have reasonable message counts
        # (indicating no instrument is blocking others)
        if len(data_counts_by_symbol) >= 2:
            counts = list(data_counts_by_symbol.values())
            min_count = min(counts)
            max_count = max(counts)
            ratio = max_count / min_count if min_count > 0 else float('inf')
            
            # Individual streams should have relatively balanced data flow
            # (no single instrument should dominate due to blocking)
            assert ratio < 10.0, f"Data distribution too uneven (ratio: {ratio:.1f}), may indicate blocking"
            
        print(f"✅ Performance characteristics look good")

    def test_verify_individual_streams_are_used(self, data_provider, test_instruments, test_channel):
        """Verify that the exchange is actually using individual streams (not bulk)."""
        # Check exchange capabilities
        exchange = data_provider._exchange
        supports_bulk = exchange.has.get("watchOHLCVForSymbols", True)
        
        print(f"Exchange supports watchOHLCVForSymbols: {supports_bulk}")
        
        # For this test, we want to verify individual streams are being used
        assert not supports_bulk, \
            "Test requires individual streams (watchOHLCVForSymbols should be False for Hyperliquid)"
        
        # Subscribe and verify individual stream behavior
        data_provider.subscribe("ohlc(1m)", test_instruments[:2])
        time.sleep(5)
        
        # Check that we can observe the individual stream behavior
        # Individual streams should create separate listeners
        symbols_with_data = set()
        for data in test_channel.received_data:
            if len(data) >= 3 and isinstance(data[2], Bar):
                symbols_with_data.add(data[0].symbol)
        
        assert len(symbols_with_data) > 0, "Should receive data when using individual streams"
        print(f"✅ Individual streams confirmed: {len(symbols_with_data)} instruments streaming independently")

    def test_data_provider_properties_hyperliquid(self, data_provider):
        """Test basic data provider properties for Hyperliquid."""
        # Should not be simulation
        assert not data_provider.is_simulation

        # Should have proper exchange ID  
        assert data_provider._exchange_id == "hyperliquid.f"

        # Should have all required components
        assert hasattr(data_provider, "_subscription_manager")
        assert hasattr(data_provider, "_connection_manager")
        assert hasattr(data_provider, "_subscription_orchestrator")
        assert hasattr(data_provider, "_data_type_handler_factory")
        assert hasattr(data_provider, "_warmup_service")

        print(f"✅ Data provider properly configured for {data_provider._exchange_id}")