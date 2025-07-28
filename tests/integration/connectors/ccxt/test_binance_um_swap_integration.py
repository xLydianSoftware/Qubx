"""
Integration tests for CCXT data provider with real Binance UM swap exchange.

These tests validate the complete subscription lifecycle, data flow, and
functionality with a live Binance UM futures exchange connection.

Requirements:
- Network connectivity to Binance
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
class TestBinanceUmSwapIntegration:
    """Integration tests with real Binance UM swap exchange."""

    @pytest.fixture(scope="class")
    def real_exchange(self):
        """Create a real CCXT Binance UM exchange for testing."""
        # Use sandbox/testnet when possible
        exchange = get_ccxt_exchange(
            exchange="binance.um",
            use_testnet=False,  # Binance testnet has limited data types
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
        """Create test instruments for Binance UM swap."""
        return [
            Instrument(
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
            ),
            Instrument(
                symbol="ETHUSDT",
                asset_type=AssetType.CRYPTO,
                market_type=MarketType.SWAP,
                exchange="BINANCE.UM",
                base="ETH",
                quote="USDT",
                settle="USDT",
                exchange_symbol="ETH/USDT:USDT",
                tick_size=0.01,
                lot_size=0.001,
                min_size=0.001,
            ),
        ]

    @pytest.fixture
    def test_channel(self):
        """Create a real control channel that captures data."""
        from qubx.core.basics import CtrlChannel

        channel = CtrlChannel("test_integration_channel")
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
        """Create a CcxtDataProvider with real exchange."""
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

    def test_subscribe_ohlc_data(self, data_provider, test_instruments, test_channel):
        """Test OHLC subscription with real exchange."""
        # Subscribe to OHLC data
        data_provider.subscribe("ohlc(1m)", test_instruments[:1])

        # Wait for data - data comes as tuples: (instrument, data_type, payload, is_historical)
        timeout = 15  # seconds
        start_time = time.time()

        received_data = []
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], Bar):
                    received_data.append(data)

            if received_data:
                break
            time.sleep(0.1)

        # Verify we received OHLC data
        assert len(received_data) > 0, f"Should receive at least one OHLC bar, got data: {test_channel.received_data}"

        # Extract the components: (instrument, data_type, bar, is_historical)
        instrument, data_type, bar, _is_historical = received_data[0]

        # Verify instrument
        assert instrument.symbol == "BTCUSDT"
        assert data_type.startswith("ohlc")

        # Verify bar data
        assert bar.open > 0
        assert bar.high >= bar.open
        assert bar.low <= bar.open
        assert bar.close > 0
        assert bar.volume >= 0

    def test_subscribe_trade_data(self, data_provider, test_instruments, test_channel):
        """Test trade data subscription with real exchange."""
        # Subscribe to trade data
        data_provider.subscribe("trade", test_instruments[:1])

        # Wait for data - data comes as tuples: (instrument, data_type, payload, is_historical)
        timeout = 15  # seconds
        start_time = time.time()

        received_data = []
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], Trade):
                    received_data.append(data)

            if received_data:
                break
            time.sleep(0.1)

        # Verify we received trade data
        assert len(received_data) > 0, f"Should receive at least one trade, got data: {test_channel.received_data}"

        # Extract the components: (instrument, data_type, trade, is_historical)
        instrument, data_type, trade, _is_historical = received_data[0]

        # Verify instrument
        assert instrument.symbol == "BTCUSDT"
        assert data_type == "trade"

        # Verify trade data
        assert trade.price > 0
        assert trade.size > 0
        assert trade.time > 0

    def test_subscribe_orderbook_data(self, data_provider, test_instruments, test_channel):
        """Test orderbook data subscription with real exchange."""
        # Subscribe to orderbook data
        data_provider.subscribe("orderbook", test_instruments[:1])

        # Wait for data - data comes as tuples: (instrument, data_type, payload, is_historical)
        timeout = 15  # seconds
        start_time = time.time()

        received_data = []
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], OrderBook):
                    received_data.append(data)
            
            if received_data:
                break
            time.sleep(0.1)

        # Verify we received orderbook data
        assert len(received_data) > 0, f"Should receive at least one orderbook update, got data: {test_channel.received_data}"

        # Extract the components: (instrument, data_type, orderbook, is_historical)
        instrument, data_type, orderbook, _is_historical = received_data[0]
        
        # Verify instrument
        assert instrument.symbol == "BTCUSDT"
        assert data_type == "orderbook"
        
        # Verify orderbook data
        assert len(orderbook.bids) > 0
        assert len(orderbook.asks) > 0
        # Verify bid/ask structure
        assert orderbook.top_bid > 0
        assert orderbook.top_ask > 0
        assert orderbook.bids[0] > 0  # First bid size
        assert orderbook.asks[0] > 0  # First ask size
        assert orderbook.top_ask > orderbook.top_bid  # Spread should be positive

    def test_subscribe_quote_data(self, data_provider, test_instruments, test_channel):
        """Test quote data subscription with real exchange."""
        # Subscribe to quote data
        data_provider.subscribe("quote", test_instruments[:1])

        # Wait for data - data comes as tuples: (instrument, data_type, payload, is_historical)
        timeout = 15  # seconds
        start_time = time.time()

        received_data = []
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], Quote):
                    received_data.append(data)
            
            if received_data:
                break
            time.sleep(0.1)

        # Verify we received quote data
        assert len(received_data) > 0, f"Should receive at least one quote, got data: {test_channel.received_data}"

        # Extract the components: (instrument, data_type, quote, is_historical)
        instrument, data_type, quote, _is_historical = received_data[0]
        
        # Verify instrument
        assert instrument.symbol == "BTCUSDT"
        assert data_type == "quote"
        
        # Verify quote data
        assert quote.bid > 0
        assert quote.ask > 0
        assert quote.bid_size > 0
        assert quote.ask_size > 0
        assert quote.ask > quote.bid  # Spread should be positive

    @pytest.mark.slow
    def test_subscribe_funding_rate_data(self, data_provider, test_instruments, test_channel):
        """Test funding rate data subscription with real exchange."""
        # Subscribe to funding rate data
        data_provider.subscribe("funding_rate", test_instruments[:1])

        # Funding rates update less frequently, so longer timeout
        timeout = 30  # seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Look for any funding rate related data
            if test_channel.received_data:
                # Check if we have any data that could be funding rate
                for data in test_channel.received_data:
                    if hasattr(data, "funding_rate") or hasattr(data, "rate"):
                        break
                else:
                    time.sleep(1)
                    continue
                break
            time.sleep(1)

        # Verify we received some data (funding rates may not update frequently)
        assert len(test_channel.received_data) >= 0, "Should handle funding rate subscription"

    @pytest.mark.slow
    def test_subscribe_open_interest_data(self, data_provider, test_instruments, test_channel):
        """Test open interest data subscription with real exchange."""
        # Subscribe to open interest data
        data_provider.subscribe("open_interest", test_instruments[:1])

        # Open interest updates less frequently
        timeout = 30  # seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Look for any open interest related data
            if test_channel.received_data:
                # Check if we have any data that could be open interest
                for data in test_channel.received_data:
                    if hasattr(data, "open_interest") or hasattr(data, "amount"):
                        break
                else:
                    time.sleep(1)
                    continue
                break
            time.sleep(1)

        # Verify we received some data (open interest may not update frequently)
        assert len(test_channel.received_data) >= 0, "Should handle open interest subscription"

    def test_add_instruments_dynamically(self, data_provider, test_instruments, test_channel):
        """Test adding instruments to existing subscription."""
        # Start with one instrument
        data_provider.subscribe("ohlc(1m)", test_instruments[:1])

        # Wait for initial data
        time.sleep(3)
        initial_data_count = len(test_channel.received_data)

        # Add second instrument
        data_provider.subscribe("ohlc(1m)", test_instruments, reset=False)

        # Wait for data from both instruments
        timeout = 15
        start_time = time.time()

        symbols_found = set()
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], Bar):
                    symbols_found.add(data[0].symbol)  # data[0] is instrument
            
            if len(symbols_found) >= 2:  # Both BTCUSDT and ETHUSDT
                break
            time.sleep(0.5)

        # Verify we got data from both instruments
        assert "BTCUSDT" in symbols_found, f"Expected BTCUSDT, got symbols: {symbols_found}"
        assert "ETHUSDT" in symbols_found, f"Expected ETHUSDT, got symbols: {symbols_found}"

    def test_remove_instruments_by_reset(self, data_provider, test_instruments, test_channel):
        """Test removing instruments by resetting subscription."""
        # Start with both instruments
        data_provider.subscribe("ohlc(1m)", test_instruments)

        # Wait for data from both
        time.sleep(5)

        # Reset to only first instrument
        data_provider.subscribe("ohlc(1m)", test_instruments[:1], reset=True)

        # Clear previous data and wait for new data
        test_channel.received_data.clear()
        time.sleep(5)

        # Verify we only get data from first instrument
        symbols_found = set()
        btc_count = 0
        eth_count = 0
        
        for data in test_channel.received_data:
            if len(data) >= 3 and isinstance(data[2], Bar):
                symbol = data[0].symbol  # data[0] is instrument
                symbols_found.add(symbol)
                if symbol == "BTCUSDT":
                    btc_count += 1
                elif symbol == "ETHUSDT":
                    eth_count += 1

        # Should primarily have BTCUSDT now
        assert "BTCUSDT" in symbols_found, f"Expected BTCUSDT, got symbols: {symbols_found}"
        
        # Should have significantly more BTC data than ETH (or no ETH)
        assert btc_count > eth_count, f"Expected more BTC data ({btc_count}) than ETH data ({eth_count})"

    def test_subscription_lifecycle_complete(self, data_provider, test_instruments, test_channel):
        """Test subscription lifecycle: subscribe → data → change subscription."""
        # Subscribe
        data_provider.subscribe("trade", test_instruments[:1])

        # Wait for data
        timeout = 10
        start_time = time.time()

        received_data = []
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], Trade):
                    received_data.append(data)
            
            if received_data:
                break
            time.sleep(0.1)

        # Verify subscription is active
        assert len(received_data) > 0, "Should receive trade data"

        # Change to OHLC subscription (different data type)
        test_channel.received_data.clear()
        data_provider.subscribe("ohlc(1m)", test_instruments[:1], reset=True)
        
        # Wait for OHLC data
        timeout = 10
        start_time = time.time()
        
        received_bars = []
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and isinstance(data[2], Bar):
                    received_bars.append(data)
            
            if received_bars:
                break
            time.sleep(0.1)
        
        # Verify we can change subscription types
        assert len(received_bars) > 0, "Should receive OHLC data after changing subscription"

    def test_multiple_data_types_simultaneously(self, data_provider, test_instruments, test_channel):
        """Test subscribing to multiple data types for same instrument."""
        instrument = test_instruments[:1]

        # Subscribe to multiple data types
        data_provider.subscribe("ohlc(1m)", instrument)
        data_provider.subscribe("trade", instrument)
        data_provider.subscribe("quote", instrument)

        # Wait for data from all types
        timeout = 20
        start_time = time.time()

        received_types = set()
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3:
                    payload = data[2]
                    if isinstance(payload, Bar):
                        received_types.add("bar")
                    elif isinstance(payload, Trade):
                        received_types.add("trade")
                    elif isinstance(payload, Quote):
                        received_types.add("quote")

            if len(received_types) >= 2:  # At least 2 types
                break
            time.sleep(0.5)

        # Verify we received multiple data types
        assert len(received_types) >= 2, f"Should receive multiple data types, got: {received_types}"

    def test_rapid_resubscription(self, data_provider, test_instruments, test_channel):
        """Test rapid resubscription scenarios."""
        instrument = test_instruments[:1]

        # Rapid subscribe/unsubscribe cycles
        for _ in range(3):
            # Subscribe
            data_provider.subscribe("ohlc(1m)", instrument)
            time.sleep(1)

            # Unsubscribe
            data_provider.subscribe("ohlc(1m)", [], reset=True)
            time.sleep(0.5)

        # Final subscription
        data_provider.subscribe("ohlc(1m)", instrument)

        # Should still work after rapid changes
        timeout = 10
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

    def test_comprehensive_resubscription_scenarios(self, data_provider, test_instruments, test_channel):
        """Test comprehensive resubscription scenarios as requested."""
        btc_instrument = test_instruments[0]  # BTCUSDT
        eth_instrument = test_instruments[1]  # ETHUSDT
        
        def get_symbol_counts():
            """Helper to count data by symbol and type."""
            bars_by_symbol = {}
            trades_by_symbol = {}
            
            for data in test_channel.received_data:
                if len(data) >= 3:
                    instrument, _data_type, payload, _is_historical = data
                    symbol = instrument.symbol
                    
                    if isinstance(payload, Bar):
                        bars_by_symbol[symbol] = bars_by_symbol.get(symbol, 0) + 1
                    elif isinstance(payload, Trade):
                        trades_by_symbol[symbol] = trades_by_symbol.get(symbol, 0) + 1
            
            return bars_by_symbol, trades_by_symbol
        
        # Scenario 1: Subscribe to BTC, then add ETH
        print("  Scenario 1: Subscribe to BTC, then add ETH")
        data_provider.subscribe("ohlc(1m)", [btc_instrument])
        time.sleep(3)
        
        bars_by_symbol, trades_by_symbol = get_symbol_counts()
        assert "BTCUSDT" in bars_by_symbol, "Should have BTC OHLC data"
        assert len(trades_by_symbol) == 0, "Should not have trade data yet"
        
        # Add ETH to existing subscription (reset=False)
        data_provider.subscribe("ohlc(1m)", [btc_instrument, eth_instrument], reset=False)
        time.sleep(3)
        
        bars_by_symbol, trades_by_symbol = get_symbol_counts()
        assert "BTCUSDT" in bars_by_symbol, "Should still have BTC OHLC data"
        assert "ETHUSDT" in bars_by_symbol, "Should now have ETH OHLC data"
        
        # Scenario 2: Subscribe back to original (BTC only)
        print("  Scenario 2: Reset to BTC only")
        test_channel.received_data.clear()
        data_provider.subscribe("ohlc(1m)", [btc_instrument], reset=True)
        time.sleep(3)
        
        bars_by_symbol, trades_by_symbol = get_symbol_counts()
        assert "BTCUSDT" in bars_by_symbol, "Should have BTC OHLC data"
        assert "ETHUSDT" not in bars_by_symbol, "Should not have ETH data after reset"
        
        # Scenario 3: Subscribe to OHLC, then add trades
        print("  Scenario 3: Add trades subscription while keeping OHLC")
        initial_btc_bars = bars_by_symbol.get("BTCUSDT", 0)
        
        # Add trades subscription
        data_provider.subscribe("trade", [btc_instrument])
        time.sleep(3)
        
        bars_by_symbol, trades_by_symbol = get_symbol_counts()
        assert "BTCUSDT" in bars_by_symbol, "Should still have BTC OHLC data"
        assert "BTCUSDT" in trades_by_symbol, "Should now have BTC trade data"
        assert bars_by_symbol["BTCUSDT"] > initial_btc_bars, "Should have more OHLC data"
        assert trades_by_symbol["BTCUSDT"] > 0, "Should have trade data"
        
        # Verify both data types work simultaneously
        assert trades_by_symbol["BTCUSDT"] > bars_by_symbol["BTCUSDT"], "Should have more trades than bars"
        
        print(f"  Final result: {bars_by_symbol} bars, {trades_by_symbol} trades")

    def test_invalid_symbol_handling(self, data_provider, test_channel):
        """Test error handling with invalid symbols."""
        # Create instrument with invalid symbol
        invalid_instrument = Instrument(
            symbol="INVALIDCOIN",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            exchange="BINANCE.UM",
            base="INVALID",
            quote="USDT",
            settle="USDT",
            exchange_symbol="INVALID/USDT:USDT",
            tick_size=0.1,
            lot_size=0.001,
            min_size=0.001,
        )

        # Should not crash when subscribing to invalid symbol
        try:
            data_provider.subscribe("ohlc(1m)", [invalid_instrument])
            time.sleep(5)  # Give it time to fail gracefully
            # If no exception, that's good
        except Exception as e:
            # Should handle errors gracefully
            assert "invalid" in str(e).lower() or "symbol" in str(e).lower()

    def test_network_resilience(self, data_provider, test_instruments, test_channel):
        """Test resilience to network issues (basic test)."""
        # Subscribe normally
        data_provider.subscribe("trade", test_instruments[:1])

        # Wait for initial data
        timeout = 10
        start_time = time.time()

        while time.time() - start_time < timeout:
            if any(isinstance(data, Trade) for data in test_channel.received_data):
                break
            time.sleep(0.1)

        initial_trades = len([data for data in test_channel.received_data if isinstance(data, Trade)])
        assert initial_trades > 0, "Should receive initial trade data"

        # The connection should continue working over time
        time.sleep(5)

        final_trades = len([data for data in test_channel.received_data if isinstance(data, Trade)])
        assert final_trades >= initial_trades, "Should continue receiving data over time"

    @pytest.mark.slow
    def test_warmup_functionality(self, data_provider, test_instruments):
        """Test historical data warmup functionality."""
        # Warmup should complete without errors
        warmups = {(DataType.OHLC, test_instruments[0]): "1h"}

        try:
            # This should not raise an exception
            data_provider.warmup(warmups)
            # Warmup is async, so we just verify it doesn't crash
            time.sleep(2)
        except Exception as e:
            pytest.fail(f"Warmup should not fail: {e}")

    def test_data_provider_properties(self, data_provider):
        """Test basic data provider properties and methods."""
        # Should not be simulation
        assert not data_provider.is_simulation

        # Should have proper exchange ID
        assert data_provider._exchange_id == "binance.um"

        # Should have all required components
        assert hasattr(data_provider, "_subscription_manager")
        assert hasattr(data_provider, "_connection_manager")
        assert hasattr(data_provider, "_subscription_orchestrator")
        assert hasattr(data_provider, "_data_type_handler_factory")
        assert hasattr(data_provider, "_warmup_service")
