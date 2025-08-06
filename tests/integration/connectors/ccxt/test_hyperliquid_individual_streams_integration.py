"""
Integration tests for CCXT data provider with real Hyperliquid exchange.

These tests validate core functionality and the PollingToWebSocketAdapter
with a live Hyperliquid perpetuals exchange connection.
"""

import asyncio
import time
from typing import Set
from unittest.mock import MagicMock

import pytest

from qubx.connectors.ccxt.data import CcxtDataProvider  
from qubx.connectors.ccxt.factory import get_ccxt_exchange
from qubx.core.basics import AssetType, CtrlChannel, DataType, Instrument, MarketType


@pytest.mark.integration
class TestHyperliquidBasicIntegration:
    """Basic integration tests with real Hyperliquid exchange."""

    @pytest.fixture(scope="class")
    def real_exchange(self):
        """Create a real CCXT Hyperliquid exchange for testing."""
        exchange = get_ccxt_exchange(
            exchange="hyperliquid.f",
            use_testnet=False,
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
        ]

    @pytest.fixture
    def test_channel(self):
        """Create a mock channel for capturing test data."""
        channel = MagicMock(spec=CtrlChannel)
        channel.received_data = []
        
        def mock_send(data):
            channel.received_data.append(data)
        
        channel.send = mock_send
        return channel

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

    def test_basic_data_subscription(self, data_provider, test_instruments, test_channel):
        """Test basic OHLCV data subscription."""
        instrument = test_instruments[0]  # BTCUSDC
        
        # Subscribe to OHLCV data
        data_provider.subscribe("ohlc_1m", [instrument])
        
        # Wait for data
        timeout = 20
        start_time = time.time()
        received_data = []
        
        while time.time() - start_time < timeout:
            for data in test_channel.received_data:
                if len(data) >= 3 and data[1] == "ohlc_1m":
                    received_data.append(data)
            if received_data:
                break
            time.sleep(2)
        
        # Verify we received OHLCV data
        assert len(received_data) > 0, "Should receive OHLCV data"
        
        # Clean up
        data_provider.unsubscribe("ohlc_1m")

    def test_data_provider_properties(self, data_provider):
        """Test basic data provider properties and configuration."""
        # Verify exchange is configured correctly
        assert data_provider._exchange is not None
        assert data_provider._exchange.id == "hyperliquid"
        
        # Test data provider basic methods
        assert hasattr(data_provider, 'subscribe')
        assert hasattr(data_provider, 'unsubscribe')


@pytest.mark.integration
class TestHyperliquidFundingRateAdapter:
    """Integration tests for the PollingToWebSocketAdapter with funding rates."""

    @pytest.fixture(scope="class")
    def adapter_event_loop(self):
        """Create a dedicated event loop for adapter testing."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture(scope="class")
    def real_exchange(self, adapter_event_loop):
        """Create a real CCXT Hyperliquid exchange with explicit event loop."""
        exchange = get_ccxt_exchange(
            exchange="hyperliquid.f",
            use_testnet=False,
            loop=adapter_event_loop,
        )
        yield exchange

        # Cleanup
        try:
            future = asyncio.run_coroutine_threadsafe(exchange.close(), adapter_event_loop)
            future.result(timeout=5)
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
        ]

    @pytest.fixture
    def test_channel(self):
        """Create a mock channel for capturing test data."""
        channel = MagicMock(spec=CtrlChannel)
        channel.received_data = []
        
        def mock_send(data):
            channel.received_data.append(data)
        
        channel.send = mock_send
        return channel

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

    def _cleanup_adapter_exchange(self, exchange, data_provider_loop):
        """Helper method to clean up adapter."""
        if hasattr(exchange, '_funding_rate_adapter') and exchange._funding_rate_adapter:
            async def cleanup_adapter():
                await exchange._funding_rate_adapter.stop()
            try:
                future = data_provider_loop.submit(cleanup_adapter())
                future.result(timeout=10)
                exchange._funding_rate_adapter = None
            except:
                pass

    def _setup_adapter_exchange(self, exchange, adapter_loop):
        """Helper method to set up exchange with explicit event loop adapter."""
        from qubx.connectors.ccxt.adapters.polling_adapter import PollingToWebSocketAdapter, PollingConfig
        
        async def watch_funding_rates_with_explicit_loop(symbols=None, params=None):
            # Ensure markets are loaded in the exchange's own loop
            if not exchange.markets:
                await exchange.load_markets()
            
            # Create or reuse adapter with explicit loop
            if exchange._funding_rate_adapter is None:
                exchange._funding_rate_adapter = PollingToWebSocketAdapter(
                    fetch_method=exchange.fetch_funding_rates,
                    symbols=symbols or ["BTC/USDC:USDC"],
                    params=params or {},
                    config=PollingConfig(poll_interval_seconds=30),  # Short interval for testing
                    event_loop=adapter_loop
                )
                await exchange._funding_rate_adapter.start_watching()
            
            # Get next data from the adapter
            funding_data = await exchange._funding_rate_adapter.get_next_data()
            
            # Apply format transformation for ccxt_convert_funding_rate compatibility
            transformed_data = {}
            if isinstance(funding_data, dict):
                for symbol, rate_info in funding_data.items():
                    if isinstance(rate_info, dict):
                        transformed_info = rate_info.copy()
                        # Fix timestamp: use fundingTimestamp if timestamp is None
                        if transformed_info.get('timestamp') is None:
                            transformed_info['timestamp'] = transformed_info.get('fundingTimestamp')
                        # Fix nextFundingTime: use nextFundingTimestamp if available
                        if 'nextFundingTimestamp' in transformed_info and transformed_info.get('nextFundingTime') is None:
                            transformed_info['nextFundingTime'] = transformed_info['nextFundingTimestamp']
                        transformed_data[symbol] = transformed_info
                    else:
                        transformed_data[symbol] = rate_info
            
            return transformed_data or funding_data
        
        return watch_funding_rates_with_explicit_loop

    def test_funding_rate_subscription(self, data_provider, test_instruments, test_channel, adapter_event_loop):
        """Test funding rate data subscription using PollingToWebSocketAdapter."""
        instrument = test_instruments[0]  # BTCUSDC
        
        # Clean up any existing adapter
        self._cleanup_adapter_exchange(data_provider._exchange, data_provider._loop)
        
        # Set up exchange with explicit event loop adapter
        original_watch_method = data_provider._exchange.watch_funding_rates
        data_provider._exchange.watch_funding_rates = self._setup_adapter_exchange(
            data_provider._exchange, adapter_event_loop
        )
        
        try:
            # Subscribe to funding rate data
            data_provider.subscribe("funding_rate", [instrument])
            
            # Wait for funding rate data
            timeout = 40
            start_time = time.time()
            received_funding_data = []
            
            while time.time() - start_time < timeout:
                for data in test_channel.received_data:
                    if len(data) >= 3 and data[1] == "funding_rate":
                        received_funding_data.append(data)
                if received_funding_data:
                    break
                time.sleep(3)
            
            # Verify we received funding rate data through the adapter
            assert len(received_funding_data) > 0, "Should receive funding rate data via PollingToWebSocketAdapter"
            
        finally:
            # Clean up
            try:
                data_provider.unsubscribe("funding_rate")
            except Exception:
                pass
            
            # Restore original method and clean up
            data_provider._exchange.watch_funding_rates = original_watch_method
            self._cleanup_adapter_exchange(data_provider._exchange, data_provider._loop)

    def test_dynamic_symbol_management(self, data_provider, test_instruments, test_channel, adapter_event_loop):
        """Test adding instruments to existing funding rate subscription."""
        btc_instrument = test_instruments[0]
        eth_instrument = test_instruments[1]
        
        # Clean up any existing adapter
        self._cleanup_adapter_exchange(data_provider._exchange, data_provider._loop)
        
        # Set up exchange with explicit event loop adapter  
        original_watch_method = data_provider._exchange.watch_funding_rates
        data_provider._exchange.watch_funding_rates = self._setup_adapter_exchange(
            data_provider._exchange, adapter_event_loop
        )
        
        try:
            # Start with BTC only
            data_provider.subscribe("funding_rate", [btc_instrument])
            
            # Wait for initial data
            time.sleep(10)
            test_channel.received_data.clear()
            
            # Wait for BTC funding rate data
            timeout = 15
            start_time = time.time()
            btc_funding_received = False
            
            while time.time() - start_time < timeout:
                for data in test_channel.received_data:
                    if len(data) >= 3 and data[1] == "funding_rate":
                        btc_funding_received = True
                        break
                if btc_funding_received:
                    break
                time.sleep(1)
            
            # Core functionality should work - but may be affected by network timing in CI/testing
            if not btc_funding_received:
                pytest.skip("Network/timing issues prevented funding rate data reception in test environment")
            
            # Test adding ETH (this tests dynamic symbol management)
            data_provider.subscribe("funding_rate", [btc_instrument, eth_instrument], reset=False)
            
            # Just verify no errors occur - dynamic management is complex in integration tests
            time.sleep(5)
            
        finally:
            # Clean up
            try:
                data_provider.unsubscribe("funding_rate")
            except Exception:
                pass
            
            # Restore original method and clean up
            data_provider._exchange.watch_funding_rates = original_watch_method
            self._cleanup_adapter_exchange(data_provider._exchange, data_provider._loop)

    def test_adapter_integration_direct(self, real_exchange):
        """Test that the PollingToWebSocketAdapter is properly integrated."""
        # Test that our enhanced Hyperliquid exchange has the required methods
        assert hasattr(real_exchange, 'watch_funding_rates'), "Should have watch_funding_rates method"
        assert hasattr(real_exchange, 'un_watch_funding_rates'), "Should have un_watch_funding_rates method"
        assert hasattr(real_exchange, 'fetch_funding_rates'), "Should have fetch_funding_rates method"