"""
Integration tests for CcxtDataProvider with real Hyperliquid exchange.

Validates:
- Basic WebSocket OHLCV subscription
- Funding rate polling via PollingToWebSocketAdapter
- Dynamic symbol management
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.core.basics import CtrlChannel, Instrument, LiveTimeProvider, MarketType
from qubx.utils.runner.accounts import AccountConfigurationManager


def _make_health_monitor():
    """
    Minimal IHealthMonitor stub — returns sane defaults for all protocol methods.
    Using MagicMock so every method call is accepted without explicit implementation.
    """
    mock = MagicMock()
    mock.is_healthy.return_value = True
    mock.get_last_event_time.return_value = None
    mock.get_last_event_times_by_exchange.return_value = {}
    mock.get_last_event_time_by_exchange.return_value = None
    mock.is_stale.return_value = False
    mock.is_exchange_stale.return_value = False
    mock.is_connected.return_value = True
    # - 'with monitor("event_type"):' → monitor("event_type") must return a context manager
    # - setting return_value=mock makes mock() return mock, which MagicMock handles as a CM automatically
    mock.return_value = mock
    return mock


def _make_instruments() -> list[Instrument]:
    return [
        Instrument(
            symbol="BTCUSDC",
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


def _make_channel() -> MagicMock:
    channel = MagicMock(spec=CtrlChannel)
    channel.received_data = []
    channel.send = lambda data: channel.received_data.append(data)
    return channel


def _make_provider(channel: MagicMock) -> CcxtDataProvider:
    """Create a CcxtDataProvider for HYPERLIQUID.F (no credentials needed for public data)."""
    return CcxtDataProvider(
        exchange_name="HYPERLIQUID.F",
        time_provider=LiveTimeProvider(),
        channel=channel,
        health_monitor=_make_health_monitor(),
        account_manager=AccountConfigurationManager(),
        max_ws_retries=3,
        warmup_timeout=30,
    )


# ------------------------------------------------------------------
# Basic integration
# ------------------------------------------------------------------


@pytest.mark.integration
class TestHyperliquidBasicIntegration:
    """Basic integration tests with a live Hyperliquid WebSocket connection."""

    @pytest.fixture
    def channel(self):
        return _make_channel()

    @pytest.fixture
    def instruments(self):
        return _make_instruments()

    @pytest.fixture
    def provider(self, channel):
        p = _make_provider(channel)
        yield p
        try:
            p.stop()
        except Exception:
            pass

    def test_exchange_configured(self, provider):
        """Exchange is created and its id contains 'hyperliquid'."""
        ex = provider._exchange_manager.exchange
        assert ex is not None
        assert "hyperliquid" in ex.id.lower()

    def test_provider_has_subscribe_unsubscribe(self, provider):
        """DataProvider exposes subscribe and unsubscribe callables."""
        assert callable(getattr(provider, "subscribe", None))
        assert callable(getattr(provider, "unsubscribe", None))

    def test_basic_ohlcv_subscription(self, provider, instruments, channel):
        """Subscribe to ohlc_1m for BTCUSDC and receive at least one bar within 20 s."""
        instrument = instruments[0]
        provider.subscribe("ohlc_1m", [instrument])

        t0 = time.time()
        received = []
        while time.time() - t0 < 20:
            received = [d for d in channel.received_data if len(d) >= 3 and d[1] == "ohlc_1m"]
            if received:
                break
            time.sleep(2)

        assert len(received) > 0, "Should receive at least one OHLCV bar within 20 s"
        provider.unsubscribe("ohlc_1m")


# ------------------------------------------------------------------
# Funding rate adapter
# ------------------------------------------------------------------


@pytest.mark.integration
class TestHyperliquidFundingRateAdapter:
    """Integration tests for PollingToWebSocketAdapter with Hyperliquid funding rates."""

    @pytest.fixture(scope="class")
    def adapter_event_loop(self):
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture
    def channel(self):
        return _make_channel()

    @pytest.fixture
    def instruments(self):
        return _make_instruments()

    @pytest.fixture
    def provider(self, channel):
        p = _make_provider(channel)
        yield p
        try:
            p.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ccxt_ex(self, provider: CcxtDataProvider):
        """Underlying ccxt Pro exchange object."""
        return provider._exchange_manager.exchange

    def _async_loop(self, provider: CcxtDataProvider):
        """asyncio loop owned by the ccxt exchange."""
        return self._ccxt_ex(provider).asyncio_loop

    def _cleanup_adapter(self, provider: CcxtDataProvider) -> None:
        ex = self._ccxt_ex(provider)
        if hasattr(ex, "_funding_rate_adapter") and ex._funding_rate_adapter:
            async def _stop():
                await ex._funding_rate_adapter.stop()
            try:
                asyncio.run_coroutine_threadsafe(_stop(), self._async_loop(provider)).result(timeout=10)
                ex._funding_rate_adapter = None
            except Exception:
                pass

    def _setup_adapter(self, provider: CcxtDataProvider, adapter_event_loop):
        """Patch watch_funding_rates on the ccxt exchange to use PollingToWebSocketAdapter."""
        from qubx.connectors.ccxt.adapters.polling_adapter import PollingConfig, PollingToWebSocketAdapter

        ex = self._ccxt_ex(provider)

        async def _watch_with_adapter(symbols=None, params=None):
            if not ex.markets:
                await ex.load_markets()
            if ex._funding_rate_adapter is None:
                ex._funding_rate_adapter = PollingToWebSocketAdapter(
                    fetch_method=ex.fetch_funding_rates,
                    symbols=symbols or ["BTC/USDC:USDC"],
                    params=params or {},
                    config=PollingConfig(poll_interval_seconds=30),
                    event_loop=adapter_event_loop,
                )
                await ex._funding_rate_adapter.start_watching()

            funding_data = await ex._funding_rate_adapter.get_next_data()

            # - normalize: fill missing timestamp / nextFundingTime from Hyperliquid-specific keys
            transformed: dict = {}
            for sym, info in (funding_data or {}).items():
                if isinstance(info, dict):
                    info = info.copy()
                    if info.get("timestamp") is None:
                        info["timestamp"] = info.get("fundingTimestamp")
                    if info.get("nextFundingTime") is None:
                        info["nextFundingTime"] = info.get("nextFundingTimestamp")
                transformed[sym] = info
            return transformed or funding_data

        return _watch_with_adapter

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_adapter_methods_present(self, provider):
        """ccxt exchange exposes watch_funding_rates, un_watch_funding_rates, fetch_funding_rates."""
        ex = self._ccxt_ex(provider)
        assert callable(getattr(ex, "watch_funding_rates", None))
        assert callable(getattr(ex, "un_watch_funding_rates", None))
        assert callable(getattr(ex, "fetch_funding_rates", None))

    def test_funding_rate_subscription(self, provider, instruments, channel, adapter_event_loop):
        """Funding rate data arrives via PollingToWebSocketAdapter within 40 s."""
        instrument = instruments[0]
        ex = self._ccxt_ex(provider)
        self._cleanup_adapter(provider)

        original = ex.watch_funding_rates
        ex.watch_funding_rates = self._setup_adapter(provider, adapter_event_loop)

        try:
            provider.subscribe("funding_rate", [instrument])

            t0 = time.time()
            received = []
            while time.time() - t0 < 40:
                received = [d for d in channel.received_data if len(d) >= 3 and d[1] == "funding_rate"]
                if received:
                    break
                time.sleep(3)

            assert len(received) > 0, "Should receive funding rate data via PollingToWebSocketAdapter within 40 s"
        finally:
            try:
                provider.unsubscribe("funding_rate")
            except Exception:
                pass
            ex.watch_funding_rates = original
            self._cleanup_adapter(provider)

    def test_dynamic_symbol_management(self, provider, instruments, channel, adapter_event_loop):
        """Adding instruments to an existing funding rate subscription does not raise."""
        btc, eth = instruments[0], instruments[1]
        ex = self._ccxt_ex(provider)
        self._cleanup_adapter(provider)

        original = ex.watch_funding_rates
        ex.watch_funding_rates = self._setup_adapter(provider, adapter_event_loop)

        try:
            provider.subscribe("funding_rate", [btc])
            time.sleep(10)
            channel.received_data.clear()

            # - wait for initial BTC data
            t0 = time.time()
            btc_received = False
            while time.time() - t0 < 15:
                if any(len(d) >= 3 and d[1] == "funding_rate" for d in channel.received_data):
                    btc_received = True
                    break
                time.sleep(1)

            if not btc_received:
                pytest.skip("Network timing prevented funding rate reception in test environment")

            # - add ETH — verify no exception on dynamic symbol management
            provider.subscribe("funding_rate", [btc, eth], reset=False)
            time.sleep(5)

        finally:
            try:
                provider.unsubscribe("funding_rate")
            except Exception:
                pass
            ex.watch_funding_rates = original
            self._cleanup_adapter(provider)
