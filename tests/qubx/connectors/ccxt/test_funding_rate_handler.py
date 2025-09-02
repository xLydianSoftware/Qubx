"""
Unit tests for FundingRateDataHandler.

Tests the simplified unified approach where both funding_rate and funding_payment
subscriptions use the same handler but always emit both data types.
"""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from qubx.connectors.ccxt.handlers.funding_rate import FundingRateDataHandler
from qubx.core.basics import AssetType, CtrlChannel, DataType, FundingPayment, FundingRate, Instrument, MarketType


@pytest.fixture
def btc_instrument():
    return Instrument(
        symbol="BTCUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="binance",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


@pytest.fixture
def eth_instrument():
    return Instrument(
        symbol="ETHUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="binance",
        base="ETH",
        quote="USDT",
        settle="USDT",
        exchange_symbol="ETHUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


@pytest.fixture
def mock_exchange():
    # Create mock CCXT exchange
    ccxt_exchange = Mock()
    ccxt_exchange.watch_funding_rates = AsyncMock()
    ccxt_exchange.un_watch_funding_rates = Mock(return_value=None)
    ccxt_exchange.name = "binance"
    ccxt_exchange.market = Mock(
        return_value={
            "symbol": "BTCUSDT",
            "type": "swap",
            "linear": True,
            "base": "BTC",
            "quote": "USDT",
            "settle": "USDT",
            "precision": {"price": 2, "amount": 3},
            "limits": {"amount": {"min": 0.001}},
        }
    )
    
    # Create mock exchange manager that returns the mock exchange
    exchange_manager = Mock()
    exchange_manager.exchange = ccxt_exchange
    return exchange_manager


@pytest.fixture
def mock_data_provider():
    data_provider = Mock()
    data_provider.time_provider.time.return_value = np.datetime64("2024-01-01T00:00:00", "s")
    data_provider._health_monitor.on_data_arrival = Mock()
    return data_provider


@pytest.fixture
def funding_handler(mock_data_provider, mock_exchange):
    return FundingRateDataHandler(
        data_provider=mock_data_provider,
        exchange_manager=mock_exchange,
        exchange_id="binance",
    )


@pytest.fixture
def rate_channel():
    return CtrlChannel("funding_rate_test")


@pytest.fixture
def payment_channel():
    return CtrlChannel("funding_payment_test")


@pytest.fixture
def sample_funding_rate():
    return FundingRate(
        time=np.datetime64("2024-01-01T00:00:00", "s"),
        rate=0.0001,
        interval="8h",
        next_funding_time=np.datetime64("2024-01-01T08:00:00", "s"),
        mark_price=50000.0,
        index_price=50005.0,
    )


class TestSimplifiedFundingHandler:
    """Test the simplified unified funding handler approach."""

    def test_funding_rate_subscription_uses_provided_name(self, funding_handler, btc_instrument, rate_channel):
        """Test that funding rate subscription uses the orchestrator-provided stream name."""
        instruments = {btc_instrument}

        config = funding_handler.prepare_subscription(
            name="orchestrator_provided_name",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )

        # Should use the orchestrator-provided name, not generate its own
        assert config.stream_name == "orchestrator_provided_name"
        assert config.subscription_type == DataType.FUNDING_RATE
        assert config.subscriber_func is not None
        assert config.unsubscriber_func is not None

    def test_funding_payment_subscription_uses_provided_name(self, funding_handler, btc_instrument, payment_channel):
        """Test that funding payment subscription uses the orchestrator-provided stream name."""
        instruments = {btc_instrument}

        config = funding_handler.prepare_subscription(
            name="different_orchestrator_name",
            sub_type=DataType.FUNDING_PAYMENT,
            channel=payment_channel,
            instruments=instruments,
        )

        # Should use the orchestrator-provided name, not generate its own
        assert config.stream_name == "different_orchestrator_name"
        assert config.subscription_type == DataType.FUNDING_PAYMENT
        assert config.subscriber_func is not None
        assert config.unsubscriber_func is not None

    def test_both_subscription_types_work_independently(
        self, funding_handler, btc_instrument, rate_channel, payment_channel
    ):
        """Test that both subscription types can be created independently."""
        instruments = {btc_instrument}

        # Create both subscription configs
        rate_config = funding_handler.prepare_subscription(
            name="rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )

        payment_config = funding_handler.prepare_subscription(
            name="payment_stream",
            sub_type=DataType.FUNDING_PAYMENT,
            channel=payment_channel,
            instruments=instruments,
        )

        # Both should work and be independent
        assert rate_config.stream_name == "rate_stream"
        assert payment_config.stream_name == "payment_stream"
        assert rate_config.subscription_type == DataType.FUNDING_RATE
        assert payment_config.subscription_type == DataType.FUNDING_PAYMENT

    @pytest.mark.asyncio
    @patch("qubx.connectors.ccxt.handlers.funding_rate.ccxt_find_instrument")
    @patch("qubx.connectors.ccxt.handlers.funding_rate.ccxt_convert_funding_rate")
    async def test_subscriber_always_emits_funding_rates(
        self, mock_convert, mock_find_instrument, funding_handler, btc_instrument, rate_channel
    ):
        """Test that the subscriber always emits funding rates regardless of subscription type."""
        instruments = {btc_instrument}

        # Setup mocks
        mock_find_instrument.return_value = btc_instrument
        mock_funding_rate = FundingRate(
            time=np.datetime64("2024-01-01T00:00:00", "s"),
            rate=0.0001,
            interval="8h",
            next_funding_time=np.datetime64("2024-01-01T08:00:00", "s"),
            mark_price=50000.0,
            index_price=50005.0,
        )
        mock_convert.return_value = mock_funding_rate

        config = funding_handler.prepare_subscription(
            name="test_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )

        # Mock exchange response
        funding_handler._exchange_manager.exchange.watch_funding_rates.return_value = {"BTCUSDT": {"some": "data"}}

        # Set up channel to capture sent data
        sent_data = []
        rate_channel.send = lambda data: sent_data.append(data)

        # Call the subscriber function
        await config.subscriber_func()

        # Should always emit funding rate
        funding_rate_data = [data for data in sent_data if data[1] == DataType.FUNDING_RATE]
        assert len(funding_rate_data) == 1

        # Verify the data
        instrument, data_type, funding_rate, is_historical = funding_rate_data[0]
        assert instrument.symbol == "BTCUSDT"
        assert data_type == DataType.FUNDING_RATE
        assert isinstance(funding_rate, FundingRate)
        assert funding_rate.rate == 0.0001
        assert not is_historical

    @pytest.mark.asyncio
    @patch("qubx.connectors.ccxt.handlers.funding_rate.ccxt_find_instrument")
    @patch("qubx.connectors.ccxt.handlers.funding_rate.ccxt_convert_funding_rate")
    async def test_subscriber_emits_payments_when_interval_changes(
        self, mock_convert, mock_find_instrument, funding_handler, btc_instrument, rate_channel
    ):
        """Test that the subscriber emits both funding rates and payments when appropriate."""
        instruments = {btc_instrument}

        # Setup mocks
        mock_find_instrument.return_value = btc_instrument

        config = funding_handler.prepare_subscription(
            name="test_stream",
            sub_type=DataType.FUNDING_PAYMENT,
            channel=rate_channel,
            instruments=instruments,
        )

        # Set up channel to capture sent data
        sent_data = []
        rate_channel.send = lambda data: sent_data.append(data)

        # First call - should not emit payment (first rate)
        first_rate = FundingRate(
            time=np.datetime64("2024-01-01T00:00:00", "s"),
            rate=0.0001,
            interval="8h",
            next_funding_time=np.datetime64("2024-01-01T08:00:00", "s"),
            mark_price=50000.0,
            index_price=50005.0,
        )
        mock_convert.return_value = first_rate
        funding_handler._exchange_manager.exchange.watch_funding_rates.return_value = {"BTCUSDT": {"data": "first"}}

        await config.subscriber_func()

        # Should have funding rate but no payment yet (first rate doesn't trigger payment)
        assert len([d for d in sent_data if d[1] == DataType.FUNDING_RATE]) == 1
        assert len([d for d in sent_data if d[1] == DataType.FUNDING_PAYMENT]) == 0

        # Second call with advanced next_funding_time - should emit payment
        sent_data.clear()
        second_rate = FundingRate(
            time=np.datetime64("2024-01-01T08:00:00", "s"),
            rate=0.0002,
            interval="8h",
            next_funding_time=np.datetime64("2024-01-01T16:00:00", "s"),  # Advanced
            mark_price=50100.0,
            index_price=50105.0,
        )
        mock_convert.return_value = second_rate
        funding_handler._exchange_manager.exchange.watch_funding_rates.return_value = {"BTCUSDT": {"data": "second"}}

        await config.subscriber_func()

        # Should emit both rate and payment (handler always emits both when appropriate)
        funding_rates = [d for d in sent_data if d[1] == DataType.FUNDING_RATE]
        funding_payments = [d for d in sent_data if d[1] == DataType.FUNDING_PAYMENT]

        assert len(funding_rates) == 1  # Always emits funding rate
        assert len(funding_payments) == 1  # Emits payment when interval changes

        # Verify payment data
        instrument, data_type, payment, _ = funding_payments[0]
        assert instrument.symbol == "BTCUSDT"
        assert data_type == DataType.FUNDING_PAYMENT
        assert isinstance(payment, FundingPayment)
        assert payment.funding_rate == 0.0001  # Previous rate
        assert payment.funding_interval_hours == 8

    @pytest.mark.asyncio
    async def test_unsubscriber_calls_exchange_cleanup(self, funding_handler, btc_instrument, rate_channel):
        """Test that the unsubscriber calls exchange cleanup."""
        instruments = {btc_instrument}

        config = funding_handler.prepare_subscription(
            name="test_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )

        # Call the unsubscriber
        await config.unsubscriber_func()

        # Should call exchange unwatch function
        funding_handler._exchange_manager.exchange.un_watch_funding_rates.assert_called_once()

    def test_payment_emission_logic(self, funding_handler, btc_instrument, sample_funding_rate):
        """Test the funding payment emission logic."""
        # First call - should not emit payment
        assert not funding_handler._should_emit_payment(btc_instrument, sample_funding_rate)

        # Second call with same next_funding_time - should not emit payment
        assert not funding_handler._should_emit_payment(btc_instrument, sample_funding_rate)

        # Third call with advanced next_funding_time - should emit payment
        advanced_rate = FundingRate(
            time=np.datetime64("2024-01-01T08:00:00", "s"),
            rate=0.0002,
            interval="8h",
            next_funding_time=np.datetime64("2024-01-01T16:00:00", "s"),  # Advanced
            mark_price=50100.0,
            index_price=50105.0,
        )
        assert funding_handler._should_emit_payment(btc_instrument, advanced_rate)

    def test_extract_interval_hours(self, funding_handler):
        """Test interval hour extraction from different formats."""
        assert funding_handler._extract_interval_hours("8h") == 8
        assert funding_handler._extract_interval_hours("4h") == 4
        assert funding_handler._extract_interval_hours("8") == 8
        assert funding_handler._extract_interval_hours("unknown") == 8  # Default
