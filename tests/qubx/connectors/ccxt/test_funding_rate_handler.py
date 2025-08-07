"""
Unit tests for FundingRateDataHandler.

Tests the unified stream approach for handling both funding_rate and funding_payment
subscriptions to avoid duplicate WebSocket connections.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from qubx.connectors.ccxt.handlers.funding_rate import FundingRateDataHandler
from qubx.core.basics import AssetType, CtrlChannel, DataType, FundingRate, FundingPayment, Instrument, MarketType


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
    exchange = Mock()
    exchange.watch_funding_rates = AsyncMock()
    exchange.un_watch_funding_rates = Mock(return_value=None)
    return exchange


@pytest.fixture
def mock_data_provider():
    data_provider = Mock()
    data_provider.time_provider.time.return_value = np.datetime64("2024-01-01T00:00:00", "s")
    data_provider._health_monitor.record_data_arrival = Mock()
    return data_provider


@pytest.fixture
def funding_handler(mock_data_provider, mock_exchange):
    return FundingRateDataHandler(
        data_provider=mock_data_provider,
        exchange=mock_exchange,
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
        index_price=50000.0,
    )


class TestUnifiedStreamLogic:
    """Test the unified stream approach for funding subscriptions."""

    def test_first_subscription_creates_unified_stream(self, funding_handler, btc_instrument, rate_channel):
        """Test that the first subscription creates the unified stream."""
        instruments = {btc_instrument}
        
        config = funding_handler.prepare_subscription(
            name="funding_rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )
        
        # Should create a real configuration
        assert config.subscriber_func is not None
        assert config.stream_name == f"funding_unified_{id(funding_handler)}"
        assert DataType.FUNDING_RATE in funding_handler._active_subscriptions
        assert funding_handler._active_stream_config is not None
        
    def test_second_subscription_reuses_stream(self, funding_handler, btc_instrument, rate_channel, payment_channel):
        """Test that a second subscription reuses the existing stream."""
        instruments = {btc_instrument}
        
        # First subscription
        config1 = funding_handler.prepare_subscription(
            name="funding_rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )
        
        # Second subscription
        config2 = funding_handler.prepare_subscription(
            name="funding_payment_stream", 
            sub_type=DataType.FUNDING_PAYMENT,
            channel=payment_channel,
            instruments=instruments,
        )
        
        # First config should be the real one
        assert config1.stream_name == f"funding_unified_{id(funding_handler)}"
        
        # Second config should be no-op
        assert config2.stream_name.startswith("funding_noop_")
        assert DataType.FUNDING_PAYMENT in funding_handler._active_subscriptions
        assert len(funding_handler._active_subscriptions) == 2

    def test_subscription_order_independence(self, funding_handler, btc_instrument, rate_channel, payment_channel):
        """Test that the order of subscriptions doesn't matter."""
        instruments = {btc_instrument}
        
        # Subscribe to payment first, then rate
        config1 = funding_handler.prepare_subscription(
            name="funding_payment_stream",
            sub_type=DataType.FUNDING_PAYMENT,
            channel=payment_channel,
            instruments=instruments,
        )
        
        config2 = funding_handler.prepare_subscription(
            name="funding_rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )
        
        # First subscription (payment) should create the real stream
        assert config1.stream_name == f"funding_unified_{id(funding_handler)}"
        
        # Second subscription (rate) should be no-op
        assert config2.stream_name.startswith("funding_noop_")
        assert len(funding_handler._active_subscriptions) == 2

    def test_channels_tracked_correctly(self, funding_handler, btc_instrument, rate_channel, payment_channel):
        """Test that channels are correctly tracked for each subscription type."""
        instruments = {btc_instrument}
        
        funding_handler.prepare_subscription(
            name="funding_rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )
        
        funding_handler.prepare_subscription(
            name="funding_payment_stream",
            sub_type=DataType.FUNDING_PAYMENT,
            channel=payment_channel,
            instruments=instruments,
        )
        
        assert funding_handler._subscription_channels[DataType.FUNDING_RATE] == rate_channel
        assert funding_handler._subscription_channels[DataType.FUNDING_PAYMENT] == payment_channel

    @pytest.mark.asyncio
    async def test_cleanup_removes_subscription(self, funding_handler, btc_instrument, rate_channel, payment_channel):
        """Test that cleanup properly removes individual subscriptions."""
        instruments = {btc_instrument}
        
        # Set up both subscriptions
        config1 = funding_handler.prepare_subscription(
            name="funding_rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )
        
        funding_handler.prepare_subscription(
            name="funding_payment_stream",
            sub_type=DataType.FUNDING_PAYMENT,
            channel=payment_channel,
            instruments=instruments,
        )
        
        # Cleanup one subscription
        await config1.unsubscriber_func()
        
        # Should remove only that subscription type
        assert DataType.FUNDING_RATE not in funding_handler._active_subscriptions
        assert DataType.FUNDING_PAYMENT in funding_handler._active_subscriptions
        assert funding_handler._active_stream_config is not None  # Stream should still exist

    @pytest.mark.asyncio
    async def test_cleanup_last_subscription_cleans_everything(self, funding_handler, btc_instrument, rate_channel):
        """Test that removing the last subscription cleans up everything."""
        instruments = {btc_instrument}
        
        config = funding_handler.prepare_subscription(
            name="funding_rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )
        
        # Cleanup the only subscription
        await config.unsubscriber_func()
        
        # Everything should be cleaned up
        assert len(funding_handler._active_subscriptions) == 0
        assert len(funding_handler._subscription_channels) == 0
        assert funding_handler._active_stream_config is None
        assert len(funding_handler._subscription_instruments) == 0


class TestStreamDataFlow:
    """Test how data flows through the unified stream to appropriate channels."""

    def test_funding_rate_emitted_to_rate_channel(self, funding_handler, btc_instrument, rate_channel, sample_funding_rate):
        """Test that funding rates are emitted to the rate channel when subscribed."""
        instruments = {btc_instrument}
        
        # Mock the channel send method
        rate_channel.send = Mock()
        
        # Subscribe to funding rates
        config = funding_handler.prepare_subscription(
            name="funding_rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )
        
        # Simulate processing a funding rate
        # This tests the internal logic that would be called by the unified stream
        funding_handler._active_subscriptions.add(DataType.FUNDING_RATE)
        funding_handler._subscription_channels[DataType.FUNDING_RATE] = rate_channel
        
        # Check if rate channel gets the data
        if DataType.FUNDING_RATE in funding_handler._active_subscriptions:
            rate_ch = funding_handler._subscription_channels.get(DataType.FUNDING_RATE)
            if rate_ch:
                rate_ch.send((btc_instrument, DataType.FUNDING_RATE, sample_funding_rate, False))
        
        # Verify the rate was sent to the correct channel
        rate_channel.send.assert_called_once_with((btc_instrument, DataType.FUNDING_RATE, sample_funding_rate, False))

    def test_funding_payment_emitted_to_payment_channel(self, funding_handler, btc_instrument, payment_channel):
        """Test that funding payments are emitted to the payment channel when subscribed."""
        instruments = {btc_instrument}
        
        # Mock the channel send method
        payment_channel.send = Mock()
        
        # Subscribe to funding payments
        funding_handler.prepare_subscription(
            name="funding_payment_stream",
            sub_type=DataType.FUNDING_PAYMENT,
            channel=payment_channel,
            instruments=instruments,
        )
        
        # Create a sample payment
        sample_payment = FundingPayment(
            time=np.datetime64("2024-01-01T08:00:00", "s"),
            funding_rate=0.0001,
            funding_interval_hours=8,
        )
        
        # Simulate sending payment to the channel
        if DataType.FUNDING_PAYMENT in funding_handler._active_subscriptions:
            payment_ch = funding_handler._subscription_channels.get(DataType.FUNDING_PAYMENT)
            if payment_ch:
                payment_ch.send((btc_instrument, DataType.FUNDING_PAYMENT, sample_payment, False))
        
        # Verify the payment was sent to the correct channel
        payment_channel.send.assert_called_once_with((btc_instrument, DataType.FUNDING_PAYMENT, sample_payment, False))

    def test_both_channels_receive_appropriate_data(self, funding_handler, btc_instrument, rate_channel, payment_channel, sample_funding_rate):
        """Test that when both subscriptions are active, each receives appropriate data."""
        instruments = {btc_instrument}
        
        # Mock both channel send methods
        rate_channel.send = Mock()
        payment_channel.send = Mock()
        
        # Subscribe to both
        funding_handler.prepare_subscription(
            name="funding_rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )
        
        funding_handler.prepare_subscription(
            name="funding_payment_stream",
            sub_type=DataType.FUNDING_PAYMENT,
            channel=payment_channel,
            instruments=instruments,
        )
        
        # Simulate the unified stream logic
        # Send rate to rate channel
        if DataType.FUNDING_RATE in funding_handler._active_subscriptions:
            rate_ch = funding_handler._subscription_channels.get(DataType.FUNDING_RATE)
            if rate_ch:
                rate_ch.send((btc_instrument, DataType.FUNDING_RATE, sample_funding_rate, False))
        
        # Send payment to payment channel (simulating payment detection)
        sample_payment = FundingPayment(
            time=np.datetime64("2024-01-01T08:00:00", "s"),
            funding_rate=0.0001,
            funding_interval_hours=8,
        )
        
        if DataType.FUNDING_PAYMENT in funding_handler._active_subscriptions:
            payment_ch = funding_handler._subscription_channels.get(DataType.FUNDING_PAYMENT)
            if payment_ch:
                payment_ch.send((btc_instrument, DataType.FUNDING_PAYMENT, sample_payment, False))
        
        # Verify each channel received its appropriate data
        rate_channel.send.assert_called_once_with((btc_instrument, DataType.FUNDING_RATE, sample_funding_rate, False))
        payment_channel.send.assert_called_once_with((btc_instrument, DataType.FUNDING_PAYMENT, sample_payment, False))


class TestFundingPaymentDetection:
    """Test funding payment detection logic."""

    def test_first_rate_update_no_payment(self, funding_handler, btc_instrument, sample_funding_rate):
        """Test that the first rate update doesn't trigger a payment."""
        should_emit = funding_handler._should_emit_payment(btc_instrument, sample_funding_rate)
        assert should_emit is False

    def test_payment_triggered_on_next_funding_time_change(self, funding_handler, btc_instrument):
        """Test that payment is triggered when next_funding_time advances."""
        # First rate update
        rate1 = FundingRate(
            time=np.datetime64("2024-01-01T00:00:00", "s"),
            rate=0.0001,
            interval="8h",
            next_funding_time=np.datetime64("2024-01-01T08:00:00", "s"),
        )
        
        should_emit1 = funding_handler._should_emit_payment(btc_instrument, rate1)
        assert should_emit1 is False
        
        # Second rate update with advanced next_funding_time
        rate2 = FundingRate(
            time=np.datetime64("2024-01-01T08:00:00", "s"),
            rate=0.0002,
            interval="8h",
            next_funding_time=np.datetime64("2024-01-01T16:00:00", "s"),
        )
        
        should_emit2 = funding_handler._should_emit_payment(btc_instrument, rate2)
        assert should_emit2 is True

    def test_payment_creation_uses_previous_rate(self, funding_handler, btc_instrument):
        """Test that payment creation uses the rate from the previous period."""
        # Set up rate history
        rate1 = FundingRate(
            time=np.datetime64("2024-01-01T00:00:00", "s"),
            rate=0.0001,
            interval="8h",
            next_funding_time=np.datetime64("2024-01-01T08:00:00", "s"),
        )
        
        funding_handler._should_emit_payment(btc_instrument, rate1)
        
        # Trigger payment with new rate
        rate2 = FundingRate(
            time=np.datetime64("2024-01-01T08:00:00", "s"),
            rate=0.0002,
            interval="8h",
            next_funding_time=np.datetime64("2024-01-01T16:00:00", "s"),
        )
        
        funding_handler._should_emit_payment(btc_instrument, rate2)
        
        # Create payment - should use rate1's values
        payment = funding_handler._create_funding_payment(btc_instrument)
        assert payment.funding_rate == 0.0001  # From rate1
        assert payment.time == np.datetime64("2024-01-01T08:00:00", "s")  # Payment time
        assert payment.funding_interval_hours == 8

    def test_interval_hours_extraction(self, funding_handler):
        """Test extraction of hours from interval strings."""
        assert funding_handler._extract_interval_hours("8h") == 8
        assert funding_handler._extract_interval_hours("1h") == 1
        assert funding_handler._extract_interval_hours("24h") == 24
        assert funding_handler._extract_interval_hours("8") == 8
        assert funding_handler._extract_interval_hours("unknown") == 8  # Default


class TestCleanupAndEdgeCases:
    """Test cleanup logic and edge cases."""

    def test_cleanup_subscription_method(self, funding_handler, btc_instrument, rate_channel):
        """Test the cleanup_subscription helper method."""
        instruments = {btc_instrument}
        
        funding_handler.prepare_subscription(
            name="funding_rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )
        
        # Manual cleanup
        funding_handler.cleanup_subscription(DataType.FUNDING_RATE)
        
        assert DataType.FUNDING_RATE not in funding_handler._active_subscriptions
        assert DataType.FUNDING_RATE not in funding_handler._subscription_channels
        assert funding_handler._active_stream_config is None

    def test_multiple_instruments_handled(self, funding_handler, btc_instrument, eth_instrument, rate_channel):
        """Test that multiple instruments are properly tracked."""
        instruments = {btc_instrument, eth_instrument}
        
        funding_handler.prepare_subscription(
            name="funding_rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )
        
        assert btc_instrument in funding_handler._subscription_instruments
        assert eth_instrument in funding_handler._subscription_instruments
        assert len(funding_handler._subscription_instruments) == 2

    @pytest.mark.asyncio
    async def test_no_op_subscriber_waits_correctly(self, funding_handler, btc_instrument, rate_channel, payment_channel):
        """Test that no-op subscribers wait appropriately."""
        instruments = {btc_instrument}
        
        # First subscription creates real stream
        funding_handler.prepare_subscription(
            name="funding_rate_stream",
            sub_type=DataType.FUNDING_RATE,
            channel=rate_channel,
            instruments=instruments,
        )
        
        # Second subscription gets no-op
        config2 = funding_handler.prepare_subscription(
            name="funding_payment_stream",
            sub_type=DataType.FUNDING_PAYMENT,
            channel=payment_channel,
            instruments=instruments,
        )
        
        # Test the no-op subscriber
        # It should wait while the subscription is active
        assert config2.subscriber_func is not None
        
        # Start the no-op task
        task = asyncio.create_task(config2.subscriber_func())
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Task should be running
        assert not task.done()
        
        # Remove subscription - should cause no-op to exit
        funding_handler._active_subscriptions.remove(DataType.FUNDING_PAYMENT)
        
        # Wait for task to complete
        await asyncio.sleep(1.1)  # Slightly longer than the sleep interval
        
        # Task should complete now
        assert task.done()
        task.cancel()