"""
Unit tests for SubscriptionOrchestrator.

Tests the orchestration logic for bulk and individual instrument subscriptions,
including resubscription behavior and cleanup.
"""

import concurrent.futures
from unittest.mock import AsyncMock, Mock

import pytest

from qubx.connectors.ccxt.connection_manager import ConnectionManager
from qubx.connectors.ccxt.subscription_config import SubscriptionConfiguration
from qubx.connectors.ccxt.subscription_manager import SubscriptionManager
from qubx.connectors.ccxt.subscription_orchestrator import SubscriptionOrchestrator
from qubx.core.basics import AssetType, CtrlChannel, Instrument, MarketType
from qubx.utils.misc import AsyncThreadLoop


@pytest.fixture
def btc_instrument():
    return Instrument(
        symbol="BTCUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="test",
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
        exchange="test",
        base="ETH",
        quote="USDT",
        settle="USDT",
        exchange_symbol="ETHUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


@pytest.fixture
def sol_instrument():
    return Instrument(
        symbol="SOLUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="test",
        base="SOL",
        quote="USDT",
        settle="USDT",
        exchange_symbol="SOLUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


@pytest.fixture
def mock_loop():
    """Mock AsyncThreadLoop that tracks submitted tasks."""
    loop = Mock(spec=AsyncThreadLoop)
    submitted_tasks = []

    def track_submit(coro):
        future = Mock(spec=concurrent.futures.Future)
        future.cancel = Mock()
        future.done = Mock(return_value=False)
        future.running = Mock(return_value=False)
        future._coro = coro  # Store coroutine for inspection
        submitted_tasks.append(future)
        return future

    loop.submit = Mock(side_effect=track_submit)
    loop.submitted_tasks = submitted_tasks
    return loop


@pytest.fixture
def subscription_manager():
    return SubscriptionManager()


@pytest.fixture
def connection_manager(subscription_manager, mock_loop):
    return ConnectionManager(
        exchange_id="TEST",
        loop=mock_loop,
        subscription_manager=subscription_manager,
    )


@pytest.fixture
def orchestrator(subscription_manager, connection_manager, mock_loop):
    return SubscriptionOrchestrator(
        exchange_id="TEST",
        subscription_manager=subscription_manager,
        connection_manager=connection_manager,
        loop=mock_loop,
    )


@pytest.fixture
def mock_exchange():
    exchange = Mock()
    exchange.name = "TEST"
    return exchange


@pytest.fixture
def ctrl_channel():
    channel = CtrlChannel("test")
    channel.control.set()
    return channel


class TestBulkSubscriptions:
    """Test bulk subscription mode (single stream for all instruments)."""

    def test_bulk_subscription_creates_single_stream(
        self, orchestrator, btc_instrument, eth_instrument, mock_exchange, ctrl_channel
    ):
        """Test that bulk subscription creates a single stream for all instruments."""
        # Arrange
        instruments = {btc_instrument, eth_instrument}
        handler = Mock()

        # For bulk subscriptions, handler returns config with subscriber_func
        # Handler's prepare_subscription will be called by orchestrator
        def mock_prepare(name, sub_type, channel, instruments, **kwargs):
            return SubscriptionConfiguration(
                subscription_type=sub_type,
                channel=None,  # Will be set by orchestrator
                subscriber_func=AsyncMock(),
                stream_name=name,  # Use the name provided by orchestrator
            )

        handler.prepare_subscription.side_effect = mock_prepare

        # Act
        orchestrator.execute_subscription(
            subscription_type="ohlc(1m)",
            instruments=instruments,
            handler=handler,
            exchange=mock_exchange,
            channel=ctrl_channel,
        )

        # Assert
        # Should submit exactly one task for bulk subscription
        assert len(orchestrator._loop.submitted_tasks) == 1

        # Should register stream with subscription manager
        stream_name = orchestrator._subscription_manager.get_subscription_stream("ohlc(1m)")
        assert stream_name is not None
        assert "ohlc(1m):" in stream_name  # Should have the pattern

        # Should register future with connection manager (it's in submitted_tasks)
        assert len(orchestrator._connection_manager._stream_to_coro) > 0

    def test_bulk_resubscription_creates_new_stream(
        self, orchestrator, btc_instrument, eth_instrument, sol_instrument, mock_exchange, ctrl_channel
    ):
        """Test that bulk resubscription creates a new stream with different instruments."""
        # First subscription
        instruments1 = {btc_instrument, eth_instrument}
        handler = Mock()

        # For bulk subscriptions, the handler should return the same config
        # The orchestrator will set the actual stream name
        def prepare_sub(name, sub_type, channel, instruments, **kwargs):
            # Return config with the name that orchestrator generates
            return SubscriptionConfiguration(
                subscription_type=sub_type,
                channel=channel,
                subscriber_func=AsyncMock(),
                stream_name=name,
            )

        handler.prepare_subscription.side_effect = prepare_sub

        orchestrator.execute_subscription(
            subscription_type="trades",
            instruments=instruments1,
            handler=handler,
            exchange=mock_exchange,
            channel=ctrl_channel,
        )

        # Get the first stream name
        first_stream_name = orchestrator._subscription_manager.get_subscription_stream("trades")
        assert first_stream_name is not None
        assert "trades:" in first_stream_name  # Should have hash suffix

        # Second subscription (different instruments)
        instruments2 = {btc_instrument, sol_instrument}

        orchestrator.execute_subscription(
            subscription_type="trades",
            instruments=instruments2,
            handler=handler,
            exchange=mock_exchange,
            channel=ctrl_channel,
        )

        # New stream should be created with different name (different hash)
        second_stream_name = orchestrator._subscription_manager.get_subscription_stream("trades")
        assert second_stream_name != first_stream_name

        # Should have created two tasks total (old one cancelled, new one created)
        # Note: The actual cancellation happens asynchronously in the implementation
        assert len(orchestrator._loop.submitted_tasks) == 2

    def test_bulk_resubscription_same_instruments_reuses_stream(
        self, orchestrator, btc_instrument, eth_instrument, mock_exchange, ctrl_channel
    ):
        """Test that resubscribing with same instruments reuses existing stream."""
        instruments = {btc_instrument, eth_instrument}
        handler = Mock()

        # First subscription
        def mock_prepare_first(name, sub_type, channel, instruments, **kwargs):
            return SubscriptionConfiguration(
                subscription_type=sub_type,
                channel=None,  # Will be set by orchestrator
                subscriber_func=AsyncMock(),
                stream_name=name,
            )

        handler.prepare_subscription.side_effect = mock_prepare_first

        orchestrator.execute_subscription(
            subscription_type="ohlc(1m)",
            instruments=instruments,
            handler=handler,
            exchange=mock_exchange,
            channel=ctrl_channel,
        )

        # Should have created one stream
        assert len(orchestrator._loop.submitted_tasks) == 1
        original_future = orchestrator._loop.submitted_tasks[0]
        first_stream_name = orchestrator._subscription_manager.get_subscription_stream("ohlc(1m)")

        # Second subscription with SAME instruments (should reuse stream)
        def mock_prepare_second(name, sub_type, channel, instruments, **kwargs):
            return SubscriptionConfiguration(
                subscription_type=sub_type,
                channel=None,
                subscriber_func=AsyncMock(),
                stream_name=name,  # Same name because same instruments
            )

        handler.prepare_subscription.side_effect = mock_prepare_second

        orchestrator.execute_subscription(
            subscription_type="ohlc(1m)",
            instruments=instruments,  # Same instruments = same hash = same name
            handler=handler,
            exchange=mock_exchange,
            channel=ctrl_channel,
        )

        # Should NOT have created a new stream (reused existing)
        assert len(orchestrator._loop.submitted_tasks) == 1

        # Stream name should remain the same
        second_stream_name = orchestrator._subscription_manager.get_subscription_stream("ohlc(1m)")
        assert second_stream_name == first_stream_name

        # Original future should NOT be cancelled (stream reused)
        original_future.cancel.assert_not_called()


class TestIndividualSubscriptions:
    """Test individual instrument subscription mode."""

    def test_individual_subscription_creates_multiple_streams(
        self, orchestrator, btc_instrument, eth_instrument, mock_exchange, ctrl_channel
    ):
        """Test that individual subscription creates separate stream per instrument."""
        # Arrange
        instruments = {btc_instrument, eth_instrument}
        handler = Mock()

        # Create individual subscribers
        btc_subscriber = AsyncMock()
        eth_subscriber = AsyncMock()

        config = SubscriptionConfiguration(
            subscription_type="orderbook",
            channel=ctrl_channel,
            instrument_subscribers={
                btc_instrument: btc_subscriber,
                eth_instrument: eth_subscriber,
            },
        )
        handler.prepare_subscription.return_value = config

        # Act
        orchestrator.execute_subscription(
            subscription_type="orderbook",
            instruments=instruments,
            handler=handler,
            exchange=mock_exchange,
            channel=ctrl_channel,
        )

        # Assert
        # Should create one stream per instrument
        assert len(orchestrator._loop.submitted_tasks) == 2

        # Should track individual streams in subscription manager
        individual_streams = orchestrator._subscription_manager.get_individual_streams("orderbook")
        assert len(individual_streams) == 2
        assert btc_instrument in individual_streams
        assert eth_instrument in individual_streams

    def test_individual_resubscription_cleans_up_and_restarts(
        self, orchestrator, btc_instrument, eth_instrument, sol_instrument, mock_exchange, ctrl_channel
    ):
        """Test that individual resubscription properly cleans up old streams and starts fresh ones."""
        # First subscription: BTC and ETH
        instruments1 = {btc_instrument, eth_instrument}
        handler = Mock()

        config1 = SubscriptionConfiguration(
            subscription_type="trades",
            channel=ctrl_channel,
            instrument_subscribers={
                btc_instrument: AsyncMock(),
                eth_instrument: AsyncMock(),
            },
        )
        handler.prepare_subscription.return_value = config1

        orchestrator.execute_subscription(
            subscription_type="trades",
            instruments=instruments1,
            handler=handler,
            exchange=mock_exchange,
            channel=ctrl_channel,
        )

        # Store references to the first two futures
        btc_future = orchestrator._loop.submitted_tasks[0]
        eth_future = orchestrator._loop.submitted_tasks[1]

        # Second subscription: BTC, ETH, and SOL (adding SOL)
        instruments2 = {btc_instrument, eth_instrument, sol_instrument}

        config2 = SubscriptionConfiguration(
            subscription_type="trades",
            channel=ctrl_channel,
            instrument_subscribers={
                btc_instrument: AsyncMock(),
                eth_instrument: AsyncMock(),
                sol_instrument: AsyncMock(),
            },
        )
        handler.prepare_subscription.return_value = config2

        orchestrator.execute_subscription(
            subscription_type="trades",
            instruments=instruments2,
            handler=handler,
            exchange=mock_exchange,
            channel=ctrl_channel,
        )

        # Old futures should NOT be cancelled - they should be reused for ongoing subscriptions
        btc_future.cancel.assert_not_called()
        eth_future.cancel.assert_not_called()

        # Should create 1 new future for SOL only (BTC and ETH streams reused)
        assert len(orchestrator._loop.submitted_tasks) == 3  # 2 original + 1 new for SOL

    def test_individual_resubscription_cleans_up_removed_instruments(
        self,
        orchestrator,
        btc_instrument,
        eth_instrument,
        sol_instrument,
        mock_exchange,
        ctrl_channel,
        connection_manager,
    ):
        """Test that removed instruments get their streams stopped."""
        # First subscription: BTC, ETH, SOL
        instruments1 = {btc_instrument, eth_instrument, sol_instrument}
        handler = Mock()

        config1 = SubscriptionConfiguration(
            subscription_type="quotes",
            channel=ctrl_channel,
            instrument_subscribers={
                btc_instrument: AsyncMock(),
                eth_instrument: AsyncMock(),
                sol_instrument: AsyncMock(),
            },
        )
        handler.prepare_subscription.return_value = config1

        orchestrator.execute_subscription(
            subscription_type="quotes",
            instruments=instruments1,
            handler=handler,
            exchange=mock_exchange,
            channel=ctrl_channel,
        )

        # Mock the connection manager's stop_stream method
        connection_manager.stop_stream = Mock()

        # Second subscription: Only BTC and ETH (removing SOL)
        instruments2 = {btc_instrument, eth_instrument}

        config2 = SubscriptionConfiguration(
            subscription_type="quotes",
            channel=ctrl_channel,
            instrument_subscribers={
                btc_instrument: AsyncMock(),
                eth_instrument: AsyncMock(),
            },
        )
        handler.prepare_subscription.return_value = config2

        orchestrator.execute_subscription(
            subscription_type="quotes",
            instruments=instruments2,
            handler=handler,
            exchange=mock_exchange,
            channel=ctrl_channel,
        )

        # stop_stream should be called for SOL's stream with wait=False
        assert connection_manager.stop_stream.called
        # Check that at least one call was made with wait=False
        calls_with_wait_false = [
            c for c in connection_manager.stop_stream.call_args_list if c.kwargs.get("wait") is False
        ]
        assert len(calls_with_wait_false) >= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_instruments_does_nothing(self, orchestrator, mock_exchange, ctrl_channel):
        """Test that empty instrument set is handled gracefully."""
        handler = Mock()

        orchestrator.execute_subscription(
            subscription_type="trades",
            instruments=set(),
            handler=handler,
            exchange=mock_exchange,
            channel=ctrl_channel,
        )

        # Should not call handler or submit any tasks
        handler.prepare_subscription.assert_not_called()
        assert len(orchestrator._loop.submitted_tasks) == 0

    def test_invalid_subscription_config_raises_error(self):
        """Test that invalid subscription configuration raises appropriate error."""
        # Config with both bulk and individual subscribers should raise error
        with pytest.raises(ValueError, match="Cannot specify both"):
            SubscriptionConfiguration(
                subscription_type="invalid",
                subscriber_func=AsyncMock(),
                instrument_subscribers={Mock(): AsyncMock()},
                stream_name="invalid",
            )
