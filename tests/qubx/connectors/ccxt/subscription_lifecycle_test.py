import asyncio
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ccxt.pro import Exchange

from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.core.basics import AssetType, CtrlChannel, DataType, Instrument, MarketType
from tests.qubx.core.utils_test import DummyTimeProvider  # type: ignore


class MockAsyncThreadLoop:
    """Mock AsyncThreadLoop for controlled testing"""

    def __init__(self):
        self.submitted_tasks = []
        self.running_futures = {}

    def submit(self, coro):
        """Submit a coroutine and return a mock Future"""
        future = MagicMock(spec=Future)
        future.running.return_value = True
        future.cancel = MagicMock()
        future.result = MagicMock(return_value=None)

        # Store the coroutine for inspection
        self.submitted_tasks.append(coro)
        self.running_futures[id(future)] = {"future": future, "coro": coro, "cancelled": False}

        return future

    def cancel_future(self, future):
        """Mark a future as cancelled"""
        future_id = id(future)
        if future_id in self.running_futures:
            self.running_futures[future_id]["cancelled"] = True
            future.running.return_value = False
            future.cancel.return_value = True


class MockExchange(Exchange):
    def __init__(self):
        self.name = "mock_exchange"
        self.asyncio_loop = asyncio.new_event_loop()
        self.watch_ohlcv_for_symbols = AsyncMock()
        self.watch_trades_for_symbols = AsyncMock()
        self.watch_order_book_for_symbols = AsyncMock()
        self.watch_bids_asks = AsyncMock()
        self.un_watch_trades_for_symbols = AsyncMock()
        self.un_watch_order_book_for_symbols = AsyncMock()
        self.find_timeframe = MagicMock(return_value="1m")
        self.fetch_ohlcv = AsyncMock(return_value=[])
        self.close = AsyncMock()
        self.has = {"watchBidsAsks": True, "watchOrderBookForSymbols": True}


@pytest.fixture
def test_instruments():
    """Create test instruments"""
    btc = Instrument(
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
    eth = Instrument(
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
    return btc, eth


@pytest.fixture
def mock_connector(test_instruments):
    """Create a CcxtDataProvider with mocked components"""
    mock_exchange = MockExchange()

    with patch("qubx.connectors.ccxt.data.AsyncThreadLoop") as mock_loop_class:
        mock_loop = MockAsyncThreadLoop()
        mock_loop_class.return_value = mock_loop

        connector = CcxtDataProvider(
            exchange=mock_exchange, time_provider=DummyTimeProvider(), channel=CtrlChannel("test")
        )

        yield connector, mock_exchange, mock_loop


class TestSubscriptionLifecycle:
    """Comprehensive tests for subscription lifecycle management"""

    def test_subscription_state_transitions(self, mock_connector, test_instruments):
        """Test subscription states: pending -> active -> cleanup"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, eth_instrument = test_instruments

        # Initially no subscriptions
        assert len(connector.get_subscriptions()) == 0
        assert len(connector.get_subscribed_instruments()) == 0

        # Subscribe to OHLC
        subscription_type = DataType.OHLC["1m"]
        instruments = [btc_instrument]

        connector.subscribe(subscription_type, instruments)

        # Should be in pending state initially
        assert not connector.has_subscription(btc_instrument, subscription_type)
        assert connector.has_pending_subscription(btc_instrument, subscription_type)

        # Simulate connection establishment
        connector._mark_subscription_active(subscription_type)

        # Should now be active
        assert connector.has_subscription(btc_instrument, subscription_type)
        assert not connector.has_pending_subscription(btc_instrument, subscription_type)
        assert btc_instrument in connector.get_subscribed_instruments(subscription_type)

    def test_resubscription_cleanup_logic(self, mock_connector, test_instruments):
        """Test the complex resubscription cleanup in _subscribe method"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, eth_instrument = test_instruments

        subscription_type = DataType.TRADE

        # First subscription
        connector.subscribe(subscription_type, [btc_instrument])

        # Verify first subscription setup
        assert len(mock_loop.submitted_tasks) == 1
        first_future = list(mock_loop.running_futures.values())[0]["future"]
        connector._mark_subscription_active(subscription_type)

        # Store references before resubscription
        old_sub_name = connector._subscription_manager._sub_to_name.get(subscription_type)

        # Second subscription (should trigger cleanup)
        connector.subscribe(subscription_type, [eth_instrument], reset=True)

        # Verify cleanup happened - should have new subscription + old subscriber cleanup
        assert len(mock_loop.submitted_tasks) == 3  # New task + cleanup task submitted

        # Old subscription should be disabled
        if old_sub_name:
            assert not connector._connection_manager._is_stream_enabled.get(old_sub_name, True)

        # Old future should be cancelled
        first_future.cancel.assert_called_once()

        # New subscription should be pending
        assert eth_instrument in connector._subscription_manager._pending_subscriptions["trade"]

    def test_subscription_name_collision_handling(self, mock_connector, test_instruments):
        """Test handling of subscription name collisions during rapid changes"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, eth_instrument = test_instruments

        subscription_type = DataType.ORDERBOOK

        # Rapid subscription changes to same instruments (name collision scenario)
        for i in range(3):
            connector.subscribe(subscription_type, [btc_instrument], reset=True)

            # Each subscription should generate a unique name
            current_name = connector._subscription_manager._sub_to_name.get(subscription_type)
            assert current_name is not None
            assert connector._connection_manager._is_stream_enabled.get(current_name, False)

        # Should have 3 subscription tasks + 2 cleanup tasks (from 2nd and 3rd resubscription)
        assert len(mock_loop.submitted_tasks) == 5

    def test_connection_ready_tracking(self, mock_connector, test_instruments):
        """Test _sub_connection_ready flag management"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, _ = test_instruments

        subscription_type = DataType.QUOTE

        # Subscribe
        connector.subscribe(subscription_type, [btc_instrument])

        # Initially not ready
        assert not connector._subscription_manager._sub_connection_ready.get(subscription_type, False)
        assert not connector.has_subscription(btc_instrument, subscription_type)

        # Mark as active (simulates successful connection)
        connector._mark_subscription_active(subscription_type)

        # Should now be ready
        assert connector._subscription_manager._sub_connection_ready.get(subscription_type, False)
        assert connector.has_subscription(btc_instrument, subscription_type)

    def test_old_subscriber_stopping_logic(self, mock_connector, test_instruments):
        """Test _stop_old_subscriber async cleanup"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, eth_instrument = test_instruments

        subscription_type = DataType.OHLC["5m"]

        # First subscription
        connector.subscribe(subscription_type, [btc_instrument])
        first_name = connector._subscription_manager._sub_to_name[subscription_type]
        first_future = list(mock_loop.running_futures.values())[0]["future"]

        # Add unsubscriber callback to test cleanup
        mock_unsubscriber = AsyncMock()
        connector._connection_manager._stream_to_unsubscriber[first_name] = mock_unsubscriber

        # Second subscription (triggers old subscriber stopping)
        connector.subscribe(subscription_type, [eth_instrument], reset=True)

        # Verify old subscription was properly disabled
        assert not connector._connection_manager._is_stream_enabled.get(first_name, True)

        # Verify future was cancelled
        first_future.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_listen_to_stream_connection_establishment(self, mock_connector, test_instruments):
        """Test _listen_to_stream connection establishment logic"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, _ = test_instruments

        # Mock subscriber that succeeds once then stops
        call_count = 0

        async def mock_subscriber():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return  # Successful call
            else:
                # Stop subscription after first success
                connector._connection_manager._is_stream_enabled["test_stream"] = False

        # Mock unsubscriber
        mock_unsubscriber = AsyncMock()

        # Set up subscription name mapping for testing
        test_sub_type = DataType.TRADE
        connector._subscription_manager._sub_to_name[test_sub_type] = "test_stream"
        connector._subscription_manager._pending_subscriptions["trade"] = {btc_instrument}

        # Run _listen_to_stream (would normally be called by AsyncThreadLoop)
        await connector._listen_to_stream(
            subscriber=mock_subscriber,
            exchange=mock_exchange,
            channel=connector.channel,
            name="test_stream",
            unsubscriber=mock_unsubscriber,
        )

        # Verify connection was marked as established
        assert connector._subscription_manager._sub_connection_ready.get(test_sub_type, False)
        assert "trade" in connector._subscription_manager._subscriptions
        assert btc_instrument in connector._subscription_manager._subscriptions["trade"]

    def test_empty_instruments_subscription(self, mock_connector, test_instruments):
        """Test subscribing with empty instruments list"""
        connector, mock_exchange, mock_loop = mock_connector

        subscription_type = DataType.TRADE

        # Subscribe with empty list
        connector.subscribe(subscription_type, [])

        # Should not create any tasks or subscriptions
        assert len(mock_loop.submitted_tasks) == 0
        assert "trade" not in connector._subscription_manager._subscriptions
        assert "trade" not in connector._subscription_manager._pending_subscriptions

    def test_subscription_state_consistency_during_rapid_changes(self, mock_connector, test_instruments):
        """Test state consistency during rapid subscription changes"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, eth_instrument = test_instruments

        subscription_type = DataType.OHLC["1m"]

        # Rapid changes
        test_sequences = [
            [btc_instrument],
            [btc_instrument, eth_instrument],
            [eth_instrument],
            [],  # Empty subscription
        ]

        for instruments in test_sequences:
            connector.subscribe(subscription_type, instruments, reset=True)

            # Verify state consistency
            if instruments:
                expected_symbols = {inst.symbol for inst in instruments}
                pending_instruments = connector._subscription_manager._pending_subscriptions.get("ohlc", set())
                pending_symbols = {inst.symbol for inst in pending_instruments}

                # Should have expected instruments in pending state
                assert expected_symbols.issubset(pending_symbols) or len(pending_symbols) >= len(expected_symbols)
            else:
                # Empty subscription should clear pending state
                assert len(connector._subscription_manager._pending_subscriptions.get("ohlc", set())) == 0

    def test_get_subscribed_instruments_with_pending_fallback(self, mock_connector, test_instruments):
        """Test get_subscribed_instruments fallback to pending subscriptions"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, _ = test_instruments

        subscription_type = DataType.TRADE

        # Subscribe (creates pending subscription)
        connector.subscribe(subscription_type, [btc_instrument])

        # Should return pending instruments when no active ones exist
        subscribed = connector.get_subscribed_instruments(subscription_type)
        assert btc_instrument in subscribed

        # Mark as active
        connector._mark_subscription_active(subscription_type)

        # Should now return active instruments
        subscribed = connector.get_subscribed_instruments(subscription_type)
        assert btc_instrument in subscribed

    def test_subscription_flag_management(self, mock_connector, test_instruments):
        """Test _is_sub_name_enabled flag management throughout lifecycle"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, eth_instrument = test_instruments

        subscription_type = DataType.ORDERBOOK

        # Subscribe to BTC
        connector.subscribe(subscription_type, [btc_instrument])

        # Get the subscription name
        sub_name = connector._subscription_manager._sub_to_name[subscription_type]

        # Flag should be enabled for new subscription
        assert connector._connection_manager._is_stream_enabled[sub_name]

        # Resubscribe to ETH (different instruments, so different name)
        old_name = sub_name
        connector.subscribe(subscription_type, [eth_instrument], reset=True)
        new_name = connector._subscription_manager._sub_to_name[subscription_type]

        # Names should be different due to different instruments
        assert old_name != new_name, f"Expected different names but got: old={old_name}, new={new_name}"

        # Old should be disabled, new should be enabled
        assert not connector._connection_manager._is_stream_enabled.get(old_name, True)
        assert connector._connection_manager._is_stream_enabled[new_name]


class TestSubscriptionErrorHandling:
    """Test error handling in subscription lifecycle"""

    def test_failed_subscription_cleanup(self, mock_connector, test_instruments):
        """Test cleanup when subscription fails"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, _ = test_instruments

        subscription_type = DataType.TRADE

        # Mock subscriber that raises an exception
        mock_exchange.watch_trades_for_symbols.side_effect = Exception("Connection failed")

        # Subscribe
        connector.subscribe(subscription_type, [btc_instrument])

        # Should still create pending subscription
        assert btc_instrument in connector._subscription_manager._pending_subscriptions["trade"]

        # Should have submitted task despite expected failure
        assert len(mock_loop.submitted_tasks) == 1

    def test_subscription_timeout_handling(self, mock_connector, test_instruments):
        """Test handling of subscription timeouts"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, _ = test_instruments

        subscription_type = DataType.OHLC["1m"]

        # First subscription
        connector.subscribe(subscription_type, [btc_instrument])
        first_future = list(mock_loop.running_futures.values())[0]["future"]

        # Simulate timeout scenario by keeping future running
        first_future.running.return_value = True

        # Second subscription (should still proceed despite timeout)
        connector.subscribe(subscription_type, [btc_instrument], reset=True)

        # Should have attempted cancellation
        first_future.cancel.assert_called_once()

        # Should have created new subscription + cleanup task
        assert len(mock_loop.submitted_tasks) == 3


@pytest.mark.asyncio
class TestAsyncSubscriptionLifecycle:
    """Async-specific tests for subscription lifecycle"""

    async def test_concurrent_subscription_changes(self, mock_connector, test_instruments):
        """Test concurrent subscription changes don't interfere"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, eth_instrument = test_instruments

        # Mock the actual async thread loop behavior
        with patch.object(connector, "_loop") as mock_async_loop:
            mock_future = AsyncMock()
            mock_async_loop.submit.return_value = mock_future
            mock_future.running.return_value = False
            mock_future.cancel.return_value = True

            # Simulate rapid concurrent changes
            subscription_type = DataType.QUOTE

            # Multiple rapid subscriptions
            connector.subscribe(subscription_type, [btc_instrument])
            connector.subscribe(subscription_type, [eth_instrument], reset=True)
            connector.subscribe(subscription_type, [btc_instrument, eth_instrument], reset=True)

            # Should have made multiple submit calls (3 subscriptions + 2 cleanup calls)
            assert mock_async_loop.submit.call_count == 5

    async def test_stop_old_subscriber_async_cleanup(self, mock_connector, test_instruments):
        """Test the async _stop_old_subscriber method"""
        connector, mock_exchange, mock_loop = mock_connector
        btc_instrument, _ = test_instruments

        # Create mock future and unsubscriber
        mock_future = AsyncMock()
        mock_future.running.return_value = True
        mock_unsubscriber = AsyncMock()

        old_name = "test_old_subscription"
        connector._connection_manager._stream_to_unsubscriber[old_name] = mock_unsubscriber
        connector._connection_manager._is_stream_enabled[old_name] = True

        # Simulate future stopping after some time
        call_count = 0

        def mock_running():
            nonlocal call_count
            call_count += 1
            return call_count <= 2  # Stop after 2 calls

        mock_future.running.side_effect = mock_running

        # Call _stop_old_subscriber
        await connector._stop_old_subscriber(old_name, mock_future)

        # Verify cleanup happened
        # Note: The flag gets deleted, so it won't exist in the dict
        assert old_name not in connector._connection_manager._is_stream_enabled
        mock_unsubscriber.assert_called_once()
