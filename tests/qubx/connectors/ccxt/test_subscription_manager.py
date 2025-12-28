"""Unit tests for SubscriptionManager."""

import pytest

from qubx.connectors.ccxt.subscription_manager import SubscriptionManager
from qubx.core.basics import AssetType, Instrument, MarketType


@pytest.fixture
def subscription_manager():
    return SubscriptionManager()


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


def _make_instrument(idx: int) -> Instrument:
    base = f"ASSET{idx}"
    quote = "USD"
    return Instrument(
        symbol=f"{base}{quote}",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="test",
        base=base,
        quote=quote,
        settle=quote,
        exchange_symbol=f"{base}{quote}",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


class TestIndividualStreamTracking:
    def test_set_and_get_individual_streams(self, subscription_manager, btc_instrument, eth_instrument):
        """Test storing and retrieving individual stream mappings."""
        subscription_type = "ohlc(1m)"
        streams = {
            btc_instrument: "BTCUSDT:ohlc(1m)",
            eth_instrument: "ETHUSDT:ohlc(1m)",
        }

        # Set streams
        subscription_manager.set_individual_streams(subscription_type, streams)

        # Get streams
        retrieved = subscription_manager.get_individual_streams(subscription_type)
        assert retrieved == streams

    def test_get_individual_streams_empty(self, subscription_manager):
        """Test getting streams for non-existent subscription returns empty dict."""
        result = subscription_manager.get_individual_streams("non_existent")
        assert result == {}

    def test_clear_individual_streams(self, subscription_manager, btc_instrument):
        """Test clearing individual stream mappings."""
        subscription_type = "trades"
        streams = {btc_instrument: "BTCUSDT:trades"}

        subscription_manager.set_individual_streams(subscription_type, streams)
        subscription_manager.clear_individual_streams(subscription_type)

        result = subscription_manager.get_individual_streams(subscription_type)
        assert result == {}

    def test_clear_subscription_state_clears_individual_streams(self, subscription_manager, btc_instrument):
        """Test that clear_subscription_state also clears individual streams."""
        subscription_type = "ohlc(5m)"
        streams = {btc_instrument: "BTCUSDT:ohlc(5m)"}

        # Add subscription and individual streams
        subscription_manager.add_subscription(subscription_type, [btc_instrument])
        subscription_manager.set_individual_streams(subscription_type, streams)

        # Clear all state
        subscription_manager.clear_subscription_state(subscription_type)

        # Verify everything is cleared
        assert subscription_manager.get_individual_streams(subscription_type) == {}
        assert subscription_manager.get_subscription_stream(subscription_type) is None
        assert subscription_type not in subscription_manager._subscriptions
        assert subscription_type not in subscription_manager._pending_subscriptions

    def test_update_individual_streams(self, subscription_manager, btc_instrument, eth_instrument):
        """Test updating individual streams replaces old mappings."""
        subscription_type = "orderbook"

        # Set initial streams
        initial_streams = {btc_instrument: "BTCUSDT:orderbook"}
        subscription_manager.set_individual_streams(subscription_type, initial_streams)

        # Update with new streams
        updated_streams = {
            btc_instrument: "BTCUSDT:orderbook",
            eth_instrument: "ETHUSDT:orderbook",
        }
        subscription_manager.set_individual_streams(subscription_type, updated_streams)

        # Verify update
        result = subscription_manager.get_individual_streams(subscription_type)
        assert result == updated_streams
        assert eth_instrument in result


class TestSubscriptionState:
    def test_add_subscription_creates_pending(self, subscription_manager, btc_instrument):
        """Test that add_subscription creates pending subscription."""
        subscription_type = "trades"

        result = subscription_manager.add_subscription(subscription_type, [btc_instrument])

        assert btc_instrument in result
        assert subscription_type in subscription_manager._pending_subscriptions
        assert not subscription_manager.is_connection_ready(subscription_type)

    def test_mark_subscription_active(self, subscription_manager, btc_instrument):
        """Test marking subscription as active moves it from pending."""
        subscription_type = "trades"

        subscription_manager.add_subscription(subscription_type, [btc_instrument])
        subscription_manager.mark_subscription_active(subscription_type)

        assert subscription_type in subscription_manager._subscriptions
        assert subscription_type not in subscription_manager._pending_subscriptions
        assert subscription_manager.is_connection_ready(subscription_type)

    def test_add_subscription_with_reset(self, subscription_manager, btc_instrument, eth_instrument):
        """Test that reset flag replaces existing subscription."""
        subscription_type = "ohlc(1m)"

        # Add initial subscription
        subscription_manager.add_subscription(subscription_type, [btc_instrument])

        # Add with reset
        result = subscription_manager.add_subscription(subscription_type, [eth_instrument], reset=True)

        assert eth_instrument in result
        assert btc_instrument not in result

    def test_remove_subscription(self, subscription_manager, btc_instrument, eth_instrument):
        """Test removing instruments from subscription."""
        subscription_type = "trades"

        # Add both instruments
        subscription_manager.add_subscription(subscription_type, [btc_instrument, eth_instrument])
        subscription_manager.mark_subscription_active(subscription_type)

        # Remove one
        subscription_manager.remove_subscription(subscription_type, [btc_instrument])

        remaining = subscription_manager.get_subscribed_instruments(subscription_type)
        assert eth_instrument in remaining
        assert btc_instrument not in remaining

    def test_add_subscription_reset_then_union_grows_set(self, subscription_manager):
        """
        Safety check for stale-refresh fix: after reset=True seeds N instruments, reset=False must union with current set.
        """
        subscription_type = "quote"
        instruments39 = [_make_instrument(i) for i in range(39)]
        stale = _make_instrument(39)

        subscription_manager.add_subscription(subscription_type, instruments39, reset=True)
        updated = subscription_manager.add_subscription(subscription_type, [stale], reset=False)

        assert len(updated) == 40
        assert stale in updated
        assert len(subscription_manager.get_subscribed_instruments(subscription_type)) == 40
