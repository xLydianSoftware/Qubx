"""
Unit tests for SubscriptionManager component.

Tests the subscription state management functionality in isolation,
focusing on state transitions, subscription grouping, and cleanup logic.
"""

from qubx.connectors.ccxt.subscription_manager import SubscriptionManager
from qubx.core.basics import DataType


class TestSubscriptionManager:
    """Test suite for SubscriptionManager component."""

    def test_initialization(self):
        """Test that SubscriptionManager initializes with empty state."""
        manager = SubscriptionManager()
        
        assert len(manager._subscriptions) == 0
        assert len(manager._pending_subscriptions) == 0
        assert len(manager._sub_connection_ready) == 0
        assert len(manager._sub_to_name) == 0
        assert len(manager._symbol_to_instrument) == 0

    def test_add_subscription_new_type(self, subscription_manager, mock_instruments):
        """Test adding instruments to a new subscription type."""
        instruments = mock_instruments[:2]  # BTC and ETH
        subscription_type = "ohlc"
        
        updated_instruments = subscription_manager.add_subscription(
            subscription_type, instruments, reset=False
        )
        
        # Should return all instruments as they're new
        assert updated_instruments == set(instruments)
        
        # Should be added to pending subscriptions
        assert subscription_type in subscription_manager._pending_subscriptions
        assert subscription_manager._pending_subscriptions[subscription_type] == set(instruments)
        
        # Should not be in active subscriptions yet
        assert subscription_type not in subscription_manager._subscriptions

    def test_add_subscription_existing_type(self, subscription_manager, mock_instruments):
        """Test adding instruments to existing subscription type."""
        instruments_batch1 = mock_instruments[:2]  # BTC and ETH
        instruments_batch2 = [mock_instruments[2]]  # ADA
        subscription_type = "ohlc"
        
        # Add first batch
        subscription_manager.add_subscription(subscription_type, instruments_batch1)
        
        # Add second batch
        updated_instruments = subscription_manager.add_subscription(
            subscription_type, instruments_batch2
        )
        
        # Should return only new instruments
        assert updated_instruments == set(instruments_batch2)
        
        # Should have all instruments pending
        all_instruments = set(instruments_batch1 + instruments_batch2)
        assert subscription_manager._pending_subscriptions[subscription_type] == all_instruments

    def test_add_subscription_with_reset(self, subscription_manager, mock_instruments):
        """Test adding instruments with reset=True replaces existing."""
        instruments_batch1 = mock_instruments[:2]  # BTC and ETH
        instruments_batch2 = [mock_instruments[2]]  # ADA
        subscription_type = "ohlc"
        
        # Add first batch
        subscription_manager.add_subscription(subscription_type, instruments_batch1)
        
        # Add second batch with reset
        updated_instruments = subscription_manager.add_subscription(
            subscription_type, instruments_batch2, reset=True
        )
        
        # Should return all instruments from second batch
        assert updated_instruments == set(instruments_batch2)
        
        # Should only have second batch pending
        assert subscription_manager._pending_subscriptions[subscription_type] == set(instruments_batch2)

    def test_mark_subscription_active(self, subscription_manager, mock_instruments):
        """Test marking subscription as active moves from pending to active."""
        instruments = mock_instruments[:2]
        subscription_type = "ohlc"
        
        # Add subscription (goes to pending)
        subscription_manager.add_subscription(subscription_type, instruments)
        
        # Mark as active
        subscription_manager.mark_subscription_active(subscription_type)
        
        # Should be in active subscriptions
        assert subscription_type in subscription_manager._subscriptions
        assert subscription_manager._subscriptions[subscription_type] == set(instruments)
        
        # Should no longer be pending
        assert subscription_type not in subscription_manager._pending_subscriptions
        
        # Connection should be marked as ready
        assert subscription_manager._sub_connection_ready[subscription_type] is True

    def test_has_subscription_active(self, subscription_manager, mock_instruments):
        """Test has_subscription returns True for active subscriptions."""
        instrument = mock_instruments[0]
        subscription_type = "ohlc"
        
        # Add and activate subscription
        subscription_manager.add_subscription(subscription_type, [instrument])
        subscription_manager.mark_subscription_active(subscription_type)
        
        # Should have active subscription
        assert subscription_manager.has_subscription(instrument, subscription_type) is True

    def test_has_subscription_pending(self, subscription_manager, mock_instruments):
        """Test has_subscription returns False for pending subscriptions."""
        instrument = mock_instruments[0] 
        subscription_type = "ohlc"
        
        # Add subscription (stays pending)
        subscription_manager.add_subscription(subscription_type, [instrument])
        
        # Should not have active subscription
        assert subscription_manager.has_subscription(instrument, subscription_type) is False

    def test_has_subscription_nonexistent(self, subscription_manager, mock_instruments):
        """Test has_subscription returns False for nonexistent subscriptions."""
        instrument = mock_instruments[0]
        subscription_type = "ohlc"
        
        # Should not have subscription
        assert subscription_manager.has_subscription(instrument, subscription_type) is False

    def test_has_pending_subscription(self, subscription_manager, mock_instruments):
        """Test has_pending_subscription correctly identifies pending state."""
        instrument = mock_instruments[0]
        subscription_type = "ohlc"
        
        # Initially no pending subscription
        assert subscription_manager.has_pending_subscription(instrument, subscription_type) is False
        
        # Add subscription (goes to pending)
        subscription_manager.add_subscription(subscription_type, [instrument])
        assert subscription_manager.has_pending_subscription(instrument, subscription_type) is True
        
        # Mark as active (no longer pending)
        subscription_manager.mark_subscription_active(subscription_type)
        assert subscription_manager.has_pending_subscription(instrument, subscription_type) is False

    def test_remove_subscription_active(self, subscription_manager, mock_instruments):
        """Test removing instruments from active subscription."""
        instruments = mock_instruments[:2]
        remove_instruments = [mock_instruments[0]]  # Remove first instrument
        subscription_type = "ohlc"
        
        # Add and activate subscription
        subscription_manager.add_subscription(subscription_type, instruments)
        subscription_manager.mark_subscription_active(subscription_type)
        
        # Remove one instrument
        subscription_manager.remove_subscription(subscription_type, remove_instruments)
        
        # Should have only the remaining instrument
        remaining_instruments = set(instruments) - set(remove_instruments)
        assert subscription_manager._subscriptions[subscription_type] == remaining_instruments

    def test_remove_subscription_all_instruments(self, subscription_manager, mock_instruments):
        """Test removing all instruments cleans up subscription completely."""
        instruments = mock_instruments[:2]
        subscription_type = "ohlc"
        
        # Add and activate subscription
        subscription_manager.add_subscription(subscription_type, instruments)
        subscription_manager.mark_subscription_active(subscription_type)
        
        # Remove all instruments
        subscription_manager.remove_subscription(subscription_type, instruments)
        
        # Subscription should be completely removed
        assert subscription_type not in subscription_manager._subscriptions
        assert subscription_manager._sub_connection_ready[subscription_type] is False

    def test_remove_subscription_pending(self, subscription_manager, mock_instruments):
        """Test removing instruments from pending subscription."""
        instruments = mock_instruments[:2]
        remove_instruments = [mock_instruments[0]]
        subscription_type = "ohlc"
        
        # Add subscription (stays pending)
        subscription_manager.add_subscription(subscription_type, instruments)
        
        # Remove one instrument
        subscription_manager.remove_subscription(subscription_type, remove_instruments)
        
        # Should have only the remaining instrument pending
        remaining_instruments = set(instruments) - set(remove_instruments)
        assert subscription_manager._pending_subscriptions[subscription_type] == remaining_instruments

    def test_get_subscriptions_for_instrument(self, subscription_manager, mock_instruments):
        """Test getting all subscription types for a specific instrument."""
        instrument = mock_instruments[0]
        
        # Add multiple subscription types for the same instrument
        subscription_manager.add_subscription("ohlc", [instrument])
        subscription_manager.add_subscription("trade", [instrument])
        subscription_manager.mark_subscription_active("ohlc")
        subscription_manager.mark_subscription_active("trade")
        
        subscriptions = subscription_manager.get_subscriptions(instrument)
        
        # Should return both active subscription types
        assert set(subscriptions) == {"ohlc", "trade"}

    def test_get_subscriptions_all(self, subscription_manager, mock_instruments):
        """Test getting all active subscription types."""
        # Add multiple subscriptions
        subscription_manager.add_subscription("ohlc", mock_instruments[:2])
        subscription_manager.add_subscription("trade", [mock_instruments[0]])
        subscription_manager.mark_subscription_active("ohlc")
        subscription_manager.mark_subscription_active("trade")
        
        subscriptions = subscription_manager.get_subscriptions()
        
        # Should return all active subscription types
        assert set(subscriptions) == {"ohlc", "trade"}

    def test_get_subscribed_instruments_by_type(self, subscription_manager, mock_instruments):
        """Test getting subscribed instruments for specific subscription type."""
        instruments = mock_instruments[:2]
        subscription_type = "ohlc"
        
        # Add and activate subscription
        subscription_manager.add_subscription(subscription_type, instruments)
        subscription_manager.mark_subscription_active(subscription_type)
        
        subscribed_instruments = subscription_manager.get_subscribed_instruments(subscription_type)
        
        assert set(subscribed_instruments) == set(instruments)

    def test_get_subscribed_instruments_all(self, subscription_manager, mock_instruments):
        """Test getting all subscribed instruments across all types."""
        # Add different subscriptions
        subscription_manager.add_subscription("ohlc", mock_instruments[:2])
        subscription_manager.add_subscription("trade", [mock_instruments[2]])
        subscription_manager.mark_subscription_active("ohlc")
        subscription_manager.mark_subscription_active("trade")
        
        all_instruments = subscription_manager.get_subscribed_instruments()
        
        # Should return all unique instruments
        assert set(all_instruments) == set(mock_instruments)

    def test_get_all_subscribed_instruments(self, subscription_manager, mock_instruments):
        """Test getting complete list of subscribed instruments."""
        # Add overlapping subscriptions
        subscription_manager.add_subscription("ohlc", mock_instruments[:2])
        subscription_manager.add_subscription("trade", mock_instruments[1:])  # Overlaps with ohlc
        subscription_manager.mark_subscription_active("ohlc")
        subscription_manager.mark_subscription_active("trade")
        
        all_instruments = subscription_manager.get_all_subscribed_instruments()
        
        # Should return all unique instruments (no duplicates)
        assert set(all_instruments) == set(mock_instruments)

    def test_prepare_resubscription(self, subscription_manager, mock_instruments):
        """Test preparing for resubscription preserves state correctly."""
        instruments = mock_instruments[:2]
        subscription_type = "ohlc"
        stream_name = "test_stream"
        
        # Add and activate subscription
        subscription_manager.add_subscription(subscription_type, instruments)
        subscription_manager.mark_subscription_active(subscription_type)
        subscription_manager.set_subscription_name(subscription_type, stream_name)
        
        # Prepare resubscription
        old_state = subscription_manager.prepare_resubscription(subscription_type)
        
        # Should return old state
        assert old_state is not None
        assert old_state["subscription_type"] == subscription_type
        assert old_state["stream_name"] == stream_name

    def test_complete_resubscription_cleanup(self, subscription_manager, mock_instruments):
        """Test completing resubscription cleanup removes old state."""
        instruments = mock_instruments[:2]
        subscription_type = "ohlc"
        old_stream_name = "old_stream"
        
        # Setup old state
        subscription_manager.add_subscription(subscription_type, instruments)
        subscription_manager.mark_subscription_active(subscription_type)
        subscription_manager.set_subscription_name(subscription_type, old_stream_name)
        
        # Complete resubscription cleanup
        subscription_manager.complete_resubscription_cleanup(subscription_type)
        
        # Old stream name should be cleaned up
        assert subscription_manager.get_subscription_name(subscription_type) is None

    def test_set_and_get_subscription_name(self, subscription_manager):
        """Test setting and getting subscription names."""
        subscription_type = "ohlc"
        stream_name = "test_stream_name"
        
        subscription_manager.set_subscription_name(subscription_type, stream_name)
        
        assert subscription_manager.get_subscription_name(subscription_type) == stream_name

    def test_symbol_to_instrument_mapping(self, subscription_manager, mock_instruments):
        """Test symbol to instrument mapping functionality."""
        instrument = mock_instruments[0]
        
        # Add subscription should create symbol mapping
        subscription_manager.add_subscription("ohlc", [instrument])
        
        # Should be able to get instrument by symbol via mapping
        mapping = subscription_manager.get_symbol_to_instrument_mapping()
        assert instrument.symbol in mapping
        assert mapping[instrument.symbol] == instrument

    def test_subscription_with_parameters(self, subscription_manager, mock_instruments):
        """Test subscription handling with DataType parameters."""
        instruments = mock_instruments[:1]
        subscription_type = "ohlc(1m)"
        
        # Parse subscription type
        parsed_type, _ = DataType.from_str(subscription_type)
        
        # Add subscription
        updated_instruments = subscription_manager.add_subscription(
            parsed_type, instruments, reset=False
        )
        
        assert updated_instruments == set(instruments)
        assert parsed_type in subscription_manager._pending_subscriptions

    def test_get_subscriptions_includes_pending_subscriptions(self, subscription_manager, mock_instruments):
        """Test that get_subscriptions returns both active and pending subscriptions.
        
        This test ensures that when DataType.ALL is used during universe updates,
        it correctly subscribes new instruments to ALL existing subscription types,
        including those that are still pending connection establishment.
        
        This addresses the issue where open_interest subscriptions were missing
        during universe updates in BaseDataGatheringStrategy.
        """
        instruments_batch1 = mock_instruments[:1]  # BTC only  
        instruments_batch2 = mock_instruments[1:2]  # ETH only
        
        # Simulate initial setup: Add subscriptions for first instrument
        # Some become active, others stay pending (like slow open_interest connections)
        subscription_manager.add_subscription("ohlc", instruments_batch1)  
        subscription_manager.add_subscription("funding_rate", instruments_batch1)
        subscription_manager.add_subscription("open_interest", instruments_batch1)
        
        # Mark some as active (fast connections like OHLC and funding_rate)
        subscription_manager.mark_subscription_active("ohlc")
        subscription_manager.mark_subscription_active("funding_rate")
        # Leave open_interest as pending (slow connection)
        
        # Test: get_subscriptions() should return ALL subscription types (active + pending)
        all_subscriptions = subscription_manager.get_subscriptions()
        expected_subscriptions = {"ohlc", "funding_rate", "open_interest"}
        assert set(all_subscriptions) == expected_subscriptions, \
            f"Expected {expected_subscriptions}, got {set(all_subscriptions)}"
        
        # Test: get_subscriptions(instrument) should also include pending for that instrument
        btc_subscriptions = subscription_manager.get_subscriptions(instruments_batch1[0])
        assert set(btc_subscriptions) == expected_subscriptions, \
            f"Expected {expected_subscriptions} for BTC, got {set(btc_subscriptions)}"
        
        # Simulate universe update: This is what happens in BaseDataGatheringStrategy.on_fit()
        # When new instruments are added, DataType.ALL uses get_subscriptions() to find
        # existing subscription types to subscribe the new instruments to
        existing_subscription_types = subscription_manager.get_subscriptions()
        
        # The fix ensures that ALL subscription types are returned, including pending ones
        assert "open_interest" in existing_subscription_types, \
            "open_interest should be included even if still pending"
        
        # Now simulate subscribing new instruments to all existing subscription types
        # (this is what DataType.ALL does internally)
        for sub_type in existing_subscription_types:
            subscription_manager.add_subscription(sub_type, instruments_batch2)
        
        # Verify that new instruments are subscribed to ALL types, including open_interest
        eth_subscriptions = subscription_manager.get_subscriptions(instruments_batch2[0])  
        assert set(eth_subscriptions) == expected_subscriptions, \
            f"New instrument should have all subscription types including pending ones: {set(eth_subscriptions)}"
            
        # Verify specific subscription states
        assert subscription_manager.has_subscription(instruments_batch2[0], "ohlc") == False, \
            "ETH ohlc should be pending initially"
        assert subscription_manager.has_pending_subscription(instruments_batch2[0], "ohlc") == True, \
            "ETH ohlc should have pending subscription"
        assert subscription_manager.has_pending_subscription(instruments_batch2[0], "open_interest") == True, \
            "ETH open_interest should have pending subscription"
            
    def test_get_subscriptions_empty_when_no_subscriptions(self, subscription_manager):
        """Test that get_subscriptions returns empty list when no subscriptions exist."""
        assert subscription_manager.get_subscriptions() == []
        assert subscription_manager.get_subscriptions(None) == []

    def test_get_subscriptions_only_returns_subscriptions_with_instruments(self, subscription_manager, mock_instruments):
        """Test that get_subscriptions only returns subscription types that have instruments."""
        instrument = mock_instruments[0]
        
        # Add subscription
        subscription_manager.add_subscription("ohlc", [instrument])
        
        # Should return the subscription type
        subscriptions = subscription_manager.get_subscriptions()
        assert "ohlc" in subscriptions
        
        # Remove all instruments from subscription
        subscription_manager.remove_subscription("ohlc", [instrument])
        
        # Should no longer return the subscription type
        subscriptions = subscription_manager.get_subscriptions()
        assert "ohlc" not in subscriptions