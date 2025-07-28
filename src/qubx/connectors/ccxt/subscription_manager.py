"""
Subscription state management for CCXT data provider.

This module handles the lifecycle and state tracking of data subscriptions,
separating subscription concerns from connection management and data handling.
"""

from collections import defaultdict
from typing import Dict, List, Set

from qubx.core.basics import DataType, Instrument


class SubscriptionManager:
    """
    Manages subscription state and lifecycle for CCXT data provider.
    
    Responsibilities:
    - Track active and pending subscriptions
    - Manage subscription state transitions (pending -> active)
    - Provide query methods for subscription status
    - Handle subscription updates and removals
    """
    
    def __init__(self):
        # Active subscriptions (connection established and receiving data)
        self._subscriptions: Dict[str, Set[Instrument]] = defaultdict(set)
        
        # Pending subscriptions (connection being established)
        self._pending_subscriptions: Dict[str, Set[Instrument]] = defaultdict(set)
        
        # Track if connection is ready for each subscription type
        self._sub_connection_ready: Dict[str, bool] = defaultdict(lambda: False)
        
        # Mapping of subscription type to stream name for connection tracking
        self._sub_to_name: Dict[str, str] = {}
        
        # Symbol to instrument mapping for quick lookups
        self._symbol_to_instrument: Dict[str, Instrument] = {}
    
    def add_subscription(
        self, 
        subscription_type: str, 
        instruments: List[Instrument], 
        reset: bool = False
    ) -> Set[Instrument]:
        """
        Add instruments to a subscription type.
        
        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
            instruments: List of instruments to subscribe to
            reset: If True, replace existing subscription; if False, add to existing
            
        Returns:
            Set of instruments that should be subscribed to (updated set)
        """
        _new_instruments = set(instruments)
        
        # Update symbol to instrument mapping
        self._symbol_to_instrument.update({i.symbol: i for i in instruments})
        
        # Determine what instruments need to be subscribed to
        _sub_type, _ = DataType.from_str(subscription_type)
        
        if reset:
            # Replace existing subscription entirely
            _updated_instruments = _new_instruments
            _instruments_to_return = _new_instruments
        else:
            # Add to existing subscription
            _current_instruments = self.get_subscribed_instruments(subscription_type)
            _updated_instruments = _new_instruments.union(_current_instruments)
            # Return only the new instruments that weren't already subscribed
            _instruments_to_return = _new_instruments - set(_current_instruments)
        
        # Mark as pending until connection is established
        self._pending_subscriptions[_sub_type] = _updated_instruments
        self._sub_connection_ready[_sub_type] = False
        
        return _instruments_to_return
    
    def remove_subscription(
        self, 
        subscription_type: str, 
        instruments: List[Instrument]
    ) -> None:
        """
        Remove instruments from a subscription type.
        
        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
            instruments: List of instruments to unsubscribe from
        """
        _sub_type, _ = DataType.from_str(subscription_type)
        
        if _sub_type in self._subscriptions:
            self._subscriptions[_sub_type] = self._subscriptions[_sub_type].difference(instruments)
            # Clean up empty subscriptions
            if not self._subscriptions[_sub_type]:
                del self._subscriptions[_sub_type]
                self._sub_connection_ready[_sub_type] = False
            
        if _sub_type in self._pending_subscriptions:
            self._pending_subscriptions[_sub_type] = self._pending_subscriptions[_sub_type].difference(instruments)
            # Clean up empty pending subscriptions
            if not self._pending_subscriptions[_sub_type]:
                del self._pending_subscriptions[_sub_type]
    
    def mark_subscription_active(self, subscription_type: str) -> None:
        """
        Mark a subscription as active once the WebSocket connection is established.
        
        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
        """
        _sub_type, _ = DataType.from_str(subscription_type)
        
        if _sub_type in self._pending_subscriptions:
            # Move from pending to active
            self._subscriptions[_sub_type] = self._pending_subscriptions[_sub_type]
            self._sub_connection_ready[_sub_type] = True
            # Clear pending subscription
            del self._pending_subscriptions[_sub_type]
    
    def clear_subscription_state(self, subscription_type: str) -> None:
        """
        Clear all state for a subscription type (used during resubscription cleanup).
        
        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
        """
        _sub_type, _ = DataType.from_str(subscription_type)
        
        # Clean up both active and pending subscriptions
        self._subscriptions.pop(_sub_type, None)
        self._pending_subscriptions.pop(_sub_type, None)
        self._sub_connection_ready.pop(_sub_type, None)
        
        # Clean up name mapping
        self._sub_to_name.pop(subscription_type, None)
    
    def set_subscription_name(self, subscription_type: str, name: str) -> None:
        """
        Set the stream name for a subscription type.
        
        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
            name: Stream name for connection tracking
        """
        self._sub_to_name[subscription_type] = name
    
    def get_subscription_name(self, subscription_type: str) -> str | None:
        """
        Get the stream name for a subscription type.
        
        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
            
        Returns:
            Stream name if exists, None otherwise
        """
        return self._sub_to_name.get(subscription_type)
    
    def get_subscriptions(self, instrument: Instrument | None = None) -> List[str]:
        """
        Get list of active subscription types.
        
        Args:
            instrument: If provided, return only subscriptions containing this instrument
            
        Returns:
            List of subscription type names
        """
        if instrument is not None:
            return [sub for sub, instrs in self._subscriptions.items() if instrument in instrs]
        return [sub for sub, instruments in self._subscriptions.items() if instruments]
    
    def get_subscribed_instruments(self, subscription_type: str | None = None) -> List[Instrument]:
        """
        Get list of instruments for a subscription type.
        
        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)"). 
                             If None, returns all subscribed instruments.
            
        Returns:
            List of subscribed instruments
        """
        if not subscription_type:
            return list(self.get_all_subscribed_instruments())
        
        # Return active subscriptions, fallback to pending if no active ones
        _sub_type, _ = DataType.from_str(subscription_type)
        if _sub_type in self._subscriptions:
            return list(self._subscriptions[_sub_type])
        elif _sub_type in self._pending_subscriptions:
            return list(self._pending_subscriptions[_sub_type])
        else:
            return []
    
    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """
        Check if an instrument has an active subscription.
        
        Args:
            instrument: Instrument to check
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
            
        Returns:
            True if subscription is active (not just pending)
        """
        sub_type, _ = DataType.from_str(subscription_type)
        # Only return True if subscription is actually active (not just pending)
        return (
            sub_type in self._subscriptions
            and instrument in self._subscriptions[sub_type]
            and self._sub_connection_ready.get(sub_type, False)
        )
    
    def has_pending_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """
        Check if an instrument has a pending subscription.
        
        Args:
            instrument: Instrument to check
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
            
        Returns:
            True if subscription is pending (connection being established)
        """
        sub_type, _ = DataType.from_str(subscription_type)
        return (
            sub_type in self._pending_subscriptions
            and instrument in self._pending_subscriptions[sub_type]
            and not self._sub_connection_ready.get(sub_type, False)
        )
    
    def get_all_subscribed_instruments(self) -> Set[Instrument]:
        """
        Get all instruments across all subscription types.
        
        Returns:
            Set of all subscribed instruments (active + pending)
        """
        active = set.union(*self._subscriptions.values()) if self._subscriptions else set()
        pending = set.union(*self._pending_subscriptions.values()) if self._pending_subscriptions else set()
        return active.union(pending)
    
    def is_connection_ready(self, subscription_type: str) -> bool:
        """
        Check if connection is ready for a subscription type.
        
        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
            
        Returns:
            True if connection is established and ready
        """
        _sub_type, _ = DataType.from_str(subscription_type)
        return self._sub_connection_ready.get(_sub_type, False)
    
    def get_symbol_to_instrument_mapping(self) -> Dict[str, Instrument]:
        """
        Get the symbol to instrument mapping.
        
        Returns:
            Dictionary mapping symbols to instruments
        """
        return self._symbol_to_instrument.copy()
    
    def find_subscription_type_by_name(self, name: str) -> str | None:
        """
        Find subscription type by stream name.
        
        Args:
            name: Stream name to look for
            
        Returns:
            Subscription type if found, None otherwise
        """
        for sub_type, stream_name in self._sub_to_name.items():
            if stream_name == name:
                return sub_type
        return None

    def prepare_resubscription(self, subscription_type: str) -> dict | None:
        """
        Prepare for resubscription by clearing old state and returning cleanup info.
        
        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
            
        Returns:
            Dict with old subscription info for cleanup, or None if no existing subscription
        """
        old_stream_name = self.get_subscription_name(subscription_type)
        if not old_stream_name:
            return None
            
        # Return info for cleanup but don't clear state yet
        # State will be cleared after new subscription is set up
        return {
            "subscription_type": subscription_type,
            "stream_name": old_stream_name
        }
    
    def complete_resubscription_cleanup(self, subscription_type: str) -> None:
        """
        Complete resubscription cleanup by clearing old active state.
        
        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
        """
        _parsed_sub_type, _ = DataType.from_str(subscription_type)
        
        # Clear active subscription state (but preserve pending subscriptions)
        self._subscriptions.pop(_parsed_sub_type, None)
        self._sub_connection_ready.pop(_parsed_sub_type, None)
        self._sub_to_name.pop(subscription_type, None)

    def setup_new_subscription(self, subscription_type: str, stream_name: str) -> None:
        """
        Set up a new subscription with its stream name.
        
        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
            stream_name: Generated stream name for this subscription
        """
        self.set_subscription_name(subscription_type, stream_name)