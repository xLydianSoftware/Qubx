"""
Subscription configuration dataclass for CCXT data provider handlers.

This module defines the configuration structure that data type handlers return
instead of directly calling connection management methods.
"""

from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

from qubx.core.basics import Instrument


@dataclass
class SubscriptionConfiguration:
    """
    Configuration data returned by data type handlers for subscription setup.
    
    This approach separates the concern of defining subscription parameters
    from actually executing the subscription, improving testability and
    maintaining clean separation of concerns.
    
    Supports both bulk subscriptions (single subscriber for all instruments) and
    individual subscriptions (separate subscriber per instrument) for exchanges
    that don't support bulk watching.
    """
    
    # Main subscription function for bulk subscriptions (existing behavior)
    subscriber_func: Optional[Callable[[], Awaitable[None]]] = None
    
    # Individual subscriber functions per instrument (for non-bulk exchanges)
    individual_subscribers: Optional[Dict[Instrument, Callable[[], Awaitable[None]]]] = None
    
    # Individual unsubscriber functions per instrument (for cleanup)
    individual_unsubscribers: Optional[Dict[Instrument, Callable[[], Awaitable[None]]]] = None
    
    # Optional cleanup function for graceful unsubscription (bulk mode)
    unsubscriber_func: Optional[Callable[[], Awaitable[None]]] = None
    
    # Stream name for logging and tracking purposes
    stream_name: str = ""
    
    # Additional metadata for subscription management
    requires_market_type_batching: bool = False
    
    
    def __post_init__(self):
        """Validate the configuration after initialization."""
        # Determine subscription mode based on what's provided
        has_individual_subscribers = self.individual_subscribers is not None
        has_bulk_subscriber = self.subscriber_func is not None
        
        # Validate that exactly one subscription mode is configured
        if has_individual_subscribers and has_bulk_subscriber:
            raise ValueError("Cannot specify both individual_subscribers and subscriber_func")
        
        if not has_individual_subscribers and not has_bulk_subscriber:
            raise ValueError("Must specify either individual_subscribers or subscriber_func")
        
        if has_individual_subscribers:
            # Validate individual subscribers are callable
            for instrument, subscriber in self.individual_subscribers.items():
                if not callable(subscriber):
                    raise ValueError(f"Individual subscriber for {instrument} must be callable")
                    
            # Validate individual unsubscribers if provided
            if self.individual_unsubscribers:
                for instrument, unsubscriber in self.individual_unsubscribers.items():
                    if not callable(unsubscriber):
                        raise ValueError(f"Individual unsubscriber for {instrument} must be callable")
        
        if has_bulk_subscriber:
            if not callable(self.subscriber_func):
                raise ValueError("subscriber_func must be callable")
        
        # Validate bulk unsubscriber if provided
        if self.unsubscriber_func is not None and not callable(self.unsubscriber_func):
            raise ValueError("unsubscriber_func must be callable if provided")
    
    def uses_individual_streams(self) -> bool:
        """Return True if this configuration uses individual instrument streams."""
        return self.individual_subscribers is not None