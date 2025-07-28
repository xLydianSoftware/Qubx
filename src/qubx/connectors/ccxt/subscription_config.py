"""
Subscription configuration dataclass for CCXT data provider handlers.

This module defines the configuration structure that data type handlers return
instead of directly calling connection management methods.
"""

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


@dataclass
class SubscriptionConfiguration:
    """
    Configuration data returned by data type handlers for subscription setup.
    
    This approach separates the concern of defining subscription parameters
    from actually executing the subscription, improving testability and
    maintaining clean separation of concerns.
    """
    
    # Main subscription function that will be called by connection manager
    subscriber_func: Callable[[], Awaitable[None]]
    
    # Optional cleanup function for graceful unsubscription
    unsubscriber_func: Optional[Callable[[], Awaitable[None]]] = None
    
    # Stream name for logging and tracking purposes
    stream_name: str = ""
    
    # Additional metadata for subscription management
    requires_market_type_batching: bool = False
    
    def __post_init__(self):
        """Validate the configuration after initialization."""
        if not callable(self.subscriber_func):
            raise ValueError("subscriber_func must be callable")
        
        if self.unsubscriber_func is not None and not callable(self.unsubscriber_func):
            raise ValueError("unsubscriber_func must be callable if provided")