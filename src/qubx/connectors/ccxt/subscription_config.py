"""
Subscription configuration dataclass for CCXT data provider handlers.

This module defines the configuration structure that data type handlers return
instead of directly calling connection management methods.
"""

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from qubx.core.basics import CtrlChannel, Instrument


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

    subscription_type: str

    # Control channel for data flow (will be set by orchestrator)
    channel: CtrlChannel | None = None

    # Main subscription function for bulk subscriptions (existing behavior)
    subscriber_func: Optional[Callable[[], Awaitable[None]]] = None

    # Optional cleanup function for graceful unsubscription (bulk mode)
    unsubscriber_func: Optional[Callable[[], Awaitable[None]]] = None

    # Individual subscriber functions per instrument (for non-bulk exchanges)
    instrument_subscribers: Optional[dict[Instrument, Callable[[], Awaitable[None]]]] = None

    # Individual unsubscriber functions per instrument (for cleanup)
    instrument_unsubscribers: Optional[dict[Instrument, Callable[[], Awaitable[None]]]] = None

    # Stream name for bulk subscriptions only (not used for individual subscriptions)
    stream_name: Optional[str] = None

    # Additional metadata for subscription management
    requires_market_type_batching: bool = False

    def __post_init__(self):
        """Validate the configuration after initialization."""
        # Determine subscription mode based on what's provided
        has_individual_subscribers = self.instrument_subscribers is not None
        has_bulk_subscriber = self.subscriber_func is not None

        # Validate that exactly one subscription mode is configured
        if has_individual_subscribers and has_bulk_subscriber:
            raise ValueError("Cannot specify both individual_subscribers and subscriber_func")

        if has_individual_subscribers and isinstance(self.instrument_subscribers, dict):
            # Validate individual subscribers are callable
            for instrument, subscriber in self.instrument_subscribers.items():
                if not callable(subscriber):
                    raise ValueError(f"Individual subscriber for {instrument} must be callable")

            # Validate individual unsubscribers if provided
            if self.instrument_unsubscribers:
                for instrument, unsubscriber in self.instrument_unsubscribers.items():
                    if not callable(unsubscriber):
                        raise ValueError(f"Individual unsubscriber for {instrument} must be callable")

        if has_bulk_subscriber:
            if not callable(self.subscriber_func):
                raise ValueError("subscriber_func must be callable")
            # Bulk subscriptions must have a stream name
            if not self.stream_name:
                raise ValueError("stream_name is required for bulk subscriptions")

        # Validate bulk unsubscriber if provided
        if self.unsubscriber_func is not None and not callable(self.unsubscriber_func):
            raise ValueError("unsubscriber_func must be callable if provided")

    @property
    def use_instrument_streams(self) -> bool:
        """Return True if this configuration uses individual instrument streams."""
        return self.instrument_subscribers is not None
