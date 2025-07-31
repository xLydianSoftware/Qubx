"""
Subscription orchestration for CCXT data provider.

This module coordinates between SubscriptionManager and ConnectionManager
to handle the complex resubscription logic and stream lifecycle management.
"""

import concurrent.futures
import time
from typing import Awaitable, Callable, Dict, List, Set

from qubx import logger
from qubx.core.basics import DataType, Instrument

from .connection_manager import ConnectionManager
from .subscription_manager import SubscriptionManager
from .utils import create_market_type_batched_subscriber


class SubscriptionOrchestrator:
    """
    Orchestrates subscription operations between SubscriptionManager and ConnectionManager.

    Responsibilities:
    - Coordinate resubscription cleanup logic
    - Handle complex subscription state transitions
    - Manage interaction between subscription state and connection lifecycle
    - Provide clean interface for subscription operations
    """

    def __init__(
        self, 
        exchange_id: str, 
        subscription_manager: SubscriptionManager, 
        connection_manager: ConnectionManager,
        cleanup_timeout: float = 3.0
    ):
        self._exchange_id = exchange_id
        self._subscription_manager = subscription_manager
        self._connection_manager = connection_manager
        self._cleanup_timeout = cleanup_timeout

    def call_by_market_type(
        self, subscriber: Callable[[List[Instrument]], Awaitable[None]], instruments: Set[Instrument]
    ) -> Callable[[], Awaitable[None]]:
        """
        Create a batched subscriber that calls the original subscriber for each market type group.
        
        This is a convenience wrapper around the utility function for backward compatibility.
        
        Args:
            subscriber: Function to call for each market type group
            instruments: Set of instruments to group by market type
            
        Returns:
            Async function that will call subscriber for each market type group
        """
        return create_market_type_batched_subscriber(subscriber, instruments)

    def execute_subscription(
        self,
        subscription_type: str,
        instruments: Set[Instrument],
        handler,
        stream_name_generator: Callable,
        async_loop_submit: Callable,
        exchange,
        channel,
        **subscriber_params,
    ) -> None:
        """
        Execute a complete subscription operation with cleanup and setup.

        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
            instruments: Set of instruments to subscribe to
            handler: Data type handler that provides subscription configuration
            stream_name_generator: Function to generate unique stream names
            async_loop_submit: Function to submit async tasks
            exchange: CCXT exchange instance
            channel: Control channel for data flow
            **subscriber_params: Additional parameters for the subscriber
        """
        if not instruments:
            logger.debug(f"<yellow>{self._exchange_id}</yellow> No instruments to subscribe to for {subscription_type}")
            return

        # 1. Handle cleanup of existing subscription
        old_stream_info = self._cleanup_old_subscription(subscription_type)
        
        # 2. Set up new subscription
        self._setup_new_subscription(
            subscription_type, instruments, handler, stream_name_generator, 
            async_loop_submit, exchange, channel, **subscriber_params
        )
        
        # 3. Schedule final cleanup of old stream (for WebSocket graceful closure)
        if old_stream_info:
            self._schedule_old_stream_cleanup(old_stream_info, async_loop_submit, subscription_type)

    def _cleanup_old_subscription(self, subscription_type: str) -> dict | None:
        """Clean up existing subscription if it exists."""
        cleanup_info = self._subscription_manager.prepare_resubscription(subscription_type)
        if not cleanup_info:
            return None
            
        old_stream_name = cleanup_info["stream_name"]
        old_future = self._connection_manager.get_stream_future(old_stream_name)
        
        if old_future:
            logger.debug(f"<yellow>{self._exchange_id}</yellow> Canceling existing {subscription_type} subscription")
            
            # Disable stream and cancel future
            self._connection_manager.disable_stream(old_stream_name)
            old_future.cancel()
            
            # Wait for cancellation
            self._wait_for_cancellation(old_future, subscription_type)
            
            # Complete cleanup in subscription manager
            self._subscription_manager.complete_resubscription_cleanup(subscription_type)
            
            return {"stream_name": old_stream_name, "future": old_future}
        
        return None

    def _setup_new_subscription(
        self, subscription_type: str, instruments: Set[Instrument], handler, 
        stream_name_generator: Callable, async_loop_submit: Callable, 
        exchange, channel, **subscriber_params
    ) -> None:
        """Set up new subscription with proper parameters."""
        # Parse subscription type and prepare parameters
        sub_type, parsed_params = DataType.from_str(subscription_type)
        handler_params = {**parsed_params, **subscriber_params}
        handler_params.pop('instruments', None)  # Avoid duplication
        
        # Generate stream name
        stream_kwargs = {"instruments": instruments, **parsed_params, **subscriber_params}
        stream_name = stream_name_generator(sub_type, **stream_kwargs)
        
        # Register with subscription manager
        self._subscription_manager.setup_new_subscription(subscription_type, stream_name)
        
        # Get subscription configuration from handler
        subscription_config = handler.prepare_subscription(
            name=stream_name,
            sub_type=subscription_type,
            channel=channel,
            instruments=instruments,
            **handler_params,
        )
        
        # Create and start subscription task
        subscription_task = self._create_subscription_task(
            subscription_config, exchange, channel, stream_name
        )
        future = async_loop_submit(subscription_task())
        
        # Register with connection manager
        self._connection_manager.register_stream_future(stream_name, future)

    def _create_subscription_task(self, subscription_config, exchange, channel, stream_name):
        """Create the async subscription task."""
        async def subscription_task():
            await self._connection_manager.listen_to_stream(
                subscriber=subscription_config.subscriber_func,
                exchange=exchange,
                channel=channel,
                stream_name=stream_name,
                unsubscriber=subscription_config.unsubscriber_func,
            )
        return subscription_task

    def _schedule_old_stream_cleanup(self, old_stream_info: dict, async_loop_submit: Callable, subscription_type: str) -> None:
        """Schedule cleanup of old WebSocket stream."""
        sub_type, _ = DataType.from_str(subscription_type)
        if sub_type != "open_interest":  # Skip cleanup for certain types
            async_loop_submit(
                self._connection_manager.stop_stream(
                    old_stream_info["stream_name"], 
                    old_stream_info.get("future"), 
                    is_resubscription=True
                )
            )

    def _wait_for_cancellation(self, future: concurrent.futures.Future, subscription_type: str) -> None:
        """Wait for future cancellation with timeout."""
        start_wait = time.time()
        while future.running() and (time.time() - start_wait) < self._cleanup_timeout:
            time.sleep(0.1)

        if future.running():
            sub_type, _ = DataType.from_str(subscription_type)
            logger.warning(f"<yellow>{self._exchange_id}</yellow> ⚠️ Old {sub_type} coroutine still running after {self._cleanup_timeout}s")

    def get_subscription_future(self, subscription_type: str) -> concurrent.futures.Future | None:
        """Get the future for a subscription type."""
        stream_name = self._subscription_manager.get_subscription_name(subscription_type)
        return self._connection_manager.get_stream_future(stream_name) if stream_name else None

    def cleanup_subscription(self, subscription_type: str) -> None:
        """Clean up all state for a subscription type."""
        self._subscription_manager.clear_subscription_state(subscription_type)

    async def stop_subscription(self, subscription_type: str) -> None:
        """Stop a subscription and clean up state."""
        stream_name = self._subscription_manager.get_subscription_name(subscription_type)
        future = self._connection_manager.get_stream_future(stream_name) if stream_name else None

        if future and stream_name:
            await self._connection_manager.stop_stream(stream_name, future)
            
        # Clean up subscription manager state
        self._subscription_manager.clear_subscription_state(subscription_type)
