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
        self, exchange_id: str, subscription_manager: SubscriptionManager, connection_manager: ConnectionManager
    ):
        self._exchange_id = exchange_id
        self._subscription_manager = subscription_manager
        self._connection_manager = connection_manager

        # Legacy state tracking (TODO: phase out)
        self._sub_to_coro: Dict[str, concurrent.futures.Future] = {}

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
        if not instruments or len(instruments) == 0:
            logger.debug(f"<yellow>{self._exchange_id}</yellow> No instruments to subscribe to for {subscription_type}")
            return

        # Check for existing subscription that needs cleanup
        cleanup_info = self._subscription_manager.prepare_resubscription(subscription_type)
        old_coro = None

        if cleanup_info and subscription_type in self._sub_to_coro:
            old_stream_name = cleanup_info["stream_name"]
            old_coro = self._sub_to_coro[subscription_type]

            logger.debug(f"<yellow>{self._exchange_id}</yellow> Canceling existing {subscription_type} subscription")

            # Clear legacy state
            del self._sub_to_coro[subscription_type]

            # Complete cleanup in subscription manager
            self._subscription_manager.complete_resubscription_cleanup(subscription_type)

            # Disable old stream and cancel coroutine
            self._connection_manager.disable_stream(old_stream_name)
            old_coro.cancel()

            # Wait for cancellation (up to 3 seconds)
            self._wait_for_cancellation(old_coro, subscription_type)

        # Generate stream name and set up new subscription
        _sub_type, _params = DataType.from_str(subscription_type)
        # Remove instruments from subscriber_params to avoid duplicate parameter
        handler_params = {**_params, **subscriber_params}
        handler_params.pop('instruments', None)
        
        kwargs = {"instruments": instruments, **_params, **subscriber_params}
        stream_name = stream_name_generator(_sub_type, **kwargs)

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

        # Create subscription task using connection manager
        async def subscription_task():
            await self._connection_manager.listen_to_stream(
                subscriber=subscription_config.subscriber_func,
                exchange=exchange,
                channel=channel,
                stream_name=stream_name,
                unsubscriber=subscription_config.unsubscriber_func,
            )

        # Start new subscriber
        future = async_loop_submit(subscription_task())
        self._sub_to_coro[subscription_type] = future
        self._connection_manager.register_stream_future(stream_name, future)

        # Schedule cleanup of old subscriber (for WebSocket subscriptions)
        if old_coro is not None and cleanup_info is not None and _sub_type != "open_interest":
            cleanup_stream_name = cleanup_info["stream_name"]
            async_loop_submit(self._connection_manager.stop_stream(cleanup_stream_name, old_coro, is_resubscription=True))

    def _wait_for_cancellation(self, coro: concurrent.futures.Future, subscription_type: str) -> None:
        """Wait for coroutine cancellation with timeout."""
        start_wait = time.time()
        while coro.running() and (time.time() - start_wait) < 3:
            time.sleep(0.1)

        if coro.running():
            _sub_type, _ = DataType.from_str(subscription_type)
            logger.warning(f"<yellow>{self._exchange_id}</yellow> ⚠️ Old {_sub_type} coroutine still running after 3s")

    def get_subscription_future(self, subscription_type: str) -> concurrent.futures.Future | None:
        """Get the future for a subscription type."""
        return self._sub_to_coro.get(subscription_type)

    def cleanup_subscription(self, subscription_type: str) -> None:
        """Clean up all state for a subscription type."""
        self._sub_to_coro.pop(subscription_type, None)
        self._subscription_manager.clear_subscription_state(subscription_type)

    async def stop_subscription(self, subscription_type: str) -> None:
        """Stop a subscription and clean up state."""
        future = self._sub_to_coro.get(subscription_type)
        stream_name = self._subscription_manager.get_subscription_name(subscription_type)

        if future and stream_name:
            # Clean up legacy state
            self._sub_to_coro.pop(subscription_type, None)
            # Stop the stream via connection manager
            await self._connection_manager.stop_stream(stream_name, future)

    # Legacy compatibility properties
    @property
    def _sub_to_coro_dict(self) -> Dict[str, concurrent.futures.Future]:
        """Legacy compatibility for tests."""
        return self._sub_to_coro
