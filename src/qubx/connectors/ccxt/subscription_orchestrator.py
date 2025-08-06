"""
Subscription orchestration for CCXT data provider.

This module coordinates between SubscriptionManager and ConnectionManager
to handle the complex resubscription logic and stream lifecycle management.
"""

import concurrent.futures
import hashlib
from typing import Awaitable, Callable

from ccxt.pro import Exchange
from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument
from qubx.utils.misc import AsyncThreadLoop

from .connection_manager import ConnectionManager
from .handlers import IDataTypeHandler
from .subscription_config import SubscriptionConfiguration
from .subscription_manager import SubscriptionManager


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
        loop: AsyncThreadLoop,
    ):
        self._exchange_id = exchange_id
        self._subscription_manager = subscription_manager
        self._connection_manager = connection_manager
        self._loop = loop

    def execute_subscription(
        self,
        subscription_type: str,
        instruments: set[Instrument],
        handler,
        exchange: Exchange,
        channel: CtrlChannel,
        **subscriber_params,
    ) -> None:
        """
        Execute a complete subscription operation with cleanup and setup.

        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
            instruments: Set of instruments to subscribe to
            handler: Data type handler that provides subscription configuration
            exchange: CCXT exchange instance
            channel: Control channel for data flow
            **subscriber_params: Additional parameters for the subscriber
        """
        if not instruments:
            logger.debug(f"<yellow>{self._exchange_id}</yellow> No instruments to subscribe to for {subscription_type}")
            return

        # Prepare subscription configuration
        subscription_config = self._prepare_subscription_config(
            subscription_type,
            instruments,
            handler,
            channel,
            **subscriber_params,
        )

        # Start subscription (unsubscription handled internally based on mode)
        self._start_subscription(
            subscription_config=subscription_config,
            exchange=exchange,
        )

    def execute_unsubscription(self, subscription_config: SubscriptionConfiguration):
        """Clean up existing subscription if it exists."""
        subscription_type = subscription_config.subscription_type

        # For bulk subscriptions, use the main stream name
        if not subscription_config.use_instrument_streams:
            stream_name = self._subscription_manager.get_subscription_stream(subscription_type)
            if stream_name:
                stream_future = self._connection_manager.get_stream_future(stream_name)
                if stream_future:
                    logger.debug(f"[{self._exchange_id}] Canceling existing {subscription_type} subscription")
                    self._connection_manager.stop_stream(stream_name)
        else:
            # For individual subscriptions, stop all individual streams
            individual_streams = self._subscription_manager.get_individual_streams(subscription_type)
            self._stop_individual_streams(individual_streams)

        self._subscription_manager.clear_subscription_state(subscription_type)

    def _prepare_subscription_config(
        self,
        subscription_type: str,
        instruments: set[Instrument],
        handler: IDataTypeHandler,
        channel: CtrlChannel,
        **subscriber_params,
    ) -> SubscriptionConfiguration:
        # Parse subscription type and prepare parameters
        _, parsed_params = DataType.from_str(subscription_type)
        handler_params = {**parsed_params, **subscriber_params}
        handler_params.pop("instruments", None)  # Avoid duplication

        # Generate stream name - will be used for bulk or ignored for individual
        stream_name = self._generate_stream_name(subscription_type, instruments)

        # Get subscription configuration from handler
        subscription_config = handler.prepare_subscription(
            name=stream_name,
            sub_type=subscription_type,
            channel=channel,
            instruments=instruments,
            **handler_params,
        )

        # Store channel and subscription type for later use
        subscription_config.subscription_type = subscription_type
        subscription_config.channel = channel

        # For bulk subscriptions, register the stream name with subscription manager
        if not subscription_config.use_instrument_streams:
            self._subscription_manager.set_subscription_name(subscription_type, stream_name)
            # Ensure bulk subscriptions have their stream name set
            if not subscription_config.stream_name:
                subscription_config.stream_name = stream_name

        return subscription_config

    def _start_subscription(
        self,
        subscription_config: SubscriptionConfiguration,
        exchange: Exchange,
    ) -> None:
        """Set up new subscription with proper parameters."""
        # Handle individual instrument streams if required
        if subscription_config.use_instrument_streams:
            self._start_instrument_streams(
                subscription_config=subscription_config,
                exchange=exchange,
            )
        else:
            self._start_bulk_stream(
                subscription_config=subscription_config,
                exchange=exchange,
            )

    def _start_bulk_stream(
        self,
        subscription_config: SubscriptionConfiguration,
        exchange: Exchange,
    ) -> None:
        # For bulk subscriptions, always stop the old stream first
        self.execute_unsubscription(subscription_config)

        # Bulk subscriptions must have a stream name
        assert subscription_config.stream_name is not None, "Bulk subscription must have stream_name"

        # Re-register with subscription manager after unsubscription cleared it
        self._subscription_manager.set_subscription_name(
            subscription_config.subscription_type, subscription_config.stream_name
        )

        # Create and start single subscription task
        subscription_task = self._create_subscription_task(
            subscription_config=subscription_config,
            exchange=exchange,
        )
        future = self._loop.submit(subscription_task())

        # Register with connection manager
        self._connection_manager.register_stream_future(subscription_config.stream_name, future)

    def _start_instrument_streams(
        self,
        subscription_config: SubscriptionConfiguration,
        exchange: Exchange,
    ) -> None:
        """
        Set up individual listen_to_stream calls for each instrument when bulk watching isn't supported.

        This creates separate independent WebSocket streams for each instrument, allowing them
        to run concurrently without waiting for each other. Supports dynamic instrument management
        for resubscriptions.
        """
        assert subscription_config.instrument_subscribers is not None
        instruments = list(subscription_config.instrument_subscribers.keys())
        subscription_type = subscription_config.subscription_type

        logger.info(f"[{self._exchange_id}] Setting up individual streams for {len(instruments)} instruments")

        # Get existing individual streams for this subscription type (for resubscription handling)
        existing_streams = self._subscription_manager.get_individual_streams(subscription_type)

        futures = {}
        active_stream_names = set()

        for instrument in subscription_config.instrument_subscribers.keys():
            # Create clean stream name for each instrument
            instrument_stream_name = self._generate_stream_name(subscription_type, {instrument})
            active_stream_names.add(instrument_stream_name)

            # Skip if stream already exists and is running (for resubscription optimization)
            if instrument in existing_streams:
                existing_future = self._connection_manager.get_stream_future(existing_streams[instrument])
                if existing_future and not existing_future.done():
                    logger.debug(
                        f"<yellow>{self._exchange_id}</yellow> Reusing existing stream: {instrument_stream_name}"
                    )
                    futures[instrument] = existing_streams[instrument]
                    continue

            # Start the individual stream
            task_coroutine = self._create_instrument_subscription_task(
                instrument=instrument,
                subscription_config=subscription_config,
                exchange=exchange,
                stream_name=instrument_stream_name,
            )
            future = self._loop.submit(task_coroutine())

            # Register each individual stream with connection manager
            self._connection_manager.register_stream_future(instrument_stream_name, future)
            futures[instrument] = instrument_stream_name

            logger.debug(f"<yellow>{self._exchange_id}</yellow> Started individual stream: {instrument_stream_name}")

        # Clean up streams for instruments that are no longer active
        removed_instruments = set(existing_streams.keys()) - set(futures.keys())
        removed_streams = {inst: existing_streams[inst] for inst in removed_instruments}
        self._stop_individual_streams(removed_streams)

        # Store individual stream mapping for future resubscriptions
        # No need for a "main stream reference" - that was a hack
        self._subscription_manager.set_individual_streams(subscription_type, futures)

    def _stop_individual_streams(self, streams: dict[Instrument, str]) -> None:
        """Stop individual streams for the given instruments."""
        for _, stream_name in streams.items():
            future = self._connection_manager.get_stream_future(stream_name)
            if future and not future.done():
                logger.debug(
                    f"<yellow>{self._exchange_id}</yellow> Stopping stream: {stream_name}"
                )
                # Don't wait for cleanup to complete - it's async cleanup
                self._connection_manager.stop_stream(stream_name, wait=False)

    def _generate_stream_name(self, subscription_type: str, instruments: set[Instrument] | None = None) -> str:
        """Generate concise, deterministic stream names."""
        if instruments and len(instruments) == 1:
            # Individual stream: "BTCUSDT:ohlc(1m)"
            symbol = next(iter(instruments)).symbol
            return f"{symbol}:{subscription_type}"
        elif instruments:
            # Bulk stream: "ohlc(1m):a3f2b1"
            symbols = sorted(i.symbol for i in instruments)
            hash_input = f"{subscription_type}:{','.join(symbols)}"
            short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
            return f"{subscription_type}:{short_hash}"
        else:
            return subscription_type

    def get_subscription_future(self, subscription_type: str) -> concurrent.futures.Future | None:
        """Get the future for a subscription type."""
        stream_name = self._subscription_manager.get_subscription_stream(subscription_type)
        return self._connection_manager.get_stream_future(stream_name) if stream_name else None

    def cleanup_subscription(self, subscription_type: str) -> None:
        """Clean up all state for a subscription type."""
        self._subscription_manager.clear_subscription_state(subscription_type)

    async def stop_subscription(self, subscription_type: str) -> None:
        """Stop a subscription and clean up all state."""
        stream_name = self._subscription_manager.get_subscription_stream(subscription_type)
        if stream_name:
            self._connection_manager.stop_stream(stream_name)
        self._subscription_manager.clear_subscription_state(subscription_type)

    def _create_subscription_task(
        self,
        subscription_config: SubscriptionConfiguration,
        exchange: Exchange,
    ) -> Callable[[], Awaitable[None]]:
        """Create the async subscription task."""

        async def subscription_task():
            assert subscription_config.subscriber_func is not None
            assert subscription_config.channel is not None, "Channel must be set before creating task"
            assert subscription_config.stream_name is not None, "Bulk subscription must have stream_name"
            await self._connection_manager.listen_to_stream(
                subscriber=subscription_config.subscriber_func,
                exchange=exchange,
                channel=subscription_config.channel,
                subscription_type=subscription_config.subscription_type,
                stream_name=subscription_config.stream_name,
                unsubscriber=subscription_config.unsubscriber_func,
            )

        return subscription_task

    def _create_instrument_subscription_task(
        self,
        instrument: Instrument,
        subscription_config: SubscriptionConfiguration,
        exchange: Exchange,
        stream_name: str,
    ) -> Callable[[], Awaitable[None]]:
        """Create the async subscription task for an individual instrument."""
        assert subscription_config.instrument_subscribers is not None
        subscriber = subscription_config.instrument_subscribers.get(instrument)
        assert subscriber is not None
        assert subscription_config.channel is not None, "Channel must be set before creating task"

        unsubscriber = None
        if subscription_config.instrument_unsubscribers:
            unsubscriber = subscription_config.instrument_unsubscribers.get(instrument)

        # Create subscription task for this instrument
        async def subscription_task():
            assert subscription_config.channel is not None, "Channel must be set before creating task"
            await self._connection_manager.listen_to_stream(
                subscriber=subscriber,
                exchange=exchange,
                channel=subscription_config.channel,
                subscription_type=subscription_config.subscription_type,
                stream_name=stream_name,
                unsubscriber=unsubscriber,
            )

        return subscription_task
