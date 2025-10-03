"""
Subscription orchestration for CCXT data provider.

This module coordinates between SubscriptionManager and ConnectionManager
to handle the complex resubscription logic and stream lifecycle management.
"""

import concurrent.futures
import uuid
from typing import Awaitable, Callable

from ccxt.pro import Exchange
from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument
from qubx.utils.misc import AsyncThreadLoop

from .connection_manager import ConnectionManager
from .exchange_manager import ExchangeManager
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
        exchange_manager: ExchangeManager,
    ):
        self._exchange_id = exchange_id
        self._subscription_manager = subscription_manager
        self._connection_manager = connection_manager
        self._exchange_manager = exchange_manager

    @property
    def _loop(self) -> AsyncThreadLoop:
        """Get current AsyncThreadLoop from exchange manager."""
        return AsyncThreadLoop(self._exchange_manager.exchange.asyncio_loop)

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
            existing_stream_name = self._subscription_manager.get_subscription_stream(subscription_type)
            new_stream_name = subscription_config.stream_name

            # Skip unsubscription if stream names match (same instruments)
            if existing_stream_name == new_stream_name:
                logger.debug(
                    f"[{self._exchange_id}] Reusing existing {subscription_type} stream: {existing_stream_name}"
                )
                return  # Skip unsubscription - reuse existing stream

            # Different instruments - proceed with cleanup
            if existing_stream_name:
                stream_future = self._connection_manager.get_stream_future(existing_stream_name)
                if stream_future:
                    logger.debug(
                        f"[{self._exchange_id}] Canceling existing {subscription_type} subscription: {existing_stream_name}"
                    )
                    self._connection_manager.stop_stream(existing_stream_name)
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

        # Generate bulk stream name (may be ignored for individual subscriptions)
        stream_name = self._generate_bulk_stream_name(subscription_type, instruments)

        # Get subscription configuration from handler
        subscription_config = handler.prepare_subscription(
            name=stream_name,
            sub_type=subscription_type,
            channel=channel,
            instruments=instruments,
            **handler_params,
        )
        subscription_config.channel = channel
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
        # Bulk subscriptions must have a stream name
        assert subscription_config.stream_name is not None, "Bulk subscription must have stream_name"

        subscription_type = subscription_config.subscription_type

        # 1. Check if we can reuse existing stream (same instruments = same hash)
        old_stream_name = self._subscription_manager.get_subscription_stream(subscription_type)

        if old_stream_name == subscription_config.stream_name:
            logger.debug(f"[{self._exchange_id}] Reusing existing bulk stream: {old_stream_name}")
            return

        # 2. Stop old stream if it exists (different instruments)
        if old_stream_name:
            future = self._connection_manager.get_stream_future(old_stream_name)
            if future:
                logger.debug(f"[{self._exchange_id}] Stopping existing bulk stream: {old_stream_name}")
                self._connection_manager.stop_stream(old_stream_name)

        # 3. Clear only old stream state, preserving pending instruments that were just added
        self._clear_old_stream_state(subscription_type)

        # 4. Register new stream with subscription manager
        self._subscription_manager.set_subscription_name(subscription_type, subscription_config.stream_name)

        # 5. Create and start new subscription task
        subscription_task = self._create_subscription_task(
            subscription_config=subscription_config,
            exchange=exchange,
        )
        future = self._loop.submit(subscription_task())

        # 6. Register with connection manager
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
            instrument_stream_name = self._generate_individual_stream_name(subscription_type, instrument)
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
        self._subscription_manager.set_individual_streams(subscription_type, futures)

    def _stop_individual_streams(self, streams: dict[Instrument, str]) -> None:
        """Stop individual streams for the given instruments."""
        for _, stream_name in streams.items():
            future = self._connection_manager.get_stream_future(stream_name)
            if future and not future.done():
                logger.debug(f"<yellow>{self._exchange_id}</yellow> Stopping stream: {stream_name}")
                # Don't wait for cleanup to complete - it's async cleanup
                self._connection_manager.stop_stream(stream_name, wait=False)

    def _generate_bulk_stream_name(self, subscription_type: str, instruments: set[Instrument]) -> str:
        """Generate bulk stream name with hash for multiple instruments."""
        if not instruments:
            return subscription_type

        return f"{subscription_type}:{len(instruments)}:{uuid.uuid4()}"

    def _generate_individual_stream_name(self, subscription_type: str, instrument: Instrument) -> str:
        """Generate individual stream name for a single instrument."""
        return f"{instrument.symbol}:{subscription_type}"

    def get_subscription_future(self, subscription_type: str) -> concurrent.futures.Future | None:
        """Get the future for a subscription type."""
        stream_name = self._subscription_manager.get_subscription_stream(subscription_type)
        return self._connection_manager.get_stream_future(stream_name) if stream_name else None

    def _clear_old_stream_state(self, subscription_type: str) -> None:
        """
        Clear only old stream state while preserving pending instruments.

        This is called during resubscription to clean up old active subscription state
        without affecting the new pending instruments that were just added.

        Args:
            subscription_type: Full subscription type (e.g., "ohlc(1m)")
        """
        # Clear old active subscription (instruments that were receiving data)
        self._subscription_manager._subscriptions.pop(subscription_type, None)

        # Reset connection readiness (new stream will mark as ready when established)
        self._subscription_manager._sub_connection_ready[subscription_type] = False

        # Clear individual stream mappings (for individual subscription mode)
        self._subscription_manager._individual_streams.pop(subscription_type, None)

        # NOTE: We deliberately DO NOT clear:
        # - _pending_subscriptions: Contains the new instruments that should be subscribed
        # - _sub_to_name: Will be updated with new stream name in next step

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
            logger.info(f"listen_to_stream finished for stream {subscription_config.stream_name}")

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
            logger.info(f"listen_to_stream finished for stream {stream_name}")

        return subscription_task
