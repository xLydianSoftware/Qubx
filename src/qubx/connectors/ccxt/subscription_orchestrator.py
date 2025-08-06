"""
Subscription orchestration for CCXT data provider.

This module coordinates between SubscriptionManager and ConnectionManager
to handle the complex resubscription logic and stream lifecycle management.
"""

import concurrent.futures
from typing import Any, Awaitable, Callable

from ccxt.pro import Exchange
from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument
from qubx.utils.misc import AsyncThreadLoop

from .connection_manager import ConnectionManager
from .handlers import IDataTypeHandler
from .subscription_config import SubscriptionConfiguration
from .subscription_manager import SubscriptionManager
from .utils import instrument_to_ccxt_symbol


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

        # 1. Handle cleanup of existing subscription
        self.execute_unsubscription(subscription_type)

        # 2. Set up new subscription
        self._start_subscription(
            subscription_type,
            instruments,
            handler,
            exchange,
            channel,
            **subscriber_params,
        )

    def execute_unsubscription(self, subscription_type: str):
        """Clean up existing subscription if it exists."""
        stream_name = self._subscription_manager.get_subscription_stream(subscription_type)
        if not stream_name:
            return None
        stream_future = self._connection_manager.get_stream_future(stream_name)
        if not stream_future:
            return None
        logger.debug(f"[{self._exchange_id}] Canceling existing {subscription_type} subscription")
        self._connection_manager.stop_stream(stream_name, subscription_type)

    def _start_subscription(
        self,
        subscription_type: str,
        instruments: set[Instrument],
        handler: IDataTypeHandler,
        exchange,
        channel,
        **subscriber_params,
    ) -> None:
        """Set up new subscription with proper parameters."""
        # Parse subscription type and prepare parameters
        sub_type, parsed_params = DataType.from_str(subscription_type)
        handler_params = {**parsed_params, **subscriber_params}
        handler_params.pop("instruments", None)  # Avoid duplication

        # Generate stream name
        stream_kwargs = {"instruments": instruments, **parsed_params, **subscriber_params}
        stream_name = self._get_subscription_name(sub_type, **stream_kwargs)

        # Register with subscription manager
        self._subscription_manager.set_subscription_name(subscription_type, stream_name)

        # Get subscription configuration from handler
        subscription_config = handler.prepare_subscription(
            name=stream_name,
            sub_type=subscription_type,
            channel=channel,
            instruments=instruments,
            **handler_params,
        )

        # Handle individual instrument streams if required
        if subscription_config.use_instrument_streams:
            self._start_instrument_streams(
                subscription_config=subscription_config,
                exchange=exchange,
                channel=channel,
                base_stream_name=stream_name,
                subscription_type=subscription_type,
            )
        else:
            self._start_bulk_stream(
                subscription_config=subscription_config,
                exchange=exchange,
                channel=channel,
                stream_name=stream_name,
                subscription_type=subscription_type,
            )

    def _start_bulk_stream(
        self,
        subscription_config: SubscriptionConfiguration,
        exchange: Exchange,
        channel: CtrlChannel,
        stream_name: str,
        subscription_type: str,
    ) -> None:
        # Create and start single subscription task
        subscription_task = self._create_subscription_task(
            subscription_config=subscription_config,
            exchange=exchange,
            channel=channel,
            stream_name=stream_name,
            subscription_type=subscription_type,
        )
        future = self._loop.submit(subscription_task())

        # Register with connection manager
        self._connection_manager.register_stream_future(stream_name, future)

    def _start_instrument_streams(
        self,
        subscription_config: SubscriptionConfiguration,
        exchange,
        channel,
        base_stream_name: str,
        subscription_type: str,
    ) -> None:
        """
        Set up individual listen_to_stream calls for each instrument when bulk watching isn't supported.

        This creates separate independent WebSocket streams for each instrument, allowing them
        to run concurrently without waiting for each other. Supports dynamic instrument management
        for resubscriptions.
        """
        assert subscription_config.instrument_subscribers is not None
        instruments = list(subscription_config.instrument_subscribers.keys())
        logger.info(f"[{self._exchange_id}] Setting up individual streams for {len(instruments)} instruments")

        # Get existing individual streams for this subscription type (for resubscription handling)
        existing_streams = self._get_existing_individual_streams(subscription_type)

        futures = []
        active_instruments = set()

        for instrument in subscription_config.instrument_subscribers.keys():
            # Create unique stream name for each instrument
            instrument_stream_name = f"{base_stream_name}_{instrument.symbol.replace('/', '_')}"
            active_instruments.add(instrument)

            # Skip if stream already exists and is running (for resubscription optimization)
            if instrument_stream_name in existing_streams:
                existing_future = existing_streams[instrument_stream_name]
                if existing_future and not existing_future.done():
                    logger.debug(
                        f"<yellow>{self._exchange_id}</yellow> Reusing existing stream: {instrument_stream_name}"
                    )
                    futures.append(existing_future)
                    continue

            # Start the individual stream
            task_coroutine = self._create_instrument_subscription_task(
                instrument=instrument,
                subscription_config=subscription_config,
                exchange=exchange,
                channel=channel,
                stream_name=instrument_stream_name,
                subscription_type=subscription_type,
            )
            future = self._loop.submit(task_coroutine())
            futures.append(future)

            # Register each individual stream with connection manager
            self._connection_manager.register_stream_future(instrument_stream_name, future)

            logger.debug(f"<yellow>{self._exchange_id}</yellow> Started individual stream: {instrument_stream_name}")

        # Clean up streams for instruments that are no longer active
        self._cleanup_removed_instrument_streams(existing_streams, active_instruments)

        # Store the main stream reference for compatibility (use first future)
        if futures:
            self._connection_manager.register_stream_future(base_stream_name, futures[0])

        # Store individual stream mapping for future resubscriptions
        self._store_individual_stream_mapping(
            subscription_type,
            {
                f"{base_stream_name}_{inst.symbol.replace('/', '_')}": fut
                for inst, fut in zip(subscription_config.instrument_subscribers.keys(), futures)
            },
        )

    def _get_existing_individual_streams(self, subscription_type: str) -> dict[str, Any]:
        """Get existing individual streams for a subscription type."""
        # This would need to be implemented based on how we store the mapping
        # For now, return empty dict - we'll enhance this later
        return {}

    def _cleanup_removed_instrument_streams(
        self, existing_streams: dict[str, Any], active_instruments: set[Instrument]
    ) -> None:
        """Clean up streams for instruments that are no longer in the subscription."""
        # Extract active stream names
        active_stream_names = {f"stream_{inst.symbol.replace('/', '_')}" for inst in active_instruments}

        for stream_name, future in existing_streams.items():
            if stream_name not in active_stream_names and future and not future.done():
                logger.debug(
                    f"<yellow>{self._exchange_id}</yellow> Cleaning up removed instrument stream: {stream_name}"
                )
                # Schedule cleanup
                self._loop.submit(self._connection_manager.stop_stream(stream_name, subscription_type))

    def _store_individual_stream_mapping(self, subscription_type: str, stream_mapping: dict[str, Any]) -> None:
        """Store mapping of individual streams for future resubscription handling."""
        # This would need to be implemented based on subscription manager capabilities
        # For now, we'll keep it simple
        pass

    def get_subscription_future(self, subscription_type: str) -> concurrent.futures.Future | None:
        """Get the future for a subscription type."""
        stream_name = self._subscription_manager.get_subscription_stream(subscription_type)
        return self._connection_manager.get_stream_future(stream_name) if stream_name else None

    def cleanup_subscription(self, subscription_type: str) -> None:
        """Clean up all state for a subscription type."""
        self._subscription_manager.clear_subscription_state(subscription_type)

    def _get_subscription_name(
        self, subscription: str, instruments: list[Instrument] | set[Instrument] | Instrument | None = None, **kwargs
    ) -> str:
        if isinstance(instruments, Instrument):
            instruments = [instruments]
        _symbols = [instrument_to_ccxt_symbol(i) for i in instruments] if instruments is not None else []
        _name = f"{','.join(_symbols)} {subscription}" if _symbols else subscription
        if kwargs:
            kwargs_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
            _name += f" ({kwargs_str})"
        return _name

    def _create_subscription_task(
        self,
        subscription_config: SubscriptionConfiguration,
        exchange: Exchange,
        channel: CtrlChannel,
        stream_name: str,
        subscription_type: str,
    ) -> Callable[[], Awaitable[None]]:
        """Create the async subscription task."""

        async def subscription_task():
            assert subscription_config.subscriber_func is not None
            await self._connection_manager.listen_to_stream(
                subscriber=subscription_config.subscriber_func,
                exchange=exchange,
                channel=channel,
                subscription_type=subscription_type,
                stream_name=stream_name,
                unsubscriber=subscription_config.unsubscriber_func,
            )

        return subscription_task

    def _create_instrument_subscription_task(
        self,
        instrument: Instrument,
        subscription_config: SubscriptionConfiguration,
        exchange: Exchange,
        channel: CtrlChannel,
        stream_name: str,
        subscription_type: str,
    ) -> Callable[[], Awaitable[None]]:
        """Create the async subscription task."""
        assert subscription_config.instrument_subscribers is not None
        subscriber = subscription_config.instrument_subscribers.get(instrument)
        assert subscriber is not None

        unsubscriber = None
        if subscription_config.instrument_unsubscribers:
            unsubscriber = subscription_config.instrument_unsubscribers.get(instrument)

        # Create subscription task for this instrument
        async def subscription_task():
            await self._connection_manager.listen_to_stream(
                subscriber=subscriber,
                exchange=exchange,
                channel=channel,
                subscription_type=subscription_type,
                stream_name=stream_name,
                unsubscriber=unsubscriber,
            )

        return subscription_task
