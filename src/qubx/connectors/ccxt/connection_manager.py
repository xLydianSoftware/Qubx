"""
Connection management for CCXT data provider.

This module handles WebSocket connections, retry logic, and stream lifecycle management,
separating connection concerns from subscription state and data handling.
"""

import asyncio
import concurrent.futures
import time
from asyncio.exceptions import CancelledError
from collections import defaultdict
from typing import Awaitable, Callable

from ccxt import ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable, NetworkError
from ccxt.pro import Exchange
from qubx import logger
from qubx.core.basics import CtrlChannel
from qubx.utils.misc import AsyncThreadLoop

from .exceptions import CcxtSymbolNotRecognized
from .exchange_manager import ExchangeManager
from .subscription_manager import SubscriptionManager


class ConnectionManager:
    """
    Manages WebSocket connections and stream lifecycle for CCXT data provider.

    Responsibilities:
    - Handle WebSocket connection establishment and management
    - Implement retry logic and error handling
    - Manage stream lifecycle (start, stop, cleanup)
    - Coordinate with SubscriptionManager for state updates
    """

    def __init__(
        self,
        exchange_id: str,
        exchange_manager: ExchangeManager,
        max_ws_retries: int = 10,
        subscription_manager: SubscriptionManager | None = None,
        cleanup_timeout: float = 3.0,
    ):
        self._exchange_id = exchange_id
        self._exchange_manager = exchange_manager
        self.max_ws_retries = max_ws_retries
        self._subscription_manager = subscription_manager
        self._cleanup_timeout = cleanup_timeout

        # Stream state management
        self._is_stream_enabled: dict[str, bool] = defaultdict(lambda: False)
        self._stream_to_unsubscriber: dict[str, Callable[[], Awaitable[None]]] = {}

        # Connection tracking
        self._stream_to_coro: dict[str, concurrent.futures.Future] = {}

    @property
    def _loop(self) -> AsyncThreadLoop:
        """Get current AsyncThreadLoop from exchange manager."""
        return AsyncThreadLoop(self._exchange_manager.exchange.asyncio_loop)

    def set_subscription_manager(self, subscription_manager: SubscriptionManager) -> None:
        """Set the subscription manager for state coordination."""
        self._subscription_manager = subscription_manager

    async def listen_to_stream(
        self,
        subscriber: Callable[[], Awaitable[None]],
        exchange: Exchange,
        channel: CtrlChannel,
        subscription_type: str,
        stream_name: str,
        unsubscriber: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """
        Listen to a WebSocket stream with error handling and retry logic.

        Args:
            subscriber: Async function that handles the stream data
            exchange: CCXT exchange instance
            channel: Control channel for data flow
            stream_name: Unique name for this stream
            unsubscriber: Optional cleanup function for graceful unsubscription
        """
        logger.info(f"<yellow>{self._exchange_id}</yellow> Listening to {stream_name}")

        # Register unsubscriber for cleanup
        if unsubscriber is not None:
            self._stream_to_unsubscriber[stream_name] = unsubscriber

        # Enable the stream
        self._is_stream_enabled[stream_name] = True
        n_retry = 0
        connection_established = False

        while channel.control.is_set() and self._is_stream_enabled[stream_name]:
            try:
                await subscriber()
                n_retry = 0  # Reset retry counter on success

                # Mark subscription as active on first successful data reception
                if not connection_established and self._subscription_manager:
                    self._subscription_manager.mark_subscription_active(subscription_type)
                    connection_established = True

                # Check if stream was disabled during subscriber execution
                if not self._is_stream_enabled[stream_name]:
                    break

            except CcxtSymbolNotRecognized:
                # Skip unrecognized symbols but continue listening
                continue
            except CancelledError:
                # Graceful cancellation
                break
            except ExchangeClosedByUser:
                # Connection closed by us, stop gracefully
                logger.info(f"<yellow>{self._exchange_id}</yellow> {stream_name} listening has been stopped")
                break
            except (NetworkError, ExchangeError, ExchangeNotAvailable) as e:
                # Network/exchange errors - retry after short delay
                logger.error(
                    f"<yellow>{self._exchange_id}</yellow> {e.__class__.__name__} :: Error in {stream_name} : {e}"
                )
                await asyncio.sleep(1)
                continue
            except Exception as e:
                # Unexpected errors
                if not channel.control.is_set() or not self._is_stream_enabled[stream_name]:
                    # Channel closed or stream disabled, exit gracefully
                    break

                logger.error(f"<yellow>{self._exchange_id}</yellow> Exception in {stream_name}: {e}")
                logger.exception(e)

                n_retry += 1
                if n_retry >= self.max_ws_retries:
                    logger.error(
                        f"<yellow>{self._exchange_id}</yellow> Max retries reached for {stream_name}. Closing connection."
                    )
                    # Clean up exchange reference to force reconnection
                    del exchange
                    break

                # Exponential backoff with cap at 60 seconds
                await asyncio.sleep(min(2**n_retry, 60))

        # Stream ended, cleanup
        logger.debug(f"<yellow>{self._exchange_id}</yellow> Stream {stream_name} ended")

    def stop_stream(self, stream_name: str, wait: bool = True) -> None:
        """
        Stop a stream (signal it to stop).

        Args:
            stream_name: Name of the stream to stop
            wait: If True, wait for stream and unsubscriber to complete (default).
                  If False, cancel asynchronously without waiting.
        """
        assert self._subscription_manager is not None

        logger.debug(f"Stopping stream: {stream_name}, wait={wait}")

        self._is_stream_enabled[stream_name] = False

        stream_future = self.get_stream_future(stream_name)
        if stream_future:
            stream_future.cancel()
            if wait:
                self._wait(stream_future, stream_name)
        else:
            logger.warning(f"[CONNECTION] No stream future found for {stream_name}")

        unsubscriber = self.get_stream_unsubscriber(stream_name)
        if unsubscriber:
            logger.debug(f"Calling unsubscriber for {stream_name}")
            unsub_task = self._loop.submit(unsubscriber())
            if wait:
                self._wait(unsub_task, f"unsubscriber for {stream_name}")
                # Wait for 1 second just in case
                self._loop.submit(asyncio.sleep(1)).result()
        else:
            logger.debug(f"No unsubscriber found for {stream_name}")

        self._is_stream_enabled.pop(stream_name, None)
        self._stream_to_coro.pop(stream_name, None)
        self._stream_to_unsubscriber.pop(stream_name, None)

    def register_stream_future(self, stream_name: str, future: concurrent.futures.Future) -> None:
        """
        Register a future for a stream for tracking and cleanup.

        Args:
            stream_name: Name of the stream
            future: Future representing the stream task
        """
        # Add done callback to handle any exceptions and prevent "Future exception was never retrieved"
        future.add_done_callback(lambda f: self._handle_stream_completion(f, stream_name))
        self._stream_to_coro[stream_name] = future

    def is_stream_enabled(self, stream_name: str) -> bool:
        """
        Check if a stream is enabled.

        Args:
            stream_name: Name of the stream to check

        Returns:
            True if stream is enabled, False otherwise
        """
        return self._is_stream_enabled.get(stream_name, False)

    def get_stream_future(self, stream_name: str) -> concurrent.futures.Future | None:
        """
        Get the future for a stream.

        Args:
            stream_name: Name of the stream

        Returns:
            Future if exists, None otherwise
        """
        return self._stream_to_coro.get(stream_name)

    def enable_stream(self, stream_name: str) -> None:
        """
        Enable a stream.

        Args:
            stream_name: Name of the stream to enable
        """
        self._is_stream_enabled[stream_name] = True

    def set_stream_unsubscriber(self, stream_name: str, unsubscriber: Callable[[], Awaitable[None]]) -> None:
        """
        Set unsubscriber function for a stream.

        Args:
            stream_name: Name of the stream
            unsubscriber: Async function to call for unsubscription
        """
        self._stream_to_unsubscriber[stream_name] = unsubscriber

    def get_stream_unsubscriber(self, stream_name: str) -> Callable[[], Awaitable[None]] | None:
        """
        Get unsubscriber function for a stream.

        Args:
            stream_name: Name of the stream

        Returns:
            Unsubscriber function if exists, None otherwise
        """
        return self._stream_to_unsubscriber.get(stream_name)

    def set_stream_coro(self, stream_name: str, coro: concurrent.futures.Future) -> None:
        """
        Set coroutine/future for a stream.

        Args:
            stream_name: Name of the stream
            coro: Future representing the stream task
        """
        self._stream_to_coro[stream_name] = coro

    def get_stream_coro(self, stream_name: str) -> concurrent.futures.Future | None:
        """
        Get coroutine/future for a stream.

        Args:
            stream_name: Name of the stream

        Returns:
            Future if exists, None otherwise
        """
        return self._stream_to_coro.get(stream_name)

    def _handle_stream_completion(self, future: concurrent.futures.Future, stream_name: str) -> None:
        """
        Handle stream future completion and any exceptions to prevent 'Future exception was never retrieved'.

        Args:
            future: The completed future
            stream_name: Name of the stream for logging
        """
        try:
            future.result()  # Retrieve result to handle any exceptions
        except Exception:
            pass  # Silent handling to prevent "Future exception was never retrieved"

    def _wait(self, future: concurrent.futures.Future, context: str) -> None:
        """Wait for future completion with timeout and exception handling."""
        start_wait = time.time()
        while future.running() and (time.time() - start_wait) < self._cleanup_timeout:
            time.sleep(0.1)

        if future.running():
            logger.warning(f"[{self._exchange_id}] {context} still running after {self._cleanup_timeout}s timeout")
        else:
            # Always retrieve result to handle exceptions properly and prevent "Future exception was never retrieved"
            try:
                future.result()  # This will raise any exception that occurred
            except Exception:
                pass  # Silent handling during cleanup - UnsubscribeError is expected
