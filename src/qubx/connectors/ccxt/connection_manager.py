"""
Connection management for CCXT data provider.

This module handles WebSocket connections, retry logic, and stream lifecycle management,
separating connection concerns from subscription state and data handling.
"""

import asyncio
import concurrent.futures
import time
from asyncio.exceptions import CancelledError
from typing import Awaitable, Callable

from ccxt import (
    BadSymbol,
    DDoSProtection,
    ExchangeClosedByUser,
    ExchangeError,
    ExchangeNotAvailable,
    NetworkError,
    RateLimitExceeded,
    UnsubscribeError,
)
from ccxt.async_support.base.ws.client import Client as _WsClient
from ccxt.pro import Exchange

from qubx import logger
from qubx.core.basics import CtrlChannel
from qubx.utils.misc import AsyncThreadLoop

from .exceptions import CcxtSymbolNotRecognized
from .exchange_manager import ExchangeManager
from .subscription_manager import SubscriptionManager


def _safe_buffer(self):
    conn = getattr(self.connection, "_conn", None)
    if not conn or not getattr(conn, "protocol", None):
        return b""
    payload = getattr(conn.protocol, "_payload", None)
    buf = getattr(payload, "_buffer", None)
    return buf if buf is not None else b""


# SAFETY PATCH: make ccxt WS buffer access resilient to closed connections
_WsClient.buffer = property(_safe_buffer)  # type: ignore


class ConnectionManager:
    """
    Manages WebSocket connections and stream lifecycle for CCXT data provider.

    Responsibilities:
    - Handle WebSocket connection establishment and management
    - Implement retry logic and error handling
    - Manage stream lifecycle (start, stop, cleanup)
    - Coordinate with SubscriptionManager for state updates

    Invariant: nothing here may block a thread on the exchange event loop without a bound, and
    nothing running *on* that loop may block waiting for it.
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

        # Plain dict on purpose: membership means "this stream is meant to be running", and a
        # defaultdict would silently re-create a popped key on read.
        self._is_stream_enabled: dict[str, bool] = {}
        self._stream_to_unsubscriber: dict[str, Callable[[], Awaitable[None]]] = {}

        # Connection tracking
        self._stream_to_coro: dict[str, concurrent.futures.Future] = {}

    @property
    def _loop(self) -> AsyncThreadLoop:
        """Get current AsyncThreadLoop from exchange manager."""
        return AsyncThreadLoop(self._exchange_manager.exchange.asyncio_loop)

    def _on_exchange_loop_thread(self) -> bool:
        """True when the caller is running *on* the exchange event loop.

        Anything that blocks waiting for that loop from this thread deadlocks the loop on itself.
        """
        try:
            return asyncio.get_running_loop() is self._exchange_manager.exchange.asyncio_loop
        except RuntimeError:
            return False

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
    ) -> None:
        """
        Listen to a WebSocket stream with error handling and retry logic.

        Only ever READS the stream registry: the orchestrator arms the enabled flag and the
        unsubscriber before submitting this task. A task that wrote the flag on its first step
        would resurrect a stream `stop_stream` already popped, because cancelling a queued future
        does not stop that first step from running.

        Args:
            subscriber: Async function that handles the stream data
            exchange: CCXT exchange instance
            channel: Control channel for data flow
            stream_name: Unique name for this stream
        """
        # "Listening to" is matched literally by the BotExchangeRecreationWedged Loki alert in
        # xlydian-platform (k8s/apps/*/loki.yaml). Do not reword this line.
        logger.info(f"<yellow>{self._exchange_id}</yellow> Listening to {stream_name}")

        n_retry = 0
        connection_established = False

        while channel.control.is_set() and self._is_stream_enabled.get(stream_name, False):
            try:
                await subscriber()
                n_retry = 0  # Reset retry counter on success

                # Mark subscription as active on first successful data reception
                if not connection_established and self._subscription_manager:
                    self._subscription_manager.mark_subscription_active(subscription_type)
                    connection_established = True

                # Check if stream was disabled during subscriber execution
                if not self._is_stream_enabled.get(stream_name, False):
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
            except BadSymbol as e:
                # Bad symbol is a permanent error - retrying will never succeed
                logger.error(f"<yellow>{self._exchange_id}</yellow> BadSymbol :: {stream_name} : {e} - stopping stream")
                break
            except UnsubscribeError as e:
                # UnsubscribeError is expected during stream restart transitions (old stream
                # still alive when new subscription starts). Log at debug level and retry.
                logger.debug(
                    f"<yellow>{self._exchange_id}</yellow> UnsubscribeError in {stream_name} (expected during restart): {e}"
                )
                await asyncio.sleep(1)
                continue
            except (RateLimitExceeded, DDoSProtection) as e:
                # WS 429/418 are IP actions that ban REST too, and a gate close is not a token debit,
                # so the next header sync does not erase it — hence the REST pool.
                logger.warning(
                    f"<yellow>{self._exchange_id}</yellow> Rate limited in {stream_name}: {e}"
                )
                if hasattr(self, "_exchange_manager") and self._exchange_manager.rate_limiter:
                    self._exchange_manager.rate_limiter.report_limit_hit(
                        pool_name="ccxt_rest", reason=f"{e.__class__.__name__}: {e}"
                    )
                await asyncio.sleep(min(2 ** (n_retry + 1), 60))
                n_retry += 1
                continue
            except (NetworkError, ExchangeError, ExchangeNotAvailable) as e:
                # Network/exchange errors - retry after short delay
                logger.error(
                    f"<yellow>{self._exchange_id}</yellow> {e.__class__.__name__} :: Error in {stream_name} : {e}"
                )
                await asyncio.sleep(1)
                continue
            except Exception as e:
                # Unexpected errors
                if not channel.control.is_set() or not self._is_stream_enabled.get(stream_name, False):
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

        if wait and self._on_exchange_loop_thread():
            # Blocking here would park the exchange loop on a coroutine only that loop can run.
            # Degrade rather than raise: callers up the synchronous subscription pipeline treat an
            # exception as a fault and would kill their own poller.
            logger.info(
                f"[{self._exchange_id}] stop_stream({stream_name}) called from the exchange loop thread - "
                "degrading to non-blocking cleanup"
            )
            wait = False

        # Tear the registry down before any blocking wait, so a wait that times out cannot leave a
        # half-removed stream behind. Popping _is_stream_enabled is the stop signal the listen loop
        # reads, and it is what is_connected() reports on.
        stream_future = self._stream_to_coro.pop(stream_name, None)
        unsubscriber = self._stream_to_unsubscriber.pop(stream_name, None)
        self._is_stream_enabled.pop(stream_name, None)

        if stream_future is not None:
            stream_future.cancel()
            if wait:
                self._wait(stream_future, stream_name)
        else:
            logger.warning(f"[CONNECTION] No stream future found for {stream_name}")

        if unsubscriber is not None:
            logger.debug(f"Calling unsubscriber for {stream_name}")
            unsub_task = self._loop.submit(unsubscriber())
            if wait and self._wait(unsub_task, f"unsubscriber for {stream_name}"):
                # Grace period so the venue acks the unsubscribe before a new subscription reuses
                # the same hashes. Plain wall-clock, not routed through the loop, which would add a
                # second thing that can block. Skipped when the unsubscribe wait timed out: no ack
                # is coming from a loop that is already not answering.
                time.sleep(1)
        else:
            logger.debug(f"No unsubscriber found for {stream_name}")

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
            future.result(timeout=0)  # the future is done here, this only re-raises
        except concurrent.futures.CancelledError:
            return
        except BaseException as e:  # noqa: BLE001 - a done-callback must never propagate onto the loop
            if self._is_stream_enabled.get(stream_name, False):
                logger.warning(f"[{self._exchange_id}] {stream_name} stopped while still subscribed: {e!r}")
            else:
                logger.debug(f"[{self._exchange_id}] {stream_name} finished with {e!r}")

    def _wait(self, future: concurrent.futures.Future, context: str) -> bool:
        """Wait for future completion with timeout and exception handling.

        Returns False iff the wait timed out (the coroutine was abandoned), True otherwise -
        including when it completed by raising, which callers treat as "the loop answered".
        """
        # A future from asyncio.run_coroutine_threadsafe goes PENDING -> FINISHED/CANCELLED and is
        # never RUNNING, so a `while future.running()` poll never loops and any timeout built on it
        # is dead code. Do not reintroduce running() here.
        try:
            future.result(timeout=self._cleanup_timeout)
        except concurrent.futures.TimeoutError:
            # Must precede `except Exception` - TimeoutError is an Exception subclass. The abandoned
            # coroutine keeps running on the loop; we only stop waiting for it.
            logger.warning(
                f"[{self._exchange_id}] {context} did not complete in {self._cleanup_timeout}s - abandoning it"
            )
            future.cancel()
            return False
        except Exception as e:
            logger.debug(f"[{self._exchange_id}] {context} finished with {e!r}")
        return True
