"""
Connection management for CCXT data provider.

This module handles WebSocket connections, retry logic, and stream lifecycle management,
separating connection concerns from subscription state and data handling.
"""

import asyncio
import concurrent.futures
from asyncio.exceptions import CancelledError
from collections import defaultdict
from typing import Awaitable, Callable, Dict

from ccxt import ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable, NetworkError
from ccxt.pro import Exchange
from qubx import logger
from qubx.core.basics import CtrlChannel

from .exceptions import CcxtSymbolNotRecognized
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
        max_ws_retries: int = 10,
        subscription_manager: SubscriptionManager | None = None
    ):
        self._exchange_id = exchange_id
        self.max_ws_retries = max_ws_retries
        self._subscription_manager = subscription_manager
        
        # Stream state management
        self._is_stream_enabled: Dict[str, bool] = defaultdict(lambda: False)
        self._stream_to_unsubscriber: Dict[str, Callable[[], Awaitable[None]]] = {}
        
        # Connection tracking
        self._stream_to_coro: Dict[str, concurrent.futures.Future] = {}
    
    def set_subscription_manager(self, subscription_manager: SubscriptionManager) -> None:
        """Set the subscription manager for state coordination."""
        self._subscription_manager = subscription_manager
    
    async def listen_to_stream(
        self,
        subscriber: Callable[[], Awaitable[None]],
        exchange: Exchange,
        channel: CtrlChannel,
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
                    subscription_type = self._subscription_manager.find_subscription_type_by_name(stream_name)
                    if subscription_type:
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
                logger.error(f"<yellow>{self._exchange_id}</yellow> {e.__class__.__name__} :: Error in {stream_name} : {e}")
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
    
    async def stop_stream(
        self, 
        stream_name: str, 
        future: concurrent.futures.Future | None = None,
        is_resubscription: bool = False
    ) -> None:
        """
        Stop a stream gracefully with proper cleanup.
        
        Args:
            stream_name: Name of the stream to stop
            future: Optional future representing the stream task
            is_resubscription: True if this is stopping an old stream during resubscription
        """
        try:
            context = "old stream" if is_resubscription else "stream"
            logger.debug(f"<yellow>{self._exchange_id}</yellow> Stopping {context} {stream_name}")
            
            # Disable the stream to signal it should stop
            self._is_stream_enabled[stream_name] = False
            
            # Wait for the stream to stop naturally
            if future:
                total_sleep_time = 0.0
                while future.running() and total_sleep_time < 20.0:
                    await asyncio.sleep(1.0)
                    total_sleep_time += 1.0
                
                if future.running():
                    logger.warning(
                        f"<yellow>{self._exchange_id}</yellow> {context.title()} {stream_name} is still running. Cancelling it."
                    )
                    future.cancel()
                else:
                    logger.debug(f"<yellow>{self._exchange_id}</yellow> {context.title()} {stream_name} has been stopped")
            
            # Run unsubscriber if available
            if stream_name in self._stream_to_unsubscriber:
                logger.debug(f"<yellow>{self._exchange_id}</yellow> Unsubscribing from {stream_name}")
                await self._stream_to_unsubscriber[stream_name]()
                del self._stream_to_unsubscriber[stream_name]
            
            # Clean up stream state
            if is_resubscription:
                # For resubscription, only clean up if the stream is actually disabled
                # (avoid interfering with new streams using the same name)
                if stream_name in self._is_stream_enabled and not self._is_stream_enabled[stream_name]:
                    del self._is_stream_enabled[stream_name]
            else:
                # For regular stops, always clean up completely
                self._is_stream_enabled.pop(stream_name, None)
                self._stream_to_coro.pop(stream_name, None)
            
            logger.debug(f"<yellow>{self._exchange_id}</yellow> {context.title()} {stream_name} stopped")
            
        except Exception as e:
            logger.error(f"<yellow>{self._exchange_id}</yellow> Error stopping {stream_name}")
            logger.exception(e)
    
    def register_stream_future(
        self, 
        stream_name: str, 
        future: concurrent.futures.Future
    ) -> None:
        """
        Register a future for a stream for tracking and cleanup.
        
        Args:
            stream_name: Name of the stream
            future: Future representing the stream task
        """
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
    
    def disable_stream(self, stream_name: str) -> None:
        """
        Disable a stream (signal it to stop).
        
        Args:
            stream_name: Name of the stream to disable
        """
        self._is_stream_enabled[stream_name] = False
    
    def enable_stream(self, stream_name: str) -> None:
        """
        Enable a stream.
        
        Args:
            stream_name: Name of the stream to enable
        """
        self._is_stream_enabled[stream_name] = True
    
    def set_stream_unsubscriber(
        self, 
        stream_name: str, 
        unsubscriber: Callable[[], Awaitable[None]]
    ) -> None:
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
    
    def set_stream_coro(
        self, 
        stream_name: str, 
        coro: concurrent.futures.Future
    ) -> None:
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

    def cleanup_all_streams(self) -> None:
        """Clean up all stream state (for shutdown)."""
        self._is_stream_enabled.clear()
        self._stream_to_unsubscriber.clear()
        self._stream_to_coro.clear()