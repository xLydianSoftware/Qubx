"""
Generic polling adapter to convert CCXT fetch_* methods into watch_* behavior.

This adapter allows any exchange's fetch_* method to be used as a watch_* method
by implementing intelligent polling with proper resource management and unwatch functionality.
"""

import asyncio
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set
from asyncio import CancelledError

from qubx import logger


class PollingToWebSocketAdapter:
    """
    Generic adapter to convert fetch_* methods to watch_* behavior using intelligent polling.
    
    This adapter provides:
    - Dynamic symbol management (add/remove symbols during runtime)
    - Comprehensive unwatch functionality (symbol-level and complete shutdown)
    - Proper resource cleanup and error handling
    - Thread-safe operations for concurrent access
    - Configurable polling intervals
    """
    
    def __init__(
        self,
        fetch_method: Callable,
        poll_interval_seconds: int = 300,  # 5 minutes default
        symbols: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        adapter_id: Optional[str] = None,
        use_time_boundaries: bool = False,  # Make boundary logic optional
        boundary_tolerance_seconds: int = 30,
    ):
        """
        Initialize the polling adapter.
        
        Args:
            fetch_method: The CCXT fetch_* method to call (e.g., self.fetch_funding_rates)
            poll_interval_seconds: How often to poll in seconds (default: 300 = 5 minutes)
            symbols: Initial list of symbols to watch
            params: Additional parameters for fetch_method
            adapter_id: Unique identifier for logging purposes
            use_time_boundaries: Whether to align polling to time boundaries (like open_interest.py)
            boundary_tolerance_seconds: Tolerance for boundary alignment (only used if use_time_boundaries=True)
        """
        self.fetch_method = fetch_method
        self.poll_interval_seconds = poll_interval_seconds
        self.params = params or {}
        self.adapter_id = adapter_id or f"adapter_{id(self)}"
        self.use_time_boundaries = use_time_boundaries
        self.boundary_tolerance_seconds = boundary_tolerance_seconds
        
        # Thread-safe symbol management
        self._symbols_lock = asyncio.Lock()
        self._symbols: Set[str] = set(symbols or [])
        
        # Polling state management
        self._polling_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._is_running = False
        
        # Statistics for monitoring
        self._poll_count = 0
        self._error_count = 0
        self._last_poll_time: Optional[float] = None
        
    async def start_watching(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Start the polling loop and yield data as it arrives.
        
        This method starts the background polling task that will continuously
        fetch data for the configured symbols and yield results like a WebSocket stream.
        
        Yields:
            Dictionary containing fetched data for symbols
        """
        if self._is_running:
            logger.warning(f"Adapter {self.adapter_id} is already running")
            return
            
        self._is_running = True
        self._stop_event.clear()
        
        logger.info(f"<green>{self.adapter_id}</green> Starting polling adapter with {len(self._symbols)} symbols")
        
        # Create a queue to receive data from polling task
        self._data_queue = asyncio.Queue()
        
        # Start the polling task
        self._polling_task = asyncio.create_task(self._polling_loop())
        
        try:
            # Yield data as it arrives
            while self._is_running:
                try:
                    # Wait for data from polling task or stop signal
                    data = await asyncio.wait_for(self._data_queue.get(), timeout=1.0)
                    yield data
                except asyncio.TimeoutError:
                    # Check if we should continue waiting
                    if not self._is_running:
                        break
                    continue
        finally:
            # Ensure cleanup
            await self._cleanup_polling_task()
        
    async def stop(self) -> None:
        """
        Stop polling completely and cleanup all resources.
        """
        if not self._is_running:
            return
            
        logger.info(f"<green>{self.adapter_id}</green> Stopping polling adapter")
        
        # Signal stop
        self._stop_event.set()
        self._is_running = False
        
        await self._cleanup_polling_task()
        
        logger.debug(f"<green>{self.adapter_id}</green> Stopped (polled {self._poll_count} times, {self._error_count} errors)")
        
    async def _cleanup_polling_task(self) -> None:
        """Clean up the polling task."""
        if self._polling_task and not self._polling_task.done():
            try:
                await asyncio.wait_for(self._polling_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"<yellow>{self.adapter_id}</yellow> Polling task didn't stop gracefully, cancelling")
                self._polling_task.cancel()
                try:
                    await self._polling_task
                except CancelledError:
                    pass
                    
        self._polling_task = None
        
    async def add_symbols(self, new_symbols: List[str]) -> None:
        """
        Add symbols to the existing watch list.
        
        Args:
            new_symbols: List of symbols to add
        """
        if not new_symbols:
            return
            
        async with self._symbols_lock:
            before_count = len(self._symbols)
            self._symbols.update(new_symbols)
            after_count = len(self._symbols)
            added_count = after_count - before_count
            
        if added_count > 0:
            logger.debug(f"<green>{self.adapter_id}</green> Added {added_count} symbols (total: {after_count})")
            
    async def remove_symbols(self, symbols_to_remove: List[str]) -> None:
        """
        Remove specific symbols from the watch list.
        
        Args:
            symbols_to_remove: List of symbols to remove
        """
        if not symbols_to_remove:
            return
            
        async with self._symbols_lock:
            before_count = len(self._symbols)
            self._symbols.difference_update(symbols_to_remove)
            after_count = len(self._symbols)
            removed_count = before_count - after_count
            
        if removed_count > 0:
            logger.debug(f"<green>{self.adapter_id}</green> Removed {removed_count} symbols (total: {after_count})")
            
        # If no symbols left, we could optionally stop polling
        # For now, we'll keep polling in case symbols are added back
            
    async def update_symbols(self, new_symbols: List[str]) -> None:
        """
        Replace entire symbol list (atomic operation).
        
        Args:
            new_symbols: New complete list of symbols to watch
        """
        async with self._symbols_lock:
            old_symbols = self._symbols.copy()
            self._symbols = set(new_symbols or [])
            
        logger.debug(f"<green>{self.adapter_id}</green> Updated symbols: {len(old_symbols)} -> {len(self._symbols)}")
        
    def is_watching(self, symbol: Optional[str] = None) -> bool:
        """
        Check if adapter has symbols configured to watch.
        
        Args:
            symbol: Optional specific symbol to check. If None, checks if has any symbols.
            
        Returns:
            True if has the specified symbol (or any symbols if symbol=None)
        """            
        if symbol is None:
            return len(self._symbols) > 0
        else:
            return symbol in self._symbols
            
    def is_running(self) -> bool:
        """
        Check if adapter is actively running (polling).
        
        Returns:
            True if adapter is currently running
        """
        return self._is_running
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics for monitoring."""
        return {
            "adapter_id": self.adapter_id,
            "is_running": self._is_running,
            "symbol_count": len(self._symbols),
            "poll_count": self._poll_count,
            "error_count": self._error_count,
            "last_poll_time": self._last_poll_time,
            "poll_interval_seconds": self.poll_interval_seconds,
        }
        
    async def _polling_loop(self) -> None:
        """
        Main polling loop that runs in the background.
        """
        logger.debug(f"<green>{self.adapter_id}</green> Starting polling loop")
        
        # Always do an initial poll immediately
        first_poll = True
        
        try:
            while not self._stop_event.is_set():
                try:
                    # Get current symbols (thread-safe)
                    async with self._symbols_lock:
                        current_symbols = list(self._symbols)
                    
                    # Skip polling if no symbols
                    if not current_symbols:
                        logger.debug(f"<green>{self.adapter_id}</green> No symbols to poll, sleeping")
                        await self._cancellable_sleep(10)  # Short sleep when no symbols
                        continue
                    
                    # Determine if we should poll now
                    should_poll = first_poll or self._should_poll_now()
                    
                    if should_poll:
                        await self._poll_once(current_symbols)
                        first_poll = False
                        
                        # Sleep longer after successful poll
                        sleep_time = self.poll_interval_seconds if not self.use_time_boundaries else 60
                        await self._cancellable_sleep(sleep_time)
                    else:
                        # Not time to poll yet, sleep briefly and check again
                        await self._cancellable_sleep(5)
                        
                except CancelledError:
                    break
                except Exception as e:
                    self._error_count += 1
                    logger.error(f"<red>{self.adapter_id}</red> Polling error: {type(e).__name__}: {e}")
                    
                    # Sleep before retry, but not too long
                    await self._cancellable_sleep(min(30, self.poll_interval_seconds // 10))
                    
        except CancelledError:
            pass
        finally:
            logger.debug(f"<green>{self.adapter_id}</green> Polling loop stopped")
            
    def _should_poll_now(self) -> bool:
        """
        Determine if we should poll now based on timing logic.
        
        Returns:
            True if we should poll now
        """
        if not self.use_time_boundaries:
            # Simple interval-based polling
            if self._last_poll_time is None:
                return True
            return (time.time() - self._last_poll_time) >= self.poll_interval_seconds
        else:
            # Boundary-based polling (like the open_interest.py logic)
            # This would need access to data provider time, which we don't have here
            # For now, fall back to simple interval polling
            # TODO: Implement boundary logic if needed when data provider is available
            if self._last_poll_time is None:
                return True
            return (time.time() - self._last_poll_time) >= self.poll_interval_seconds
            
    async def _poll_once(self, symbols: List[str]) -> None:
        """
        Perform a single polling operation.
        
        Args:
            symbols: List of symbols to poll for
        """
        self._poll_count += 1
        self._last_poll_time = time.time()
        
        logger.debug(f"<green>{self.adapter_id}</green> Polling {len(symbols)} symbols (poll #{self._poll_count})")
        
        try:
            # Call the fetch method with symbols and params
            result = await self.fetch_method(symbols, **self.params)
            
            logger.debug(f"<green>{self.adapter_id}</green> Successfully fetched data for {len(symbols)} symbols")
            
            # Store result and put it in queue for yielding
            self._last_result = result
            
            # Put data in queue for start_watching to yield
            if hasattr(self, '_data_queue'):
                try:
                    self._data_queue.put_nowait(result)
                except asyncio.QueueFull:
                    logger.warning(f"<yellow>{self.adapter_id}</yellow> Data queue full, dropping data")
            
        except Exception as e:
            # Re-raise to be handled by polling loop
            logger.error(f"<red>{self.adapter_id}</red> Fetch failed for symbols {symbols}: {e}")
            raise
            
    async def _cancellable_sleep(self, seconds: float) -> None:
        """
        Sleep that can be interrupted by the stop event.
        
        Args:
            seconds: Number of seconds to sleep
        """
        if seconds <= 0:
            return
            
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            # Timeout is expected - means we slept for the full duration
            pass