"""
Generic polling adapter to convert CCXT fetch_* methods into watch_* behavior.

This adapter allows any exchange's fetch_* method to be used as a watch_* method
by implementing intelligent polling with proper resource management and unwatch functionality.
"""

import asyncio
import concurrent.futures
import time
from asyncio import CancelledError
from typing import Any, Callable, Dict, List, Optional, Set

from qubx import logger
from qubx.utils.misc import AsyncThreadLoop


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
        use_time_boundaries: bool = False,  # Make boundary logic optional
        boundary_tolerance_seconds: int = 30,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,  # Optional explicit loop
    ):
        """
        Initialize the polling adapter.

        Args:
            fetch_method: The CCXT fetch_* method to call (e.g., self.fetch_funding_rates)
            poll_interval_seconds: How often to poll in seconds (default: 300 = 5 minutes)
            symbols: Initial list of symbols to watch
            params: Additional parameters for fetch_method
            use_time_boundaries: Whether to align polling to time boundaries (like open_interest.py)
            boundary_tolerance_seconds: Tolerance for boundary alignment (only used if use_time_boundaries=True)
            event_loop: Optional explicit asyncio event loop to use for background tasks
        """
        self.fetch_method = fetch_method
        self.poll_interval_seconds = poll_interval_seconds
        self.params = params or {}
        self.adapter_id = f"polling_adapter_{id(self)}"  # Auto-generated for logging
        self.use_time_boundaries = use_time_boundaries
        self.boundary_tolerance_seconds = boundary_tolerance_seconds
        self.event_loop = event_loop  # Store explicit loop if provided

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

        # Data management for awaitable pattern
        self._data_queue: asyncio.Queue = asyncio.Queue()
        self._latest_data: Optional[Dict[str, Any]] = None
        self._data_condition = asyncio.Condition()

    async def start_watching(self) -> None:
        """
        Start the polling loop in background.

        This method starts the background polling task that will continuously
        fetch data for the configured symbols and store latest results.
        """
        if self._is_running:
            logger.warning(f"Adapter {self.adapter_id} is already running")
            return

        self._is_running = True
        self._stop_event.clear()

        logger.debug(f"Starting polling adapter {self.adapter_id} with {len(self._symbols)} symbols")

        # Do immediate initial poll before starting background loop
        async with self._symbols_lock:
            current_symbols = list(self._symbols)
        
        if current_symbols:
            try:
                await self._poll_once(current_symbols)
            except Exception as e:
                logger.error(f"Initial poll failed for adapter {self.adapter_id}: {e}")

        # Start the polling task for subsequent polls
        if self.event_loop is not None:
            # Use explicit loop if provided (for testing environments)
            
            # Always use AsyncThreadLoop for cross-thread task submission
            # This handles both same-thread and cross-thread cases properly
            async_loop = AsyncThreadLoop(self.event_loop)
            self._polling_task = async_loop.submit(self._polling_loop())
        else:
            # Use current event loop (normal operation)
            self._polling_task = asyncio.create_task(self._polling_loop())

    async def get_next_data(self) -> Dict[str, Any]:
        """
        Get the next available data (CCXT awaitable pattern).
        
        This is the method that watch_* methods should call to get data.
        It waits for new data from the polling task.
        
        Returns:
            Dictionary containing fetched data for symbols
        """
        if not self._is_running:
            logger.debug(f"Starting adapter {self.adapter_id} on first data request")
            await self.start_watching()
            # Give the background task a chance to start and do initial poll
            await asyncio.sleep(0.5)
            
        # Wait for new data from polling task
        try:
            # First, check if we have data immediately available
            try:
                data = self._data_queue.get_nowait()
                return data
            except asyncio.QueueEmpty:
                pass
            
            # If no immediate data, wait with timeout
            data = await asyncio.wait_for(self._data_queue.get(), timeout=30.0)
            return data
        except asyncio.TimeoutError:
            logger.debug(f"Timeout waiting for data from adapter {self.adapter_id}, falling back to cached data")
            
            # If no new data, return latest data if available
            if self._latest_data is not None:
                return self._latest_data
            else:
                # Try a manual poll as fallback - this handles pytest environment issues
                async with self._symbols_lock:
                    current_symbols = list(self._symbols)
                if current_symbols:
                    try:
                        await self._poll_once(current_symbols)
                        if self._latest_data is not None:
                            return self._latest_data
                    except Exception as e:
                        logger.error(f"Manual poll failed for adapter {self.adapter_id}: {e}")
                
                raise TimeoutError(f"No data available from polling adapter {self.adapter_id}")

    async def stop(self) -> None:
        """
        Stop polling completely and cleanup all resources.
        """
        if not self._is_running:
            return

        logger.debug(f"Stopping polling adapter {self.adapter_id}")

        # Signal stop
        self._stop_event.set()
        self._is_running = False

        # Clear data queue to prevent stale results from being processed
        while not self._data_queue.empty():
            try:
                self._data_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        await self._cleanup_polling_task()

        logger.debug(f"Adapter {self.adapter_id} stopped (polled {self._poll_count} times, {self._error_count} errors)")

    async def _cleanup_polling_task(self) -> None:
        """Clean up the polling task."""
        if self._polling_task and not self._polling_task.done():
            try:
                # Handle both Task (normal operation) and Future (explicit event loop) objects
                if hasattr(self._polling_task, 'result') and not hasattr(self._polling_task, '__await__'):
                    # This is a Future from AsyncThreadLoop.submit()
                    self._polling_task.result(timeout=5.0)
                else:
                    # This is a regular Task
                    await asyncio.wait_for(self._polling_task, timeout=5.0)
            except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
                logger.debug(f"Polling task for adapter {self.adapter_id} didn't stop gracefully, cancelling")
                self._polling_task.cancel()
                try:
                    if hasattr(self._polling_task, 'result') and not hasattr(self._polling_task, '__await__'):
                        # Future cancellation is handled by the future itself
                        pass
                    else:
                        # Task cancellation
                        await self._polling_task
                except (CancelledError, concurrent.futures.CancelledError):
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
            logger.debug(f"Added {added_count} symbols to adapter {self.adapter_id} (total: {after_count})")

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
            logger.debug(f"Removed {removed_count} symbols from adapter {self.adapter_id} (total: {after_count})")

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

        logger.debug(f"Updated symbols for adapter {self.adapter_id}: {len(old_symbols)} -> {len(self._symbols)}")

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
        # Always do an initial poll immediately  
        first_poll = True
        
        # Ensure we yield control to allow other tasks to run
        await asyncio.sleep(0)

        try:
            while not self._stop_event.is_set():
                try:
                    # Get current symbols (thread-safe)
                    async with self._symbols_lock:
                        current_symbols = list(self._symbols)

                    # Skip polling if no symbols
                    if not current_symbols:
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
                    logger.error(f"Polling error in adapter {self.adapter_id}: {e}")

                    # Sleep before retry, but not too long
                    # Ensure minimum 1 second sleep to avoid tight retry loops
                    sleep_time = max(1, min(30, self.poll_interval_seconds // 10))
                    await self._cancellable_sleep(sleep_time)

        except CancelledError:
            pass
        finally:
            logger.debug(f"Polling loop stopped for adapter {self.adapter_id}")

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

        logger.debug(f"Polling {len(symbols)} symbols for adapter {self.adapter_id}")

        try:
            # Filter out adapter-specific parameters before calling fetch method
            # These parameters are for the adapter, not the underlying fetch method
            adapter_params = {'pollInterval', 'interval', 'updateInterval'}
            fetch_params = {k: v for k, v in self.params.items() if k not in adapter_params}
            
            # Call the fetch method with symbols and filtered params
            result = await self.fetch_method(symbols, **fetch_params)

            # Check if we've been stopped while the fetch was happening
            if self._stop_event.is_set():
                return

            # Store result and put it in queue for get_next_data to return
            self._latest_data = result

            # Put data in queue for get_next_data to return
            try:
                self._data_queue.put_nowait(result)
            except asyncio.QueueFull:
                # Clear old data and add new
                try:
                    self._data_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self._data_queue.put_nowait(result)

        except Exception as e:
            # Check if we've been stopped during the error
            if self._stop_event.is_set():
                return
            # Re-raise to be handled by polling loop
            logger.error(f"Fetch failed for adapter {self.adapter_id}: {e}")
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
