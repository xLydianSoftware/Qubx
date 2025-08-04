"""
Generic polling adapter to convert CCXT fetch_* methods into watch_* behavior.

This adapter allows any exchange's fetch_* method to be used as a watch_* method
by implementing intelligent polling with proper resource management and unwatch functionality.
"""

import asyncio
import concurrent.futures
import time
from asyncio import CancelledError
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Union

from qubx import logger
from qubx.utils.misc import AsyncThreadLoop

# Constants
DEFAULT_POLL_INTERVAL = 300  # 5 minutes
DEFAULT_TIMEOUT = 30.0
DEFAULT_QUEUE_SIZE = 100
MIN_POLL_INTERVAL = 0.1  # Allow 100ms for testing
MAX_POLL_INTERVAL = 3600  # 1 hour
INITIAL_DATA_WAIT_TIMEOUT = 10.0
TASK_CLEANUP_TIMEOUT = 5.0


@dataclass
class PollingConfig:
    """
    Configuration for polling adapter.
    
    This class provides structured configuration with validation
    for all polling adapter parameters.
    """
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL
    timeout_seconds: float = DEFAULT_TIMEOUT
    queue_size: int = DEFAULT_QUEUE_SIZE
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not MIN_POLL_INTERVAL <= self.poll_interval_seconds <= MAX_POLL_INTERVAL:
            raise ValueError(
                f"poll_interval_seconds must be between {MIN_POLL_INTERVAL} and {MAX_POLL_INTERVAL}, "
                f"got {self.poll_interval_seconds}"
            )
        
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
            
        if self.queue_size <= 0:
            raise ValueError(f"queue_size must be positive, got {self.queue_size}")


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
        symbols: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[PollingConfig] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Initialize the polling adapter.

        Args:
            fetch_method: The CCXT fetch_* method to call (e.g., self.fetch_funding_rates)
            symbols: Initial list of symbols to watch
            params: Additional parameters for fetch_method
            config: PollingConfig instance (uses default if None)
            event_loop: Optional explicit asyncio event loop to use for background tasks
        """
        # Handle configuration
        self.config = config if config is not None else PollingConfig()
        
        self.fetch_method = fetch_method
        self.params = params or {}
        self.adapter_id = f"polling_adapter_{id(self)}"  # Auto-generated for logging
        self.event_loop = event_loop  # Store explicit loop if provided

        # Thread-safe symbol management
        self._symbols_lock = asyncio.Lock()
        self._symbols: Set[str] = set(symbols or [])
        self._symbols_changed_event = asyncio.Event()  # Track when symbols change

        # Polling state management
        self._polling_task: Optional[Union[asyncio.Task, concurrent.futures.Future]] = None
        self._stop_event = asyncio.Event()
        self._force_poll_event = asyncio.Event()
        self._is_running = False

        # Statistics for monitoring
        self._poll_count = 0
        self._error_count = 0
        self._last_poll_time: Optional[float] = None

        # Data management for awaitable pattern
        self._data_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.queue_size)
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
            
        # If symbols recently changed, wait for the forced poll to complete
        if self._symbols_changed_event.is_set():
            logger.debug(f"Symbols changed, waiting for forced poll to complete for adapter {self.adapter_id}")
            start_time = asyncio.get_event_loop().time()
            timeout = 10.0  # 10 second timeout
            
            while self._symbols_changed_event.is_set() and self._is_running:
                if (asyncio.get_event_loop().time() - start_time) > timeout:
                    logger.warning(f"Timeout waiting for forced poll after symbol change for adapter {self.adapter_id}")
                    break
                await asyncio.sleep(0.1)
            
            if not self._symbols_changed_event.is_set():
                logger.debug(f"Forced poll completed for adapter {self.adapter_id}")
        
        # Wait for new data from polling task
        try:
            # First, check if we have data immediately available
            try:
                data = self._data_queue.get_nowait()
                return data
            except asyncio.QueueEmpty:
                pass
            
            # If no immediate data, wait with timeout
            data = await asyncio.wait_for(self._data_queue.get(), timeout=self.config.timeout_seconds)
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
        self._force_poll_event.clear()  # Clear any pending force polls
        self._symbols_changed_event.clear()  # Clear symbols changed event
        self._is_running = False

        # Clear data queue to prevent stale results from being processed
        self._clear_data_queue()

        await self._cleanup_polling_task()

        logger.debug(f"Adapter {self.adapter_id} stopped (polled {self._poll_count} times, {self._error_count} errors)")

    def _clear_data_queue(self) -> None:
        """Clear stale data from the queue and cached data."""
        cleared_count = 0
        while not self._data_queue.empty():
            try:
                self._data_queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        
        # Also clear cached latest data since symbols changed
        if self._latest_data is not None:
            logger.debug(f"Clearing cached latest data for adapter {self.adapter_id}")
            self._latest_data = None
            
        if cleared_count > 0:
            logger.debug(f"Cleared {cleared_count} stale items from data queue for adapter {self.adapter_id}")

    async def _cleanup_polling_task(self) -> None:
        """Clean up the polling task."""
        if self._polling_task and not self._polling_task.done():
            try:
                # Handle both Task (normal operation) and Future (explicit event loop) objects
                if hasattr(self._polling_task, 'result') and not hasattr(self._polling_task, '__await__'):
                    # This is a Future from AsyncThreadLoop.submit()
                    self._polling_task.result(timeout=TASK_CLEANUP_TIMEOUT)
                else:
                    # This is a regular Task
                    await asyncio.wait_for(self._polling_task, timeout=TASK_CLEANUP_TIMEOUT)
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

    async def add_symbols(self, new_symbols: List[str], immediate_poll: bool = True) -> None:
        """
        Add symbols to the existing watch list.

        Args:
            new_symbols: List of symbols to add
            immediate_poll: Whether to trigger immediate poll if symbols were added (default: True)
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
            
            # Trigger immediate poll if symbols were added
            if immediate_poll and self._is_running:
                logger.debug(f"Triggering immediate poll for adapter {self.adapter_id} due to new symbols")
                # Clear stale data from queue since symbols changed
                self._clear_data_queue()
                self._symbols_changed_event.set()  # Mark that symbols changed
                self._force_poll_event.set()

    async def remove_symbols(self, symbols_to_remove: List[str], immediate_poll: bool = True) -> None:
        """
        Remove specific symbols from the watch list.

        Args:
            symbols_to_remove: List of symbols to remove
            immediate_poll: Whether to trigger immediate poll if symbols were removed (default: True)
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
            
            # Trigger immediate poll if symbols were removed
            if immediate_poll and self._is_running:
                logger.debug(f"Triggering immediate poll for adapter {self.adapter_id} due to removed symbols")
                # Clear stale data from queue since symbols changed
                self._clear_data_queue()
                self._symbols_changed_event.set()  # Mark that symbols changed
                self._force_poll_event.set()

        # If no symbols left, we could optionally stop polling
        # For now, we'll keep polling in case symbols are added back

    async def update_symbols(self, new_symbols: List[str], immediate_poll: bool = True) -> None:
        """
        Replace entire symbol list (atomic operation).

        Args:
            new_symbols: New complete list of symbols to watch
            immediate_poll: Whether to trigger immediate poll if symbols changed (default: True)
        """
        async with self._symbols_lock:
            old_symbols = self._symbols.copy()
            self._symbols = set(new_symbols or [])
            symbols_changed = old_symbols != self._symbols

        logger.debug(f"Updated symbols for adapter {self.adapter_id}: {len(old_symbols)} -> {len(self._symbols)}")
        
        # Trigger immediate poll if symbols changed
        if symbols_changed and immediate_poll and self._is_running:
            logger.debug(f"Triggering immediate poll for adapter {self.adapter_id} due to symbol change")
            # Clear stale data from queue since symbols changed
            self._clear_data_queue()
            self._symbols_changed_event.set()  # Mark that symbols changed - get_next_data should wait
            self._force_poll_event.set()

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

    async def force_poll(self) -> Optional[Dict[str, Any]]:
        """
        Force an immediate poll and return result.
        
        This method can be used to get immediate data without waiting
        for the next scheduled poll interval.
        
        Returns:
            Dictionary containing fetched data for symbols, or None if no symbols
        """
        if not self._is_running:
            await self.start_watching()
            
        async with self._symbols_lock:
            current_symbols = list(self._symbols)
            
        if not current_symbols:
            logger.warning(f"No symbols configured for adapter {self.adapter_id}")
            return None
            
        try:
            await self._poll_once(current_symbols)
            return self._latest_data
        except Exception as e:
            logger.error(f"Force poll failed for adapter {self.adapter_id}: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics for monitoring."""
        return {
            "adapter_id": self.adapter_id,
            "is_running": self._is_running,
            "symbol_count": len(self._symbols),
            "poll_count": self._poll_count,
            "error_count": self._error_count,
            "last_poll_time": self._last_poll_time,
            "poll_interval_seconds": self.config.poll_interval_seconds,
            "timeout_seconds": self.config.timeout_seconds,
            "queue_size": self.config.queue_size,
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

                    # Check if we should poll (scheduled OR forced)
                    force_poll = self._force_poll_event.is_set()
                    scheduled_poll = first_poll or self._should_poll_now()
                    
                    if force_poll or scheduled_poll:
                        if force_poll:
                            self._force_poll_event.clear()
                            logger.debug(f"Executing forced poll for adapter {self.adapter_id}")
                        
                        await self._poll_once(current_symbols)
                        first_poll = False
                        
                        # Clear symbols changed event after successful forced poll
                        if force_poll:
                            self._symbols_changed_event.clear()
                            logger.debug(f"Cleared symbols changed event after forced poll for adapter {self.adapter_id}")
                        
                        # Only sleep full interval after scheduled polls, not forced
                        if not force_poll:
                            await self._cancellable_sleep(self.config.poll_interval_seconds)
                        else:
                            # Brief pause after forced poll to prevent tight loops
                            await self._cancellable_sleep(1)
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
                    sleep_time = max(1, min(30, self.config.poll_interval_seconds // 10))
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
        # Simple interval-based polling
        if self._last_poll_time is None:
            return True
        return (time.time() - self._last_poll_time) >= self.config.poll_interval_seconds

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
            adapter_params = {'pollInterval', 'interval', 'updateInterval', 'poll_interval_minutes'}
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
        Sleep that can be interrupted by the stop event or force poll event.

        Args:
            seconds: Number of seconds to sleep
        """
        if seconds <= 0:
            return

        try:
            # Wait for either stop_event or force_poll_event
            done, pending = await asyncio.wait(
                [asyncio.create_task(self._stop_event.wait()),
                 asyncio.create_task(self._force_poll_event.wait())],
                timeout=seconds,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        except asyncio.TimeoutError:
            # Timeout is expected - means we slept for the full duration
            pass
