"""
Simplified polling adapter to convert CCXT fetch_* methods into watch_* behavior.

This adapter provides a much simpler approach:
- No background tasks or queues
- get_next_data() waits until it's time to poll, then polls synchronously
- Time-aligned polling (e.g., 11:30, 11:35, 11:40 for 5-minute intervals)
- Immediate polling when symbols change
"""

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from qubx import logger

# Constants
DEFAULT_POLL_INTERVAL = 300  # 5 minutes
MIN_POLL_INTERVAL = 1  # 1 second minimum
MAX_POLL_INTERVAL = 3600  # 1 hour maximum


@dataclass
class PollingConfig:
    """Configuration for polling adapter."""

    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not MIN_POLL_INTERVAL <= self.poll_interval_seconds <= MAX_POLL_INTERVAL:
            raise ValueError(
                f"poll_interval_seconds must be between {MIN_POLL_INTERVAL} and {MAX_POLL_INTERVAL}, "
                f"got {self.poll_interval_seconds}"
            )


class PollingToWebSocketAdapter:
    """
    Simplified polling adapter that polls synchronously when data is requested.

    Key features:
    - No background tasks or queues
    - Time-aligned polling (respects clock boundaries)
    - Immediate polling when symbols change
    - Thread-safe symbol management
    """

    def __init__(
        self,
        fetch_method: Callable,
        symbols: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[PollingConfig] = None,
    ):
        """
        Initialize the simplified polling adapter.

        Args:
            fetch_method: The CCXT fetch_* method to call
            symbols: Initial list of symbols to watch
            params: Additional parameters for fetch_method
            config: PollingConfig instance (uses default if None)
        """
        self.config = config if config is not None else PollingConfig()
        self.fetch_method = fetch_method
        self.params = params or {}
        self.adapter_id = f"polling_adapter_{id(self)}"

        # Thread-safe symbol management
        self._symbols_lock = asyncio.Lock()
        self._symbols: Set[str] = set(symbols or [])

        # Polling state
        self._last_poll_time: Optional[float] = None
        self._symbols_changed = False  # Flag to trigger immediate poll

        # Statistics
        self._poll_count = 0
        self._error_count = 0

    async def get_next_data(self) -> Dict[str, Any]:
        """
        Get the next available data by waiting until it's time to poll, then polling.

        This method:
        1. Checks if symbols changed (immediate poll)
        2. Calculates when next poll should happen based on time alignment
        3. Waits until that time
        4. Polls and returns fresh data

        Returns:
            Dictionary containing fetched data for symbols
        """
        async with self._symbols_lock:
            current_symbols = list(self._symbols)
            symbols_changed = self._symbols_changed

        # If symbols changed, poll immediately
        if symbols_changed:
            logger.debug(f"Symbols changed, polling immediately for adapter {self.adapter_id}")
            async with self._symbols_lock:
                self._symbols_changed = False
            return await self._poll_now(current_symbols)

        # Calculate wait time for next aligned poll
        wait_time = self._calculate_wait_time()

        if wait_time > 0:
            logger.debug(f"Waiting {wait_time:.1f}s for next poll cycle for adapter {self.adapter_id}")
            await asyncio.sleep(wait_time)

        # Time to poll
        logger.debug(f"Polling now for adapter {self.adapter_id}")
        return await self._poll_now(current_symbols)

    def _calculate_wait_time(self) -> float:
        """
        Calculate how long to wait until the next aligned poll time.

        For intervals >= 1 minute: aligns to clock boundaries (11:30, 11:35, 11:40)
        For intervals < 1 minute: uses simple interval-based timing

        Returns:
            Number of seconds to wait (0 if should poll now)
        """
        current_time = time.time()
        interval_seconds = self.config.poll_interval_seconds

        # First poll is always immediate
        if self._last_poll_time is None:
            return 0

        if interval_seconds >= 60:
            # Time-aligned polling for intervals >= 1 minute using UTC
            # Calculate next boundary based on seconds since epoch
            next_boundary = math.ceil(current_time / interval_seconds) * interval_seconds
            wait_time = next_boundary - current_time
            return max(0, wait_time)
        else:
            # Simple interval-based polling for sub-minute intervals
            next_poll_time = self._last_poll_time + interval_seconds
            wait_time = next_poll_time - current_time
            return max(0, wait_time)

    async def _poll_now(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Perform a poll operation immediately.

        Args:
            symbols: List of symbols to poll for

        Returns:
            Dictionary containing fetched data for symbols
        """
        self._poll_count += 1
        self._last_poll_time = time.time()

        logger.debug(f"Polling {len(symbols) if symbols else 'all'} symbols for adapter {self.adapter_id}")

        try:
            # Filter out adapter-specific parameters
            adapter_params = {"pollInterval", "interval", "updateInterval", "poll_interval_minutes"}
            fetch_params = {k: v for k, v in self.params.items() if k not in adapter_params}

            # Call the fetch method
            result = await self.fetch_method(symbols, **fetch_params)

            logger.debug(f"Poll completed successfully for adapter {self.adapter_id}")
            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"Poll failed for adapter {self.adapter_id}: {e}")
            raise

    async def update_symbols(self, new_symbols: List[str]) -> None:
        """
        Update the symbol list.

        If symbols changed, the next call to get_next_data() will poll immediately.

        Args:
            new_symbols: New complete list of symbols to watch
        """
        async with self._symbols_lock:
            old_symbols = self._symbols.copy()
            self._symbols = set(new_symbols or [])
            symbols_changed = old_symbols != self._symbols

            if symbols_changed:
                self._symbols_changed = True
                logger.debug(
                    f"Symbols updated for adapter {self.adapter_id}: {len(old_symbols)} -> {len(self._symbols)}"
                )

    async def add_symbols(self, new_symbols: List[str]) -> None:
        """Add symbols to the existing watch list."""
        if not new_symbols:
            return

        async with self._symbols_lock:
            before_count = len(self._symbols)
            self._symbols.update(new_symbols)
            after_count = len(self._symbols)

            if after_count > before_count:
                self._symbols_changed = True
                logger.debug(f"Added {after_count - before_count} symbols to adapter {self.adapter_id}")

    async def remove_symbols(self, symbols_to_remove: List[str]) -> None:
        """Remove symbols from the watch list."""
        if not symbols_to_remove:
            return

        async with self._symbols_lock:
            before_count = len(self._symbols)
            self._symbols.difference_update(symbols_to_remove)
            after_count = len(self._symbols)

            if after_count < before_count:
                self._symbols_changed = True
                logger.debug(f"Removed {before_count - after_count} symbols from adapter {self.adapter_id}")

    def is_watching(self, symbol: Optional[str] = None) -> bool:
        """Check if adapter has symbols configured to watch."""
        if symbol is None:
            return len(self._symbols) > 0
        else:
            return symbol in self._symbols

    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics for monitoring."""
        return {
            "adapter_id": self.adapter_id,
            "symbol_count": len(self._symbols),
            "poll_count": self._poll_count,
            "error_count": self._error_count,
            "last_poll_time": self._last_poll_time,
            "poll_interval_seconds": self.config.poll_interval_seconds,
        }

    async def stop(self) -> None:
        """Stop the adapter (cleanup method for compatibility)."""
        logger.debug(f"Adapter {self.adapter_id} stopped (polled {self._poll_count} times, {self._error_count} errors)")
