"""
Data throttling utilities for rate-limiting high-frequency updates.

Used to reduce processing overhead by throttling updates to a maximum frequency,
typically for quote/orderbook data that updates many times per second.
"""

import time
from collections import defaultdict

from qubx.core.basics import Instrument


class InstrumentThrottler:
    """
    Throttles data updates per data type and instrument to a maximum frequency.

    Maintains separate throttle state for each (data_type, instrument) pair,
    allowing independent rate limiting across different data types and instruments.

    Example:
        config = {
            "quote": 2.0,      # Max 2 quote updates/sec per instrument
            "orderbook": 5.0,  # Max 5 orderbook updates/sec per instrument
        }
        throttler = InstrumentThrottler(config)

        if throttler.should_send("quote", instrument):
            # Process the quote
            process_data(data)
    """

    def __init__(self, throttle_config: dict[str, float] | None = None):
        """
        Initialize throttler with per-data-type frequency limits.

        Args:
            throttle_config: Dictionary mapping data type to max frequency (Hz).
                           Example: {"quote": 2.0, "orderbook": 5.0}
                           If None or empty, no throttling is applied.
        """
        self._throttle_config = throttle_config or {}
        self._min_intervals: dict[str, float] = {}
        self._last_send_time: dict[tuple[str, Instrument], float] = defaultdict(lambda: 0.0)

        # Precompute minimum intervals for each data type
        for data_type, max_freq_hz in self._throttle_config.items():
            if max_freq_hz > 0:
                self._min_intervals[data_type] = 1.0 / max_freq_hz
            else:
                self._min_intervals[data_type] = 0.0

    def should_send(self, data_type: str, instrument: Instrument) -> bool:
        """
        Check if data should be processed based on throttle interval.

        If data type is not configured for throttling, always returns True.
        For configured data types, throttles to the specified frequency per instrument.

        Args:
            data_type: Type of data (e.g., "quote", "orderbook", "trade")
            instrument: Instrument for the data

        Returns:
            True if data should be processed, False if throttled
        """
        # If this data type is not configured for throttling, allow all updates
        if data_type not in self._min_intervals:
            return True

        min_interval = self._min_intervals[data_type]
        key = (data_type, instrument)
        current_time = time.time()
        last_time = self._last_send_time[key]

        # Check if enough time has passed
        if current_time - last_time >= min_interval:
            self._last_send_time[key] = current_time
            return True

        # Throttled
        return False

    def is_throttled(self, data_type: str) -> bool:
        """
        Check if a data type is configured for throttling.

        Args:
            data_type: Type of data to check

        Returns:
            True if this data type has throttling enabled
        """
        return data_type in self._min_intervals

    def get_frequency(self, data_type: str) -> float | None:
        """
        Get the configured max frequency for a data type.

        Args:
            data_type: Type of data

        Returns:
            Max frequency in Hz, or None if not throttled
        """
        return self._throttle_config.get(data_type)

    def reset(self, data_type: str | None = None, instrument: Instrument | None = None) -> None:
        """
        Reset throttle state.

        Args:
            data_type: If provided, reset only this data type. If None, reset all.
            instrument: If provided, reset only this instrument. If None, reset all.
        """
        if data_type is None and instrument is None:
            self._last_send_time.clear()
        elif data_type is not None and instrument is None:
            # Reset all instruments for this data type
            keys_to_remove = [k for k in self._last_send_time.keys() if k[0] == data_type]
            for key in keys_to_remove:
                del self._last_send_time[key]
        elif data_type is None and instrument is not None:
            # Reset this instrument for all data types
            keys_to_remove = [k for k in self._last_send_time.keys() if k[1] == instrument]
            for key in keys_to_remove:
                del self._last_send_time[key]
        else:
            # Reset specific (data_type, instrument) pair
            if data_type is not None and instrument is not None:
                key = (data_type, instrument)
                self._last_send_time.pop(key, None)

    def __repr__(self) -> str:
        throttled_types = ", ".join(f"{dt}:{freq}Hz" for dt, freq in self._throttle_config.items())
        return f"InstrumentThrottler({throttled_types}, tracked_pairs={len(self._last_send_time)})"
