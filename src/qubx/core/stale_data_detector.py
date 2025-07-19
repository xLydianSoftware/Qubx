"""
Stale Data Detection Module

This module provides functionality to detect when instrument prices haven't moved
for a specified period of time, indicating stale or inactive market data.
"""

import pandas as pd

from qubx import logger
from qubx.core.basics import Instrument, dt_64, td_64
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.interfaces import ITimeProvider


class InstrumentStaleState:
    """
    Tracks stale data analysis state for a single instrument to enable incremental analysis.
    """
    
    def __init__(self):
        # Time of the last bar that was analyzed
        self.last_checked_bar_time: dt_64 | None = None
        
        # Duration of consecutive stale data (in nanoseconds)
        self.consecutive_stale_duration: int = 0
        
        # Price of the last analyzed bar (to check for price continuity)
        self.last_bar_price: float | None = None
        
        # Whether we're currently in a stale state
        self.is_currently_stale: bool = False
        
        # When the current stale period started
        self.stale_start_time: dt_64 | None = None
        
        # Time when this state was last updated
        self.last_analysis_time: dt_64 | None = None
        
    def reset(self) -> None:
        """Reset all state for this instrument."""
        self.last_checked_bar_time = None
        self.consecutive_stale_duration = 0
        self.last_bar_price = None
        self.is_currently_stale = False
        self.stale_start_time = None
        self.last_analysis_time = None


class StaleDataDetector:
    """
    Detects stale market data by checking if instrument prices haven't moved
    for a specified period of time.
    
    Stale data is defined as:
    1. All OHLC values are equal for each bar (open == high == low == close)
    2. Consecutive bars have the same price values
    3. This condition persists for the entire detection period
    """
    
    def __init__(
        self,
        cache: CachedMarketDataHolder,
        time_provider: ITimeProvider,
        detection_period: td_64 | str = td_64(2, "h"),
        check_interval: td_64 | str = td_64(10, "m"),
        min_bars_required: int = 3,
    ):
        """
        Initialize the stale data detector.
        
        Args:
            cache: Market data cache holder
            time_provider: Time provider for current time
            detection_period: Period over which to check for stale data (td_64 or string like "2h", "5Min")
            check_interval: Interval between stale data checks (td_64 or string like "10m", "30s")
            min_bars_required: Minimum number of bars required to declare data stale
        """
        self._cache = cache
        self._time_provider = time_provider
        
        # Parse string parameters to td_64 if needed
        if isinstance(detection_period, str):
            self._detection_period = td_64(pd.Timedelta(detection_period).value, "ns")
        else:
            self._detection_period = detection_period
            
        if isinstance(check_interval, str):
            self._check_interval = td_64(pd.Timedelta(check_interval).value, "ns")
        else:
            self._check_interval = check_interval
            
        self._min_bars_required = min_bars_required
        
        # Track last check time per instrument
        self._last_check_time: dict[Instrument, dt_64] = {}
        
        # Track stale analysis state per instrument for incremental analysis
        self._instrument_states: dict[Instrument, InstrumentStaleState] = {}
    
    def is_instrument_stale(self, instrument: Instrument) -> bool:
        """
        Check if the given instrument has stale data using incremental analysis.
        Only analyzes new bars since the last check to improve performance.
        
        Args:
            instrument: The instrument to check
            
        Returns:
            bool: True if data is stale, False otherwise
        """
        try:
            # Get the OHLCV data for the default timeframe
            ohlcv = self._cache.get_ohlcv(instrument, self._cache.default_timeframe)
            
            if len(ohlcv) < self._min_bars_required:
                return False  # Not enough data to determine staleness
            
            # Get or create state for this instrument
            if instrument not in self._instrument_states:
                self._instrument_states[instrument] = InstrumentStaleState()
            
            state = self._instrument_states[instrument]
            current_time = self._time_provider.time()
            
            # Get timeframe in nanoseconds for duration calculations
            timeframe_ns = int(self._cache.default_timeframe / td_64(1, "ns"))
            detection_period_ns = int(self._detection_period / td_64(1, "ns"))
            
            # Check if we need to reset state (first time or data reset)
            needs_reset = (
                state.last_checked_bar_time is None or
                len(ohlcv) == 0
            )
            
            # Additional reset check: if we have previous state but the current data doesn't look like
            # a continuation of what we analyzed before, reset
            if not needs_reset and state.last_checked_bar_time is not None:
                # If the most recent bar we can find is much older than what we last checked,
                # or if the price structure has changed dramatically, reset
                newest_bar_time = ohlcv[0].time if ohlcv else None
                if (newest_bar_time is None or 
                    newest_bar_time < state.last_checked_bar_time or
                    # Check if the data structure looks fundamentally different
                    (state.last_bar_price is not None and ohlcv and
                     abs(ohlcv[0].close - state.last_bar_price) > state.last_bar_price * 0.5)):  # 50% price change indicates reset
                    needs_reset = True
            
            if needs_reset:
                state.reset()
            
            # If no previous analysis, we need to do a full initial analysis
            if state.last_checked_bar_time is None:
                return self._perform_initial_analysis(ohlcv, state, timeframe_ns, detection_period_ns, current_time)
            
            # Incremental analysis: only check new bars using efficient .loc operation
            # Use .loc to slice from timestamp onwards - this uses binary search internally
            try:
                # Get all data from the last checked timestamp onwards, then filter out the exact timestamp
                candidate_bars_slice = ohlcv.loc[state.last_checked_bar_time:]
                candidate_bars = list(candidate_bars_slice) if candidate_bars_slice else []
                # Filter to only get bars after the last checked time (exclude exact match)
                new_bars = [bar for bar in candidate_bars if bar.time > state.last_checked_bar_time]
            except (KeyError, IndexError, OverflowError):
                # Handle case where timestamp is beyond available data or overflow
                # Fall back to original method for safety
                new_bars = [bar for bar in ohlcv if bar.time > state.last_checked_bar_time]
            
            if not new_bars:
                # No new data, return current stale status
                state.last_analysis_time = current_time
                return state.consecutive_stale_duration >= detection_period_ns
            
            # Process new bars incrementally (they're already in chronological order from .loc)
            for bar in new_bars:
                self._process_bar_incrementally(bar, state, timeframe_ns)
            
            # Update analysis time
            state.last_analysis_time = current_time
            
            # Return true if we've been stale for the detection period
            return state.consecutive_stale_duration >= detection_period_ns
            
        except Exception as e:
            logger.warning(f"Error checking stale data for {instrument.symbol}: {e}")
            return False
    
    def _analyze_staleness(self, ohlcv, bars_to_check: int) -> bool:
        """
        Analyze staleness using the original detection algorithm.
        
        Args:
            ohlcv: OHLCV data array
            bars_to_check: Number of bars to check
            
        Returns:
            bool: True if stale, False otherwise
        """
        if len(ohlcv) < self._min_bars_required:
            return False
        
        # Check the last N bars (original algorithm)
        bars_checked = 0
        prev_bar = None
        
        for i in range(min(bars_to_check, len(ohlcv))):
            bar = ohlcv[i]  # Latest bars are at index 0
            
            # Check if this bar is flat (price hasn't moved within the bar)
            if not (bar.open == bar.high == bar.low == bar.close):
                return False  # Found a bar with intra-bar price movement
            
            # Check if consecutive bars have the same price
            if prev_bar is not None:
                if not (bar.open == prev_bar.open == prev_bar.high == prev_bar.low == prev_bar.close):
                    return False  # Found price movement between bars
            
            prev_bar = bar
            bars_checked += 1
            
            # If we've checked enough bars and they're all flat with same prices, it's stale
            if bars_checked >= bars_to_check:
                return True
        
        # If we checked all available bars and they're all flat with same prices,
        # only consider it stale if we've covered enough time period
        return bool(bars_checked >= bars_to_check and bars_checked >= self._min_bars_required)
    
    def _perform_initial_analysis(self, ohlcv, state: InstrumentStaleState, timeframe_ns: int, detection_period_ns: int, current_time: dt_64) -> bool:
        """
        Perform initial analysis when no previous state exists.
        This builds up the stale state from scratch by checking recent bars.
        
        Args:
            ohlcv: OHLCV data array
            state: Instrument state to update
            timeframe_ns: Timeframe in nanoseconds
            detection_period_ns: Detection period in nanoseconds
            current_time: Current analysis time
            
        Returns:
            bool: True if stale, False otherwise
        """
        if not ohlcv:
            return False
            
        # Calculate how many bars we need to check to cover the detection period
        bars_needed = max(int(detection_period_ns / timeframe_ns), self._min_bars_required)
        bars_to_check = min(len(ohlcv), bars_needed)
        
        # Start from the most recent bar and work backwards
        state.consecutive_stale_duration = 0
        state.is_currently_stale = False
        state.stale_start_time = None
        
        # Check bars in reverse chronological order (most recent first)
        prev_bar = None
        consecutive_stale_bars = 0
        
        for i in range(bars_to_check):
            bar = ohlcv[i]
            
            # Check if this bar is stale (flat within bar and same as previous)
            is_bar_stale = self._is_bar_stale(bar, prev_bar)
            
            if is_bar_stale:
                # This bar is stale
                consecutive_stale_bars += 1
                if state.consecutive_stale_duration == 0:
                    # Starting a new stale period
                    state.stale_start_time = bar.time
                    state.is_currently_stale = True
                
                state.consecutive_stale_duration += timeframe_ns
                state.last_bar_price = bar.close
            else:
                # Non-stale bar found, stop looking back
                break
            
            prev_bar = bar
        
        # Update state with the most recent bar
        if ohlcv:
            state.last_checked_bar_time = ohlcv[0].time
        state.last_analysis_time = current_time
        
        return state.consecutive_stale_duration >= detection_period_ns
    
    def _process_bar_incrementally(self, bar, state: InstrumentStaleState, timeframe_ns: int) -> None:
        """
        Process a single new bar incrementally, updating the stale state.
        
        Args:
            bar: The new bar to process
            state: Instrument state to update
            timeframe_ns: Timeframe in nanoseconds
        """
        # Check if this bar is stale (flat and same price as previous if we have one)
        is_bar_flat = (bar.open == bar.high == bar.low == bar.close)
        
        # Check if price is same as previous bar (if we have previous price)
        price_same_as_previous = (
            state.last_bar_price is None or 
            bar.open == state.last_bar_price
        )
        
        is_bar_stale = is_bar_flat and price_same_as_previous
        
        if is_bar_stale and state.is_currently_stale:
            # Continue stale period
            state.consecutive_stale_duration += timeframe_ns
        elif is_bar_stale and not state.is_currently_stale:
            # Start new stale period
            state.is_currently_stale = True
            state.stale_start_time = bar.time
            state.consecutive_stale_duration = timeframe_ns
        else:
            # Non-stale bar, reset stale tracking
            state.is_currently_stale = False
            state.consecutive_stale_duration = 0
            state.stale_start_time = None
        
        # Update tracking variables
        state.last_checked_bar_time = bar.time
        state.last_bar_price = bar.close
    
    def _is_bar_stale(self, bar, prev_bar) -> bool:
        """
        Check if a single bar is stale (flat within bar and same price as previous).
        
        Args:
            bar: Current bar to check
            prev_bar: Previous bar for comparison (can be None)
            
        Returns:
            bool: True if bar is stale, False otherwise
        """
        # Check if bar is flat (no intra-bar movement)
        if not (bar.open == bar.high == bar.low == bar.close):
            return False
        
        # If we have a previous bar, check if prices are the same
        if prev_bar is not None:
            if not (bar.open == prev_bar.close):  # Compare current bar price to previous bar's close
                return False
        
        # If no previous bar (first bar), it's stale if it's flat
        return True
    
    
    def should_check_instrument(self, instrument: Instrument) -> bool:
        """
        Check if enough time has passed since the last check for this instrument.
        
        Args:
            instrument: The instrument to check
            
        Returns:
            bool: True if the instrument should be checked, False otherwise
        """
        current_time = self._time_provider.time()
        last_check = self._last_check_time.get(instrument)
        
        if last_check is None:
            return True  # Never checked before
        
        time_since_last_check = current_time - last_check
        return bool(time_since_last_check >= self._check_interval)
    
    def detect_stale_instruments(self, instruments: list[Instrument]) -> list[Instrument]:
        """
        Detect stale instruments from the given list.
        
        Args:
            instruments: List of instruments to check
            
        Returns:
            list[Instrument]: List of instruments that were detected as stale
        """
        current_time = self._time_provider.time()
        stale_instruments = []
        
        for instrument in instruments:
            # Check if we need to check this instrument (based on check interval)
            if not self.should_check_instrument(instrument):
                continue  # Too soon to check again
            
            # Check if this instrument has stale data
            if self.is_instrument_stale(instrument):
                stale_instruments.append(instrument)
            
            # Update the last check time
            self._last_check_time[instrument] = current_time
        
        return stale_instruments
    
    def reset_check_time(self, instrument: Instrument) -> None:
        """
        Reset the last check time for an instrument and clear its cached state.
        
        Args:
            instrument: The instrument to reset
        """
        self._last_check_time.pop(instrument, None)
        if instrument in self._instrument_states:
            self._instrument_states[instrument].reset()
    
    def get_last_check_time(self, instrument: Instrument) -> dt_64 | None:
        """
        Get the last check time for an instrument.
        
        Args:
            instrument: The instrument to check
            
        Returns:
            dt_64 | None: The last check time or None if never checked
        """
        return self._last_check_time.get(instrument)