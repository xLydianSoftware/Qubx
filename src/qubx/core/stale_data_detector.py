"""
Stale Data Detection Module

This module provides functionality to detect when instrument prices haven't moved
for a specified period of time, indicating stale or inactive market data.
"""

from qubx import logger
from qubx.core.basics import Instrument, dt_64, td_64
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.interfaces import ITimeProvider


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
        detection_period: td_64 = td_64(2, "h"),
        check_interval: td_64 = td_64(10, "m"),
        min_bars_required: int = 3,
    ):
        """
        Initialize the stale data detector.
        
        Args:
            cache: Market data cache holder
            time_provider: Time provider for current time
            detection_period: Period over which to check for stale data
            check_interval: Interval between stale data checks
            min_bars_required: Minimum number of bars required to declare data stale
        """
        self._cache = cache
        self._time_provider = time_provider
        self._detection_period = detection_period
        self._check_interval = check_interval
        self._min_bars_required = min_bars_required
        
        # Track last check time per instrument
        self._last_check_time: dict[Instrument, dt_64] = {}
    
    def is_instrument_stale(self, instrument: Instrument) -> bool:
        """
        Check if the given instrument has stale data.
        
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
            
            # Calculate how many bars we need to check based on timeframe
            timeframe_seconds = self._cache.default_timeframe / td_64(1, "s")
            detection_period_seconds = self._detection_period / td_64(1, "s")
            bars_to_check = max(int(detection_period_seconds / timeframe_seconds), self._min_bars_required)
            
            # Check the last N bars
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
            
        except Exception as e:
            logger.warning(f"Error checking stale data for {instrument.symbol}: {e}")
            return False
    
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
        Reset the last check time for an instrument.
        
        Args:
            instrument: The instrument to reset
        """
        self._last_check_time.pop(instrument, None)
    
    def get_last_check_time(self, instrument: Instrument) -> dt_64 | None:
        """
        Get the last check time for an instrument.
        
        Args:
            instrument: The instrument to check
            
        Returns:
            dt_64 | None: The last check time or None if never checked
        """
        return self._last_check_time.get(instrument)