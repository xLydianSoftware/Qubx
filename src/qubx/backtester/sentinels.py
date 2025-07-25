from __future__ import annotations

from typing import Optional


class NoDataContinue:
    """Sentinel indicating no data streams available but simulation should continue.
    
    This is used when all instruments are unsubscribed but there may still be
    scheduled events to process before the simulation stop time.
    """
    
    def __init__(self, next_scheduled_time: Optional[int] = None):
        """Initialize the sentinel.
        
        Args:
            next_scheduled_time: The next scheduled event time in nanoseconds,
                                or None if no scheduled events exist.
        """
        self.next_scheduled_time = next_scheduled_time
    
    def __repr__(self) -> str:
        return f"NoDataContinue(next_scheduled_time={self.next_scheduled_time})"