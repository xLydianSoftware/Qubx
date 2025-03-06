"""
Strategy initialization utilities.

This module provides classes for initializing strategies, including setting up
schedules, warmup periods, and position mismatch resolution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from qubx.core.interfaces import IStrategyInitializer, PositionMismatchResolver, StartTimeFinder


@dataclass
class BasicStrategyInitializer(IStrategyInitializer):
    """
    Basic implementation of the IStrategyInitializer interface.

    This class stores configuration information set during strategy initialization,
    such as schedules, warmup periods, and position mismatch resolvers.
    """

    # Default values for all fields
    base_subscription: Optional[str] = None
    fit_schedule: Optional[str] = None
    event_schedule: Optional[str] = None
    warmup_period: Optional[str] = None
    start_time_finder: Optional[StartTimeFinder] = None
    mismatch_resolver: Optional[PositionMismatchResolver] = None

    # Additional configuration that might be needed
    config: Dict[str, Any] = field(default_factory=dict)

    _auto_subscribe: Optional[bool] = None

    def set_base_subscription(self, subscription_type: str) -> None:
        self.base_subscription = subscription_type

    @property
    def auto_subscribe(self) -> bool | None:
        return self._auto_subscribe

    @auto_subscribe.setter
    def auto_subscribe(self, value: bool) -> None:
        self._auto_subscribe = value

    def set_fit_schedule(self, schedule: str) -> None:
        self.fit_schedule = schedule

    def set_event_schedule(self, schedule: str) -> None:
        self.event_schedule = schedule

    def set_warmup(self, period: str, start_time_finder: StartTimeFinder | None = None) -> None:
        self.warmup_period = period
        self.start_time_finder = start_time_finder

    def set_mismatch_resolver(self, resolver: PositionMismatchResolver) -> None:
        self.mismatch_resolver = resolver

    def set_config(self, key: str, value: Any) -> None:
        self.config[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
