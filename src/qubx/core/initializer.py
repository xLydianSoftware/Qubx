"""
Strategy initialization utilities.

This module provides classes for initializing strategies, including setting up
schedules, warmup periods, and position mismatch resolution.
"""

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from qubx.core.basics import Instrument, td_64
from qubx.core.interfaces import IStrategyInitializer, StartTimeFinderProtocol, StateResolverProtocol
from qubx.core.utils import recognize_timeframe

if TYPE_CHECKING:
    from qubx.core.interfaces import IStrategyContext


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
    start_time_finder: Optional[StartTimeFinderProtocol] = None
    mismatch_resolver: Optional[StateResolverProtocol] = None
    auto_subscribe: Optional[bool] = None
    simulation: Optional[bool] = None
    subscription_warmup: Optional[dict[Any, str]] = None
    data_cache_config: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "prefetch_period": "1w", "cache_size_mb": 1000}
    )

    # Stale data detection configuration
    stale_data_detection_enabled: bool = False
    stale_data_detection_period: Optional[str] = None
    stale_data_check_interval: Optional[str] = None

    # Additional configuration that might be needed
    config: Dict[str, Any] = field(default_factory=dict)

    _pending_global_subscriptions: set[str] = field(default_factory=set)
    _pending_instrument_subscriptions: dict[str, set[Instrument]] = field(default_factory=dict)
    _custom_schedules: dict[str, tuple[str, Callable[["IStrategyContext"], None]]] = field(default_factory=dict)

    def set_base_subscription(self, subscription_type: str) -> None:
        self.base_subscription = subscription_type

    def get_base_subscription(self) -> str | None:
        return self.base_subscription

    def set_auto_subscribe(self, value: bool) -> None:
        self.auto_subscribe = value

    def get_auto_subscribe(self) -> bool | None:
        return self.auto_subscribe

    def set_fit_schedule(self, schedule: str) -> None:
        self.fit_schedule = schedule

    def get_fit_schedule(self) -> str | None:
        return self.fit_schedule

    def set_event_schedule(self, schedule: str) -> None:
        self.event_schedule = schedule

    def get_event_schedule(self) -> str | None:
        return self.event_schedule

    def set_warmup(self, period: str, start_time_finder: StartTimeFinderProtocol | None = None) -> None:
        self.warmup_period = period
        self.start_time_finder = start_time_finder

    def get_warmup(self) -> td_64 | None:
        return td_64(recognize_timeframe(self.warmup_period), "ns") if self.warmup_period else None

    def set_start_time_finder(self, finder: StartTimeFinderProtocol) -> None:
        self.start_time_finder = finder

    def get_start_time_finder(self) -> StartTimeFinderProtocol | None:
        return self.start_time_finder

    def get_state_resolver(self) -> StateResolverProtocol | None:
        return self.mismatch_resolver

    def set_state_resolver(self, resolver: StateResolverProtocol) -> None:
        self.mismatch_resolver = resolver

    def set_config(self, key: str, value: Any) -> None:
        self.config[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    @property
    def is_simulation(self) -> bool | None:
        return self.simulation

    def subscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None) -> None:
        if instruments is None:
            self._pending_global_subscriptions.add(subscription_type)
            return

        if isinstance(instruments, Instrument):
            instruments = [instruments]

        self._pending_instrument_subscriptions[subscription_type].update(instruments)

    def get_pending_global_subscriptions(self) -> set[str]:
        return self._pending_global_subscriptions

    def get_pending_instrument_subscriptions(self) -> dict[str, set[Instrument]]:
        return self._pending_instrument_subscriptions

    def set_subscription_warmup(self, configs: dict[Any, str]) -> None:
        self.subscription_warmup = configs

    def get_subscription_warmup(self) -> dict[Any, str]:
        return self.subscription_warmup if self.subscription_warmup else {}

    def set_data_cache_config(
        self, enabled: bool = True, prefetch_period: str = "1w", cache_size_mb: int = 1000
    ) -> None:
        """
        Configure CachedPrefetchReader for aux data readers.

        Args:
            enabled: Whether to enable data caching
            prefetch_period: Period to prefetch ahead (e.g., "1w", "2d")
            cache_size_mb: Maximum cache size in MB
        """
        self.data_cache_config = {
            "enabled": enabled,
            "prefetch_period": prefetch_period,
            "cache_size_mb": cache_size_mb,
        }

    def get_data_cache_config(self) -> Dict[str, Any]:
        """
        Get CachedPrefetchReader configuration.

        Returns:
            Dictionary with cache configuration
        """
        return self.data_cache_config

    def schedule(self, cron_schedule: str, method: Callable[["IStrategyContext"], None]) -> None:
        """
        Schedule a custom method to be called at specified times.

        Args:
            cron_schedule: Cron-like schedule string (e.g., "0 0 * * *" for daily at midnight)
            method: Method to call - should accept IStrategyContext as parameter
        """
        schedule_id = str(uuid.uuid4())
        self._custom_schedules[schedule_id] = (cron_schedule, method)

    def get_custom_schedules(self) -> dict[str, tuple[str, Callable[["IStrategyContext"], None]]]:
        """
        Get all custom scheduled methods.

        Returns:
            Dictionary mapping schedule IDs to (cron_schedule, method) tuples
        """
        return self._custom_schedules.copy()

    def set_stale_data_detection(self, enabled: bool, detection_period: str = "1d", check_interval: str = "1d") -> None:
        """
        Configure stale data detection settings.

        Args:
            enabled: Whether to enable stale data detection
            detection_period: Period to consider data as stale (e.g., "5Min", "1h"). If None, uses default.
            check_interval: Interval between stale data checks (e.g., "30s", "1Min"). If None, uses default.
        """
        self.stale_data_detection_enabled = enabled
        self.stale_data_detection_period = detection_period
        self.stale_data_check_interval = check_interval

    def get_stale_data_detection_config(self) -> tuple[bool, str | None, str | None]:
        """
        Get current stale data detection configuration.

        Returns:
            tuple: (enabled, detection_period, check_interval)
        """
        return (self.stale_data_detection_enabled, self.stale_data_detection_period, self.stale_data_check_interval)
