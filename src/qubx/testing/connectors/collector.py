"""Monkeypatch instrumentation and event collection for connector verification."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from qubx.core.basics import Bar, Instrument

if TYPE_CHECKING:
    from qubx.core.interfaces import IStrategyContext
    from qubx.core.mixins.market import CachedMarketDataHolder


@dataclass
class BarEvent:
    """A single bar update captured at CachedMarketDataHolder.update_by_bar."""

    wall_time: float
    instrument_symbol: str
    bar_time_ns: int
    bar_close: float
    bar_open: float
    bar_high: float
    bar_low: float
    bar_volume: float
    last_bar_time_ns: int | None  # _last_bar.time before this call, None if no _last_bar
    series_head_time_ns: int | None  # OHLCV.times[0] before this call
    was_skipped: bool  # True if the guard at market.py:248 skipped this bar
    is_warmup: bool


@dataclass
class LifecycleEvent:
    """A lifecycle event (warmup finished, strategy started, etc.)."""

    wall_time: float
    event: str


@dataclass
class EventCollector:
    """Collects bar events and lifecycle events via monkeypatching."""

    bar_events: list[BarEvent] = field(default_factory=list)
    lifecycle_events: list[LifecycleEvent] = field(default_factory=list)
    _original_update_by_bar: Any = field(default=None, repr=False)
    _cache_ref: CachedMarketDataHolder | None = field(default=None, repr=False)
    _ctx_ref: IStrategyContext | None = field(default=None, repr=False)

    def install(self, ctx: IStrategyContext) -> None:
        """Monkeypatch CachedMarketDataHolder.update_by_bar on the context's cache instance."""
        import types

        from qubx.core.mixins.market import CachedMarketDataHolder

        self._ctx_ref = ctx
        cache = cast(CachedMarketDataHolder, ctx.get_market_data_cache())
        self._cache_ref = cache
        self._original_update_by_bar = cache.update_by_bar

        # Bind our interceptor to the instance
        cache.update_by_bar = types.MethodType(self._make_interceptor(), cache)

    def _make_interceptor(self):
        """Create the patched update_by_bar method."""
        collector = self
        original = self._original_update_by_bar

        def intercepted_update_by_bar(cache_self, instrument: Instrument, bar: Bar):
            # Capture state before the call
            _last_bar = cache_self._last_bar[instrument]
            last_bar_time_ns = _last_bar.time if _last_bar is not None else None

            # Check series head
            series_head_time_ns = None
            if instrument in cache_self._ohlcvs:
                for ser in cache_self._ohlcvs[instrument].values():
                    if ser.times:
                        series_head_time_ns = ser.times[0]
                    break  # just check the first (default tf) series

            # Predict if the guard will skip this bar
            was_skipped = _last_bar is not None and _last_bar.time > bar.time

            # Detect phase
            is_warmup = (
                collector._ctx_ref is not None
                and hasattr(collector._ctx_ref, "_strategy_state")
                and collector._ctx_ref._strategy_state.is_warmup_in_progress
            )

            collector.bar_events.append(
                BarEvent(
                    wall_time=time.monotonic(),
                    instrument_symbol=instrument.symbol,
                    bar_time_ns=bar.time,
                    bar_close=bar.close,
                    bar_open=bar.open,
                    bar_high=bar.high,
                    bar_low=bar.low,
                    bar_volume=bar.volume,
                    last_bar_time_ns=last_bar_time_ns,
                    series_head_time_ns=series_head_time_ns,
                    was_skipped=was_skipped,
                    is_warmup=is_warmup,
                )
            )

            # Call original
            return original(instrument, bar)

        return intercepted_update_by_bar

    def uninstall(self) -> None:
        """Restore the original update_by_bar method."""
        if self._cache_ref is not None and self._original_update_by_bar is not None:
            self._cache_ref.update_by_bar = self._original_update_by_bar
            self._cache_ref = None
            self._original_update_by_bar = None

    def record_lifecycle(self, event: str) -> None:
        self.lifecycle_events.append(LifecycleEvent(wall_time=time.monotonic(), event=event))
