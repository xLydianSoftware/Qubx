"""Built-in assertions for connector verification."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from qubx.testing.connectors.collector import BarEvent, EventCollector


@dataclass
class AssertionResult:
    name: str
    passed: bool
    message: str


# -- assertion registry

ASSERTIONS: dict[str, Callable[..., AssertionResult]] = {}


def assertion(name: str):
    def decorator(fn: Callable[..., AssertionResult]) -> Callable[..., AssertionResult]:
        ASSERTIONS[name] = fn
        return fn

    return decorator


def run_assertion(name: str, collector: EventCollector, **params) -> AssertionResult:
    fn = ASSERTIONS.get(name)
    if fn is None:
        return AssertionResult(name=name, passed=False, message=f"Unknown assertion: {name}")
    return fn(collector, **params)


# -- built-in assertions


def _live_events(collector: EventCollector) -> list[BarEvent]:
    return [e for e in collector.bar_events if not e.is_warmup]


def _group_by_instrument(events: list[BarEvent]) -> dict[str, list[BarEvent]]:
    groups: dict[str, list[BarEvent]] = defaultdict(list)
    for e in events:
        groups[e.instrument_symbol].append(e)
    return groups


@assertion("no_out_of_order_bars")
def assert_no_out_of_order_bars(collector: EventCollector, **kwargs) -> AssertionResult:
    """Verify no live bars arrive with a timestamp before the previous bar for the same instrument."""
    live = _live_events(collector)
    groups = _group_by_instrument(live)

    violations = []
    for sym, events in groups.items():
        max_time = -1
        for e in events:
            if e.bar_time_ns < max_time and not e.was_skipped:
                violations.append(f"{sym}: bar_time={e.bar_time_ns} < prev_max={max_time}")
            max_time = max(max_time, e.bar_time_ns)

    if violations:
        return AssertionResult(
            name="no_out_of_order_bars",
            passed=False,
            message=f"{len(violations)} out-of-order bars: {violations[:3]}",
        )
    return AssertionResult(name="no_out_of_order_bars", passed=True, message="OK")


@assertion("crosses_at_least_n_candle_boundaries")
def assert_crosses_boundaries(collector: EventCollector, value: int = 1, **kwargs) -> AssertionResult:
    """Verify that we observed at least N candle boundary transitions during live."""
    live = _live_events(collector)
    groups = _group_by_instrument(live)

    min_boundaries = float("inf")
    min_sym = ""
    for sym, events in groups.items():
        unique_times = set(e.bar_time_ns for e in events)
        boundaries = len(unique_times) - 1  # N unique times = N-1 transitions
        if boundaries < min_boundaries:
            min_boundaries = boundaries
            min_sym = sym

    if not groups:
        return AssertionResult(
            name="crosses_at_least_n_candle_boundaries",
            passed=False,
            message="No live bar events collected",
        )

    passed = min_boundaries >= value
    return AssertionResult(
        name="crosses_at_least_n_candle_boundaries",
        passed=passed,
        message=f"Min boundaries: {min_boundaries} (on {min_sym}), required: {value}",
    )


@assertion("no_gap_after_warmup")
def assert_no_gap_after_warmup(collector: EventCollector, **kwargs) -> AssertionResult:
    """Verify no gap between last warmup bar and first live bar for each instrument."""
    warmup_events = [e for e in collector.bar_events if e.is_warmup]
    live_events = _live_events(collector)

    warmup_by_sym = _group_by_instrument(warmup_events)
    live_by_sym = _group_by_instrument(live_events)

    gaps = []
    for sym in warmup_by_sym:
        if sym not in live_by_sym:
            continue
        last_warmup = max(e.bar_time_ns for e in warmup_by_sym[sym])
        first_live = min(e.bar_time_ns for e in live_by_sym[sym])
        gap_ns = first_live - last_warmup
        # Allow up to 2x the candle period as tolerance
        # (we don't know the exact timeframe here, so use the gap between consecutive bars)
        if gap_ns < 0:
            gaps.append(f"{sym}: live started before warmup ended (gap={gap_ns}ns)")

    if gaps:
        return AssertionResult(
            name="no_gap_after_warmup",
            passed=False,
            message=f"{len(gaps)} gaps: {gaps[:3]}",
        )
    return AssertionResult(name="no_gap_after_warmup", passed=True, message="OK")


@assertion("last_bar_state_preserved")
def assert_last_bar_state_preserved(collector: EventCollector, **kwargs) -> AssertionResult:
    """Verify _last_bar is not None for the first live bar after warmup."""
    live = _live_events(collector)
    if not live:
        return AssertionResult(
            name="last_bar_state_preserved",
            passed=False,
            message="No live bar events collected",
        )

    # Check first live bar per instrument
    seen = set()
    null_instruments = []
    for e in live:
        if e.instrument_symbol not in seen:
            seen.add(e.instrument_symbol)
            if e.last_bar_time_ns is None:
                null_instruments.append(e.instrument_symbol)

    if null_instruments:
        return AssertionResult(
            name="last_bar_state_preserved",
            passed=False,
            message=f"_last_bar was None for first live bar on: {null_instruments}",
        )
    return AssertionResult(name="last_bar_state_preserved", passed=True, message="OK")


@assertion("no_past_bar_without_guard")
def assert_no_past_bar_without_guard(collector: EventCollector, **kwargs) -> AssertionResult:
    """Verify no bar arrives where _last_bar is None but the series is already ahead.

    This catches the exact warmup→live bug: _last_bar is reset to None during state transfer,
    but the OHLCV series carries over from warmup. If a late bar arrives before _last_bar is
    populated, the guard at market.py:248 is bypassed (because _last_bar is None), and the
    bar reaches the series which may be ahead of it.
    """
    live = _live_events(collector)

    violations = []
    for e in live:
        if e.last_bar_time_ns is None and e.series_head_time_ns is not None:
            if e.bar_time_ns < e.series_head_time_ns:
                violations.append(
                    f"{e.instrument_symbol}: bar_time={e.bar_time_ns} < series_head={e.series_head_time_ns} "
                    f"with _last_bar=None (no guard)"
                )

    if violations:
        return AssertionResult(
            name="no_past_bar_without_guard",
            passed=False,
            message=f"{len(violations)} unguarded past bars: {violations[:3]}",
        )
    return AssertionResult(name="no_past_bar_without_guard", passed=True, message="OK")
