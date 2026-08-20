"""
Warmup must not reach the metric store.

Warmup rebinds the live emitter to a simulated context whose clock sits in the warmup
window, so anything emitted there lands timestamped hours before it was written.
"""

from unittest.mock import MagicMock

import pandas as pd

from qubx.core.basics import dt_64
from qubx.core.interfaces import IStrategyContext
from qubx.emitters.inmemory import InMemoryMetricEmitter


def _context(is_simulation: bool, is_warmup: bool, time: str) -> MagicMock:
    ctx = MagicMock(spec=IStrategyContext)
    ctx.is_simulation = is_simulation
    ctx.is_live = not is_simulation
    ctx.is_warmup_in_progress = is_warmup
    ctx.is_live_or_warmup = (not is_simulation) or is_warmup
    ctx.time.return_value = dt_64(pd.Timestamp(time))
    return ctx


def test_emit_is_suppressed_during_warmup():
    em = InMemoryMetricEmitter()
    em.set_context(_context(is_simulation=True, is_warmup=True, time="2026-08-18 15:03:00"))

    em.emit("position_pnl", 1.0)

    assert em.get_dataframe().empty


def test_emit_resumes_once_warmup_finishes():
    em = InMemoryMetricEmitter()
    live = _context(is_simulation=False, is_warmup=False, time="2026-08-18 22:38:00")
    em.set_context(_context(is_simulation=True, is_warmup=True, time="2026-08-18 15:03:00"))

    em.emit("position_pnl", 1.0)
    em.set_context(live)
    em.emit("position_pnl", 2.0)

    df = em.get_dataframe()
    assert list(df["value"]) == [2.0]
    assert list(df["timestamp"]) == [pd.Timestamp("2026-08-18 22:38:00")]


def test_emit_still_works_in_a_plain_simulation():
    # - a backtest is is_simulation without warmup; InMemoryMetricEmitter is used for research there
    em = InMemoryMetricEmitter()
    em.set_context(_context(is_simulation=True, is_warmup=False, time="2021-06-01 00:00:00"))

    em.emit("position_pnl", 3.0)

    assert list(em.get_dataframe()["value"]) == [3.0]


def test_strategy_stats_are_suppressed_during_warmup():
    em = InMemoryMetricEmitter(stats_interval="1Sec")
    ctx = _context(is_simulation=True, is_warmup=True, time="2026-08-18 15:03:00")
    em.notify(ctx)
    ctx.time.return_value = dt_64(pd.Timestamp("2026-08-18 15:04:00"))

    em.notify(ctx)

    assert em.get_dataframe().empty
