"""
Regression guard for BundledSeries timestamp alignment.

These tests cover two bugs fixed by the `align_timeframe` parameter and the
rewrite of `_initial_data_recalculate_bundled()` from inner-join+dropna to
floor+outer-join+ffill:

  ISSUE.BS  — sources carry the same logical bar but stamp it at different
              offsets within the bar (e.g. OHLC open-stamped at 14:00:00 and
              vendor data close-stamped at 14:14:59 for a 15-min timeframe).
              The old inner-join on raw timestamps produced zero matching rows
              → empty bundle → downstream indicators saw nothing.

  ISSUE.BS2 — sources have different cadences (e.g. 15m + 1h). The old
              inner-join dropped every 15m-only row; bundle effectively ran at
              the slow cadence. Correct behaviour: emit at the fast cadence
              with the slow source forward-filled.
"""

import numpy as np
import pandas as pd

from qubx.core.series import BundledSeries, TimeSeries

# - short internal timeframe so we can place source ticks at arbitrary offsets
# - within the logical bar; the bundle's align_timeframe controls binning
_FINE_TF = "1s"

# - base timestamp for every scenario (midnight-aligned so 15m / 1h bins are clean)
_BASE_NS = np.datetime64("2024-01-01T14:00:00", "ns").astype("int64")
_S1 = 1_000_000_000
_TF_15M_NS = 15 * 60 * _S1
_TF_1H_NS = 60 * 60 * _S1


def test_bundle_aligns_ohlc_with_close_shifted_vendor_on_15m_bins():
    """
    Simulates the xmetals prod scenario: OHLC open-stamped at bar-start
    vs vendor data close-stamped at bar-end-1s within the same 15m bin.
    Before the fix: inner-join on raw stamps produced zero rows.
    After the fix: both floor to the same 15m bin via align_timeframe default.
    """
    ts_close = TimeSeries("close", _FINE_TF)
    ts_oi = TimeSeries("oi", _FINE_TF)

    # - three 15m bars worth of data, with OI offset inside each bar
    for i in range(3):
        bar_start = _BASE_NS + i * _TF_15M_NS
        bar_end_minus_1s = bar_start + _TF_15M_NS - _S1
        ts_close.update(bar_start, 100.0 + i)  # 14:00, 14:15, 14:30
        ts_oi.update(bar_end_minus_1s, 1000.0 + i * 10)  # 14:14:59, 14:29:59, 14:44:59

    # - align_timeframe defaults to timeframe ("15m") → both sources floor identically
    bundle = BundledSeries("b", "15m", {"close": ts_close, "oi": ts_oi})

    df = bundle.to_series()
    assert len(df) == 3, f"expected 3 aligned rows, got {len(df)} — bundle is stale/empty"

    # - each row should have BOTH fields from the same logical bar
    rows = [df.iloc[i].to_dict() for i in range(len(df))]
    assert rows[0] == {"close": 100.0, "oi": 1000.0}
    assert rows[1] == {"close": 101.0, "oi": 1010.0}
    assert rows[2] == {"close": 102.0, "oi": 1020.0}

    # - row timestamps should be the floored bin starts
    expected_ts = [pd.Timestamp(_BASE_NS + i * _TF_15M_NS) for i in range(3)]
    assert list(df.index) == expected_ts


def test_bundle_mixed_timeframe_forward_fills_slow_source():
    """
    Mixing 15m + 1h sources must emit at 15m cadence with the 1h source ffilled.
    Before the fix: inner-join on raw timestamps kept only rows where both had
    a tick → 2 hourly rows.
    After the fix: align_timeframe=15m floors both, outer-join keeps all 15m
    rows, ffill carries the 1h value forward across the 4 in-bin bars.
    """
    ts_close = TimeSeries("close", _FINE_TF)
    ts_oi = TimeSeries("oi", _FINE_TF)

    # - 15m close over 2h (8 bars)
    for i in range(8):
        ts_close.update(_BASE_NS + i * _TF_15M_NS, 100.0 + i)

    # - 1h OI at bar-open (2 bars)
    ts_oi.update(_BASE_NS, 1000.0)  # 14:00 → 14:00-15:00 bar
    ts_oi.update(_BASE_NS + _TF_1H_NS, 1010.0)  # 15:00 → 15:00-16:00 bar

    bundle = BundledSeries("b", "15m", {"close": ts_close, "oi": ts_oi})

    df = bundle.to_series()
    assert len(df) == 8, f"expected 8 rows at 15m cadence, got {len(df)}"

    # - close values strictly increasing across 15m bins
    close_vals = df["close"].tolist()
    assert close_vals == [100.0 + i for i in range(8)]

    # - OI ffilled: first 4 bars = 1000 (from 14:00), next 4 = 1010 (from 15:00)
    oi_vals = df["oi"].tolist()
    assert oi_vals == [1000.0, 1000.0, 1000.0, 1000.0, 1010.0, 1010.0, 1010.0, 1010.0]


def test_bundle_explicit_align_timeframe_overrides_bundle_timeframe():
    """
    `align_timeframe` is independent of the bundle's own timeframe: it controls
    only the binning used to align source timestamps during initial recalc.
    Verifies the new parameter is plumbed correctly.
    """
    ts_a = TimeSeries("a", _FINE_TF)
    ts_b = TimeSeries("b", _FINE_TF)

    # - two logical 1h bars, source A stamped at bar-open, source B at bar-open+45min
    ts_a.update(_BASE_NS, 1.0)  # 14:00
    ts_b.update(_BASE_NS + 45 * 60 * _S1, 10.0)  # 14:45
    ts_a.update(_BASE_NS + _TF_1H_NS, 2.0)  # 15:00
    ts_b.update(_BASE_NS + _TF_1H_NS + 45 * 60 * _S1, 20.0)  # 15:45

    # - without align_timeframe="1h", default = "15m" would bin A@14:00 and B@14:45 into
    # - different 15m bins → only ffill would save it. With explicit "1h" align they
    # - floor to the same bin (14:00, 15:00) without needing ffill at all.
    bundle = BundledSeries("b", "15m", {"a": ts_a, "b": ts_b}, align_timeframe="1h")

    df = bundle.to_series()
    assert len(df) == 2
    assert df["a"].tolist() == [1.0, 2.0]
    assert df["b"].tolist() == [10.0, 20.0]
    # - row timestamps are the floored hour bin starts
    assert list(df.index) == [pd.Timestamp(_BASE_NS), pd.Timestamp(_BASE_NS + _TF_1H_NS)]
