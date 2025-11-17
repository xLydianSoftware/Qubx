import pandas as pd

import qubx.ta.indicators as ta
from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import DataType, Instrument, MarketEvent, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, IStrategyInitializer
from qubx.core.series import OHLCV, Indicator
from qubx.data import loader
from qubx.data.registry import StorageRegistry
from qubx.pandaz.utils import shift_series


class Test0(IStrategy):
    cusum_threshold: float = 0.3
    cusum_timeframe = "1d"
    cusum_period = 30
    timeframe = "1h"

    _cs_filters: dict[Instrument, Indicator] = {}
    _vol_data: dict[Instrument, Indicator] = {}

    def create_cusum_filter(
        self,
        ctx: IStrategyContext,
        instrument: Instrument,
        timeframe: str,
        cusum_threshold: float,
        cusum_timeframe: str,
        cusum_period: int,
    ):
        # - Calculate minimum bars needed for StdEma to stabilize
        # - cusum_period = 30 days, so need ~90 days = 2160 hours minimum
        _bars_per_period = pd.Timedelta(cusum_timeframe) // pd.Timedelta(timeframe)
        _min_bars = self.cusum_period * 3 * _bars_per_period
        _base_data = ctx.ohlc(instrument, timeframe, _min_bars)

        _volt_data = _base_data.resample(self.cusum_timeframe)

        vol = ta.stdema(ta.pct_change(_volt_data.close), cusum_period)
        return ta.cusum_filter(_base_data.close, vol * cusum_threshold), vol

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent) -> list[Signal] | Signal | None:
        # - data (it may be unclosed bar) that was used for updating internal ctx series
        bar_data = data.data

        # - DEBUG: Log market data around 2022-01-01 to understand what simulator feeds
        if hasattr(bar_data, "time"):
            bar_time = pd.Timestamp(bar_data.time)
            if bar_time >= pd.Timestamp("2022-01-01 00:00") and bar_time <= pd.Timestamp("2022-01-01 02:00"):
                logger.info(f"[on_market_data] {bar_time} -> open={bar_data.open:.2f}, close={bar_data.close:.2f}")

                # - Check what ctx.ohlc() returns after this update
                _base_data = ctx.ohlc(ctx.instruments[0], self.timeframe, 3)
                if len(_base_data) >= 3:
                    logger.info(f"  ctx.ohlc(3) last 3 bars:")
                    for i in range(min(3, len(_base_data))):
                        bar = _base_data[i]
                        logger.info(f"    [{i}] {pd.Timestamp(bar.time)} close={bar.close:.2f}")

    def on_init(self, initializer: IStrategyInitializer) -> None:
        initializer.set_base_subscription(DataType.OHLC[self.timeframe])
        initializer.set_event_schedule(self.timeframe + " -1s")

    def on_start(self, ctx: IStrategyContext):
        self._cs_filters = {}
        self._vol_data = {}
        _base_frame = self.timeframe
        for i in ctx.instruments:
            cf, vl = self.create_cusum_filter(
                ctx, i, _base_frame, self.cusum_threshold, self.cusum_timeframe, self.cusum_period
            )
            self._cs_filters[i] = cf
            self._vol_data[i] = vl

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        for i in ctx.instruments:
            ctx.emitter.emit("cusum", self._cs_filters[i][0], instrument=i, tags={"type": "filter"})
            # - emit the LAST value of volatility series (most recent)
            ctx.emitter.emit("volatility", self._vol_data[i][0], instrument=i, tags={"type": "volatility"})


def test_cusum_streaming_vs_static():
    """Test that replicates test_cusum_filter logic with strategy's data"""
    # - Load same data as strategy will use
    reader = StorageRegistry.get("csv::tests/data/storages/csv_longer")["BINANCE.UM", "SWAP"]
    raw = reader.read("ETHUSDT", "ohlc(1h)", "2021-12-01", "2022-01-11")
    T = slice("2022-01-01", "2022-01-10")

    # - Non-streaming: calculate on data all at once
    ohlc = raw.to_ohlc()
    _volt_data = ohlc.resample("1d")
    vol = ta.stdema(ta.pct_change(_volt_data.close), 30)
    ns_csf = ta.cusum_filter(ohlc.close, vol * 0.3).pd()[T]

    # - Streaming: feed bars one by one (like test_cusum_filter does)
    H1 = OHLCV("s1", "1h")
    D1 = H1.resample("1d")
    vol1 = ta.stdema(ta.pct_change(D1.close), 30)
    s_csf = ta.cusum_filter(H1.close, vol1 * 0.3)

    # - Feed bars in order
    bars = ohlc.pd()
    for idx in bars.index:
        bar = bars.loc[idx]
        H1.update_by_bar(int(idx.value), bar["open"], bar["high"], bar["low"], bar["close"], bar.get("volume", 0))

    stream_cs = s_csf.pd()[T]

    # - Compare volatility
    vol_diff = (vol.pd()[T] - vol1.pd()[T]).dropna()
    print(f"\n=== VOLATILITY COMPARISON ===")
    print(f"Max vol diff: {vol_diff.abs().max():.15f}")

    # - Compare CUSUM events
    ns_events = ns_csf[ns_csf == 1]
    s_events = stream_cs[stream_cs == 1]
    print(f"\n=== CUSUM EVENTS COMPARISON ===")
    print(f"Non-streaming events: {len(ns_events)}, Streaming events: {len(s_events)}")

    # - This MUST be 100% match like test_cusum_filter
    assert all(stream_cs[stream_cs == 1] == ns_csf[ns_csf == 1]), (
        f"Streaming and non-streaming CUSUM must match exactly!"
    )
    print(f"✅ Perfect match! Streaming and non-streaming produce identical results.")


def test_cusum_with_preloaded_data():
    """Test that replicates EXACTLY what strategy does: preload data, attach indicators, then stream"""
    reader = StorageRegistry.get("csv::tests/data/storages/csv_longer")["BINANCE.UM", "SWAP"]

    # - Split data to EXACTLY replicate simulator behavior:
    # - Historical: Load 2160 bars which will include simulation start (2022-01-01 00:00)
    # - New: Stream from 2022-01-01 00:00, so there's 1 bar overlap
    # - This matches what ctx.ohlc() does: returns bars UP TO AND INCLUDING current time
    historical = reader.read("ETHUSDT", "ohlc(1h)", "2021-10-03 01:00", "2022-01-01")
    # - Stream all bars including the overlapping one
    new_data = reader.read("ETHUSDT", "ohlc(1h)", "2022-01-01", "2022-01-11")

    print(f"\n=== DATA RANGES ===")
    hist_ohlc_check = historical.to_ohlc()
    new_ohlc_check = new_data.to_ohlc()
    print(f"Historical: {len(hist_ohlc_check)} bars")
    print(f"  First: {pd.Timestamp(hist_ohlc_check[-1].time)} (oldest)")
    print(f"  Last: {pd.Timestamp(hist_ohlc_check[0].time)} (newest)")
    print(f"New data: {len(new_ohlc_check)} bars")
    print(f"  First: {pd.Timestamp(new_ohlc_check[-1].time)} (oldest)")
    print(f"  Last: {pd.Timestamp(new_ohlc_check[0].time)} (newest)")
    T = slice("2022-01-01", "2022-01-10")

    # - Approach 1: Pre-load historical data, THEN attach indicators (like strategy does)
    print(f"\n=== APPROACH 1: Pre-load data, then attach indicators ===")

    # - Create OHLCV series and feed historical bars
    H1_preloaded = OHLCV("preloaded", "1h")
    hist_ohlc = historical.to_ohlc()
    hist_bars = hist_ohlc.pd()
    print(f"Pre-loading {len(hist_ohlc)} historical bars")
    print(f"  First bar to feed: {hist_bars.index[0]}")
    print(f"  Last bar to feed: {hist_bars.index[-1]}")
    for idx in hist_bars.index:
        bar = hist_bars.loc[idx]
        H1_preloaded.update_by_bar(
            int(idx.value), bar["open"], bar["high"], bar["low"], bar["close"], bar.get("volume", 0)
        )

    # - NOW attach indicators to series with existing data (this is what strategy does!)
    print(f"Attaching indicators to series with {len(H1_preloaded)} existing bars")
    print(f"First bar: {pd.Timestamp(H1_preloaded[0].time)}, Last bar: {pd.Timestamp(H1_preloaded[-1].time)}")
    D1_preloaded = H1_preloaded.resample("1d")
    print(f"Daily resampled series has {len(D1_preloaded)} bars")
    vol_preloaded = ta.stdema(ta.pct_change(D1_preloaded.close), 30)
    print(f"After attaching stdema, volatility has {len(vol_preloaded)} bars")
    csf_preloaded = ta.cusum_filter(H1_preloaded.close, vol_preloaded * 0.3)
    print(f"After attaching cusum, filter has {len(csf_preloaded)} bars")

    # - Now feed new bars (simulating strategy streaming during simulation)
    # - IMPORTANT: Skip the first bar (2022-01-01 00:00) which is already in historical data
    new_bars = new_data.to_ohlc()
    bars_pd = new_bars.pd()
    print(f"Streaming {len(new_bars)} new bars (skipping first overlapping bar)")
    for i, idx in enumerate(bars_pd.index):
        if i == 0:  # Skip first bar (overlapping)
            print(f"  Skipping first bar: {idx} (already in historical data)")
            continue
        bar = bars_pd.loc[idx]
        H1_preloaded.update_by_bar(
            int(idx.value), bar["open"], bar["high"], bar["low"], bar["close"], bar.get("volume", 0)
        )

    result_preloaded = csf_preloaded.pd()[T]
    events_preloaded = result_preloaded[result_preloaded == 1]
    print(f"Events from preloaded approach: {len(events_preloaded)}")

    # - Approach 2: Feed ALL bars incrementally (proven to work)
    print("\n=== APPROACH 2: Feed all bars incrementally ===")
    all_data = reader.read("ETHUSDT", "ohlc(1h)", "2021-10-03 01:00", "2022-01-11")
    all_ohlc = all_data.to_ohlc()

    H1_incremental = OHLCV("incremental", "1h")
    D1_incremental = H1_incremental.resample("1d")
    vol_incremental = ta.stdema(ta.pct_change(D1_incremental.close), 30)
    csf_incremental = ta.cusum_filter(H1_incremental.close, vol_incremental * 0.3)

    all_bars = all_ohlc.pd()
    print(f"Feeding all {len(all_ohlc)} bars incrementally")
    for idx in all_bars.index:
        bar = all_bars.loc[idx]
        H1_incremental.update_by_bar(
            int(idx.value), bar["open"], bar["high"], bar["low"], bar["close"], bar.get("volume", 0)
        )

    result_incremental = csf_incremental.pd()[T]
    events_incremental = result_incremental[result_incremental == 1]
    print(f"Events from incremental approach: {len(events_incremental)}")

    # - Compare
    print("\n=== COMPARISON ===")
    vol_diff = (vol_preloaded.pd()[T] - vol_incremental.pd()[T]).dropna()
    print(f"Volatility max diff: {vol_diff.abs().max():.15f}")

    preloaded_times = set(events_preloaded.index)
    incremental_times = set(events_incremental.index)
    common = preloaded_times & incremental_times
    overlap_pct = len(common) / max(len(preloaded_times), len(incremental_times)) * 100

    print(f"Event overlap: {len(common)}/{max(len(preloaded_times), len(incremental_times))} ({overlap_pct:.1f}%)")

    if overlap_pct < 100:
        missing_in_preloaded = incremental_times - preloaded_times
        extra_in_preloaded = preloaded_times - incremental_times
        print(f"\nMissing in preloaded: {len(missing_in_preloaded)}")
        print(f"Extra in preloaded: {len(extra_in_preloaded)}")

        if missing_in_preloaded:
            print(f"\nFirst 5 missing events:")
            for t in sorted(missing_in_preloaded)[:5]:
                print(f"  {t}")
                # - Check CUSUM values at this time
                preloaded_val = result_preloaded.get(t, None)
                incremental_val = result_incremental.get(t, None)
                print(f"    Preloaded: {preloaded_val}, Incremental: {incremental_val}")

        if extra_in_preloaded:
            print(f"\nFirst 5 extra events:")
            for t in sorted(extra_in_preloaded)[:5]:
                print(f"  {t}")

        # - Check CUSUM values around the boundary (2022-01-01)
        print(f"\n=== CUSUM VALUES AROUND BOUNDARY (2022-01-01) ===")
        boundary_slice = slice("2021-12-31 22:00", "2022-01-01 04:00")
        print(f"\nPreloaded CUSUM:")
        print(csf_preloaded.pd()[boundary_slice])
        print(f"\nIncremental CUSUM:")
        print(csf_incremental.pd()[boundary_slice])

    # - MUST be 100% match!
    assert overlap_pct == 100, f"Preloaded vs incremental must match 100%! Got {overlap_pct:.1f}%"
    print("✅ Perfect match! Pre-loading data works correctly.")


def test_cusum_in_strategy():
    # - Load from CSV
    ldr = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_longer_1h")

    # - Run strategy in simulator
    r = simulate(
        {"test0": Test0(timeframe="1h", cusum_threshold=0.3)},
        data={"ohlc(1h)": ldr},
        capital=10000,
        enable_inmemory_emitter=True,
        instruments=["BINANCE.UM:ETHUSDT"],
        debug="INFO",
        start="2022-01-01",
        stop="2022-01-11",
        commissions="vip0_usdt",
    )

    # - get data from strategy emmitter
    dT = slice("2022-01-01", "2022-01-10")
    emd = r[0].emitter_data
    cs_strat = emd[emd["type"] == "filter"].set_index("timestamp").value[dT]
    vol_strat = emd[emd["type"] == "volatility"].set_index("timestamp").value[dT]

    cs_strat_events = cs_strat[cs_strat == 1]
    cs_strat_events.index = cs_strat_events.index.round("1min")

    # - calculate on static data (all at once, not streaming)
    # - Load enough historical data (90 days before test period) for volatility calculation
    reader = StorageRegistry.get("csv::tests/data/storages/csv_longer")["BINANCE.UM", "SWAP"]
    D0 = reader.read("ETHUSDT", "ohlc(1h)", "2021-10-03 01:00", "2022-01-11")
    ohlc = D0.to_ohlc()

    T = slice("2022-01-01", "2022-01-10")
    _volt_data = ohlc.resample("1d")
    vol = ta.stdema(ta.pct_change(_volt_data.close), 30)
    cs_static = ta.cusum_filter(ohlc.close, vol * 0.3).pd()[T]
    cs_static = shift_series(cs_static, "1h")
    cs_static_events = cs_static[cs_static == 1]

    # - compare daily volatility values (take last value of each day)
    vol_strat_daily = vol_strat.resample("1D").last()
    vol_static_daily = vol.pd()[T]
    diff = (vol_strat_daily - vol_static_daily).dropna()

    assert (diff.abs() < 1e-10).all(), "Volatility values do not match exactly !!"
    assert len(cs_strat_events) == len(cs_static_events), "Strategy events doesn't match Static events"

    # - find events that are in static but not in strategy
    strat_times = set(cs_strat_events.index)
    static_times = set(cs_static_events.index)

    # - Check for exact match
    common_events = strat_times & static_times
    overlap_pct = len(common_events) / max(len(strat_times), len(static_times)) * 100
    print(f"Event overlap: {len(common_events)}/{max(len(strat_times), len(static_times))} ({overlap_pct:.1f}%)")

    assert overlap_pct >= 100, f"Overlap too low: {overlap_pct:.1f}%"
