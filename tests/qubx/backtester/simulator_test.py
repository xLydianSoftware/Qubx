import inspect
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
import pandas as pd

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import DataType, Instrument, MarketEvent, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, IStrategyInitializer
from qubx.data.cache import CachedReader
from qubx.data.registry import StorageRegistry
from qubx.data.storage import IReader, IStorage, Transformable
from qubx.data.storages.handy import HandyReader, HandyStorage
from qubx.utils.runner.configs import PrefetchConfig


class _CountingStorage(IStorage):
    """
    Transparent IStorage wrapper that records every raw read() call.
    Wraps each IReader returned by get_reader() so all inner-layer reads are counted,
    regardless of how many CachedReader / TimeGuardedReader layers are stacked on top.
    """

    read_calls: list[tuple]  # - each entry: (data_id, dtype, start, stop)

    def __init__(self, inner: IStorage) -> None:
        self._inner = inner
        self.read_calls = []

    def get_exchanges(self) -> list[str]:
        return self._inner.get_exchanges()

    def get_market_types(self, exchange: str) -> list[str]:
        return self._inner.get_market_types(exchange)

    def get_reader(self, exchange: str, market: str) -> IReader:
        inner_reader = self._inner.get_reader(exchange, market)
        outer = self

        class _CountingReader(IReader):
            def read(
                self,
                data_id: str | list[str],
                dtype: DataType | str,
                start: str | None = None,
                stop: str | None = None,
                chunksize: int = 0,
                **kwargs,
            ) -> Iterator[Transformable] | Transformable:
                outer.read_calls.append((data_id, str(dtype), start, stop))

                resul = inner_reader.read(data_id, dtype, start, stop, chunksize=chunksize, **kwargs)

                # - the cache is updated by CachedReader._store_result() AFTER this read()
                # - returns — so dumping cache here would still miss the just-fetched data.
                # - fix: walk the stack to find the wrapping CachedReader and patch its
                # - _store_result with a one-shot wrapper that dumps cache AFTER the store.
                frame = inspect.currentframe().f_back
                try:
                    while frame is not None:
                        f_self = frame.f_locals.get("self")
                        if isinstance(f_self, CachedReader):
                            _cr = f_self
                            _call_n = len(outer.read_calls)
                            _req_ids = data_id
                            _orig_store = _cr._store_result  # - bound method

                            def _post_store_dump(
                                cache_key,
                                result,
                                start_s,
                                stop_s,
                                _cr=_cr,
                                _n=_call_n,
                                _ids=_req_ids,
                                _orig=_orig_store,
                            ):
                                _orig(cache_key, result, start_s, stop_s)  # - store first
                                _cr._store_result = _orig  # - restore immediately
                                cache = _cr._cache
                                stored = {ck: sorted(v.keys()) for ck, v in cache._data.items()}
                                ranges = {ck: v for ck, v in cache._ranges.items()}
                                logger.info(
                                    f"\n[_CountingReader] after inner read #{_n}"
                                    f"  ids={_ids!r}  start={start_s}  stop={stop_s}\n"
                                    f"  cache._data  = {stored}\n"
                                    f"  cache._ranges = {ranges}"
                                )

                            _cr._store_result = _post_store_dump
                            break
                        frame = frame.f_back
                finally:
                    del frame  # - avoid reference cycle

                return resul

            def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
                return inner_reader.get_data_id(dtype)

            def get_data_types(self, data_id: str) -> list[DataType]:
                return inner_reader.get_data_types(data_id)

            def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[np.datetime64, np.datetime64]:
                return inner_reader.get_time_range(data_id, dtype)

            def close(self) -> None:
                inner_reader.close()

        return _CountingReader()


class DataSubscriptionStrategy(IStrategy):
    base: str = "ohlc(1h)"
    additional: list[str] = []
    schedule: str = "1h"

    _n_features: int
    _n_events: int
    _data_hits: dict[str, int]

    def on_init(self, initializer: IStrategyInitializer):
        initializer.set_base_subscription(self.base)

        # - subscribe on some arbitrary type
        for sb in self.additional:
            initializer.subscribe(sb)

        if self.schedule:
            initializer.set_event_schedule(self.schedule)

        self._data_hits = defaultdict(lambda: 0)

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent) -> list[Signal] | Signal | None:
        self._data_hits[data.type] += 1

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | Signal | None:
        self._data_hits["event"] += 1


class TestSimulator:
    def testing_csv_storage(self, path="tests/data/storages/multi/"):
        return StorageRegistry.get(f"csv::{path}")

    def test_multi_subscriptions(self):
        simulate(
            (s := DataSubscriptionStrategy(base="ohlc_quotes(1h)", additional=["ohlc_trades(1h)"], schedule=None)),
            data=self.testing_csv_storage(),
            capital=1000,
            start="2026-01-01 00:00",
            stop="2026-01-01 05:00",
            instruments=["BINANCE.UM:SWAP:BTCUSDT"],
            debug="INFO",
            silent=True,
        )
        # - 4 quotes per bar (open, mid1, mid2, close) × 5 bars = 20 (stop is exclusive: timestamp < stop)
        assert s._data_hits["quote"] >= 16, f"Expected >= 16 quote events, got {s._data_hits['quote']}"

        # - 3 trades per bar (open, mid1, mid2) × 5 bars = 15 (stop is exclusive: timestamp < stop)
        assert s._data_hits["trade"] >= 12, f"Expected >= 12 trade events, got {s._data_hits['trade']}"

    def test_external_subscription(self):
        stor = self.testing_csv_storage()

        rr = stor.get_reader("BINANCE.UM", "SWAP")
        assert "features" in rr.get_data_types("BTCUSDT")

        simulate(
            (s := DataSubscriptionStrategy(base="ohlc(1h)", additional=["features"], schedule="1h")),
            data=stor,
            capital=1000,
            start="2026-01-01 00:00",
            stop="2026-01-01 05:00",
            instruments=["BINANCE.UM:SWAP:BTCUSDT"],
            debug="INFO",
            silent=True,
        )

        # - scheduled events every 1h over 5h window
        assert s._data_hits["event"] >= 4, f"Expected >= 4 scheduled events, got {s._data_hits['event']}"

        # - features data arrives at each hour (01:00-04:00), stop at 05:00 is exclusive
        assert s._data_hits["features"] == 4, f"Expected 4 features events, got {s._data_hits['features']}"

    def test_ohlc_data(self):
        """
        Test simplest OHLC data retrieving
        """

        class _TestOhlc(IStrategy):
            _results: dict[Instrument, pd.DataFrame] = {}

            def on_init(self, initializer: IStrategyInitializer):
                initializer.set_event_schedule("1h -1s")  # - trigger every hour at xx:59:59
                initializer.set_base_subscription("ohlc(1h)")

            def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | Signal | None:
                for i in ctx.instruments:
                    self._results[i] = ctx.ohlc(i, "1h", 10).pd()
                    logger.info(
                        f" === Time: <r>{ctx.time()}</r> ===\n<g>"
                        + str(self._results[i][["open", "high", "low", "close", "volume"]])
                        + "</g>\n - - - - - - -"
                    )

        stor = self.testing_csv_storage()
        simulate(
            s := _TestOhlc(),
            data=stor,
            capital=1000,
            start="2026-01-01 00:00",
            stop="2026-01-01 01:00",
            instruments=[
                "BINANCE.UM:SWAP:BTCUSDT",
                "KRAKEN.F:SWAP:BTCUSD",
                "HYPERLIQUID:SWAP:BTCUSDC",
            ],
            debug="DEBUG",
        )

        for i, d in s._results.items():
            original = (
                stor.get_reader(i.exchange, i.market_type)
                .read(i.symbol, "ohlc(1h)", "2025-12-27 21:00", "2026-01-01 00:00")
                .to_pd()  # type: ignore
                .tail(11)
            )
            assert all(
                (d[["open", "high", "low", "close", "volume"]] - original[["open", "high", "low", "close", "volume"]])
                == 0
            )

    def test_aux_reader_from_data_storage(self):
        """
        When aux_data is not provided, ctx.get_aux_reader() should read from the main data storage.
        Sim window crosses midnight (Dec 31 22:59 → Jan 1 03:59) so daily fundamental data
        should grow as time advances past the day boundary. TimeGuard checked on every event.
        """

        class _AuxFromDataStorage(IStrategy):
            # - collect per-event: (sim_time, row_count, last_ts, correct_symbols)
            checks: list[tuple[str, int, pd.Timestamp, bool]]

            def on_init(self, initializer: IStrategyInitializer):
                initializer.set_base_subscription("ohlc(1h)")
                initializer.set_event_schedule("1h -1s")
                self.checks = []

            def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | Signal | None:
                sim_time = ctx.time()
                reader = ctx.get_aux_reader("COINGECKO", "FUNDAMENTAL")
                result = reader.read(DataType.ALL, "fundamental", "2020-01-01", "now").to_pd(False)

                last_ts = result.index.max() if len(result) > 0 else pd.Timestamp("NaT")
                correct_syms = set(result["symbol"].unique()) == {"BCH", "BTC", "ETH"} if len(result) > 0 else False
                self.checks.append((str(sim_time), len(result), last_ts, correct_syms))

        stor = self.testing_csv_storage()
        simulate(
            s := _AuxFromDataStorage(),
            data=stor,
            capital=1000,
            # - crosses midnight: events at 22:59, 23:59 (Jun 10) then 00:59, 01:59, 02:59 (Jun 11)
            start="2025-06-10 22:00",
            stop="2025-06-11 03:00",
            instruments=["BINANCE.UM:SWAP:BTCUSDT"],
            debug="INFO",
            silent=True,
        )

        assert len(s.checks) > 0, "Strategy should have received events"

        row_counts = []
        for sim_time_str, row_count, last_ts, correct_syms in s.checks:
            # - every event must have data with correct symbols
            assert row_count > 0, f"No data at sim_time={sim_time_str}"
            assert correct_syms, f"Expected BCH/BTC/ETH at sim_time={sim_time_str}"
            # - time guard: last data timestamp <= sim time
            assert last_ts <= pd.Timestamp(sim_time_str), (
                f"TimeGuard violated: last_ts={last_ts} > sim_time={sim_time_str}"
            )
            row_counts.append(row_count)

        # - daily data should grow after crossing midnight (2025-06-11 row becomes visible)
        assert row_counts[-1] > row_counts[0], (
            f"Aux data should grow after midnight crossing: first={row_counts[0]}, last={row_counts[-1]}"
        )

    def test_aux_reader_from_separate_storage(self):
        """
        When aux_data is provided explicitly, ctx.get_aux_reader() should use that storage
        instead of the main data storage. Sim crosses midnight so HandyStorage daily data
        should grow. TimeGuard checked on every event.
        """

        class _AuxFromSeparateStorage(IStrategy):
            # - collect per-event: (sim_time, row_count, last_ts, is_from_handy)
            checks: list[tuple[str, int, pd.Timestamp, bool]]

            def on_init(self, initializer: IStrategyInitializer):
                initializer.set_base_subscription("ohlc(1h)")
                initializer.set_event_schedule("1h -1s")
                self.checks = []

            def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | Signal | None:
                sim_time = ctx.time()
                reader = ctx.get_aux_reader("COINGECKO", "FUNDAMENTAL")
                result = reader.read(DataType.ALL, "fundamental", "2020-01-01", "now").to_pd(False)

                last_ts = result.index.max() if len(result) > 0 else pd.Timestamp("NaT")
                from_handy = (
                    (list(result["symbol"].unique()) == ["BTC"] and list(result["metric"].unique()) == ["market_cap"])
                    if len(result) > 0
                    else False
                )
                self.checks.append((str(sim_time), len(result), last_ts, from_handy))

        # - build HandyStorage with daily data spanning Jun 8 → Jun 17 (10 rows)
        # - sim crosses midnight at Jun 11, so rows after sim time must be clamped
        dates = pd.date_range("2025-06-08", periods=10, freq="1D", name="timestamp")
        fundamental_df = pd.DataFrame(
            {
                "symbol": ["BTC"] * 10,
                "asset": ["BTC"] * 10,
                "metric": ["market_cap"] * 10,
                "value": [float(i) for i in range(10)],
            },
            index=dates,
        )
        aux_stor = HandyStorage({}, exchange="COINGECKO:FUNDAMENTAL")
        aux_stor._readers["COINGECKO"]["FUNDAMENTAL"] = HandyReader()
        aux_stor._readers["COINGECKO"]["FUNDAMENTAL"].add("__ALL__", "fundamental", fundamental_df)

        stor = self.testing_csv_storage()
        simulate(
            s := _AuxFromSeparateStorage(),
            data=stor,
            aux_data=aux_stor,
            capital=1000,
            # - crosses midnight: events at 22:59, 23:59 (Jun 10) then 00:59, 01:59, 02:59 (Jun 11)
            start="2025-06-10 22:00",
            stop="2025-06-11 03:00",
            instruments=["BINANCE.UM:SWAP:BTCUSDT"],
            debug="INFO",
            silent=True,
        )

        assert len(s.checks) > 0, "Strategy should have received events"

        row_counts = []
        for sim_time_str, row_count, last_ts, from_handy in s.checks:
            assert row_count > 0, f"No data at sim_time={sim_time_str}"
            # - must be from HandyStorage, not CSV
            assert from_handy, f"Expected HandyStorage data (BTC/market_cap) at sim_time={sim_time_str}"
            # - time guard: last data timestamp <= sim time
            assert last_ts <= pd.Timestamp(sim_time_str), (
                f"TimeGuard violated: last_ts={last_ts} > sim_time={sim_time_str}"
            )
            # - always fewer than 10 rows (future data clamped)
            assert row_count < 10, f"Expected < 10 rows (clamped), got {row_count} at sim_time={sim_time_str}"
            row_counts.append(row_count)

        # - daily data should grow after crossing midnight (2025-06-11 row becomes visible)
        assert row_counts[-1] > row_counts[0], (
            f"Aux data should grow after midnight crossing: first={row_counts[0]}, last={row_counts[-1]}"
        )

    def test_simulation_with_prefetch(self):
        class _TestStrategy(IStrategy):
            # - collect per-event: (sim_time, row_count, last_ts, correct_symbols)
            checks: list[tuple[str, object]]

            def on_init(self, initializer: IStrategyInitializer):
                initializer.set_base_subscription("ohlc(1h)")
                initializer.set_event_schedule("1h -1s")
                self.checks = []

            def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | Signal | None:
                sim_time = ctx.time()
                self.checks.append((str(sim_time), True))

        stor = self.testing_csv_storage()
        simulate(
            s := _TestStrategy(),
            data=stor,
            capital=1000,
            # - crosses midnight: events at 22:59, 23:59 (Jun 10) then 00:59, 01:59, 02:59 (Jun 11)
            start="2025-06-10 22:00",
            stop="2025-06-11 03:00",
            instruments=["BINANCE.UM:SWAP:BTCUSDT"],
            debug="DEBUG",
            silent=True,
            prefetch_config=PrefetchConfig(enabled=True),
        )
        assert len(s.checks) > 0, "Strategy should have received events"

    def test_prefetch_mid_sim_subscribe(self):
        """
        Verify that prefetch + cache behaves correctly when a second symbol is subscribed
        mid-simulation (DataPump.restart_read path).

        Scenario
        --------
        - sim window: 2025-06-10 22:00 → 2025-06-11 03:00  (5 h, scheduled events at xx:59:59)
        - initial universe: BTCUSDT only
        - at the 00:59:59 event (first time >= 2025-06-11 00:59), subscribe ETHUSDT
        - prefetch enabled (CachedStorage wraps the raw CSV reader)

        Expected behaviour
        ------------------
        1. BTCUSDT gets events for the full window: 22:59, 23:59, 00:59, 01:59, 02:59  (5 events)
        2. ETHUSDT only gets events after subscription:  01:59, 02:59  (2 events)
        3. First ETHUSDT event is strictly after the subscribe threshold
        4. BTCUSDT has events *before* the subscribe threshold (proving it ran independently)

        Cache read-call assertions (inner CSV reader, below CachedStorage)
        ------------------------------------------------------------------
        - Call 1  — initial DataPump.start_read:  data_id=["BTCUSDT"]
        - Call 2  — DataPump.restart_read after ETHUSDT is subscribed: data_id=["ETHUSDT"]
          (partial-hit path: BTCUSDT is already cached, so only the missing ETHUSDT is fetched)
        So exactly 2 raw reads reach the inner CSV layer; all other sim ticks serve from cache.
        """

        _SUBSCRIBE_AT = "2025-06-11 00:59"  # - first event at or after this time triggers subscribe

        class _TestStrategy(IStrategy):
            _events: dict[str, list[str]]  # - symbol -> list[sim_time_str]
            _subscribed_second: bool

            def on_init(self, initializer: IStrategyInitializer):
                initializer.set_base_subscription("ohlc(1h)")
                initializer.set_event_schedule("1h -1s")
                self._events = defaultdict(list)
                self._subscribed_second = False

            def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | Signal | None:
                # - record which symbols fired at this sim tick
                for i in ctx.instruments:
                    self._events[i.symbol].append(str(ctx.time()))

                # - add ETHUSDT to universe once we pass the threshold (only once)
                # - add_instruments() adds to ctx.instruments AND subscribes to data feed,
                # - triggering DataPump.restart_read() for the cache-miss path
                if not self._subscribed_second and ctx.time() >= pd.Timestamp(_SUBSCRIBE_AT):
                    eth = ctx.query_instrument("ETHUSDT")
                    if eth is not None:
                        ctx.add_instruments([eth])
                        self._subscribed_second = True

        # - wrap CSV storage in a counter so we can assert inner reader calls later
        counting_stor = _CountingStorage(self.testing_csv_storage())

        simulate(
            s := _TestStrategy(),
            data=counting_stor,
            capital=1000,
            start="2025-06-10 22:00",
            stop="2025-06-11 03:00",
            instruments=["BINANCE.UM:SWAP:BTCUSDT"],
            debug="DEBUG",
            silent=True,
            prefetch_config=PrefetchConfig(enabled=True),
        )

        btc_times = [pd.Timestamp(t) for t in s._events["BTCUSDT"]]
        eth_times = [pd.Timestamp(t) for t in s._events["ETHUSDT"]]
        subscribe_ts = pd.Timestamp(_SUBSCRIBE_AT)

        # - BTCUSDT: must have events (ran throughout)
        assert len(btc_times) > 0, "BTCUSDT produced no events"

        # - ETHUSDT: must have events (subscription worked)
        assert len(eth_times) > 0, "ETHUSDT produced no events after mid-sim subscribe"

        # - BTCUSDT has more events than ETHUSDT (subscribed earlier)
        assert len(btc_times) > len(eth_times), f"Expected BTCUSDT ({len(btc_times)}) > ETHUSDT ({len(eth_times)})"

        # - BTCUSDT had events *before* the subscribe threshold
        btc_before = [t for t in btc_times if t < subscribe_ts]
        assert len(btc_before) > 0, "BTCUSDT should have events before the subscribe threshold"

        # - first ETHUSDT event is strictly after the subscribe threshold
        assert eth_times[0] > subscribe_ts, (
            f"First ETHUSDT event {eth_times[0]} must be after subscribe_ts {subscribe_ts}"
        )

        # - all ETHUSDT events are after the subscribe threshold
        assert all(t > subscribe_ts for t in eth_times), (
            f"ETHUSDT event before subscribe threshold: {[t for t in eth_times if t <= subscribe_ts]}"
        )

        # - cache read-call verification:
        #   call 1 — initial start_read  for [BTCUSDT]
        #   call 2 — restart_read        for [BTCUSDT, ETHUSDT] (miss because ETHUSDT absent)
        #   anything beyond 2 means data was NOT served from cache
        inner_calls = counting_stor.read_calls
        assert len(inner_calls) == 2, (
            f"Expected exactly 2 inner reader calls (1 initial + 1 on restart), got {len(inner_calls)}:\n"
            + "\n".join(f"  {c}" for c in inner_calls)
        )

        # - first call: only BTCUSDT
        first_ids = inner_calls[0][0]
        assert "BTCUSDT" in first_ids and "ETHUSDT" not in first_ids, (
            f"First inner read should only contain BTCUSDT, got: {first_ids}"
        )

        # - second call: ETHUSDT only (partial hit — BTCUSDT already cached, only new symbol fetched)
        second_ids = inner_calls[1][0]
        assert "ETHUSDT" in second_ids and "BTCUSDT" not in second_ids, (
            f"Second inner read should contain only ETHUSDT (partial hit), got: {second_ids}"
        )

    def test_prefetch_subscribe_unsubscribe_cycle(self):
        """
        Verify cache behavior through a full subscribe → unsubscribe → re-subscribe cycle
        with prefetch enabled (chunksize > 0).

        Universe timeline (events fire at xx:59:59 with 1h -1s schedule):
          22:59:59  [BTCUSDT]              — phase 0 (initial)
          23:59:59  [BTCUSDT]              — phase 0
          00:59:59  [BTCUSDT]   → +ETHUSDT — phase 0→1 transition at END of event
          01:59:59  [BTC, ETH]  → -BTCUSDT — phase 1→2 transition at END of event
          02:59:59  [ETHUSDT]   → +BTCUSDT — phase 2→3 transition at END of event
          03:59:59  [BTC, ETH]             — phase 3 (both active again)
          04:59:59  [BTC, ETH]             — phase 3

        Expected BTCUSDT events: 22:59:59, 23:59:59, 00:59:59, 01:59:59, 03:59:59, 04:59:59
        Expected ETHUSDT events: 01:59:59, 02:59:59, 03:59:59, 04:59:59

        Cache read-call expectations (inner CSV reader, below CachedStorage layer):
          Call 1 — initial DataPump.start_read:  data_id=[BTCUSDT]
                   BTCUSDT stored in cache for full sim range.
          Call 2 — DataPump.restart_read on ETHUSDT add:  data_id=[ETHUSDT]
                   Partial hit: range already covered (from call 1), BTCUSDT already cached.
                   Only the missing ETHUSDT is fetched from the inner reader.
                   After this call both symbols are fully in cache.
          NO call 3 — re-add BTCUSDT (phase 2→3):
                   restart_read([ETHUSDT, BTCUSDT]) — both already in cache → pure cache hit.
        Total inner calls: exactly 2.
        """

        _ADD_ETH_AT = "2025-06-11 00:59"  # - trigger at 00:59:59 event (phase 0→1)
        _REMOVE_BTC_AT = "2025-06-15 01:59"  # - trigger at 01:59:59 event (phase 1→2)
        _READD_BTC_AT = "2025-06-28 02:59"  # - trigger at 02:59:59 event (phase 2→3)
        _READD_CACHE = "2025-06-30 00:00"  # - read cache

        class _CycleStrategy(IStrategy):
            _events: dict[str, list[str]]  # - symbol → list[sim_time_str]
            _phase: int  # - 0: BTC only, 1: BTC+ETH, 2: ETH only, 3: BTC+ETH

            def on_init(self, initializer: IStrategyInitializer):
                initializer.set_base_subscription("ohlc(1h)")
                initializer.set_event_schedule("1h -1s")
                self._events = defaultdict(list)
                self._phase = 0

            # def on_market_data(self, ctx: IStrategyContext, data: MarketEvent) -> None:
                # logger.info(f"MD: {data}")

            def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | Signal | None:
                t = ctx.time()
                # - record universe membership at the START of this tick (before any change)
                for i in ctx.instruments:
                    self._events[i.symbol].append(str(t))

                # - phase 0 → 1: add ETHUSDT to universe
                if self._phase == 0 and t >= pd.Timestamp(_ADD_ETH_AT):
                    eth = ctx.query_instrument("ETHUSDT")
                    if eth is not None:
                        ctx.add_instruments([eth])
                        self._phase = 1

                # - phase 1 → 2: remove BTCUSDT (keep only ETHUSDT)
                elif self._phase == 1 and t >= pd.Timestamp(_REMOVE_BTC_AT):
                    eth_only = [i for i in ctx.instruments if i.symbol == "ETHUSDT"]
                    if eth_only:
                        ctx.set_universe(eth_only)
                        self._phase = 2

                # - phase 2 → 3: re-add BTCUSDT — should be served from cache
                elif self._phase == 2 and t >= pd.Timestamp(_READD_BTC_AT):
                    btc = ctx.query_instrument("BTCUSDT")
                    if btc is not None:
                        ctx.add_instruments([btc])
                        self._phase = 3

                elif self._phase == 3 and t >= pd.Timestamp(_READD_CACHE):
                    logger.info("Final - read")
                    ctx.ohlc(ctx.instruments[0], "1h", 10)
                    self._phase = 4

        counting_stor = _CountingStorage(self.testing_csv_storage())

        simulate(
            s := _CycleStrategy(),
            data=counting_stor,
            capital=1000,
            start="2025-06-10 22:00",
            stop="2025-07-11 05:00",  # - 7h window → 7 events: 22:59, 23:59, 00:59 … 04:59
            instruments=["BINANCE.UM:SWAP:BTCUSDT"],
            debug="INFO",
            silent=True,
            prefetch_config=PrefetchConfig(enabled=True),
        )

        btc_times = [pd.Timestamp(t) for t in s._events["BTCUSDT"]]
        eth_times = [pd.Timestamp(t) for t in s._events["ETHUSDT"]]
        add_eth_ts = pd.Timestamp(_ADD_ETH_AT)
        remove_btc_ts = pd.Timestamp(_REMOVE_BTC_AT)
        readd_btc_ts = pd.Timestamp(_READD_BTC_AT)

        # ---- universe membership assertions ----

        # - BTCUSDT must have events (phases 0, 1, 3)
        assert len(btc_times) > 0, "BTCUSDT should have events"
        # - ETHUSDT must have events after subscribe (phases 1, 2, 3)
        assert len(eth_times) > 0, "ETHUSDT should have events after mid-sim subscribe"

        # - BTCUSDT has events BEFORE ETHUSDT was added (phase 0)
        btc_before_eth = [t for t in btc_times if t < add_eth_ts]
        assert len(btc_before_eth) > 0, "BTCUSDT must have events before ETHUSDT subscribe"

        # - ETHUSDT has no events before subscribe threshold
        eth_before_add = [t for t in eth_times if t <= add_eth_ts]
        assert len(eth_before_add) == 0, f"ETHUSDT must not have events before subscribe threshold: {eth_before_add}"

        # - BTCUSDT is absent during the ETH-only phase (June 15 → June 28)
        # - Note: BTCUSDT IS legitimately present at the removal event (01:59:59 on June 15) —
        #   recording happens BEFORE ctx.set_universe() fires at the end of that event.
        # - The ETH-only window is therefore strictly (June 15 01:59:59, June 28 02:59:59].
        _phase1_last_event = pd.Timestamp(_REMOVE_BTC_AT).replace(second=59)
        _eth_only_last_event = pd.Timestamp(_READD_BTC_AT).replace(second=59)
        btc_during_eth_only = [t for t in btc_times if _phase1_last_event < t <= _eth_only_last_event]
        assert len(btc_during_eth_only) == 0, (
            f"BTCUSDT must not have events during ETH-only phase "
            f"({_phase1_last_event} → {_eth_only_last_event}): {btc_during_eth_only[:5]}"
        )

        # - BTCUSDT returns after re-add (phase 3 starts after _eth_only_last_event)
        btc_after_readd = [t for t in btc_times if t > _eth_only_last_event]
        assert len(btc_after_readd) > 0, "BTCUSDT must have events after re-add (phase 3)"

        # - during the overlap phase (1) both symbols are active → same event count
        # - overlap starts at eth_times[0]: ETHUSDT is added at END of the 00:59:59 event,
        #   so it first appears at the NEXT tick (01:59:59); BTCUSDT is still in universe there
        # - overlap ends at _phase1_last_event (June 15 01:59:59): BTCUSDT is recorded BEFORE
        #   removal fires at end of that event, so both are still present at that tick
        _overlap_start = eth_times[0]  # - June 11 01:59:59
        btc_in_overlap = [t for t in btc_times if _overlap_start <= t <= _phase1_last_event]
        eth_in_overlap = [t for t in eth_times if _overlap_start <= t <= _phase1_last_event]
        assert len(btc_in_overlap) == len(eth_in_overlap), (
            f"During overlap phase both symbols must fire on every tick: "
            f"BTCUSDT={len(btc_in_overlap)}, ETHUSDT={len(eth_in_overlap)}"
        )
        assert len(btc_in_overlap) > 0, "Both symbols must have events during overlap phase"

        # - ETHUSDT runs continuously (no removal gap) so may have MORE total events than
        #   BTCUSDT, which was absent for ~13 days during the ETH-only phase
        eth_only_events = [t for t in eth_times if _phase1_last_event < t <= _eth_only_last_event]
        assert len(eth_only_events) > 0, "ETHUSDT must have events during BTC-absent phase"

        # ---- cache inner-reader call assertions ----
        inner_calls = counting_stor.read_calls

        # - exactly 2 inner reads despite 3 subscription-change events:
        #   call 1 — initial start_read:        [BTCUSDT]
        #   call 2 — restart on ETHUSDT add:    [ETHUSDT] only  (partial hit: BTC already cached)
        #   NO call 3 — re-add BTCUSDT:         both symbols in cache → pure cache hit
        assert len(inner_calls) == 2, (
            f"Expected exactly 2 inner reader calls despite 3 subscription events:\n"
            f"  call 1 — initial [BTCUSDT]\n"
            f"  call 2 — partial hit, only [ETHUSDT] fetched\n"
            f"  NO call 3 — re-add BTCUSDT served from cache\n"
            f"Got {len(inner_calls)} call(s):\n"
            + "\n".join(
                f"  call {i + 1}: ids={c[0]}, dtype={c[1]}, start={c[2]}, stop={c[3]}"
                for i, c in enumerate(inner_calls)
            )
        )

        # - call 1: BTCUSDT only (initial subscription)
        first_ids = inner_calls[0][0]
        assert "BTCUSDT" in first_ids and "ETHUSDT" not in first_ids, f"Call 1 should be BTCUSDT-only, got: {first_ids}"

        # - call 2: ETHUSDT only — partial hit, BTCUSDT was already cached and is NOT re-read
        second_ids = inner_calls[1][0]
        assert "ETHUSDT" in second_ids and "BTCUSDT" not in second_ids, (
            f"Call 2 should contain only ETHUSDT (partial hit), got: {second_ids}"
        )
