from collections import defaultdict

import pandas as pd

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import DataType, Instrument, MarketEvent, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, IStrategyInitializer
from qubx.data.registry import StorageRegistry
from qubx.data.storages.handy import HandyReader, HandyStorage


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
