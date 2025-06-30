from collections import defaultdict
from typing import Any, cast

import numpy as np
import pandas as pd

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.backtester.utils import SetupTypes, recognize_simulation_configuration, recognize_simulation_data_config
from qubx.core.basics import DataType, Instrument, MarketEvent, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext
from qubx.core.lookups import lookup
from qubx.core.series import OHLCV, Quote
from qubx.data import loader
from qubx.data.helpers import InMemoryCachedReader
from qubx.data.readers import AsOhlcvSeries, CsvStorageDataReader, InMemoryDataFrameReader
from qubx.pandaz.utils import shift_series
from qubx.ta.indicators import ema
from qubx.trackers.riskctrl import AtrRiskTracker


class Issue1(IStrategy):
    exchange: str = "BINANCE.UM"
    _idx = 0
    _err = False
    _to_test: list[list[Instrument]] = []

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(DataType.OHLC["1h"])
        ctx.set_fit_schedule("59 22 * */1 L7")  # Run at 22:59 every month on Sunday
        ctx.set_event_schedule("55 23 * * *")  # Run at 23:55 every day
        self._to_test = [
            [self.find_instrument(s) for s in ["BTCUSDT", "ETHUSDT"]],
            [self.find_instrument(s) for s in ["BTCUSDT", "BCHUSDT", "LTCUSDT"]],
            [self.find_instrument(s) for s in ["BTCUSDT", "AAVEUSDT", "ETHUSDT"]],
        ]

    def on_fit(self, ctx: IStrategyContext):
        ctx.set_universe(self._to_test[self._idx])
        logger.info(f" -> SET NEW UNIVERSE {','.join(i.symbol for i in self._to_test[self._idx])}")
        self._idx += 1
        if self._idx > 2:
            self._idx = 0

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        for s in ctx.instruments:
            q = ctx.quote(s)
            # - quotes should be in ctx already !!!
            if q is None:
                # print(f"\n{s.symbol} -> NO QUOTE\n")
                logger.error(f"\n{s.symbol} -> NO QUOTE\n")
                self._err = True

        return []

    def find_instrument(self, symbol: str) -> Instrument:
        i = lookup.find_symbol(self.exchange, symbol)
        assert i is not None, f"Could not find {self.exchange}:{symbol}"
        return i


class Issue2(IStrategy):
    _fits_called = 0
    _events_called = 0

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(DataType.OHLC["1h"])
        ctx.set_fit_schedule("59 22 * * *")  # Run at 22:59 every month on Sunday
        ctx.set_event_schedule("55 23 * * *")  # Run at 23:55 every day
        self._fits_called = 0
        self._events_called = 0

    def on_fit(self, ctx: IStrategyContext):
        logger.info(f" > [{ctx.time()}] On Fit is called")
        self._fits_called += 1

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        logger.info(f" > [{ctx.time()}] On event is called")
        self._events_called += 1
        return []


class Issue3(IStrategy):
    _fits_called = 0
    _triggers_called = 0
    _market_quotes_called = 0
    _market_ohlc_called = 0
    _last_trigger_event: TriggerEvent | None = None
    _last_market_event: MarketEvent | None = None
    _market_events: list[MarketEvent]

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(DataType.OHLC["1h"])
        self._fits_called = 0
        self._triggers_called = 0
        self._market_ohlc_called = 0
        self._market_events = []

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent):
        self._triggers_called += 1
        self._last_trigger_event = event

    def on_market_data(self, ctx: IStrategyContext, event: MarketEvent):
        # print(event.type)
        if event.type == DataType.QUOTE:
            self._market_quotes_called += 1

        if event.type == DataType.OHLC:
            self._market_ohlc_called += 1

        self._market_events.append(event)
        self._last_market_event = event


class Issue3_OHLC_TICKS(IStrategy):
    _fits_called = 0
    _triggers_called = 0
    _market_quotes_called = 0
    _market_ohlc_called = 0
    _last_trigger_event: TriggerEvent | None = None
    _last_market_event: MarketEvent | None = None
    _market_events: list[MarketEvent]

    def on_init(self, ctx: IStrategyContext) -> None:
        # - this will creates quotes from OHLC
        # ctx.set_base_subscription(DataType.OHLC_QUOTES["1h"])
        self._fits_called = 0
        self._triggers_called = 0
        self._market_events = []

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent):
        self._triggers_called += 1
        self._last_trigger_event = event

    def on_market_data(self, ctx: IStrategyContext, event: MarketEvent):
        assert event.instrument is not None, "Got wrong instrument"
        logger.info(f"{event.instrument.symbol} market event ::: {event.type}\t:::  -> {event.data}")
        if event.type == DataType.QUOTE:
            self._market_quotes_called += 1

        if event.type == DataType.OHLC:
            self._market_ohlc_called += 1

        self._market_events.append(event)
        self._last_market_event = event


class Issue4(IStrategy):
    _issues = 0

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(DataType.OHLC["1h"])

    def on_market_data(self, ctx: IStrategyContext, event: MarketEvent):
        try:
            if event.type != DataType.QUOTE:
                return
            quote = event.data
            assert isinstance(quote, Quote)
            assert event.instrument is not None, "Got wrong instrument"
            _ohlc = ctx.ohlc(event.instrument)
            assert _ohlc[0].close == quote.mid_price(), f"OHLC: {_ohlc[0].close} != Quote: {quote.mid_price()}"
        except:
            self._issues += 1


class Issue5(IStrategy):
    _err_time: int = 0
    _err_bars: int = 0
    _out: OHLCV | None = None

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(DataType.OHLC["1d"])
        ctx.set_event_schedule("0 0 * * *")  # Run at 00:00 every day

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        # logger.info(f"On Event: {ctx.time()}")

        data = ctx.ohlc(ctx.instruments[0], "1d", 10)
        # logger.info(f"On Event: {len(data)} -> {data[0].open} ~ {data[0].close}")
        print(f"On Event: {ctx.time()}\n{str(data)}")

        # - at 00:00 bar[1] must be previous day's bar !
        if data[1].time > ctx.time().item() - pd.Timedelta("1d").asm8:
            self._err_time += 1

        # - check bar's consitency
        if data[1].open == data[1].close and data[1].open == data[1].low and data[1].open == data[1].low:
            self._err_bars += 1

        self._out = data
        return []


class Test6_HistOHLC(IStrategy):
    _U: list[list[Instrument]] = []
    _out: dict[Any, OHLCV] = {}
    _out_fit: dict[Any, Any] = {}

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(DataType.OHLC["1d"])
        # ctx.set_fit_schedule("59 22 * */1 L7")
        # ctx.set_event_schedule("55 23 * * *")
        # ctx.set_fit_schedule("0 0 * */1 L1")
        ctx.set_fit_schedule("0 0 * * *")
        ctx.set_event_schedule("0 0 * * *")
        # ctx.set_warmup(SubscriptionType.OHLC, "1d")
        self._U = [
            [self.find_instrument(s) for s in ["BTCUSDT", "ETHUSDT"]],
            [self.find_instrument(s) for s in ["BTCUSDT", "BCHUSDT", "LTCUSDT"]],
            [self.find_instrument(s) for s in ["BTCUSDT", "AAVEUSDT", "ETHUSDT"]],
        ]
        self._out = {}
        self._out_fit = {}

    def find_instrument(self, symbol: str) -> Instrument:
        assert (i := lookup.find_symbol("BINANCE.UM", symbol)) is not None, f"Could not find BINANCE.UM:{symbol}"
        return i

    def on_fit(self, ctx: IStrategyContext):
        logger.info(f" - - - - - - - ||| <r>FIT</r> at {ctx.time()}")
        self._out_fit |= self.get_ohlc(ctx, self._U[0])
        ctx.set_universe(self._U[0])

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        logger.info(f" - - - - - - - ||| <g>TRIGGER</g> at {ctx.time()}")
        self._out |= self.get_ohlc(ctx, ctx.instruments)
        return []

    def get_ohlc(self, ctx: IStrategyContext, instruments: list[Instrument]) -> dict:
        closes = defaultdict(dict)
        # Sort instruments by symbol to ensure consistent ordering
        sorted_instruments = sorted(instruments, key=lambda x: x.symbol)
        for i in sorted_instruments:
            data = ctx.ohlc(i, "1d", 4)
            logger.info(f":: :: :: {i.symbol} :: :: ::\n{str(data)}")
            closes[pd.Timestamp(data[0].time, unit="ns")] |= {i.symbol: data[0].close}
        return closes


class QuotesUpdatesTest(IStrategy):
    _market_quotes_called = 0

    def on_init(self, ctx: IStrategyContext) -> None:
        self._market_quotes_called = 0
        self._market_natural_spread = 0

    def on_market_data(self, ctx: IStrategyContext, event: MarketEvent):
        if event.type == DataType.QUOTE:
            q = event.data
            s = q.ask - q.bid
            # print(q, f"{s:.2f} | {event.instrument.tick_size:.2f}")
            assert event.instrument is not None, "Got wrong instrument"
            self._market_natural_spread += s > event.instrument.tick_size
            self._market_quotes_called += 1


class TestSimulator:
    def test_fit_event_quotes(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        simulate(
            {
                "fail1": (stg := Issue1()),
            },
            ld,
            aux_data=ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-08-01",
            debug="DEBUG",
            n_jobs=1,
        )

        assert not stg._err, "Got Errors during the simulation"

    def test_scheduled_events(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        test0 = simulate(
            {
                "fail2": (stg := Issue2()),
            },
            ld,
            aux_data=ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-06-10",
            debug="DEBUG",
            n_jobs=1,
        )

        assert stg._fits_called >= 9, "Got Errors during the simulation"
        assert stg._events_called >= 9, "Got Errors during the simulation"

    def test_market_updates(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        simulate(
            {
                "fail3": (stg := Issue3()),
            },
            ld,
            aux_data=ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-06-10",
            debug="DEBUG",
            n_jobs=1,
        )

        # - no quotes arrived - it subscribed to OHLC only
        assert stg._market_quotes_called == 0, "Got Errors during the simulation"
        assert (stg._triggers_called) * 4 + 3 == stg._market_ohlc_called, "Got Errors during the simulation"

    def test_ohlc_quote_update(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        # fmt: off
        test0 = simulate(
            { "fail4": (stg := Issue4()), },
            ld, aux_data=ld,
            capital=100_000, instruments=["BINANCE.UM:BTCUSDT"], commissions="vip0_usdt", debug="DEBUG", n_jobs=1,
            start="2023-06-01", stop="2023-06-10",
        )
        # fmt: on

        assert stg._issues == 0, "Got Errors during the simulation"

    def test_ohlc_data(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        test4 = simulate(
            {
                "Issue5": (stg := Issue5()),
            },
            ld,
            aux_data=ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-07-01",
            stop="2023-07-10",
            debug="DEBUG",
            silent=True,
            n_jobs=1,
        )
        assert stg._err_time == 0, "Got wrong OHLC bars time"
        assert stg._err_bars == 0, "OHLC bars were not consistent"
        assert isinstance(stg, Issue5)
        assert isinstance(stg._out, OHLCV)

        r = ld[["BTCUSDT"], "2023-06-22":"2023-07-10"]("1d")["BTCUSDT"]  # type: ignore
        assert all(stg._out.pd()[["open", "high", "low", "close"]] == r[["open", "high", "low", "close"]]), (
            "Out OHLC differ"
        )

    def test_ohlc_hist_data(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)
        ld_test = loader("BINANCE.UM", "1d", source="csv::tests/data/csv_1h/", n_jobs=1)

        # fmt: off
        simulate({ "Issue5": (stg := Test6_HistOHLC()), },
            ld, 
            capital=100_000, instruments=["BINANCE.UM:BTCUSDT", "BINANCE.UM:ETHUSDT"], commissions="vip0_usdt",
            start="2023-07-01", stop="2023-07-04 23:59",
            debug="DEBUG", silent=True, n_jobs=1,
        )
        # fmt: on

        r = ld_test[["BTCUSDT", "ETHUSDT"], "2023-07-01":"2023-07-04"]("1d")  # type: ignore
        assert all(pd.DataFrame.from_dict(stg._out_fit, orient="index") == r.open)

        r = ld_test[["BTCUSDT", "ETHUSDT"], "2023-07-02":"2023-07-04"]("1d")  # type: ignore
        assert all(pd.DataFrame.from_dict(stg._out, orient="index") == r.open)

    def test_ohlc_tick_data_subscription(self):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # TODO: need to check how it's passed in simulator
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)
        # fmt: off
        simulate(
            { "fail3_ohlc_ticks": (stg := Issue3_OHLC_TICKS()), },
            # ld, 
            {'quote': ld}, 
            aux_data=ld, capital=100_000, debug="DEBUG", n_jobs=1, instruments=["BINANCE.UM:BTCUSDT"], commissions="vip0_usdt",
            start="2023-06-01", stop="2023-06-02 1:00",
            silent=True
        )
        # fmt: on

        assert (stg._triggers_called + 1) * 4 - 1 == stg._market_quotes_called, "Got Errors during the test"

    def test_quotes_updates(self):
        """
        Test that we receive real quote / not restored
        """
        ld = loader("BINANCE.UM", None, source="csv::tests/data/csv_quotes/", n_jobs=1)
        test0 = simulate(
            {"test4": (stg := QuotesUpdatesTest())},
            ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            debug="DEBUG",
            n_jobs=1,
            start="2017-08-24 13:01:11",
            stop="2017-08-24 13:09:31",
        )

        d = ld["BTCUSDT", "2017-08-24 13:01:12":"2017-08-24 13:09:31"]
        assert stg._market_natural_spread == sum((d.ask - d.bid) > 0.1), "Got Errors during the test"

    def test_simple_strategy_simulation(self):
        class CrossOver(IStrategy):
            timeframe: str = "1Min"
            fast_period = 5
            slow_period = 12

            def on_init(self, ctx: IStrategyContext):
                ctx.set_base_subscription(DataType.OHLC[self.timeframe])

            def on_event(self, ctx: IStrategyContext, event: TriggerEvent):
                for i in ctx.instruments:
                    ohlc = ctx.ohlc(i, self.timeframe)
                    fast = ema(ohlc.close, self.fast_period)
                    slow = ema(ohlc.close, self.slow_period)
                    pos = ctx.positions[i].quantity
                    if pos <= 0:
                        if (fast[1] > slow[1]) and (fast[2] < slow[2]):
                            ctx.trade(i, abs(pos) + i.min_size * 10)
                    if pos >= 0:
                        if (fast[1] < slow[1]) and (fast[2] > slow[2]):
                            ctx.trade(i, -pos - i.min_size * 10)
                return None

            def ohlcs(self, timeframe: str) -> dict[str, pd.DataFrame]:
                return {s.symbol: self.ctx.ohlc(s, timeframe).pd() for s in self.ctx.instruments}

        r = CsvStorageDataReader("tests/data/csv")
        ohlc = r.read("BINANCE.UM:BTCUSDT", "2024-01-01", "2024-01-02", AsOhlcvSeries("5Min"))
        fast = ema(ohlc.close, 5)  # type: ignore
        slow = ema(ohlc.close, 15)  # type: ignore
        sigs = (((fast > slow) + (fast.shift(1) < slow.shift(1))) == 2) - (
            ((fast < slow) + (fast.shift(1) > slow.shift(1))) == 2
        )
        sigs = sigs.pd()
        sigs = sigs[sigs != 0]
        i1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert i1 is not None
        # s2 = shift_series(sigs, "4Min59Sec").rename(i1) / 100  # type: ignore
        s2 = shift_series(sigs, "5Min").rename(i1) / 100  # type: ignore

        # fmt: off
        rep1 = simulate(
            {
                # - generated signals as series
                "test0": CrossOver(timeframe="5Min", fast_period=5, slow_period=15),
                "test1": s2,
            },
            {'ohlc(5Min)': r}, 10000, ["BINANCE.UM:BTCUSDT"], "vip0_usdt", "2024-01-01", "2024-01-02", n_jobs=1
        ) 
        # fmt:on

        assert all(
            rep1[0].executions_log[["filled_qty", "price", "side"]]
            == rep1[1].executions_log[["filled_qty", "price", "side"]]
        )


class TestSimulatorHelpers:
    def test_recognize_simulation_configuration(self):
        # fmt: off
        setups = recognize_simulation_configuration(
            "X1",
            {
                "S1": pd.Series([1, 2, 3], name="BTCUSDT"), 
                "S2": pd.Series([1, 2, 3], name="BINANCE.UM:LTCUSDT"),
                "S3": [pd.DataFrame({"BTCUSDT": [1, 2, 3], "BCHUSDT": [4, 5, 6]}), AtrRiskTracker(None, None, '1h', 10)],
                "S4": [IStrategy(), AtrRiskTracker(None, None, '1h', 10)],
                "S5": IStrategy(),
                "S6": { 'A': IStrategy(), 'B': IStrategy(), }
            }, # type: ignore
            [lookup.find_symbol("BINANCE.UM", s) for s in ["BTCUSDT", "BCHUSDT", "LTCUSDT"]],  # type: ignore
            ["BINANCE.UM"],
            10_000, "USDT", "vip0_usdt", "1Min", True)

        assert setups[0].setup_type == SetupTypes.SIGNAL, "Got wrong setup type"
        assert setups[1].setup_type == SetupTypes.SIGNAL, "Got wrong setup type"
        assert setups[2].setup_type == SetupTypes.SIGNAL_AND_TRACKER, "Got wrong setup type"
        assert setups[3].setup_type == SetupTypes.STRATEGY_AND_TRACKER, "Got wrong setup type"
        assert setups[4].setup_type == SetupTypes.STRATEGY, "Got wrong setup type"
        assert setups[5].setup_type == SetupTypes.STRATEGY, "Got wrong setup type"
        assert setups[5].name == "X1/S6/A", "Got wrong setup type"
        assert setups[6].setup_type == SetupTypes.STRATEGY, "Got wrong setup type"
        assert setups[6].name == "X1/S6/B", "Got wrong setup type"
        # fmt: on

    def test_recognize_simulation_input_data(self):
        l1 = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h", n_jobs=1)
        l2 = loader("BINANCE.UM", "1d", source="csv::tests/data/csv_1h", n_jobs=1)

        idx = pd.date_range(start="2023-06-01 00:00", end="2023-07-30", freq="1h", name="timestamp")
        c_data = pd.DataFrame({"value1": np.random.randn(len(idx)), "value2": np.random.randn(len(idx))}, index=idx)
        custom_reader = InMemoryDataFrameReader({"BINANCE.UM:BTCUSDT": c_data})

        idx = pd.date_range(start="2023-06-01 00:00", end="2023-07-30", freq="1h", name="timestamp")
        q_data = pd.DataFrame({"bid": np.random.randn(len(idx)), "ask": np.random.randn(len(idx))}, index=idx)
        qts_reader = InMemoryDataFrameReader({"BINANCE.UM:BTCUSDT": q_data})

        instrs = [lookup.find_symbol("BINANCE.UM", s) for s in ["BTCUSDT", "BCHUSDT", "LTCUSDT"]]  # type: ignore
        assert all([x and isinstance(x, Instrument) for x in instrs]), "Got wrong instruments"
        instrs = cast(list[Instrument], instrs)

        l1 = cast(InMemoryCachedReader, l1)
        C1 = l1
        cfg = recognize_simulation_data_config(C1, instrs)
        assert cfg.default_trigger_schedule == "0 */1 * * *"
        assert cfg.default_base_subscription == "ohlc(1h)"

        C2 = l1[["BTCUSDT", "ETHUSDT"], "2023-06-01":"2023-07-30"]
        cfg = recognize_simulation_data_config(C2, instrs)
        assert cfg.default_trigger_schedule == "0 */1 * * *"
        assert cfg.default_base_subscription == "ohlc(1h)"

        C3 = {"ohlc(15Min)": l2}
        cfg = recognize_simulation_data_config(C3, instrs)
        assert cfg.default_trigger_schedule == "59 23 */1 * * 59"
        assert cfg.default_base_subscription == "ohlc(1D)"

        try:
            C3 = {"ohlc(1h)": l1, "ohlc(15Min)": l2}
            cfg = recognize_simulation_data_config(C3, instrs)
            assert False, "Shoud not pass !"
        except:  # noqa: E722
            assert True

        Ci = {"ohlc(1Min)": qts_reader}
        cfg = recognize_simulation_data_config(Ci, instrs)
        assert cfg.default_trigger_schedule == "*/1 * * * *"
        assert cfg.default_base_subscription == "quote"

        Ci = {"ohlc": l1[["BTCUSDT", "ETHUSDT"], "2023-06-01":"2023-07-30"]}
        cfg = recognize_simulation_data_config(Ci, instrs)
        assert cfg.default_trigger_schedule == "0 */1 * * *"
        assert cfg.default_base_subscription == "ohlc(1h)"

        Ci = {"quote": qts_reader}
        cfg = recognize_simulation_data_config(Ci, instrs)
        assert cfg.default_trigger_schedule == ""
        assert cfg.default_base_subscription == "quote"

        Ci = {"trade": l1}
        cfg = recognize_simulation_data_config(Ci, instrs)
        assert cfg.default_trigger_schedule == "0 */1 * * *"
        assert cfg.default_base_subscription == "ohlc_trades"

        Ci = {"trade": l1, "quote": l1}
        cfg = recognize_simulation_data_config(Ci, instrs)
        assert cfg.default_trigger_schedule == "0 */1 * * *"
        assert cfg.default_base_subscription == "ohlc_quotes"  # quotes has higher priority

        Ci = {"ohlc(1Min)": qts_reader, "quote": l1}
        cfg = recognize_simulation_data_config(Ci, instrs)
        assert cfg.default_trigger_schedule == "*/1 * * * *"
        assert cfg.default_base_subscription == "quote"  # quotes has higher priority

        Ci = {
            "ohlc(1d23h45Min30Sec)": l1,
            "trade": l1,
            "custom": custom_reader,
        }
        cfg = recognize_simulation_data_config(Ci, instrs)
        assert cfg.default_trigger_schedule == "45 23 */1 * * 30"
        assert cfg.default_base_subscription == "ohlc(1h)"
        assert "custom" in cfg.data_providers
