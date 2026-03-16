import pandas as pd

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import DataType
from qubx.core.interfaces import (
    Instrument,
    IStrategy,
    IStrategyContext,
    IStrategyInitializer,
    Signal,
    TriggerEvent,
)
from qubx.data import CsvStorage
from qubx.trackers.riskctrl import StopTakePositionTracker

# - path to the IStorage-structured CSV test data (EXCHANGE/MARKET_TYPE/SYMBOL.TYPE.csv.gz)
_CSV_STORAGE = "tests/data/storages/csv/"


class Test_SetUniverseLogic(IStrategy):
    commands = [
        ("fit", "nope", None, None),  # - skip 1'st fit
        ("fit", "set", 0, "close"),
        ("event", "trade", 0, 0.25),
        ("fit", "set", 1, "close"),
    ]
    asserts = []

    def on_init(self, initializer: IStrategyInitializer) -> None:
        self.asserts = []
        self.setup_schedules(initializer)
        self.commands.insert(0, ("fit", "nope", None, None))  # - skip 1'st fit

    def setup_schedules(self, initializer: IStrategyInitializer):
        initializer.set_base_subscription("ohlc(1d)")
        initializer.set_event_schedule("1d")
        initializer.set_fit_schedule("3D @ 23:59")

    def on_fit(self, ctx: IStrategyContext):
        logger.info(f" - <r>FIT</r> at {ctx.time()} -")
        self.run_cmd("fit", ctx)

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | Signal:
        # logger.info(f" - <g>EVENT</g> at {ctx.time()} -")
        return self.run_cmd("event", ctx)

    def _i_by_symbol(self, ctx: IStrategyContext, symbol: str) -> Instrument:
        for i in ctx.instruments:
            if i.symbol == symbol:
                return i
        raise ValueError(f"Instrument {symbol} not found")

    def query_instruments(self, ctx: IStrategyContext, symbols: list[str]) -> list[Instrument]:
        return [i for s in symbols if (i := ctx.query_instrument(s)) is not None]

    def run_cmd(self, scope: str, ctx: IStrategyContext) -> list[Signal] | Signal:
        if self.commands and scope == self.commands[0][0]:
            _, c, a0, a1 = self.commands.pop(0)
            # logger.info(f"\t<r>COMMAND</r> for {scope} ::: <r>{c}</r> ({a0}, {a1})")

            match c:
                case "set":
                    n_universe = self.query_instruments(ctx, a0)
                    logger.info(
                        f"\t\t>>> <r>COMMAND</r> <cyan>SET UNIVERSE: {','.join([i.symbol for i in n_universe])}</cyan>"
                    )
                    ctx.set_universe(n_universe, if_has_position_then=a1)

                case "trade":
                    instr = self._i_by_symbol(ctx, a0)
                    logger.info(f"\t\t>>> <r>COMMAND</r> <y>TRADE: {a0} -> {a1}</y> ::: {ctx.quote(instr)}")
                    return instr.signal(ctx, a1)

                case "show-universe":
                    logger.info(
                        f"\t\t>>> <r>COMMAND</r> <y>SHOW UNIVERSE: {','.join([i.symbol for i in ctx.instruments])}</y>"
                    )

                case "check-universe":
                    logger.info(f"\t\t>>> <r>COMMAND</r> <y>CHECK UNIVERSE: {','.join(a0)}</y>")
                    self.asserts.append(a := set(a0) == set([i.symbol for i in ctx.instruments]))
                    assert a, f"Universe mismatch: {a0} != {','.join([i.symbol for i in ctx.instruments])}"

        return []


class RiskManager:
    def tracker(self, ctx: IStrategyContext):
        return StopTakePositionTracker(stop_risk=10, take_target=10)


class TestSetUniverseInSimulator:
    def test_set_universe_policies(self):
        ld = CsvStorage(_CSV_STORAGE)
        s0, s1, s2, s3 = None, None, None, None

        # fmt: off
        r = simulate(
            {
                "SetUniverse0": (
                    s0 := Test_SetUniverseLogic(
                        commands=[
                            ("fit", "set", ["BTCUSDT", "ETHUSDT"], "close"),  # - set initial universe
                            ("event", "show-universe", None, None),
                            ("event", "trade", "BTCUSDT", 0.25),
                            ("fit", "set", ["ETHUSDT", "LTCUSDT"], "close"),  # - this must close BTC position
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["ETHUSDT", "LTCUSDT"], None),
                        ]
                    )
                ),

                "SetUniverse1": (
                    s1 := Test_SetUniverseLogic(
                        commands=[
                            ("fit", "set", ["BTCUSDT", "ETHUSDT"], "close"),                      # - set initial universe
                            ("event", "show-universe", None, None),
                            ("event", "trade", "BTCUSDT", 0.25),                                  # - open BTC position
                            ("fit", "set", ["ETHUSDT", "LTCUSDT"], "wait_for_close"),             # - wait before closing BTC position
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["BTCUSDT", "ETHUSDT", "LTCUSDT"], None), # - universe must be unchanged as it waits for BTC position to be closed
                            ("event", "trade", "BTCUSDT", 0),                                     # - close BTC position
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["ETHUSDT", "LTCUSDT"], None),            # - universe must be changed as BTC position is closed
                        ]
                    )
                ),

                "SetUniverse2": (
                    s2 := Test_SetUniverseLogic(
                        commands=[
                            ("fit", "set", ["BTCUSDT", "ETHUSDT"], "close"),
                            ("event", "show-universe", None, None),
                            ("event", "trade", "BTCUSDT", 0.25),
                            ("fit", "set", ["ETHUSDT", "LTCUSDT"], "wait_for_change"),            # - wait before try to change BTC position somehow
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["BTCUSDT", "ETHUSDT", "LTCUSDT"], None), # - universe must be unchanged
                            ("event", "trade", "BTCUSDT", -2),                                    # - try to change BTC position somehow
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["ETHUSDT", "LTCUSDT"], None),            # - universe must be changed as BTC position is closed
                        ]
                    )
                ),

                "SetUniverse3": (
                    s3 := Test_SetUniverseLogic(
                        commands=[
                            ("fit", "set", ["BTCUSDT", "ETHUSDT"], "close"),
                            ("event", "show-universe", None, None),
                            ("event", "trade", "BTCUSDT", 0.25),
                            ("fit", "set", ["ETHUSDT", "LTCUSDT"], "wait_for_close"),             # - wait before try to change BTC position somehow
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["BTCUSDT", "ETHUSDT", "LTCUSDT"], None), # - universe must be unchanged
                            ("fit", "set", ["BTCUSDT", "ETHUSDT", "LTCUSDT"], "close"),           # - this should reset BTC removal as we want to add it again 
                            ("event", "trade", "BTCUSDT", 0), # - try to change BTC position
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["BTCUSDT", "ETHUSDT", "LTCUSDT"], None),  # - universe must be unchanged
                        ]
                    )
                ),
            },
            ld,
            capital=100_000, instruments=["BINANCE.UM:BTCUSDT"], commissions="vip0_usdt",
            start="2023-06-01", stop="+10d",
            debug="DEBUG", silent=True, n_jobs=-1,
        )
        # fmt: on
        for ri in r:
            logger.info(f"Executions {ri.name}\n" + ri.executions_log.to_string())
            logger.info(f"Signals {ri.name}\n" + ri.signals_log.to_string())

        assert all(s0.asserts) if s0 else True
        assert all(s1.asserts) if s1 else True
        assert all(s2.asserts) if s2 else True
        assert all(s3.asserts) if s3 else True

    def test_set_universe_policies_with_risk_manager(self):
        ld = CsvStorage(_CSV_STORAGE)
        s0, s1, s2, s3 = None, None, None, None

        # fmt: off
        r = simulate(
            {
                "SetUniverse0": (
                    s0 := (Test_SetUniverseLogic + RiskManager)(
                        commands=[
                            ("fit", "set", ["BTCUSDT", "ETHUSDT"], "close"),  # - set initial universe
                            ("event", "show-universe", None, None),
                            ("event", "trade", "BTCUSDT", 0.25),
                            ("fit", "set", ["ETHUSDT", "LTCUSDT"], "close"),  # - this must close BTC position
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["ETHUSDT", "LTCUSDT"], None),
                        ]
                    )
                ),

                "SetUniverse1": (
                    s1 := (Test_SetUniverseLogic + RiskManager)(
                        commands=[
                            ("fit", "set", ["BTCUSDT", "ETHUSDT"], "close"),                      # - set initial universe
                            ("event", "show-universe", None, None),
                            ("event", "trade", "BTCUSDT", 0.25),                                  # - open BTC position
                            ("fit", "set", ["ETHUSDT", "LTCUSDT"], "wait_for_close"),             # - wait before closing BTC position
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["BTCUSDT", "ETHUSDT", "LTCUSDT"], None), # - universe must be unchanged as it waits for BTC position to be closed
                            ("event", "trade", "BTCUSDT", 0),                                     # - close BTC position
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["ETHUSDT", "LTCUSDT"], None),            # - universe must be changed as BTC position is closed
                        ]
                    )
                ),

                "SetUniverse2": (
                    s2 := (Test_SetUniverseLogic + RiskManager)(
                        commands=[
                            ("fit", "set", ["BTCUSDT", "ETHUSDT"], "close"),
                            ("event", "show-universe", None, None),
                            ("event", "trade", "BTCUSDT", 0.25),
                            ("fit", "set", ["ETHUSDT", "LTCUSDT"], "wait_for_change"),            # - wait before try to change BTC position somehow
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["BTCUSDT", "ETHUSDT", "LTCUSDT"], None), # - universe must be unchanged
                            ("event", "trade", "BTCUSDT", -2),                                    # - try to change BTC position somehow
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["ETHUSDT", "LTCUSDT"], None),            # - universe must be changed as BTC position is closed
                        ]
                    )
                ),

                "SetUniverse3": (
                    s3 := (Test_SetUniverseLogic + RiskManager)(
                        commands=[
                            ("fit", "set", ["BTCUSDT", "ETHUSDT"], "close"),
                            ("event", "show-universe", None, None),
                            ("event", "trade", "BTCUSDT", 0.25),
                            ("fit", "set", ["ETHUSDT", "LTCUSDT"], "wait_for_close"),             # - wait before try to change BTC position somehow
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["BTCUSDT", "ETHUSDT", "LTCUSDT"], None), # - universe must be unchanged
                            ("fit", "set", ["BTCUSDT", "ETHUSDT", "LTCUSDT"], "close"),           # - this should reset BTC removal as we want to add it again 
                            ("event", "trade", "BTCUSDT", 0), # - try to change BTC position
                            ("event", "show-universe", None, None),
                            ("event", "check-universe", ["BTCUSDT", "ETHUSDT", "LTCUSDT"], None),  # - universe must be unchanged
                        ]
                    )
                ),
            },
            ld,
            capital=100_000, instruments=["BINANCE.UM:BTCUSDT"], commissions="vip0_usdt",
            start="2023-06-01", stop="+10d",
            debug="DEBUG", silent=True, n_jobs=1,
        )
        # fmt: on

        assert all(s0.asserts) if s0 else True
        assert all(s1.asserts) if s1 else True
        assert all(s2.asserts) if s2 else True
        assert all(s3.asserts) if s3 else True

    def test_resubscibe_first_quote(self):
        ld = CsvStorage(_CSV_STORAGE)

        class Test_SetUniverseLogic_QuoteCheck(Test_SetUniverseLogic):
            use_warmup = False

            def setup_schedules(self, initializer: IStrategyInitializer):
                # - subscription warmup
                if self.use_warmup:
                    initializer.set_subscription_warmup({DataType.OHLC["1h"]: "2h"})

                initializer.set_fit_schedule("3D @ 23:59")
                initializer.set_base_subscription(DataType.OHLC["1h"])
                initializer.set_event_schedule("1h -1s")

        commands = [
            ("fit", "set", ["BTCUSDT", "ETHUSDT"], "close"),  # - set initial universe
            ("event", "show-universe", None, None),
            ("fit", "set", ["ETHUSDT", "LTCUSDT"], "close"),  # - this must close BTC position
            ("event", "show-universe", None, None),
            ("event", "check-universe", ["ETHUSDT", "LTCUSDT"], None),
            ("fit", "set", ["ETHUSDT", "BTCUSDT"], "close"),  # - add BTC again
            ("event", "trade", "BTCUSDT", 0.25),
        ]

        # fmt: off
        r = simulate(
            {
                "RebalanceUniverse-Qtest-no_warmup": (
                    s0 := Test_SetUniverseLogic_QuoteCheck(commands=list(commands), use_warmup = False)
                ),
                "RebalanceUniverse-Qtest-warmup": (
                    s1 := Test_SetUniverseLogic_QuoteCheck(commands=list(commands), use_warmup = True)
                ),
            },
            ld, capital=100_000, instruments=["BINANCE.UM:BTCUSDT"], commissions="vip0_usdt",
            debug="DEBUG", silent=True, n_jobs=1,
            start="2023-06-01", stop="+10d",
        )
        # fmt: on

        exs0 = r[0].executions_log
        exs1 = r[1].executions_log

        # - both simulations must have exactly 1 execution (the "trade BTCUSDT 0.25" after re-subscription)
        assert len(exs0) == 1, f"NO-WARMUP: expected 1 execution, got {len(exs0)}"
        assert len(exs1) == 1, f"WARMUP: expected 1 execution, got {len(exs1)}"

        exec_price0 = exs0.iloc[0].price
        exec_time0 = exs0.index[0]
        exec_price1 = exs1.iloc[0].price
        exec_time1 = exs1.index[0]
        logger.info(f" NO-WARMUP -> {exec_time0} :: {exec_price0}")
        logger.info(f"    WARMUP -> {exec_time1} :: {exec_price1}")

        # - universe check-universe commands must have all passed in both variants
        assert s0.asserts and all(s0.asserts), f"NO-WARMUP: check-universe failed: {s0.asserts}"
        assert s1.asserts and all(s1.asserts), f"WARMUP: check-universe failed: {s1.asserts}"

        # - trade must happen AFTER BTCUSDT was re-added (3rd fit fires at 2023-06-07T23:59)
        _resubscribe_time = pd.Timestamp("2023-06-07 23:59:00")
        assert exec_time0 > _resubscribe_time, (
            f"NO-WARMUP: trade at {exec_time0} must be after re-subscription fit at {_resubscribe_time}"
        )
        assert exec_time1 > _resubscribe_time, (
            f"WARMUP: trade at {exec_time1} must be after re-subscription fit at {_resubscribe_time}"
        )

        # - key assertion: both variants must produce the same result — subscription warmup makes
        # - no difference for quote availability, the simulator always provides a fresh quote from
        # - the current OHLC bar when an instrument is re-added to the universe
        assert exec_time0 == exec_time1, f"Execution times must match: NO-WARMUP={exec_time0}, WARMUP={exec_time1}"
        assert exec_price0 == exec_price1, f"Execution prices must match: NO-WARMUP={exec_price0}, WARMUP={exec_price1}"
