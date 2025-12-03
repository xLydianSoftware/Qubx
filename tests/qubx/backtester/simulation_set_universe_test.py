import pandas as pd

from qubx import logger
from qubx.backtester import simulate
from qubx.core.basics import DataType
from qubx.core.interfaces import (
    Instrument,
    IStrategy,
    IStrategyContext,
    IStrategyInitializer,
    Signal,
    TriggerEvent,
)
from qubx.data import loader
from qubx.trackers.riskctrl import StopTakePositionTracker


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
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)
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
            {"ohlc(1d)": ld},
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
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)
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
            {"ohlc(1d)": ld},
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
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

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
            {"ohlc(1d)": ld}, capital=100_000, instruments=["BINANCE.UM:BTCUSDT"], commissions="vip0_usdt",
            debug="INFO", silent=True, n_jobs=1,
            start="2023-06-01", stop="+10d",
        )
        # fmt: on

        exs0 = r[0].executions_log
        exs1 = r[1].executions_log

        # - only one trade
        exec_price0 = exs0.iloc[0].price
        exec_time0 = exs0.index[0]
        exec_price1 = exs1.iloc[0].price
        exec_time1 = exs1.index[0]
        logger.info(f" NO-WARMUP -> {exec_time0} :: {exec_price0}")
        logger.info(f"    WARMUP -> {exec_time1} :: {exec_price1}")

        logger.info(
            "OHLC:\n" + str(ld["BTCUSDT", "2023-06-07 23:00":"2023-06-08 01:00"][["open", "high", "low", "close"]])
        )

        # - DEBUG: Print execution info
        logger.info(f"NO-WARMUP executions: {len(exs0)}")
        logger.info(f"WARMUP executions: {len(exs1)}")

        if len(exs0) > 0 and len(exs1) > 0:
            # - Verify that both executions happen at the same price (no stale quotes)
            assert abs(exec_price0 - exec_price1) < 1.0, (
                f"Execution prices should be similar! "
                f"NO-WARMUP: {exec_price0}, WARMUP: {exec_price1}, "
                f"Difference: {abs(exec_price0 - exec_price1)}"
            )
        else:
            logger.warning(f"Missing executions! NO-WARMUP: {len(exs0)}, WARMUP: {len(exs1)}")
