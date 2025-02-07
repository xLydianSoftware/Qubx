from collections import defaultdict

import pandas as pd

from qubx import logger, lookup
from qubx.backtester import simulate
from qubx.core.basics import DataType
from qubx.core.interfaces import (
    OHLCV,
    Instrument,
    IStrategy,
    IStrategyContext,
    Signal,
    TriggerEvent,
)
from qubx.data import loader


class Test_SetUniverseLogic(IStrategy):
    commands = [
        ("fit", "nope", None, None),  # - skip 1'st fit
        ("fit", "set", 0, "close"),
        ("event", "trade", 0, 0.25),
        ("fit", "set", 1, "close"),
    ]
    asserts = []

    def on_init(self, ctx: IStrategyContext) -> None:
        self.asserts = []
        ctx.set_fit_schedule("3D @ 23:59")

        self.commands.insert(0, ("fit", "nope", None, None))  # - skip 1'st fit

    def on_fit(self, ctx: IStrategyContext):
        logger.info(f" - <r>FIT</r> at {ctx.time()} -")
        self.run_cmd("fit", ctx)

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        logger.info(f" - <g>EVENT</g> at {ctx.time()} -")
        return self.run_cmd("event", ctx)

    def _i_by_symbol(self, ctx: IStrategyContext, symbol: str) -> Instrument:
        for i in ctx.instruments:
            if i.symbol == symbol:
                return i
        raise ValueError(f"Instrument {symbol} not found")

    def query_instruments(self, ctx: IStrategyContext, symbols: list[str]) -> list[Instrument]:
        return [i for s in symbols if (i := ctx.query_instrument(s)) is not None]

    def run_cmd(self, scope: str, ctx: IStrategyContext) -> list[Signal]:
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
                    logger.info(f"\t\t>>> <r>COMMAND</r> <cyan>TRADE: {a0} -> {a1}</cyan>")
                    return self._i_by_symbol(ctx, a0).signal(a1)

                case "show-universe":
                    logger.info(
                        f"\t\t>>> <r>COMMAND</r> <cyan>SHOW UNIVERSE: {','.join([i.symbol for i in ctx.instruments])}</cyan>"
                    )

                case "check-universe":
                    logger.info(f"\t\t>>> <r>COMMAND</r> <cyan>CHECK UNIVERSE: {','.join(a0)}</cyan>")
                    self.asserts.append(a := set(a0) == set([i.symbol for i in ctx.instruments]))
                    assert a, f"Universe mismatch: {a0} != {','.join([i.symbol for i in ctx.instruments])}"

        return []


class TestSetUniverseInSimulator:
    def test_set_universe_1(self):
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
