from collections import defaultdict
from typing import Any

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
    _U: list[list[Instrument]] = []
    commands = [
        ("fit", "nope", None, None),  # - skip 1'st fit
        ("fit", "set", 0, "close"),
        ("event", "trade", 0, 0.25),
        ("fit", "set", 1, "close"),
    ]

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_fit_schedule("3D @ 23:59")
        self._U = [
            [i for s in ["BTCUSDT", "ETHUSDT"] if (i := ctx.query_instrument(s)) is not None],
            [i for s in ["ETHUSDT", "LTCUSDT"] if (i := ctx.query_instrument(s)) is not None],
        ]

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

    def run_cmd(self, scope: str, ctx: IStrategyContext) -> list[Signal]:
        if self.commands and scope == self.commands[0][0]:
            _, c, a0, a1 = self.commands.pop(0)
            logger.info(f"\t<r>COMMAND</r> for {scope} ::: <r>{c}</r> ({a0}, {a1})")
            match c:
                case "set":
                    logger.info(f"\t\t>>> <cyan>SET UNIVERSE: {','.join([i.symbol for i in self._U[a0]])}</cyan>")
                    ctx.set_universe(self._U[a0], if_has_position_then=a1)
                    return []

                case "trade":
                    return self._i_by_symbol(ctx, a0).signal(a1)

                case "show-universe":
                    logger.info(f"\t\t>>> <cyan>UNIVERSE: {','.join([i.symbol for i in ctx.instruments])}</cyan>")
                    return []

        return []


class TestSetUniverseInSimulator:
    def test_set_universe_1(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        # fmt: off
        r = simulate(
            {
                "SetUniverse1": (
                    s1 := Test_SetUniverseLogic(
                        commands=[
                            ("fit", "nope", None, None),  # - skip 1'st fit
                            ("fit", "set", 0, "close"),
                            ("event", "show-universe", None, None),
                            ("event", "trade", "BTCUSDT", 0.25),
                            ("fit", "set", 1, "wait_for_close"),
                            # ("event", "nope", None, None),
                            # ("fit", "set", 1, "close"),
                            ("event", "show-universe", None, None),
                            ("event", "trade", "BTCUSDT", 0),
                            ("event", "show-universe", None, None),
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
        logger.info("Executions\n" + r[0].executions_log.to_string())
        logger.info("Signals\n" + r[0].signals_log.to_string())
