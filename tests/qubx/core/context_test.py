from typing import Any

from qubx import logger
from qubx.backtester.broker import SimulatedAccountProcessor
from qubx.backtester.simulator import simulate
from qubx.backtester.utils import SimulatedScheduler, SimulatedTimeProvider, recognize_simulation_data_config
from qubx.core.basics import DataType, Instrument, ITimeProvider, Signal, TriggerEvent, dt_64
from qubx.core.interfaces import IStrategy, IStrategyContext
from qubx.core.lookups import lookup
from qubx.data.helpers import loader


class Tester1(IStrategy):
    _idx = 0
    _err = False
    _to_test: list[list[Instrument]] = []

    def on_init(self, ctx: IStrategyContext) -> None:
        self._exch = ctx.exchanges[0]
        logger.info(f"Exchange: {self._exch}")

    def on_fit(self, ctx: IStrategyContext):
        instr = [ctx.query_instrument(s, ctx.exchanges[0]) for s in ["BTCUSDT", "ETHUSDT", "BCHUSDT"]]
        logger.info(str(instr))
        # logger.info(f" -> SET NEW UNIVERSE {','.join(i.symbol for i in self._to_test[self._idx])}")
        # self._idx += 1

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        for s in ctx.instruments:
            q = ctx.quote(s)
            if q is None:
                logger.error(f"\n{s.symbol} -> NO QUOTE\n")
                self._err = True

        return []


class TestStrategyContext:
    def test_context_exchanges(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        simulate(
            {
                "fail1": (stg := Tester1()),
            },
            {"ohlc(4h)": ld},
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-08-01",
            debug="DEBUG",
            n_jobs=1,
        )

        assert stg._exch == "BINANCE.UM", "Got Errors during the simulation"
