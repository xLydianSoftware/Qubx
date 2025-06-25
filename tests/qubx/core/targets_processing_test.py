from collections import deque

import pandas as pd

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import Instrument, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext
from qubx.data.helpers import loader


class SignalsGenerator(IStrategy):
    _idx = 0
    when_send_signals = [
        ("signal", "2023-06-03 23:59:59", +1.0, None, None, None),  # mkt order
        ("check", "2023-06-04 01:00:00", None, None, None, None),  # check
    ]
    _queue: deque = deque()

    def on_start(self, ctx: IStrategyContext) -> None:
        self._queue = deque(self.when_send_signals)
        self._idx = 0

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        s = ctx.instruments[0]
        if self._queue:
            cmd, tm, pos, price, tk, stp = self._queue[0]
            if pd.Timestamp(tm) <= ctx.time():
                logger.info(f"\n\n\t>>> <R>-{cmd.upper()}-</R> {s} {pos} @ {price} (tk={tk}, stp={stp})\n")
                self._queue.popleft()

                match cmd:
                    case "signal":
                        return [s.signal(ctx, pos, price, take=tk, stop=stp)]

                    case "check":
                        for _s, _tp in ctx.get_active_targets().items():
                            logger.info(f"\n\n\t>>> {_s} ::: <G>-{str(_tp)}-</G>\n")

                    case _:
                        pass

        return []


class TestTargetsProcessing:
    def test_signals_processing(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        simulate(
            {
                "signals_generator": (stg := SignalsGenerator()),
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

        # assert stg._exch == "BINANCE.UM", "Got Errors during the simulation"
