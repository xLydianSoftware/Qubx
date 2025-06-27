from collections import deque

import pandas as pd

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import InitializingSignal, Instrument, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, PositionsTracker
from qubx.data.helpers import loader
from qubx.trackers.sizers import FixedSizer


class SignalsGenerator(IStrategy):
    actions = []
    _cmd_queue: deque = deque()

    def on_start(self, ctx: IStrategyContext) -> None:
        self._cmd_queue = deque(self.actions)

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        s = ctx.instruments[0]

        if self._cmd_queue:
            cmd, tm, pos, price, tk, stp = self._cmd_queue[0]
            if pd.Timestamp(tm) <= ctx.time():
                logger.info(f"\n\n\t>>> <R>-{cmd.upper()}-</R> {s} {pos} @ {price} (tk={tk}, stp={stp})\n")
                self._cmd_queue.popleft()

                match cmd:
                    case "signal":
                        return [s.signal(ctx, pos, price, take=tk, stop=stp)]

                    case "check-active-targets":
                        for _s, _tp in ctx.get_active_targets().items():
                            logger.info(f"\n\n\t>>> {_s} ::: <G>-{str(_tp)}-</G>\n")

                    case "init-signal":
                        return [InitializingSignal(ctx.time(), s, pos, price, take=tk, stop=stp)]

                    case "emit-init-signal":
                        ctx.emit_signal(InitializingSignal(ctx.time(), s, pos, price, take=tk, stop=stp))
                        return []

                    case _:
                        pass

        return []

    def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
        return PositionsTracker(FixedSizer(10000.0))


class TestPostWarmupInitializationTestTargetsProcessing:
    def test_initializing_signals_processing(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        simulate(
            {
                "signals_generator": (
                    s := SignalsGenerator(
                        actions=[
                            # ("init-signal", "2023-06-03 23:59:59", +10.0, None, None, None),  # mkt order
                            # ("emit-init-signal", "2023-06-03 23:59:59", +10.0, None, None, None),  # mkt order
                            ("signal", "2023-06-03 23:59:59", +1.0, None, None, None),  # mkt order
                            ("check-active-targets", "2023-06-04 01:00:00", None, None, None, None),  # check
                        ]
                    )
                ),
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
