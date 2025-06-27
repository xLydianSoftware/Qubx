from collections import deque

import pandas as pd

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import InitializingSignal, Instrument, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, PositionsTracker
from qubx.data.helpers import loader
from qubx.trackers.riskctrl import SignalRiskPositionTracker
from qubx.trackers.sizers import FixedSizer


class SignalsGenerator(IStrategy):
    actions = []
    _cmd_queue: deque = deque()

    def on_start(self, ctx: IStrategyContext) -> None:
        self._cmd_queue = deque(self.actions)

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        signals = []
        s = ctx.instruments[0]

        if self._cmd_queue:
            _c_a = self._cmd_queue[0]
            _fn = lambda *a: True
            tm, cmd, pos, price, tk, stp = _c_a[:6]
            if len(_c_a) > 6:
                _fn = _c_a[6]

            if pd.Timestamp(tm) <= ctx.time():
                _d_str = f">>> <R>-{cmd.upper()}-</R> {s}"
                _res = ""
                self._cmd_queue.popleft()

                match cmd:
                    case "signal":
                        signals = [s.signal(ctx, pos, price, take=tk, stop=stp)]

                    case "check-active-targets":
                        for _s, _tp in ctx.get_active_targets().items():
                            _res += f"\n\t<G>{str(_tp)}</G>"
                        _res = _res or "<Y>NO TARGETS</Y>"

                    case "init-signal":
                        return [InitializingSignal(ctx.time(), s, pos, price, take=tk, stop=stp)]

                    case "emit-init-signal":
                        ctx.emit_signal(InitializingSignal(ctx.time(), s, pos, price, take=tk, stop=stp))
                        return []

                    case _:
                        pass

                logger.info(_d_str + " -> " + f"{_res}" + f" {_fn(ctx, s)}")

        return signals

    def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
        return SignalRiskPositionTracker(FixedSizer(10000.0))


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
                            # - open by signal
                            ("2023-06-03 23:59:59", "signal", +1.0, None, None, None),
                            (
                                "2023-06-04 01:00:00",
                                "check-active-targets",
                                None,
                                None,
                                None,
                                None,
                                lambda c, s: s in c.get_active_targets(),
                            ),
                            # - close by signal
                            ("2023-06-04 03:00:00", "signal", +0.0, None, None, None),
                            (
                                "2023-06-04 04:00:00",
                                "check-active-targets",
                                None,
                                None,
                                None,
                                None,
                                lambda c, s: s not in c.get_active_targets(),
                            ),
                            # - open by signal
                            ("2023-06-05 13:00:00", "signal", +1.0, None, None, 26000.0),
                            (
                                "2023-06-05 14:00:00",
                                "check-active-targets",
                                None,
                                None,
                                None,
                                None,
                                lambda c, s: s in c.get_active_targets(),
                            ),
                            (
                                "2023-06-06 00:00:00",
                                "check-active-targets",
                                None,
                                None,
                                None,
                                None,
                                lambda c, s: s not in c.get_active_targets(),
                            ),
                        ]
                    )
                ),
            },
            {"ohlc(1h)": ld},
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-08-01",
            debug="DEBUG",
            n_jobs=1,
        )

        # assert stg._exch == "BINANCE.UM", "Got Errors during the simulation"
