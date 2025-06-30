from collections import deque

import pandas as pd

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import InitializingSignal, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, PositionsTracker
from qubx.data.helpers import loader
from qubx.trackers.riskctrl import SignalRiskPositionTracker
from qubx.trackers.sizers import FixedSizer


class SignalsGenerator(IStrategy):
    actions = []
    _cmd_queue: deque = deque()
    _errors = []

    def on_start(self, ctx: IStrategyContext) -> None:
        self._cmd_queue = deque(self.actions)
        self._errors = []

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        signals = []
        s = ctx.instruments[0]

        if self._cmd_queue:
            _c_a = self._cmd_queue[0]
            _fn = None
            tm, cmd, pos, price, tk, stp = _c_a[:6]
            if len(_c_a) > 6:
                _fn = _c_a[6]

            if pd.Timestamp(tm) <= ctx.time():
                _d_str = f">>> <R>-{cmd.upper()}-</R> {s}"
                _res = ""
                self._cmd_queue.popleft()

                match cmd:
                    case "signal":
                        signals = [s := s.signal(ctx, pos, price, take=tk, stop=stp)]
                        _res = str(s)

                    case "check-active-targets":
                        for _s, _tp in ctx.get_active_targets().items():
                            _res += f"\n\t<G>{str(_tp)}</G>"
                        _res = _res or "<Y>NO TARGETS</Y>"

                    case "check-position":
                        _res = "position is " + str(ctx.get_position(s))

                    case "init-signal":
                        signals = [s := InitializingSignal(ctx.time(), s, pos, price, take=tk, stop=stp)]
                        _res = str(s)

                    case "emit-init-signal":
                        ctx.emit_signal(InitializingSignal(ctx.time(), s, pos, price, take=tk, stop=stp))

                    case "test-condition":
                        # - check any condition
                        pass

                    case _:
                        pass

                if _fn:
                    check = _fn(ctx, s)
                    control = "<G> PASSED </G>" if check else "<R> FAILED </R>"
                    logger.info(_d_str + " -> " + f"{_res}" + f" : {control}")
                    if not check:
                        self._errors.append((_c_a, control))
                else:
                    logger.info(_d_str + " -> " + f"{_res}")

        return signals  # type: ignore

    def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
        return SignalRiskPositionTracker(FixedSizer(10000.0))


class TestPostWarmupInitializationTestTargetsProcessing:
    def run_test(self, scenario: list, start: str, stop: str):
        simulate(
            {"signals_generator": (s := SignalsGenerator(actions=scenario))},
            {"ohlc(1h)": loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)},
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            n_jobs=1,
            debug="DEBUG",
            start=start,
            stop=stop,
        )
        return s

    def test_active_targets_processing(self):
        # fmt: off
        scenario = [
            ( # - open by signal
                "2023-06-03 23:59:59", "signal", 
                +1.0, None, None, None
            ),
            ( # - it should be active target for instrument
                "2023-06-04 01:00:00", "check-active-targets",
                None, None, None, None,
                lambda c, s: s in c.get_active_targets(),
            ),
            ( # - close by signal
                "2023-06-04 03:00:00", "signal", 
                +0.0, None, None, None
            ),
            ( # - no active targets
                "2023-06-04 04:00:00", "check-active-targets",
                None, None, None, None,
                lambda c, s: s not in c.get_active_targets(),
            ),
            ( # - open by signal with take
                "2023-06-05 11:00:00", "signal", +1.0, None, None, 26000.0
            ),
            ( # - target must be active
                "2023-06-05 13:00:00", "check-active-targets",
                None, None, None, None,
                lambda c, s: s in c.get_active_targets(), 
            ),

            ( # - position must be open
                "2023-06-05 14:00:00", "check-position",
                None, None, None, None,
                lambda c, s: c.get_position(s).is_open(), 
            ),
            ( # - no active target - must be closed by take
                "2023-06-06 00:00:00", "check-active-targets",
                None, None, None, None,
                lambda c, s: s not in c.get_active_targets(),
            ),
        ]
        # fmt: on

        s = self.run_test(scenario, "2023-06-01", "2023-08-01")
        assert not s._errors, "\n".join(list(map(str, s._errors)))

    def test_initilization_stage(self):
        # fmt: off
        scenario = [
            (  # send initializing signal
                "2023-06-06 12:00:00", "init-signal",
                +0.25, None, 26500.0, None
            ),

            (  # check position: it must be equal to size of signal 
                "2023-06-06 13:00:00", "check-position",
                None, None, None, None,
                lambda c, s: c.get_position(s).quantity == +0.25, 
            ),

            (  # check orders: should be one for take 
                "2023-06-06 14:00:00", "check-condition",
                None, None, None, None,
                lambda c, s: c.get_orders(s), 
            ),

            (  # check position: now it should be closed by take 
                "2023-06-06 18:00:00", "check-position",
                None, None, None, None,
                lambda c, s: c.get_position(s).quantity == 0.0, 
            ),
            
            ( # - now send standard signal
                "2023-06-06 19:00:00", "signal", 
                +1.0, None, None, None
            ),
            (  # check position: now it should be closed by take 
                "2023-06-06 20:00:00", "check-position",
                None, None, None, None,
                lambda c, s: c.get_position(s).quantity > 0.0, 
            ),
        ]
        # fmt: on

        s = self.run_test(scenario, "2023-06-06", "2023-06-08")
        assert not s._errors, "\n".join(list(map(str, s._errors)))

    def test_initilization_stage_reset_by_standard_signal(self):
        # fmt: off
        scenario = [
            (  # send initializing signal
                "2023-06-06 12:00:00", "init-signal",
                +0.25, None, 27500.0, None
            ),

            (  # check position: it must be equal to size of signal 
                "2023-06-06 13:00:00", "check-position",
                None, None, None, None,
                lambda c, s: c.get_position(s).quantity == +0.25, 
            ),

            (  # check orders: should be one for take 
                "2023-06-06 14:00:00", "check-condition",
                None, None, None, None,
                lambda c, s: c.get_orders(s), 
            ),

            (  # send standard signal - it should reset initialization stage
                "2023-06-07 00:00:00", "signal",
                -1, None, None, None
            ),

            (  # check orders: init take should be canceled
                "2023-06-07 01:00:00", "Is Open Orders",
                None, None, None, None,
                lambda c, s: not c.get_orders(s), 
            ),
            (  # check position: short must be open
                "2023-06-07 01:00:00", "check-position",
                None, None, None, None,
                lambda c, s: c.get_position(s).quantity < 0.0, 
            ),
        ]
        # fmt: on

        s = self.run_test(scenario, "2023-06-06", "2023-06-08")
        assert not s._errors, "\n".join(list(map(str, s._errors)))

    def test_initilization_stage_attempts_when_active_target_is_present(self):
        # fmt: off
        scenario = [
            (  # send standard signal - it opens position
                "2023-06-06 12:00:00", "signal",
                +1, None, None, None
            ),

            (  # check position: it must be equal to size of signal 
                "2023-06-06 13:00:00", "check-position",
                None, None, None, None,
                lambda c, s: c.get_position(s).is_open(), 
            ),

            (  # send initializing signal - it should be skipped
                "2023-06-07 00:00:00", "init-signal",
                -1, None, None, None
            ),

            (  # check position: still long must be open as initialization stage is not started
                "2023-06-07 01:00:00", "check-position",
                None, None, None, None,
                lambda c, s: c.get_position(s).quantity > 0.0,
            ),
        ]
        # fmt: on

        s = self.run_test(scenario, "2023-06-06", "2023-06-08")
        assert not s._errors, "\n".join(list(map(str, s._errors)))
