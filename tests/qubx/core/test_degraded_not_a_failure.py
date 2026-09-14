import numpy as np
import pytest

from qubx.core.basics import TriggerEvent
from qubx.core.exceptions import QubxDegradedState, StrategyExceededMaxNumberOfRuntimeFailuresError
from qubx.core.mixins.processing import ProcessingManager
from tests.qubx.core.mixins.boot_pipeline_test import drive, make_pm

T0 = np.datetime64("2025-01-01T00:00:00", "ns")


def _trigger_event() -> TriggerEvent:
    return TriggerEvent(time=T0, type="time", instrument=None, data=None)


def test_degraded_state_does_not_stop_the_run():
    pm, _, strategy = make_pm()
    drive(pm)  # boot to trading
    strategy.on_event.side_effect = QubxDegradedState("venue in maintenance", ())

    event = _trigger_event()
    for _ in range(ProcessingManager.MAX_NUMBER_OF_STRATEGY_FAILURES * 2):
        pm._run_strategy_pipeline(event)

    assert pm._fails_counter == 0


def test_other_exceptions_still_stop_the_run():
    pm, _, strategy = make_pm()
    drive(pm)  # boot to trading
    strategy.on_event.side_effect = ValueError("real bug")

    event = _trigger_event()
    with pytest.raises(StrategyExceededMaxNumberOfRuntimeFailuresError):
        for _ in range(ProcessingManager.MAX_NUMBER_OF_STRATEGY_FAILURES):
            pm._run_strategy_pipeline(event)
