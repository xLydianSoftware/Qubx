from pytest_mock import MockerFixture

from qubx.core.basics import Instrument
from qubx.core.context import StrategyContext


def test_is_instrument_listed_delegates_to_market_manager(mocker: MockerFixture):
    ctx = StrategyContext.__new__(StrategyContext)  # bypass heavy __init__
    mm = mocker.Mock()
    mm.is_instrument_listed.return_value = False
    ctx._market_data_provider = mm
    instr = mocker.Mock(spec=Instrument)

    result = ctx.is_instrument_listed(instr)

    assert result is False
    mm.is_instrument_listed.assert_called_once_with(instr)
