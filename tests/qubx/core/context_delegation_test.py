import pytest
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


def test_convert_currency_delegates_to_trading_manager(mocker: MockerFixture):
    ctx = StrategyContext.__new__(StrategyContext)  # bypass heavy __init__
    tm = mocker.Mock()
    tm.convert_currency.return_value = "conv-1"
    ctx._trading_manager = tm
    ctx._fit_state = mocker.Mock()
    ctx._fit_state.is_fit_thread.return_value = False

    result = ctx.convert_currency("BINANCE.UM", "USDC", "USDT", 100.0, limit_price=0.995)

    assert result == "conv-1"
    tm.convert_currency.assert_called_once_with(
        "BINANCE.UM", "USDC", "USDT", 100.0, limit_price=0.995, max_slippage_bps=10.0
    )


def test_convert_currency_is_barred_from_the_fit_thread(mocker: MockerFixture):
    """A blocking venue write from a threaded on_fit would mutate ProcessorThread state."""
    ctx = StrategyContext.__new__(StrategyContext)
    ctx._trading_manager = mocker.Mock()
    ctx._fit_state = mocker.Mock()
    ctx._fit_state.is_fit_thread.return_value = True

    with pytest.raises(RuntimeError, match="fit thread"):
        ctx.convert_currency("BINANCE.UM", "USDC", "USDT", 100.0)

    ctx._trading_manager.convert_currency.assert_not_called()
