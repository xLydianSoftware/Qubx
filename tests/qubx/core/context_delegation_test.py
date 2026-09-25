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


def test_get_collateral_equity_delegates_to_account(mocker: MockerFixture):
    ctx = StrategyContext.__new__(StrategyContext)  # bypass heavy __init__
    ctx.account = mocker.Mock()
    ctx.account.get_collateral_equity.return_value = 4200.0

    assert ctx.get_collateral_equity("BINANCE.UM") == 4200.0
    ctx.account.get_collateral_equity.assert_called_once_with("BINANCE.UM")


def test_convert_currency_is_barred_from_the_fit_thread(mocker: MockerFixture):
    """A blocking venue write from a threaded on_fit would mutate ProcessorThread state."""
    ctx = StrategyContext.__new__(StrategyContext)
    ctx._trading_manager = mocker.Mock()
    ctx._fit_state = mocker.Mock()
    ctx._fit_state.is_fit_thread.return_value = True

    with pytest.raises(RuntimeError, match="fit thread"):
        ctx.convert_currency("BINANCE.UM", "USDC", "USDT", 100.0)

    ctx._trading_manager.convert_currency.assert_not_called()


def test_move_funds_delegates_to_trading_manager(mocker: MockerFixture):
    ctx = StrategyContext.__new__(StrategyContext)  # bypass heavy __init__
    tm = mocker.Mock()
    tm.move_funds.return_value = "mv-1"
    ctx._trading_manager = tm
    ctx._fit_state = mocker.Mock()
    ctx._fit_state.is_fit_thread.return_value = False

    assert ctx.move_funds("BINANCE.PM", "USDT", "futures_um", "margin", 100.0) == "mv-1"
    tm.move_funds.assert_called_once_with("BINANCE.PM", "USDT", "futures_um", "margin", 100.0)


def test_move_funds_is_barred_from_the_fit_thread(mocker: MockerFixture):
    ctx = StrategyContext.__new__(StrategyContext)
    ctx._trading_manager = mocker.Mock()
    ctx._fit_state = mocker.Mock()
    ctx._fit_state.is_fit_thread.return_value = True

    with pytest.raises(RuntimeError, match="fit thread"):
        ctx.move_funds("BINANCE.PM", None, "futures_um", "margin")

    ctx._trading_manager.move_funds.assert_not_called()


def test_wallet_moves_delegates_to_trading_manager(mocker: MockerFixture):
    ctx = StrategyContext.__new__(StrategyContext)
    tm = mocker.Mock()
    tm.wallet_moves.return_value = []
    ctx._trading_manager = tm

    assert ctx.wallet_moves("BINANCE.PM") == []
    tm.wallet_moves.assert_called_once_with("BINANCE.PM")


def test_repay_debt_delegates_to_trading_manager(mocker: MockerFixture):
    ctx = StrategyContext.__new__(StrategyContext)  # bypass heavy __init__
    tm = mocker.Mock()
    tm.repay_debt.return_value = "rp-1"
    ctx._trading_manager = tm
    ctx._fit_state = mocker.Mock()
    ctx._fit_state.is_fit_thread.return_value = False

    assert ctx.repay_debt("BINANCE.PM", "USDT", 3.5) == "rp-1"
    tm.repay_debt.assert_called_once_with("BINANCE.PM", "USDT", 3.5)


def test_repay_debt_is_barred_from_the_fit_thread(mocker: MockerFixture):
    ctx = StrategyContext.__new__(StrategyContext)
    ctx._trading_manager = mocker.Mock()
    ctx._fit_state = mocker.Mock()
    ctx._fit_state.is_fit_thread.return_value = True

    with pytest.raises(RuntimeError, match="fit thread"):
        ctx.repay_debt("BINANCE.PM", "USDT")

    ctx._trading_manager.repay_debt.assert_not_called()


def test_debt_repayments_delegates_to_trading_manager(mocker: MockerFixture):
    ctx = StrategyContext.__new__(StrategyContext)
    tm = mocker.Mock()
    tm.debt_repayments.return_value = ["borrowed", "interest"]
    ctx._trading_manager = tm

    assert ctx.debt_repayments("BINANCE.PM") == ["borrowed", "interest"]
    tm.debt_repayments.assert_called_once_with("BINANCE.PM")
