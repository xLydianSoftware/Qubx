from pytest_mock import MockerFixture

from qubx.core.basics import Instrument
from qubx.core.mixins.market import MarketManager


def _instr(mocker: MockerFixture, exchange: str = "BINANCE.UM"):
    i = mocker.Mock(spec=Instrument)
    i.exchange = exchange
    i.symbol = "TONUSDT"
    return i


def _market_manager_with(provider):
    mm = MarketManager.__new__(MarketManager)  # bypass heavy __init__
    mm._exchange_to_data_provider = {} if provider is None else {"BINANCE.UM": provider}
    return mm


def test_listed_delegates_to_provider_true(mocker: MockerFixture):
    dp = mocker.Mock()
    dp.is_instrument_listed.return_value = True
    mm = _market_manager_with(dp)
    assert mm.is_instrument_listed(_instr(mocker)) is True


def test_not_listed_delegates_to_provider_false(mocker: MockerFixture):
    dp = mocker.Mock()
    dp.is_instrument_listed.return_value = False
    mm = _market_manager_with(dp)
    assert mm.is_instrument_listed(_instr(mocker)) is False


def test_fail_open_when_no_provider(mocker: MockerFixture):
    mm = _market_manager_with(None)
    assert mm.is_instrument_listed(_instr(mocker)) is True
