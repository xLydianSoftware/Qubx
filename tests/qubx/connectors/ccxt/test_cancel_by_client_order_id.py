from unittest.mock import AsyncMock, MagicMock, Mock, patch

import ccxt
import pytest

from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.core.basics import CtrlChannel
from qubx.health.dummy import DummyHealthMonitor
from tests.qubx.core.utils_test import DummyTimeProvider


def _create_mock_exchange():
    """Create a mock CCXT exchange with commonly used methods."""
    exchange = Mock()
    exchange.name = "binanceusdm"
    exchange.asyncio_loop = Mock()
    exchange.cancel_order = AsyncMock(return_value={})
    exchange.cancel_order_ws = AsyncMock(return_value={})
    exchange.cancel_order_with_client_order_id = AsyncMock(return_value={})
    return exchange


def _create_mock_exchange_manager(exchange):
    """Create a mock exchange manager with the given exchange."""
    exchange_manager = Mock()
    exchange_manager.exchange = exchange
    return exchange_manager


def _create_mock_account_manager():
    """Create a mock account configuration manager."""
    account_manager = Mock()
    creds = Mock()
    creds.api_key = "test_api_key"
    creds.secret = "test_secret"
    creds.testnet = False
    account_manager.get_exchange_credentials = Mock(return_value=creds)
    return account_manager


@pytest.mark.asyncio
async def test_cancel_with_exchange_order_id_calls_cancel_order() -> None:
    exchange = _create_mock_exchange()
    exchange_manager = _create_mock_exchange_manager(exchange)
    account_manager = _create_mock_account_manager()

    with patch("qubx.connectors.ccxt.factory.get_ccxt_exchange_manager", return_value=exchange_manager):
        broker = CcxtBroker(
            exchange_name="BINANCE.UM",
            channel=Mock(spec=CtrlChannel),
            time_provider=DummyTimeProvider(),
            account=Mock(),
            data_provider=Mock(),
            account_manager=account_manager,
            health_monitor=DummyHealthMonitor(),
            enable_cancel_order_ws=False,
        )

    instrument = Mock()
    instrument.exchange = "BINANCE.UM"
    instrument.symbol = "BTCUSDT"

    ok = await broker._cancel_order_with_retry(order_id="123", client_order_id=None, instrument=instrument)
    assert ok is True
    exchange.cancel_order.assert_awaited()
    exchange.cancel_order_with_client_order_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_with_client_order_id_calls_cancel_order_with_client_order_id() -> None:
    exchange = _create_mock_exchange()
    exchange_manager = _create_mock_exchange_manager(exchange)
    account_manager = _create_mock_account_manager()

    with patch("qubx.connectors.ccxt.factory.get_ccxt_exchange_manager", return_value=exchange_manager):
        broker = CcxtBroker(
            exchange_name="BINANCE.UM",
            channel=Mock(spec=CtrlChannel),
            time_provider=DummyTimeProvider(),
            account=Mock(),
            data_provider=Mock(),
            account_manager=account_manager,
            health_monitor=DummyHealthMonitor(),
            enable_cancel_order_ws=False,
        )

    instrument = Mock()
    instrument.exchange = "BINANCE.UM"
    instrument.symbol = "BTCUSDT"

    ok = await broker._cancel_order_with_retry(order_id=None, client_order_id="client_1", instrument=instrument)
    assert ok is True
    exchange.cancel_order_with_client_order_id.assert_awaited()


@pytest.mark.asyncio
async def test_cancel_with_client_order_id_fast_fails_on_missing_order_id_error() -> None:
    exchange = _create_mock_exchange()
    exchange.cancel_order_with_client_order_id = AsyncMock(
        side_effect=ccxt.BadRequest(
            'binanceusdm {"code":-1102,"msg":"Mandatory parameter \'orderId\' was not sent, was empty/null, or malformed."}'
        )
    )
    exchange_manager = _create_mock_exchange_manager(exchange)
    account_manager = _create_mock_account_manager()

    with patch("qubx.connectors.ccxt.factory.get_ccxt_exchange_manager", return_value=exchange_manager):
        broker = CcxtBroker(
            exchange_name="BINANCE.UM",
            channel=Mock(spec=CtrlChannel),
            time_provider=DummyTimeProvider(),
            account=Mock(),
            data_provider=Mock(),
            account_manager=account_manager,
            health_monitor=DummyHealthMonitor(),
            enable_cancel_order_ws=False,
        )

    instrument = Mock()
    instrument.exchange = "BINANCE.UM"
    instrument.symbol = "BTCUSDT"

    ok = await broker._cancel_order_with_retry(order_id=None, client_order_id="client_2", instrument=instrument)
    assert ok is False
    assert exchange.cancel_order_with_client_order_id.await_count == 1
