from unittest.mock import Mock, patch

import pandas as pd
import pytest

from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.core.errors import OrderCreationError
from qubx.health.dummy import DummyHealthMonitor

# Try to import qubx_lighter - skip tests if not available
try:
    from qubx_lighter.broker import LighterBroker

    HAS_QUBX_LIGHTER = True
except ImportError:
    HAS_QUBX_LIGHTER = False
    LighterBroker = None  # type: ignore


def _create_mock_account_manager():
    """Create a mock account configuration manager."""
    account_manager = Mock()
    creds = Mock()
    creds.api_key = "test_api_key"
    creds.secret = "test_secret"
    creds.testnet = False
    account_manager.get_exchange_credentials = Mock(return_value=creds)
    return account_manager


def _create_mock_exchange_manager():
    """Create a mock exchange manager."""
    exchange = Mock()
    exchange.name = "binanceusdm"
    exchange.asyncio_loop = Mock()
    exchange_manager = Mock()
    exchange_manager.exchange = exchange
    return exchange_manager


def test_ccxt_order_creation_error_includes_client_id():
    channel = Mock()
    time_provider = Mock()
    time_provider.time.return_value = pd.Timestamp("2024-01-01").asm8

    account_manager = _create_mock_account_manager()
    exchange_manager = _create_mock_exchange_manager()

    with patch("qubx.connectors.ccxt.factory.get_ccxt_exchange_manager", return_value=exchange_manager):
        broker = CcxtBroker(
            exchange_name="BINANCE.UM",
            channel=channel,
            time_provider=time_provider,
            account=Mock(),
            data_provider=Mock(),
            account_manager=account_manager,
            health_monitor=DummyHealthMonitor(),
        )

    instrument = Mock()
    instrument.symbol = "XRPUSDT"

    broker._post_order_error_to_databus(
        Exception("rejected"),
        instrument,
        "SELL",
        "LIMIT",
        100.0,
        1.0,
        "cid_123",
        "gtx",
    )

    assert channel.send.called
    sent = channel.send.call_args[0][0]
    assert isinstance(sent[2], OrderCreationError)
    assert sent[2].client_id == "cid_123"


def _create_mock_lighter_account_manager():
    """Create a mock account configuration manager for XLighter."""
    account_manager = Mock()
    creds = Mock()
    creds.api_key = "test_api_key"
    creds.secret = "test_secret"
    creds.base_currency = "USDC"
    creds.get_extra_field = Mock(side_effect=lambda key, default=None: {"account_index": 123, "api_key_index": 0}.get(key, default))
    account_manager.get_exchange_credentials = Mock(return_value=creds)
    settings = Mock()
    settings.testnet = False
    account_manager.get_exchange_settings = Mock(return_value=settings)
    return account_manager


@pytest.mark.skipif(not HAS_QUBX_LIGHTER, reason="qubx_lighter package not installed")
def test_lighter_order_creation_error_includes_client_id():
    from unittest.mock import patch

    channel = Mock()
    time_provider = Mock()
    time_provider.time.return_value = pd.Timestamp("2024-01-01").asm8

    mock_client = Mock()
    mock_client._loop = Mock()
    mock_client.testnet = False

    account_manager = _create_mock_lighter_account_manager()

    with patch("qubx_lighter.broker.get_lighter_client", return_value=mock_client):
        with patch("qubx_lighter.broker.get_lighter_ws_manager", return_value=Mock()):
            broker = LighterBroker(
                exchange_name="XLIGHTER",
                channel=channel,
                time_provider=time_provider,
                account=Mock(),
                data_provider=Mock(),
                account_manager=account_manager,
                health_monitor=DummyHealthMonitor(),
            )

    instrument = Mock()
    instrument.symbol = "XRPUSDT"

    broker._post_order_error_to_channel(
        Exception("rejected"),
        instrument,
        "SELL",
        "LIMIT",
        100.0,
        1.0,
        "cid_123",
        "gtx",
    )

    assert channel.send.called
    sent = channel.send.call_args[0][0]
    assert isinstance(sent[2], OrderCreationError)
    assert sent[2].client_id == "cid_123"
