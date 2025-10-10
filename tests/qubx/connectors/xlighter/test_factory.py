"""Tests for xlighter factory functions"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qubx.core.basics import CtrlChannel, LiveTimeProvider
from qubx.connectors.xlighter.factory import (
    create_xlighter_components,
    get_xlighter_account,
    get_xlighter_broker,
    get_xlighter_client,
    get_xlighter_data_provider,
)


class TestGetXLighterClient:
    """Test xlighter client factory"""

    @patch("qubx.connectors.xlighter.factory.LighterClient")
    def test_get_client(self, mock_client_cls):
        """Test creating a lighter client"""
        mock_client = MagicMock()
        mock_client.account_index = 12345
        mock_client.api_key_index = 1
        mock_client_cls.return_value = mock_client

        client = get_xlighter_client(
            api_key="0xTestAddress",
            secret="0xTestPrivateKey",
            account_index=12345,
            api_key_index=1,
        )

        assert client is not None
        assert client.account_index == 12345
        assert client.api_key_index == 1

        # Verify client created with correct parameters
        mock_client_cls.assert_called_once_with(
            api_key="0xTestAddress",
            private_key="0xTestPrivateKey",
            account_index=12345,
            api_key_index=1,
        )

    @patch("qubx.connectors.xlighter.factory.LighterClient")
    def test_get_client_default_api_key_index(self, mock_client_cls):
        """Test creating client with default api_key_index"""
        mock_client = MagicMock()
        mock_client.api_key_index = 0
        mock_client_cls.return_value = mock_client

        client = get_xlighter_client(
            api_key="0xTestAddress",
            secret="0xTestPrivateKey",
            account_index=12345,
        )

        assert client is not None
        assert client.api_key_index == 0  # Default

        # Verify default api_key_index was used
        mock_client_cls.assert_called_once_with(
            api_key="0xTestAddress",
            private_key="0xTestPrivateKey",
            account_index=12345,
            api_key_index=0,
        )


class TestGetXLighterDataProvider:
    """Test xlighter data provider factory"""

    @patch("qubx.connectors.xlighter.instruments.LighterInstrumentLoader")
    @patch("qubx.connectors.xlighter.factory.LighterDataProvider")
    def test_get_data_provider(self, mock_dp_cls, mock_loader_cls):
        """Test creating a lighter data provider"""
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader.load_instruments = AsyncMock(return_value={})
        mock_loader_cls.return_value = mock_loader

        mock_dp = MagicMock()
        mock_dp_cls.return_value = mock_dp

        mock_client = MagicMock()
        mock_client.get_markets = AsyncMock(return_value=[])
        time_provider = LiveTimeProvider()
        channel = CtrlChannel("test")

        # Create data provider
        data_provider = get_xlighter_data_provider(
            client=mock_client,
            time_provider=time_provider,
            channel=channel,
        )

        assert data_provider is not None

        # Should create loader with client
        mock_loader_cls.assert_called_once_with(client=mock_client)

        # Should create data provider
        call_kwargs = mock_dp_cls.call_args.kwargs
        assert call_kwargs["client"] is mock_client
        assert call_kwargs["time_provider"] is time_provider
        assert call_kwargs["channel"] is channel


class TestGetXLighterAccount:
    """Test xlighter account processor factory"""

    @patch("qubx.connectors.xlighter.factory.LighterAccountProcessor")
    @patch("qubx.connectors.xlighter.factory.LighterWebSocketManager", create=True)
    @patch("qubx.connectors.xlighter.factory.LighterInstrumentLoader", create=True)
    def test_get_account(self, mock_loader_cls, mock_ws_cls, mock_account_cls):
        """Test creating a lighter account processor"""
        # Setup mocks
        mock_account = MagicMock()
        mock_account_cls.return_value = mock_account

        mock_loader = MagicMock()
        mock_loader.load_instruments = AsyncMock(return_value={})
        mock_loader_cls.return_value = mock_loader

        mock_ws = MagicMock()
        mock_ws_cls.return_value = mock_ws

        mock_client = MagicMock()
        mock_client.account_index = 12345
        mock_client.get_markets = AsyncMock(return_value=[])
        time_provider = LiveTimeProvider()
        channel = CtrlChannel("test")

        account = get_xlighter_account(
            client=mock_client,
            channel=channel,
            time_provider=time_provider,
            base_currency="USDC",
            initial_capital=50000.0,
        )

        assert account is not None

        # Verify account created with correct parameters
        call_kwargs = mock_account_cls.call_args.kwargs
        assert call_kwargs["client"] is mock_client
        assert call_kwargs["channel"] is channel
        assert call_kwargs["time_provider"] is time_provider
        assert call_kwargs["base_currency"] == "USDC"
        assert call_kwargs["initial_capital"] == 50000.0
        assert call_kwargs["account_id"] == "12345"

    @patch("qubx.connectors.xlighter.factory.LighterAccountProcessor")
    @patch("qubx.connectors.xlighter.factory.LighterWebSocketManager", create=True)
    @patch("qubx.connectors.xlighter.factory.LighterInstrumentLoader", create=True)
    def test_get_account_default_values(self, mock_loader_cls, mock_ws_cls, mock_account_cls):
        """Test creating account with default values"""
        mock_account = MagicMock()
        mock_account_cls.return_value = mock_account

        mock_loader = MagicMock()
        mock_loader.load_instruments = AsyncMock(return_value={})
        mock_loader_cls.return_value = mock_loader

        mock_ws = MagicMock()
        mock_ws_cls.return_value = mock_ws

        mock_client = MagicMock()
        mock_client.account_index = 12345
        mock_client.get_markets = AsyncMock(return_value=[])
        time_provider = LiveTimeProvider()
        channel = CtrlChannel("test")

        account = get_xlighter_account(
            client=mock_client,
            channel=channel,
            time_provider=time_provider,
        )

        assert account is not None

        # Verify defaults were used
        call_kwargs = mock_account_cls.call_args.kwargs
        assert call_kwargs["base_currency"] == "USDC"
        assert call_kwargs["initial_capital"] == 100_000.0


class TestGetXLighterBroker:
    """Test xlighter broker factory"""

    @patch("qubx.connectors.xlighter.factory.LighterBroker")
    @patch("qubx.connectors.xlighter.factory.LighterInstrumentLoader", create=True)
    def test_get_broker(self, mock_loader_cls, mock_broker_cls):
        """Test creating a lighter broker"""
        # Setup mocks
        mock_broker = MagicMock()
        mock_broker_cls.return_value = mock_broker

        mock_loader = MagicMock()
        mock_loader.load_instruments = AsyncMock(return_value={})
        mock_loader_cls.return_value = mock_loader

        mock_client = MagicMock()
        mock_client.get_markets = AsyncMock(return_value=[])
        mock_account = MagicMock()

        # Import the real class to use isinstance check
        from qubx.connectors.xlighter.data import LighterDataProvider
        mock_data_provider = MagicMock(spec=LighterDataProvider)
        mock_data_provider.instrument_loader = mock_loader

        time_provider = LiveTimeProvider()
        channel = CtrlChannel("test")

        broker = get_xlighter_broker(
            client=mock_client,
            channel=channel,
            time_provider=time_provider,
            account=mock_account,
            data_provider=mock_data_provider,
        )

        assert broker is not None

        # Verify broker created with correct parameters
        call_kwargs = mock_broker_cls.call_args.kwargs
        assert call_kwargs["client"] is mock_client
        assert call_kwargs["channel"] is channel
        assert call_kwargs["time_provider"] is time_provider
        assert call_kwargs["account"] is mock_account
        assert call_kwargs["data_provider"] is mock_data_provider


class TestCreateXLighterComponents:
    """Test creating all components together"""

    @patch("qubx.connectors.xlighter.factory.get_xlighter_client")
    @patch("qubx.connectors.xlighter.factory.get_xlighter_data_provider")
    @patch("qubx.connectors.xlighter.factory.get_xlighter_account")
    @patch("qubx.connectors.xlighter.factory.get_xlighter_broker")
    def test_create_all_components(
        self, mock_broker_fn, mock_account_fn, mock_data_provider_fn, mock_client_fn
    ):
        """Test creating all components in correct order"""
        # Setup mocks
        mock_client = MagicMock()
        mock_data_provider = MagicMock()
        mock_account = MagicMock()
        mock_broker = MagicMock()

        mock_client_fn.return_value = mock_client
        mock_data_provider_fn.return_value = mock_data_provider
        mock_account_fn.return_value = mock_account
        mock_broker_fn.return_value = mock_broker

        # Create components
        time_provider = LiveTimeProvider()
        channel = CtrlChannel("test")

        client, data_provider, account, broker = create_xlighter_components(
            api_key="0xTestAddress",
            secret="0xTestPrivateKey",
            account_index=12345,
            api_key_index=1,
            channel=channel,
            time_provider=time_provider,
            base_currency="USDC",
            initial_capital=75000.0,
        )

        # Verify all components created
        assert client is mock_client
        assert data_provider is mock_data_provider
        assert account is mock_account
        assert broker is mock_broker

        # Verify correct factory calls
        mock_client_fn.assert_called_once()
        mock_data_provider_fn.assert_called_once()
        mock_account_fn.assert_called_once()
        mock_broker_fn.assert_called_once()

        # Verify client passed to other components
        data_provider_call = mock_data_provider_fn.call_args
        assert data_provider_call.kwargs["client"] is mock_client

        account_call = mock_account_fn.call_args
        assert account_call.kwargs["client"] is mock_client
        assert account_call.kwargs["base_currency"] == "USDC"
        assert account_call.kwargs["initial_capital"] == 75000.0

        broker_call = mock_broker_fn.call_args
        assert broker_call.kwargs["client"] is mock_client
        assert broker_call.kwargs["account"] is mock_account
        assert broker_call.kwargs["data_provider"] is mock_data_provider
