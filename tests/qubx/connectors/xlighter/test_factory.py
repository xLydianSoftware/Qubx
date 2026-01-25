"""Tests for xlighter factory functions"""

from unittest.mock import MagicMock, patch

from qubx.connectors.xlighter.factory import (
    clear_lighter_cache,
    get_lighter_client,
    get_lighter_instrument_loader,
    get_lighter_ws_manager,
)


class TestGetLighterClient:
    """Test lighter client factory with caching"""

    def setup_method(self):
        """Clear cache before each test"""
        clear_lighter_cache()

    def teardown_method(self):
        """Clear cache after each test"""
        clear_lighter_cache()

    @patch("qubx.connectors.xlighter.factory.LighterClient")
    def test_get_client(self, mock_client_cls):
        """Test creating a lighter client"""
        mock_client = MagicMock()
        mock_client.account_index = 12345
        mock_client.api_key_index = 1
        mock_client_cls.return_value = mock_client

        client = get_lighter_client(
            api_key="0xTestAddress",
            private_key="0xTestPrivateKey",
            account_index=12345,
            api_key_index=1,
        )

        assert client is not None
        assert client.account_index == 12345
        assert client.api_key_index == 1

        mock_client_cls.assert_called_once_with(
            api_key="0xTestAddress",
            private_key="0xTestPrivateKey",
            account_index=12345,
            api_key_index=1,
            testnet=False,
            loop=None,
        )

    @patch("qubx.connectors.xlighter.factory.LighterClient")
    def test_get_client_caching(self, mock_client_cls):
        """Test that clients are cached and reused"""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        client1 = get_lighter_client(
            api_key="0xTestAddress",
            private_key="0xTestPrivateKey",
            account_index=12345,
        )
        client2 = get_lighter_client(
            api_key="0xTestAddress",
            private_key="0xTestPrivateKey",
            account_index=12345,
        )

        assert client1 is client2
        assert mock_client_cls.call_count == 1

    @patch("qubx.connectors.xlighter.factory.LighterClient")
    def test_different_params_create_different_clients(self, mock_client_cls):
        """Test that different parameters create separate clients"""
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        mock_client_cls.side_effect = [mock_client1, mock_client2]

        client1 = get_lighter_client(
            api_key="0xTestAddress1",
            private_key="0xTestPrivateKey1",
            account_index=12345,
        )
        client2 = get_lighter_client(
            api_key="0xTestAddress2",
            private_key="0xTestPrivateKey2",
            account_index=67890,
        )

        assert client1 is not client2
        assert mock_client_cls.call_count == 2


class TestGetLighterWsManager:
    """Test lighter WebSocket manager factory with caching"""

    def setup_method(self):
        """Clear cache before each test"""
        clear_lighter_cache()

    def teardown_method(self):
        """Clear cache after each test"""
        clear_lighter_cache()

    @patch("qubx.connectors.xlighter.factory.LighterWebSocketManager")
    @patch("qubx.connectors.xlighter.factory.LighterClient")
    def test_get_ws_manager(self, mock_client_cls, mock_ws_cls):
        """Test creating a WebSocket manager"""
        mock_client = MagicMock()
        mock_client.testnet = False
        mock_client_cls.return_value = mock_client

        mock_ws = MagicMock()
        mock_ws_cls.return_value = mock_ws

        ws_manager = get_lighter_ws_manager(
            api_key="0xTestAddress",
            private_key="0xTestPrivateKey",
            account_index=12345,
        )

        assert ws_manager is not None
        mock_ws_cls.assert_called_once()

    @patch("qubx.connectors.xlighter.factory.LighterWebSocketManager")
    @patch("qubx.connectors.xlighter.factory.LighterClient")
    def test_ws_manager_caching(self, mock_client_cls, mock_ws_cls):
        """Test that WebSocket managers are cached and reused"""
        mock_client = MagicMock()
        mock_client.testnet = False
        mock_client_cls.return_value = mock_client

        mock_ws = MagicMock()
        mock_ws_cls.return_value = mock_ws

        ws1 = get_lighter_ws_manager(
            api_key="0xTestAddress",
            private_key="0xTestPrivateKey",
            account_index=12345,
        )
        ws2 = get_lighter_ws_manager(
            api_key="0xTestAddress",
            private_key="0xTestPrivateKey",
            account_index=12345,
        )

        assert ws1 is ws2
        assert mock_ws_cls.call_count == 1


class TestGetLighterInstrumentLoader:
    """Test lighter instrument loader factory with caching"""

    def setup_method(self):
        """Clear cache before each test"""
        clear_lighter_cache()

    def teardown_method(self):
        """Clear cache after each test"""
        clear_lighter_cache()

    @patch("qubx.connectors.xlighter.factory.LighterInstrumentLoader")
    def test_get_instrument_loader(self, mock_loader_cls):
        """Test creating an instrument loader"""
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader

        loader = get_lighter_instrument_loader()

        assert loader is not None
        mock_loader_cls.assert_called_once()

    @patch("qubx.connectors.xlighter.factory.LighterInstrumentLoader")
    def test_instrument_loader_caching(self, mock_loader_cls):
        """Test that instrument loaders are cached and reused"""
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader

        loader1 = get_lighter_instrument_loader()
        loader2 = get_lighter_instrument_loader()

        assert loader1 is loader2
        assert mock_loader_cls.call_count == 1


class TestClearCache:
    """Test cache clearing functionality"""

    @patch("qubx.connectors.xlighter.factory.LighterClient")
    def test_clear_cache(self, mock_client_cls):
        """Test that clearing cache allows new instances to be created"""
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        mock_client_cls.side_effect = [mock_client1, mock_client2]

        client1 = get_lighter_client(
            api_key="0xTestAddress",
            private_key="0xTestPrivateKey",
            account_index=12345,
        )

        clear_lighter_cache()

        client2 = get_lighter_client(
            api_key="0xTestAddress",
            private_key="0xTestPrivateKey",
            account_index=12345,
        )

        assert client1 is not client2
        assert mock_client_cls.call_count == 2
