"""Tests for BaseWebSocketManager"""
import asyncio
import json

import pytest
import websockets
from websockets.server import serve

from qubx.utils.websocket_manager import (
    BaseWebSocketManager,
    ConnectionState,
    ReconnectionConfig,
)


class MockWebSocketServer:
    """Mock WebSocket server for testing"""

    def __init__(self, port: int = 0):
        self.port = port
        self.server = None
        self.clients = []
        self.messages_received = []
        self.should_close_connection = False
        self.response_messages = []

    async def handler(self, websocket):
        """Handle client connections"""
        self.clients.append(websocket)
        try:
            async for message in websocket:
                self.messages_received.append(json.loads(message))

                # Send responses if any
                for response in self.response_messages:
                    await websocket.send(json.dumps(response))
                self.response_messages.clear()

                # Simulate connection close if requested
                if self.should_close_connection:
                    await websocket.close()
                    break

        except Exception:
            pass
        finally:
            if websocket in self.clients:
                self.clients.remove(websocket)

    async def start(self):
        """Start the mock server"""
        self.server = await serve(self.handler, "127.0.0.1", self.port)
        # Get actual port if port was 0 (auto-assign)
        if self.port == 0:
            self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self):
        """Stop the mock server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def send_to_all(self, message: dict):
        """Send message to all connected clients"""
        for client in self.clients:
            await client.send(json.dumps(message))


@pytest.fixture
async def mock_server():
    """Fixture providing a mock WebSocket server"""
    server = MockWebSocketServer()
    await server.start()
    yield server
    await server.stop()


@pytest.mark.asyncio
class TestBaseWebSocketManager:
    """Test BaseWebSocketManager functionality"""

    async def test_connect(self, mock_server):
        """Test basic connection"""
        manager = BaseWebSocketManager(f"ws://localhost:{mock_server.port}")

        assert manager.state == ConnectionState.DISCONNECTED
        assert not manager.is_connected

        await manager.connect()

        assert manager.state == ConnectionState.CONNECTED
        assert manager.is_connected
        assert len(mock_server.clients) == 1

        await manager.disconnect()

    async def test_send_message(self, mock_server):
        """Test sending messages"""
        manager = BaseWebSocketManager(f"ws://localhost:{mock_server.port}")
        await manager.connect()

        test_message = {"type": "test", "data": "hello"}
        await manager.send(test_message)

        # Give server time to receive
        await asyncio.sleep(0.1)

        assert len(mock_server.messages_received) == 1
        assert mock_server.messages_received[0] == test_message

        await manager.disconnect()

    async def test_subscribe(self, mock_server):
        """Test subscription mechanism"""
        manager = BaseWebSocketManager(f"ws://localhost:{mock_server.port}")
        await manager.connect()

        received_messages = []

        async def handler(msg: dict):
            received_messages.append(msg)

        await manager.subscribe("test_channel", handler)

        assert "test_channel" in manager.subscriptions

        # Verify subscription message was sent
        await asyncio.sleep(0.1)
        assert len(mock_server.messages_received) >= 1
        sub_msg = mock_server.messages_received[-1]
        assert sub_msg["type"] == "subscribe"
        assert sub_msg["channel"] == "test_channel"

        await manager.disconnect()

    async def test_receive_message(self, mock_server):
        """Test receiving and routing messages"""
        manager = BaseWebSocketManager(f"ws://localhost:{mock_server.port}")
        await manager.connect()

        received_messages = []

        async def handler(msg: dict):
            received_messages.append(msg)

        await manager.subscribe("test_channel", handler)
        await asyncio.sleep(0.1)

        # Send message from server
        test_message = {"channel": "test_channel", "data": "test_data"}
        await mock_server.send_to_all(test_message)

        # Wait for message to be processed
        await asyncio.sleep(0.2)

        assert len(received_messages) == 1
        assert received_messages[0] == test_message

        await manager.disconnect()

    async def test_unsubscribe(self, mock_server):
        """Test unsubscription"""
        manager = BaseWebSocketManager(f"ws://localhost:{mock_server.port}")
        await manager.connect()

        async def handler(msg: dict):
            pass

        await manager.subscribe("test_channel", handler)
        assert "test_channel" in manager.subscriptions

        await manager.unsubscribe("test_channel")

        assert "test_channel" not in manager.subscriptions

        # Verify unsubscribe message was sent
        await asyncio.sleep(0.1)
        unsub_msg = mock_server.messages_received[-1]
        assert unsub_msg["type"] == "unsubscribe"
        assert unsub_msg["channel"] == "test_channel"

        await manager.disconnect()

    async def test_disconnect(self, mock_server):
        """Test graceful disconnection"""
        manager = BaseWebSocketManager(f"ws://localhost:{mock_server.port}")
        await manager.connect()

        assert manager.is_connected

        await manager.disconnect()

        assert manager.state == ConnectionState.CLOSED
        assert not manager.is_connected

    async def test_reconnection_disabled(self, mock_server):
        """Test behavior when reconnection is disabled"""
        config = ReconnectionConfig(enabled=False)
        manager = BaseWebSocketManager(
            f"ws://localhost:{mock_server.port}", reconnection=config
        )

        await manager.connect()
        assert manager.is_connected

        # Force close from client side to trigger reconnection logic
        if manager._ws:
            await manager._ws.close()

        # Wait for the listener to detect the closure
        await asyncio.sleep(0.5)

        # With reconnection disabled, stop event should be set
        assert manager._stop_event.is_set()
        # Should not be connected
        assert not manager.is_connected

        await manager.disconnect()

    async def test_reconnection_with_backoff(self, mock_server):
        """Test reconnection with exponential backoff"""
        config = ReconnectionConfig(
            enabled=True, max_retries=3, initial_delay=0.1, max_delay=1.0, exponential_base=2.0
        )
        manager = BaseWebSocketManager(
            f"ws://localhost:{mock_server.port}", reconnection=config
        )

        await manager.connect()
        initial_connection_count = len(mock_server.clients)

        # Close connection from server side
        if mock_server.clients:
            await mock_server.clients[0].close()

        # Wait for reconnection attempt
        await asyncio.sleep(0.5)

        # Should have reconnected (new connection)
        assert manager.state in [ConnectionState.CONNECTED, ConnectionState.RECONNECTING]

        await manager.disconnect()

    async def test_resubscribe_after_reconnection(self, mock_server):
        """Test that subscriptions are restored after reconnection"""
        config = ReconnectionConfig(enabled=True, max_retries=2, initial_delay=0.1, jitter=0.0)
        manager = BaseWebSocketManager(
            f"ws://localhost:{mock_server.port}", reconnection=config
        )

        await manager.connect()

        received_messages = []

        async def handler(msg: dict):
            received_messages.append(msg)

        await manager.subscribe("test_channel", handler)
        await asyncio.sleep(0.1)

        # Clear received messages
        mock_server.messages_received.clear()

        # Close connection from server
        if mock_server.clients:
            await mock_server.clients[0].close()

        # Poll for reconnection and resubscription with timeout
        max_wait = 2.0  # seconds
        poll_interval = 0.1  # seconds
        elapsed = 0.0

        while elapsed < max_wait:
            subscribe_messages = [msg for msg in mock_server.messages_received if msg.get("type") == "subscribe"]
            if len(subscribe_messages) >= 1:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Check that resubscription happened
        # Should have subscribe message again
        subscribe_messages = [msg for msg in mock_server.messages_received if msg.get("type") == "subscribe"]
        assert len(subscribe_messages) >= 1

        await manager.disconnect()

    async def test_multiple_subscriptions(self, mock_server):
        """Test managing multiple channel subscriptions"""
        manager = BaseWebSocketManager(f"ws://localhost:{mock_server.port}")
        await manager.connect()

        channel1_messages = []
        channel2_messages = []

        async def handler1(msg: dict):
            channel1_messages.append(msg)

        async def handler2(msg: dict):
            channel2_messages.append(msg)

        await manager.subscribe("channel1", handler1)
        await manager.subscribe("channel2", handler2)

        assert len(manager.subscriptions) == 2
        assert "channel1" in manager.subscriptions
        assert "channel2" in manager.subscriptions

        await asyncio.sleep(0.1)

        # Send messages to different channels
        await mock_server.send_to_all({"channel": "channel1", "data": "msg1"})
        await mock_server.send_to_all({"channel": "channel2", "data": "msg2"})

        await asyncio.sleep(0.2)

        assert len(channel1_messages) == 1
        assert len(channel2_messages) == 1
        assert channel1_messages[0]["data"] == "msg1"
        assert channel2_messages[0]["data"] == "msg2"

        await manager.disconnect()

    async def test_send_without_connection(self):
        """Test that sending without connection raises error"""
        manager = BaseWebSocketManager("ws://localhost:9999")

        with pytest.raises(ConnectionError):
            await manager.send({"type": "test"})

    async def test_subscribe_without_connection(self):
        """Test that subscribing without connection raises error"""
        manager = BaseWebSocketManager("ws://localhost:9999")

        async def handler(msg: dict):
            pass

        with pytest.raises(ConnectionError):
            await manager.subscribe("channel", handler)


@pytest.mark.asyncio
class TestReconnectionConfig:
    """Test ReconnectionConfig"""

    def test_default_config(self):
        """Test default reconnection configuration"""
        config = ReconnectionConfig()
        assert config.enabled is True
        assert config.max_retries == 10
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0

    def test_custom_config(self):
        """Test custom reconnection configuration"""
        config = ReconnectionConfig(
            enabled=False, max_retries=5, initial_delay=0.5, max_delay=30.0, exponential_base=1.5
        )
        assert config.enabled is False
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
