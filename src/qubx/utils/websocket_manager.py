"""
Generic WebSocket manager for exchange connectors.

Provides reusable WebSocket connection management with:
- Automatic reconnection with exponential backoff
- Channel multiplexing support
- Event-based subscription system
- Thread-safe operations
- Graceful shutdown handling
"""

import asyncio
import json
from asyncio.exceptions import CancelledError
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

from qubx import logger


class ConnectionState(Enum):
    """WebSocket connection states"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class ChannelSubscription:
    """Represents a channel subscription"""

    channel: str
    handler: Callable[[dict], Awaitable[None]]
    params: dict[str, Any]


@dataclass
class ReconnectionConfig:
    """Configuration for reconnection behavior"""

    enabled: bool = True
    max_retries: int = 10
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0


class BaseWebSocketManager:
    """
    Base WebSocket manager with reconnection and multiplexing support.

    This class provides generic WebSocket connection management that can be
    extended by specific exchange implementations.

    Example usage:
        ```python
        async def message_handler(msg: dict):
            print(f"Received: {msg}")

        manager = BaseWebSocketManager("wss://example.com/stream")
        await manager.connect()
        await manager.subscribe("channel_name", message_handler)

        # Keep connection alive
        await manager.wait_until_closed()
        ```
    """

    def __init__(
        self,
        url: str,
        reconnection_config: Optional[ReconnectionConfig] = None,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
    ):
        """
        Initialize WebSocket manager.

        Args:
            url: WebSocket URL to connect to
            reconnection_config: Configuration for reconnection behavior
            ping_interval: Interval between pings (seconds)
            ping_timeout: Timeout for ping response (seconds)
        """
        self.url = url
        self.reconnection_config = reconnection_config or ReconnectionConfig()
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        # Connection state
        self._ws: Optional[WebSocketClientProtocol] = None
        self._state = ConnectionState.DISCONNECTED
        self._lock = asyncio.Lock()

        # Subscription management
        self._subscriptions: dict[str, ChannelSubscription] = {}
        self._subscription_lock = asyncio.Lock()

        # Reconnection tracking
        self._retry_count = 0
        self._last_error: Optional[Exception] = None

        # Tasks
        self._listener_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    @property
    def state(self) -> ConnectionState:
        """Get current connection state"""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connection is established"""
        return self._state == ConnectionState.CONNECTED and self._ws is not None

    @property
    def subscriptions(self) -> list[str]:
        """Get list of active subscription channels"""
        return list(self._subscriptions.keys())

    async def connect(self) -> None:
        """
        Establish WebSocket connection.

        Raises:
            ConnectionError: If connection fails
        """
        async with self._lock:
            if self._state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
                logger.debug(f"Already connected or connecting (state: {self._state})")
                return

            self._state = ConnectionState.CONNECTING
            logger.info(f"Connecting to {self.url}")

            try:
                self._ws = await websockets.connect(
                    self.url, ping_interval=self.ping_interval, ping_timeout=self.ping_timeout
                )
                self._state = ConnectionState.CONNECTED
                self._retry_count = 0
                self._last_error = None
                logger.info(f"Connected to {self.url}")

                # Start listener task
                if self._listener_task is None or self._listener_task.done():
                    self._listener_task = asyncio.create_task(self._listen())

            except Exception as e:
                self._state = ConnectionState.DISCONNECTED
                self._last_error = e
                logger.error(f"Failed to connect: {e}")
                raise ConnectionError(f"Failed to connect to {self.url}") from e

    async def disconnect(self) -> None:
        """Gracefully disconnect from WebSocket"""
        async with self._lock:
            if self._state == ConnectionState.DISCONNECTED:
                return

            self._state = ConnectionState.CLOSING
            logger.info("Disconnecting from WebSocket")

            # Signal stop
            self._stop_event.set()

            # Cancel listener
            if self._listener_task and not self._listener_task.done():
                self._listener_task.cancel()
                try:
                    await self._listener_task
                except CancelledError:
                    pass

            # Close WebSocket
            if self._ws:
                await self._ws.close()
                self._ws = None

            self._state = ConnectionState.CLOSED
            logger.info("Disconnected from WebSocket")

    async def subscribe(
        self, channel: str, handler: Callable[[dict], Awaitable[None]], **params
    ) -> None:
        """
        Subscribe to a channel.

        Args:
            channel: Channel name/identifier
            handler: Async function to handle messages
            **params: Additional parameters for subscription

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to WebSocket")

        async with self._subscription_lock:
            if channel in self._subscriptions:
                logger.warning(f"Already subscribed to {channel}, updating handler")

            self._subscriptions[channel] = ChannelSubscription(
                channel=channel, handler=handler, params=params
            )
            logger.debug(f"Subscribed to channel: {channel}")

            # Send subscription message
            await self._send_subscription_message(channel, params)

    async def unsubscribe(self, channel: str) -> None:
        """
        Unsubscribe from a channel.

        Args:
            channel: Channel name/identifier
        """
        async with self._subscription_lock:
            if channel not in self._subscriptions:
                logger.warning(f"Not subscribed to {channel}")
                return

            # Send unsubscription message
            await self._send_unsubscription_message(channel)

            del self._subscriptions[channel]
            logger.debug(f"Unsubscribed from channel: {channel}")

    async def send(self, message: dict) -> None:
        """
        Send a message through the WebSocket.

        Args:
            message: Message to send (will be JSON encoded)

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected or not self._ws:
            raise ConnectionError("Not connected to WebSocket")

        await self._ws.send(json.dumps(message))
        logger.debug(f"Sent message: {message}")

    async def wait_until_closed(self) -> None:
        """Wait until connection is closed"""
        await self._stop_event.wait()

    async def _listen(self) -> None:
        """
        Main listener loop.

        Handles incoming messages and reconnection.
        """
        logger.debug("Starting listener loop")

        while not self._stop_event.is_set():
            try:
                if not self._ws:
                    logger.warning("WebSocket not connected, waiting...")
                    await asyncio.sleep(1)
                    continue

                # Receive message
                message = await self._ws.recv()
                await self._handle_message(message)

            except ConnectionClosedOK:
                logger.info("Connection closed gracefully")
                if not self._stop_event.is_set():
                    await self._handle_reconnection()
                else:
                    break

            except (ConnectionClosed, ConnectionClosedError) as e:
                logger.warning(f"Connection closed: {e}")
                if not self._stop_event.is_set():
                    await self._handle_reconnection()
                else:
                    break

            except CancelledError:
                logger.debug("Listener cancelled")
                break

            except Exception as e:
                logger.error(f"Error in listener loop: {e}")
                if not self._stop_event.is_set():
                    await self._handle_reconnection()
                else:
                    break

        logger.debug("Listener loop stopped")

    async def _handle_message(self, raw_message: str | bytes) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            raw_message: Raw message from WebSocket
        """
        try:
            # Parse message
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")

            message = json.loads(raw_message)

            # Route to appropriate handler
            channel = self._extract_channel(message)
            if channel and channel in self._subscriptions:
                subscription = self._subscriptions[channel]
                await subscription.handler(message)
            else:
                # Let subclass handle unknown messages
                await self._handle_unknown_message(message)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_reconnection(self) -> None:
        """Handle reconnection with exponential backoff"""
        if not self.reconnection_config.enabled:
            logger.info("Reconnection disabled, stopping")
            self._stop_event.set()
            return

        if self._retry_count >= self.reconnection_config.max_retries:
            logger.error(
                f"Max reconnection retries ({self.reconnection_config.max_retries}) reached, stopping"
            )
            self._stop_event.set()
            return

        self._retry_count += 1
        self._state = ConnectionState.RECONNECTING

        # Calculate backoff delay
        delay = min(
            self.reconnection_config.initial_delay
            * (self.reconnection_config.exponential_base ** (self._retry_count - 1)),
            self.reconnection_config.max_delay,
        )

        logger.info(
            f"Reconnecting in {delay:.1f}s (attempt {self._retry_count}/{self.reconnection_config.max_retries})"
        )
        await asyncio.sleep(delay)

        try:
            # Close old connection
            if self._ws:
                await self._ws.close()
                self._ws = None

            # Reconnect
            await self.connect()

            # Resubscribe to all channels
            await self._resubscribe_all()

            logger.info("Reconnection successful")

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self._last_error = e
            # Will retry in next iteration

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all channels after reconnection"""
        if not self._subscriptions:
            return

        logger.info(f"Resubscribing to {len(self._subscriptions)} channels")

        for channel, subscription in self._subscriptions.items():
            try:
                await self._send_subscription_message(channel, subscription.params)
                logger.debug(f"Resubscribed to {channel}")
            except Exception as e:
                logger.error(f"Failed to resubscribe to {channel}: {e}")

    async def _send_subscription_message(self, channel: str, params: dict[str, Any]) -> None:
        """
        Send subscription message to the exchange.

        This method should be overridden by subclasses to implement
        exchange-specific subscription protocol.

        Args:
            channel: Channel to subscribe to
            params: Subscription parameters
        """
        # Default implementation - override in subclass
        await self.send({"type": "subscribe", "channel": channel, **params})

    async def _send_unsubscription_message(self, channel: str) -> None:
        """
        Send unsubscription message to the exchange.

        This method should be overridden by subclasses to implement
        exchange-specific unsubscription protocol.

        Args:
            channel: Channel to unsubscribe from
        """
        # Default implementation - override in subclass
        await self.send({"type": "unsubscribe", "channel": channel})

    def _extract_channel(self, message: dict) -> Optional[str]:
        """
        Extract channel identifier from message.

        This method should be overridden by subclasses to implement
        exchange-specific message format.

        Args:
            message: Parsed message dict

        Returns:
            Channel identifier or None
        """
        # Default implementation - override in subclass
        return message.get("channel")

    async def _handle_unknown_message(self, message: dict) -> None:
        """
        Handle messages that don't match any subscription.

        This method can be overridden by subclasses to handle
        exchange-specific system messages (heartbeats, errors, etc.)

        Args:
            message: Parsed message dict
        """
        logger.debug(f"Unhandled message: {message}")
