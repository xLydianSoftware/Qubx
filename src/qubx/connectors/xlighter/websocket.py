"""Lighter-specific WebSocket manager"""

from typing import Any, Awaitable, Callable, Optional

from qubx import logger
from qubx.utils.websocket_manager import BaseWebSocketManager, ReconnectionConfig

from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_PING_INTERVAL,
    DEFAULT_PING_TIMEOUT,
    WS_BASE_MAINNET,
    WS_BASE_TESTNET,
    WS_MSG_TYPE_CONNECTED,
)


class LighterWebSocketManager(BaseWebSocketManager):
    """
    Lighter-specific WebSocket manager.

    Extends BaseWebSocketManager with Lighter's subscription protocol:
    - Subscription: {"type": "subscribe", "channel": "order_book/0"}
    - Unsubscription: {"type": "unsubscribe", "channel": "order_book/0"}

    Supported channels:
    - order_book/{market_id}: L2 orderbook updates
    - trade/{market_id}: Trade feed
    - market_stats/{market_id} or market_stats/all: Market statistics
    - account_all/{account_id}: Account updates (positions, balances)
    - user_stats/{account_id}: User statistics
    - executed_transaction: Global transaction executions
    """

    def __init__(
        self,
        testnet: bool = False,
        reconnection_config: Optional[ReconnectionConfig] = None,
        ping_interval: float = DEFAULT_PING_INTERVAL,
        ping_timeout: float = DEFAULT_PING_TIMEOUT,
    ):
        """
        Initialize Lighter WebSocket manager.

        Args:
            testnet: If True, connect to testnet. Otherwise mainnet.
            reconnection_config: Configuration for reconnection behavior
            ping_interval: Interval between pings (seconds)
            ping_timeout: Timeout for ping response (seconds)
        """
        url = WS_BASE_TESTNET if testnet else WS_BASE_MAINNET

        if reconnection_config is None:
            reconnection_config = ReconnectionConfig(max_retries=DEFAULT_MAX_RETRIES)

        super().__init__(
            url=url,
            reconnection_config=reconnection_config,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
        )

        self.testnet = testnet
        self._on_connected_callback: Optional[Callable[[], Awaitable[None]]] = None

    def set_on_connected_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """
        Set callback to be called when connection is established.

        Args:
            callback: Async function to call on connection
        """
        self._on_connected_callback = callback

    async def subscribe_orderbook(
        self, market_id: int, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """
        Subscribe to orderbook updates for a market.

        Args:
            market_id: Lighter market ID
            handler: Async function to handle orderbook updates
        """
        channel = f"order_book/{market_id}"
        await self.subscribe(channel, handler)

    async def subscribe_trades(
        self, market_id: int, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """
        Subscribe to trade feed for a market.

        Args:
            market_id: Lighter market ID
            handler: Async function to handle trade updates
        """
        channel = f"trade/{market_id}"
        await self.subscribe(channel, handler)

    async def subscribe_market_stats(
        self, market_id: int | str, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """
        Subscribe to market statistics.

        Args:
            market_id: Lighter market ID or "all" for all markets
            handler: Async function to handle market stats
        """
        channel = f"market_stats/{market_id}"
        await self.subscribe(channel, handler)

    async def subscribe_account(
        self, account_id: int, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """
        Subscribe to account updates.

        Args:
            account_id: Lighter account ID
            handler: Async function to handle account updates
        """
        channel = f"account_all/{account_id}"
        await self.subscribe(channel, handler)

    async def subscribe_user_stats(
        self, account_id: int, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """
        Subscribe to user statistics.

        Args:
            account_id: Lighter account ID
            handler: Async function to handle user stats
        """
        channel = f"user_stats/{account_id}"
        await self.subscribe(channel, handler)

    async def subscribe_executed_transactions(
        self, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """
        Subscribe to executed transactions (global).

        Args:
            handler: Async function to handle transaction executions
        """
        channel = "executed_transaction"
        await self.subscribe(channel, handler)

    async def _send_subscription_message(self, channel: str, params: dict[str, Any]) -> None:
        """
        Send Lighter-specific subscription message.

        Format: {"type": "subscribe", "channel": "order_book/0"}

        Args:
            channel: Channel to subscribe to
            params: Additional parameters (unused for Lighter)
        """
        message = {"type": "subscribe", "channel": channel}
        await self.send(message)
        logger.debug(f"Sent Lighter subscription: {message}")

    async def _send_unsubscription_message(self, channel: str) -> None:
        """
        Send Lighter-specific unsubscription message.

        Format: {"type": "unsubscribe", "channel": "order_book/0"}

        Args:
            channel: Channel to unsubscribe from
        """
        message = {"type": "unsubscribe", "channel": channel}
        await self.send(message)
        logger.debug(f"Sent Lighter unsubscription: {message}")

    def _extract_channel(self, message: dict) -> Optional[str]:
        """
        Extract channel from Lighter message.

        Lighter messages have format:
        - {"type": "update/order_book", "channel": "order_book:0", ...}
        - {"type": "subscribed/order_book", "channel": "order_book:0", ...}

        The channel field uses colon separator, we need to convert back to slash.

        Args:
            message: Parsed message dict

        Returns:
            Channel identifier or None
        """
        channel = message.get("channel")
        if channel:
            # Convert "order_book:0" back to "order_book/0"
            channel = channel.replace(":", "/")
        return channel

    async def _handle_unknown_message(self, message: dict) -> None:
        """
        Handle Lighter system messages.

        Handles:
        - Connection confirmation: {"type": "connected"}
        - Subscription confirmations
        - Errors

        Args:
            message: Parsed message dict
        """
        msg_type = message.get("type")

        if msg_type == WS_MSG_TYPE_CONNECTED:
            logger.info("Lighter WebSocket connected")
            # Call on_connected callback if set
            if self._on_connected_callback:
                await self._on_connected_callback()

        elif msg_type and msg_type.startswith("subscribed/"):
            # Subscription confirmation
            channel = self._extract_channel(message)
            logger.info(f"Lighter subscription confirmed: {channel}")

        elif msg_type == "error":
            # Error message
            error_msg = message.get("message", "Unknown error")
            logger.error(f"Lighter WebSocket error: {error_msg}")

        else:
            # Log unknown messages for debugging
            logger.debug(f"Unhandled Lighter message: {message}")
