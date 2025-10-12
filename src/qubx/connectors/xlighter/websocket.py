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

    async def send_tx(self, tx_type: int, tx_info: str, tx_id: str | None = None) -> dict:
        """
        Send a single signed transaction via WebSocket.

        Args:
            tx_type: Transaction type (e.g., 14 for CREATE_ORDER, 15 for CANCEL_ORDER)
            tx_info: Signed transaction info JSON string from SignerClient
            tx_id: Optional transaction ID for tracking (auto-generated if not provided)

        Returns:
            Response dict from WebSocket

        Example:
            >>> tx_info, err = signer_client.sign_create_order(...)
            >>> response = await ws.send_tx(tx_type=14, tx_info=tx_info)
        """
        import json
        import uuid

        if tx_id is None:
            tx_id = str(uuid.uuid4())

        # Parse tx_info if it's a string
        tx_info_dict = json.loads(tx_info) if isinstance(tx_info, str) else tx_info

        message = {
            "type": "jsonapi/sendtx",
            "data": {"id": tx_id, "tx_type": tx_type, "tx_info": tx_info_dict},
        }

        logger.debug(f"Sending transaction via WebSocket: type={tx_type}, id={tx_id}")
        await self.send(message)

        # Wait for response (next message should be the tx response)
        # Note: In production, we should have a proper response tracking system
        # For now, we'll return the sent message as confirmation
        return {"tx_id": tx_id, "tx_type": tx_type, "status": "sent"}

    async def send_batch_tx(
        self, tx_types: list[int], tx_infos: list[str], batch_id: str | None = None
    ) -> dict:
        """
        Send multiple signed transactions in a single batch via WebSocket.

        Up to 50 transactions can be sent in one batch.

        Args:
            tx_types: List of transaction types
            tx_infos: List of signed transaction info JSON strings
            batch_id: Optional batch ID for tracking (auto-generated if not provided)

        Returns:
            Response dict from WebSocket

        Raises:
            ValueError: If tx_types and tx_infos have different lengths or exceed 50

        Example:
            >>> tx_info1, _ = signer_client.sign_create_order(...)
            >>> tx_info2, _ = signer_client.sign_create_order(...)
            >>> response = await ws.send_batch_tx(
            ...     tx_types=[14, 14],
            ...     tx_infos=[tx_info1, tx_info2]
            ... )
        """
        import json
        import uuid

        if len(tx_types) != len(tx_infos):
            raise ValueError(f"tx_types and tx_infos must have same length: {len(tx_types)} != {len(tx_infos)}")

        if len(tx_types) > 50:
            raise ValueError(f"Batch size cannot exceed 50 transactions, got {len(tx_types)}")

        if len(tx_types) == 0:
            raise ValueError("Cannot send empty batch")

        if batch_id is None:
            batch_id = str(uuid.uuid4())

        # Parse tx_infos if they're strings
        tx_infos_dicts = [json.loads(info) if isinstance(info, str) else info for info in tx_infos]

        message = {
            "type": "jsonapi/sendtxbatch",
            "data": {"id": batch_id, "tx_types": tx_types, "tx_infos": tx_infos_dicts},
        }

        logger.info(f"Sending transaction batch via WebSocket: count={len(tx_types)}, id={batch_id}")
        await self.send(message)

        # Return confirmation
        return {"batch_id": batch_id, "count": len(tx_types), "status": "sent"}

    async def unsubscribe_orderbook(self, market_id: int) -> None:
        """
        Unsubscribe from orderbook updates.

        Args:
            market_id: Lighter market ID
        """
        channel = f"order_book/{market_id}"
        await self.unsubscribe(channel)

    async def unsubscribe_trades(self, market_id: int) -> None:
        """
        Unsubscribe from trade feed.

        Args:
            market_id: Lighter market ID
        """
        channel = f"trade/{market_id}"
        await self.unsubscribe(channel)

    async def _send_subscription_message(self, channel: str, params: dict[str, Any]) -> None:
        """
        Send Lighter-specific subscription message.

        Format: {"type": "subscribe", "channel": "order_book/0"}
        With auth: {"type": "subscribe", "channel": "account_all/123", "auth": "TOKEN"}

        Args:
            channel: Channel to subscribe to
            params: Additional parameters (e.g., "auth" for authentication token)
        """
        message = {"type": "subscribe", "channel": channel}

        # Add auth token if provided
        if "auth" in params:
            message["auth"] = params["auth"]

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
        - Application-level ping/pong: {"type": "ping"} -> {"type": "pong"}
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

        elif msg_type == "ping":
            # Application-level ping - must respond with pong
            logger.debug("Received application-level ping, sending pong")
            await self.send({"type": "pong"})

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
