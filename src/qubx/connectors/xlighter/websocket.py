"""Lighter-specific WebSocket manager"""

import json
import uuid
from typing import Any, Awaitable, Callable, Optional, cast

import pandas as pd

from qubx import logger
from qubx.utils.rate_limiter import RateLimiterRegistry, TokenBucketRateLimiter, rate_limited
from qubx.utils.time import now_utc
from qubx.utils.websocket_manager import BaseWebSocketManager, ReconnectionConfig

from .client import LighterClient
from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_PING_TIMEOUT,
    WS_BASE_MAINNET,
    WS_BASE_TESTNET,
    WS_MSG_TYPE_CONNECTED,
)
from .nonce import LighterNonceProvider
from .rate_limits import DEFAULT_WS_SUB_LIMIT


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
        client: LighterClient,
        testnet: bool = False,
        reconnection_config: Optional[ReconnectionConfig] = None,
        ping_interval: float | None = None,
        ping_timeout: float | None = DEFAULT_PING_TIMEOUT,
        application_ping_interval: float | None = 20.0,
        ws_subscription_rate_limit: int | None = None,
    ):
        """
        Initialize Lighter WebSocket manager.

        Args:
            testnet: If True, connect to testnet. Otherwise mainnet.
            reconnection_config: Configuration for reconnection behavior
            ping_interval: Interval for protocol-level pings (seconds), or None to disable (default: None)
            ping_timeout: Timeout for ping response (seconds)
            application_ping_interval: Interval for application-level pings (seconds), or None to disable (default: 20.0)
            ws_subscription_rate_limit: Max subscriptions per second (default: 20)
        """
        if reconnection_config is None:
            reconnection_config = ReconnectionConfig(max_retries=DEFAULT_MAX_RETRIES)

        super().__init__(
            url=WS_BASE_TESTNET if testnet else WS_BASE_MAINNET,
            reconnection=reconnection_config,
            ping_interval=None,  # use app-level pings
            ping_timeout=None,
            max_size=16 * 1024 * 1024,
            max_queue=None,
            compression=None,
            inbox_size=5000,
            workers=1,
            app_ping_interval=20.0,  # send {"type":"ping"} every 20s
            no_rx_reconnect_after=60.0,
        )

        self.testnet = testnet
        self._client = client
        self._auth_token: Optional[str] = None
        self._auth_token_expiry: Optional[pd.Timestamp] = None
        self._on_connected_callback: Optional[Callable[[], Awaitable[None]]] = None

        # Initialize rate limiters for WebSocket subscriptions
        self._rate_limiters = RateLimiterRegistry()
        ws_limit = ws_subscription_rate_limit if ws_subscription_rate_limit is not None else DEFAULT_WS_SUB_LIMIT
        ws_limiter = TokenBucketRateLimiter(
            capacity=float(ws_limit),
            refill_rate=float(ws_limit),
            name="lighter_ws_sub",
        )
        self._rate_limiters.register_limiter("ws_sub", ws_limiter)

        self._nonce_provider = LighterNonceProvider(client=self._client)

    def set_on_connected_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """
        Set callback to be called when connection is established.

        Args:
            callback: Async function to call on connection
        """
        self._on_connected_callback = callback

    @property
    def auth_token(self) -> str:
        if self._auth_token_expiry is None:
            self._update_auth_token()
            return cast(str, self._auth_token)

        if self._auth_token_expiry < now_utc():
            self._update_auth_token()
            return cast(str, self._auth_token)

        return cast(str, self._auth_token)

    async def next_nonce(self) -> int:
        return await self._nonce_provider.get_nonce()

    @rate_limited("ws_sub", weight=1.0)
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
        if tx_id is None:
            tx_id = str(uuid.uuid4())

        # Parse tx_info if it's a string
        tx_info_dict = json.loads(tx_info) if isinstance(tx_info, str) else tx_info

        message = {
            "type": "jsonapi/sendtx",
            "data": {"id": tx_id, "tx_type": tx_type, "tx_info": tx_info_dict},
        }

        await self.send(message)

        # Wait for response (next message should be the tx response)
        # Note: In production, we should have a proper response tracking system
        # For now, we'll return the sent message as confirmation
        return {"tx_id": tx_id, "tx_type": tx_type, "status": "sent"}

    @rate_limited("ws_sub", weight=1.0)
    async def send_batch_tx(self, tx_types: list[int], tx_infos: list[str], batch_id: str | None = None) -> dict:
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

        await self.send(message)

        # Return confirmation
        return {"batch_id": batch_id, "count": len(tx_types), "status": "sent"}

    async def subscribe_orderbook(self, market_id: int, handler: Callable[[dict], Awaitable[None]]) -> None:
        await self.subscribe(f"order_book/{market_id}", handler)

    async def subscribe_trades(self, market_id: int, handler: Callable[[dict], Awaitable[None]]) -> None:
        await self.subscribe(f"trade/{market_id}", handler)

    async def subscribe_market_stats(self, market_id: int | str, handler: Callable[[dict], Awaitable[None]]) -> None:
        await self.subscribe(f"market_stats/{market_id}", handler)

    async def unsubscribe_orderbook(self, market_id: int) -> None:
        await self.unsubscribe(f"order_book/{market_id}")

    async def unsubscribe_trades(self, market_id: int) -> None:
        await self.unsubscribe(f"trade/{market_id}")

    async def unsubscribe_market_stats(self, market_id: int | str) -> None:
        await self.unsubscribe(f"market_stats/{market_id}")

    async def subscribe_account_all(self, account_id: int, handler: Callable[[dict], Awaitable[None]]) -> None:
        """
        Subscribe to account_all channel for positions and trades (primary channel).

        Requires authentication token.

        This is the single source of truth for positions and trade history.
        Updates positions directly from position data, sends trades as Deals
        through channel for strategy notification.

        Message format:
        {
            "account": 225671,
            "channel": "account_all:225671",
            "type": "update/account_all",
            "positions": {
                "24": {
                    "market_id": 24,
                    "sign": -1,  # 1 for long, -1 for short
                    "position": "1.00",
                    "avg_entry_price": "40.1342",
                    ...
                }
            },
            "trades": {
                "24": [
                    {
                        "trade_id": 225067334,
                        "market_id": 24,
                        "size": "1.00",
                        "price": "40.1342",
                        "timestamp": 1760287839079,
                        ...
                    }
                ]
            },
            "funding_histories": {}
        }
        """
        await self.subscribe(f"account_all/{account_id}", handler, auth=self.auth_token)

    async def subscribe_account_all_orders(self, account_id: int, handler: Callable[[dict], Awaitable[None]]) -> None:
        """
        Subscribe to account_all_orders channel for order updates across all markets.

        Requires authentication token.

        Message format:
        {
            "channel": "account_all_orders:225671",
            "type": "update/account_all_orders",
            "orders": {
                "24": [  # market_index
                    {
                        "order_id": "7036874567748225",
                        "status": "filled",
                        ...
                    }
                ]
            }
        }
        """
        await self.subscribe(f"account_all_orders/{account_id}", handler, auth=self.auth_token)

    async def subscribe_user_stats(self, account_id: int, handler: Callable[[dict], Awaitable[None]]) -> None:
        """
        Subscribe to user_stats channel for account statistics.

        Requires authentication token.

        Message format:
        {
            "channel": "user_stats:225671",
            "type": "update/user_stats",
            "stats": {
                "collateral": "998.888700",
                "portfolio_value": "998.901500",
                "available_balance": "990.920600",
                "leverage": "0.04",
                "margin_usage": "0.80",
                ...
            }
        }
        """
        await self.subscribe(f"user_stats/{account_id}", handler, auth=self.auth_token)

    @rate_limited("ws_sub", weight=1.0)
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
            # take a new one just in case
            message["auth"] = self.auth_token

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
        try:
            channel = message.get("channel", None)
            if channel:
                # Convert "order_book:0" back to "order_book/0"
                channel = channel.replace(":", "/")
            return channel
        except Exception:
            return None

    async def _handle_error_message(self, error: dict) -> None:
        """
        Handle Lighter error message.

        - Errors:
           1. {'error': {'code': 20013, 'message': 'invalid auth: expired token: account_all_orders:225671'}}

        Args:
            error: Error dict
        """
        error_code = error.get("code", -1)
        error_message = error.get("message", "Unknown error")
        match error_code:
            case 20013:
                # expired token
                logger.warning("Auth token expired, generating new one")
                self._update_auth_token()
            case 30003:
                # Alread subscribed
                logger.debug(f"Already subscribed: {error}")
            case 21104:
                # nonce too small
                logger.warning("Nonce too small, resyncing nonce provider")
                await self._nonce_provider.resync()
            case _:
                logger.warning(f"Lighter WebSocket error [{error_code}] {error_message}")

    def _update_auth_token(self):
        """
        Generate authentication token for WebSocket subscriptions.

        Auth tokens are required for account-specific channels:
        - account_all/{account_id}
        - account_all_orders/{account_id}
        - user_stats/{account_id}

        Note: create_auth_token_with_expiry() returns (auth_token_string, error_string)
        where auth_token_string is the token itself (not an object).
        Default expiry is 10 minutes from creation time.
        """
        try:
            logger.info("Generating auth token...")

            # Use SignerClient to create auth token
            # Returns (token_string, error_string)
            signer = self._client.signer_client
            auth_token, error = signer.create_auth_token_with_expiry()

            if error:
                raise RuntimeError(f"Failed to generate auth token: {error}")

            if not auth_token:
                raise RuntimeError("Auth token is empty")

            # Store token (it's already a string, not an object)
            self._auth_token = auth_token
            # default expiry is 10 minutes from creation time, we need to resubscribe before that
            self._auth_token_expiry = cast(pd.Timestamp, now_utc() + pd.Timedelta(minutes=9))

            logger.info(f"Auth token generated successfully (expires at: {self._auth_token_expiry})")

        except Exception as e:
            logger.error(f"Failed to generate auth token: {e}")
            raise

    async def _handle_unknown_message(self, message: dict) -> None:
        """
        Handle Lighter system messages.

        Handles:
        - Connection confirmation: {"type": "connected"}
        - Application-level ping/pong: {"type": "ping"} -> {"type": "pong"}
        - Subscription confirmations
        - Errors  {'error': {'code': 20013, 'message': 'invalid auth: expired token: account_all_orders:225671'}}

        Args:
            message: Parsed message dict
        """
        is_error = "error" in message
        if is_error:
            await self._handle_error_message(message["error"])

        msg_type = message.get("type")

        if msg_type == WS_MSG_TYPE_CONNECTED:
            logger.info("Lighter WebSocket connected")
            # Call on_connected callback if set
            if self._on_connected_callback:
                await self._on_connected_callback()

        elif msg_type == "pong":
            pass

        elif msg_type == "ping":
            # Application-level ping - must respond with pong
            logger.debug("Received application-level ping, sending pong")
            await self.send({"type": "pong"})

        elif msg_type and msg_type.startswith("subscribed/"):
            # Subscription confirmation
            channel = self._extract_channel(message)
            if channel:
                logger.debug(f"Lighter subscription confirmed: {channel}")
            else:
                logger.debug(f"Received subscription confirmation with no channel: {message}")

        elif msg_type == "error":
            # Error message
            error_msg = message.get("message", "Unknown error")
            logger.error(f"Lighter WebSocket error: {error_msg}")

        else:
            # Log unknown messages for debugging
            # logger.debug(f"Unhandled Lighter message: {message}")
            pass

    def _app_ping_payload(self) -> Optional[dict]:
        return {"type": "ping"}
