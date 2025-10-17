"""
LighterAccountProcessor - IAccountProcessor implementation for Lighter exchange.

Tracks account state via WebSocket subscriptions:
- Positions and trades via account_all channel (requires auth)
- Orders via account_all_orders channel (requires auth)
- Balances via user_stats channel (requires auth)

Uses account_all as single source of truth for positions. Deals are sent
through channel for strategy notification but do not update positions.
"""

import asyncio
import time
from typing import Optional

import numpy as np

from qubx import logger
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import (
    CtrlChannel,
    DataType,
    Deal,
    Instrument,
    ITimeProvider,
    Order,
    TransactionCostsCalculator,
)
from qubx.core.interfaces import ISubscriptionManager
from qubx.utils.misc import AsyncThreadLoop

from .client import LighterClient
from .instruments import LighterInstrumentLoader
from .parsers import (
    parse_account_all_message,
    parse_account_all_orders_message,
    parse_user_stats_message,
)
from .websocket import LighterWebSocketManager


class LighterAccountProcessor(BasicAccountProcessor):
    """
    Account processor for Lighter exchange.

    Subscribes to WebSocket channels (all require authentication):
    - `account_all/{account_id}` - Positions, trades, and funding (primary channel)
    - `account_all_orders/{account_id}` - Order updates for all markets
    - `user_stats/{account_id}` - Account statistics (balance, leverage, margin)

    Uses account_all as single source of truth for positions:
    - Positions are updated directly from account_all position data
    - Trades are parsed into Deals and sent through channel for strategy notification
    - Deals do NOT update positions (process_deals is overridden to prevent this)
    """

    def __init__(
        self,
        account_id: str,
        client: LighterClient,
        instrument_loader: LighterInstrumentLoader,
        ws_manager: LighterWebSocketManager,
        channel: CtrlChannel,
        time_provider: ITimeProvider,
        loop: asyncio.AbstractEventLoop,
        base_currency: str = "USDC",
        tcc: TransactionCostsCalculator = None,
        initial_capital: float = 100_000,
        max_retries: int = 10,
        connection_timeout: int = 30,
    ):
        """
        Initialize Lighter account processor.

        Args:
            account_id: Account identifier (Lighter account_index as string)
            client: LighterClient for REST API access
            instrument_loader: Instrument loader with market mappings
            ws_manager: WebSocket manager for subscriptions
            channel: Control channel for sending events
            time_provider: Time provider for timestamps
            loop: Event loop for async operations (from client)
            base_currency: Base currency (always USDC for Lighter)
            tcc: Transaction costs calculator
            initial_capital: Initial capital (used if no balance data)
            max_retries: Maximum retry attempts for subscriptions
            connection_timeout: Connection timeout in seconds
        """
        if tcc is None:
            from qubx.core.basics import ZERO_COSTS

            tcc = ZERO_COSTS

        super().__init__(
            account_id=account_id,
            time_provider=time_provider,
            base_currency=base_currency,
            tcc=tcc,
            initial_capital=0,  # Will be updated from WebSocket
        )

        self.client = client
        self.instrument_loader = instrument_loader
        self.ws_manager = ws_manager
        self.channel = channel
        self.max_retries = max_retries
        self.connection_timeout = connection_timeout

        # Async thread loop for submitting tasks to client's event loop
        self._async_loop = AsyncThreadLoop(loop)

        # State tracking
        self._is_running = False
        self._subscription_manager: Optional[ISubscriptionManager] = None
        self._subscription_tasks: list = []

        # Lighter-specific IDs and auth
        self._lighter_account_index = int(account_id)
        self._auth_token: Optional[str] = None
        self._auth_token_expiry: Optional[int] = None
        self._processed_tx_hashes: set[str] = set()  # Track processed transaction hashes

        logger.info(f"Initialized LighterAccountProcessor for account {account_id}")

    def set_subscription_manager(self, manager: ISubscriptionManager) -> None:
        """Set the subscription manager (required by interface)"""
        self._subscription_manager = manager

    def start(self):
        """Start WebSocket subscriptions for account data"""
        if self._is_running:
            logger.debug("Account processor is already running")
            return

        if not self.channel or not self.channel.control.is_set():
            logger.warning("Channel not set or control not active, cannot start")
            return

        self._is_running = True

        # Start subscription tasks using AsyncThreadLoop
        logger.info("Starting Lighter account subscriptions")

        # Submit connection and subscription tasks to the event loop
        self._async_loop.submit(self._start_subscriptions())

        logger.info("Lighter account subscriptions started")

    async def _start_subscriptions(self):
        """Connect to WebSocket and start all subscriptions"""
        try:
            # Ensure WebSocket is connected
            if not self.ws_manager.is_connected:
                logger.info("Connecting to Lighter WebSocket...")
                await self.ws_manager.connect()
                logger.info("Connected to Lighter WebSocket")

            # Generate auth token for authenticated channels
            await self._generate_auth_token()

            # Start all subscriptions
            await self._subscribe_account_all()
            await self._subscribe_account_all_orders()
            await self._subscribe_user_stats()

        except Exception as e:
            logger.error(f"Failed to start subscriptions: {e}")
            self._is_running = False
            raise

    def stop(self):
        """Stop all WebSocket subscriptions"""
        if not self._is_running:
            return

        logger.info("Stopping Lighter account subscriptions")

        # Cancel all subscription tasks
        for task in self._subscription_tasks:
            if not task.done():
                task.cancel()

        self._subscription_tasks.clear()
        self._is_running = False

        logger.info("Lighter account subscriptions stopped")

    async def _generate_auth_token(self):
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
            signer = self.client.signer_client
            auth_token, error = signer.create_auth_token_with_expiry()

            if error:
                raise RuntimeError(f"Failed to generate auth token: {error}")

            if not auth_token:
                raise RuntimeError("Auth token is empty")

            # Store token (it's already a string, not an object)
            self._auth_token = auth_token
            # Calculate expiry (default is 10 minutes from now)
            self._auth_token_expiry = int(time.time()) + 10 * 60

            logger.info(f"Auth token generated successfully (expires at: {self._auth_token_expiry})")

        except Exception as e:
            logger.error(f"Failed to generate auth token: {e}")
            raise

    async def _subscribe_account_all(self):
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
        channel = f"account_all/{self._lighter_account_index}"
        logger.info(f"Subscribing to {channel} (with auth)")

        try:
            # Subscribe with auth token
            await self.ws_manager.subscribe(
                channel=channel, handler=self._handle_account_all_message, auth=self._auth_token
            )
            logger.info(f"Successfully subscribed to {channel}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            raise

    async def _subscribe_account_all_orders(self):
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
        channel = f"account_all_orders/{self._lighter_account_index}"
        logger.info(f"Subscribing to {channel} (with auth)")

        try:
            # Subscribe with auth token
            await self.ws_manager.subscribe(
                channel=channel, handler=self._handle_account_all_orders_message, auth=self._auth_token
            )
            logger.info(f"Successfully subscribed to {channel}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            raise

    async def _subscribe_user_stats(self):
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
        channel = f"user_stats/{self._lighter_account_index}"
        logger.info(f"Subscribing to {channel} (with auth)")

        try:
            # Subscribe with auth token
            await self.ws_manager.subscribe(
                channel=channel, handler=self._handle_user_stats_message, auth=self._auth_token
            )
            logger.info(f"Successfully subscribed to {channel}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            raise

    async def _handle_account_all_message(self, message: dict):
        """
        Handle account_all WebSocket messages (primary channel).

        Parses positions and trades:
        - Positions: Synced from server state including quantity and avg_entry_price
          (single source of truth for position state)
        - Trades: Parsed into Deals and sent through channel for strategy notification
          (process_deals tracks fees but does NOT update positions)

        This ensures positions are always consistent with Lighter's server state.
        """
        try:
            # Parse message into positions dict, deals list, and funding payments
            position_states, deals, funding_payments = parse_account_all_message(
                message, self.instrument_loader, self._lighter_account_index
            )

            # Sync position state from server (single source of truth)
            # Update both quantity and average entry price
            for instrument, pos_state in position_states.items():
                position = self.get_position(instrument)

                # Sync quantity and position_avg_price from Lighter's authoritative data
                position.quantity = pos_state.quantity
                position.position_avg_price = pos_state.avg_entry_price

                # Update market price for unrealized PnL recalculation
                # Use the avg_entry_price as a reference if no better price available
                # (market price updates will come from market data feed)
                if position.last_update_price == 0 or np.isnan(position.last_update_price):
                    position.last_update_price = pos_state.avg_entry_price

                logger.debug(
                    f"Synced position: {instrument.symbol} = {pos_state.quantity:+.4f} "
                    f"@ avg_price={pos_state.avg_entry_price:.4f}"
                )

            # Send deals through channel for strategy notification
            # Note: process_deals is overridden to track fees without updating positions
            for instrument, deal in deals:
                # Send: (instrument, "deals", [deal], False)
                # False means not a snapshot
                self.channel.send((instrument, "deals", [deal], False))

                logger.debug(
                    f"Sent deal: {instrument.symbol} {deal.amount:+.4f} @ {deal.price:.4f} "
                    f"fee={deal.fee_amount:.6f} (id={deal.id})"
                )

            # Send funding payments through channel
            for instrument, payments in funding_payments.items():
                for payment in payments:
                    # Send: (instrument, DataType.FUNDING_PAYMENT, payment, False)
                    self.channel.send((instrument, DataType.FUNDING_PAYMENT, payment, False))

                    logger.debug(
                        f"Sent funding payment: {instrument.symbol} rate={payment.funding_rate:.6f} at {payment.time}"
                    )

        except Exception as e:
            logger.error(f"Error handling account_all message: {e}")
            logger.exception(e)

    async def _handle_account_all_orders_message(self, message: dict):
        """
        Handle account_all_orders WebSocket messages.

        Parses order updates and sends Order objects through channel.

        Follows CCXT pattern: parse → send through channel → framework handles rest.
        """
        try:
            # Parse message into list of (Instrument, Order) tuples
            orders = parse_account_all_orders_message(message, self.instrument_loader)

            # Send each order through channel (CCXT pattern)
            for instrument, order in orders:
                # Send: (instrument, "order", order, False)
                # False means not a snapshot
                self.channel.send((instrument, "order", order, False))

                logger.debug(
                    f"Sent order: {instrument.symbol} {order.side} {order.quantity:+.4f} @ {order.price:.4f} "
                    f"[{order.status}] (order_id={order.id})"
                )

        except Exception as e:
            logger.error(f"Error handling account_all_orders message: {e}")
            logger.exception(e)

    async def _handle_user_stats_message(self, message: dict):
        """
        Handle user_stats WebSocket messages.

        Parses balance information and updates account balance.
        """
        try:
            # Parse message into balances dict
            balances = parse_user_stats_message(message)

            # Update balances (should have USDC)
            for currency, balance in balances.items():
                self.update_balance(currency, balance.total, balance.locked)

                logger.debug(
                    f"Updated balance - {currency}: total={balance.total:.2f}, "
                    f"free={balance.free:.2f}, locked={balance.locked:.2f}"
                )

        except Exception as e:
            logger.error(f"Error handling user_stats message: {e}")
            logger.exception(e)

    def process_deals(self, instrument: Instrument, deals: list[Deal], is_snapshot: bool = False) -> None:
        """
        Override process_deals to track fees WITHOUT updating positions.

        In Lighter, positions are synced directly from account_all channel
        (single source of truth for quantity and avg_entry_price). However,
        we still need to track fees/commissions from deals.

        This prevents double position updates while ensuring commission tracking.

        Args:
            instrument: The instrument for the deals
            deals: List of Deal objects
            is_snapshot: Whether this is a snapshot or incremental update
        """
        # Do NOT call super().process_deals() - that would update positions
        # Instead, manually track fees for the position
        if not deals:
            return

        position = self.get_position(instrument)

        for deal in deals:
            # Track commission from the deal
            if deal.fee_amount and deal.fee_amount > 0:
                # Add fee to position's commission tracking
                position.commissions += deal.fee_amount

                logger.debug(
                    f"Tracked fee for {instrument.symbol}: {deal.fee_amount:.6f} {deal.fee_currency} "
                    f"(total commissions: {position.commissions:.6f})"
                )

        logger.debug(
            f"Processed {len(deals)} deal(s) for {instrument.symbol} - fees tracked, positions synced from account_all"
        )

    def process_order(self, order: Order, update_locked_value: bool = True) -> None:
        """
        Override process_order to handle Lighter's server-assigned order IDs.

        Lighter assigns server IDs different from our client_id. When an order
        update arrives with a new server ID but matching client_id, we need to
        migrate the order from client_id key to server_id key while preserving
        the same object instance (for external references).

        Args:
            order: Order update from WebSocket
            update_locked_value: Whether to update locked capital tracking
        """
        # Check if order exists under client_id (migration case)
        if order.client_id and order.client_id in self._active_orders:
            # Get the existing order stored under client_id
            existing_order = self._active_orders[order.client_id]

            logger.debug(f"Migrating order: client_id={order.client_id} → server_id={order.id}")

            # Remove from old location
            self._active_orders.pop(order.client_id)

            # Store it under the new server ID before base class processing
            # This allows base class merge logic to find and update it in place
            self._active_orders[order.id] = existing_order

            # Also migrate locked capital tracking if present
            if order.client_id in self._locked_capital_by_order:
                locked_value = self._locked_capital_by_order.pop(order.client_id)
                self._locked_capital_by_order[order.id] = locked_value

        # Let base class handle the rest (merge, store, lock/unlock, etc.)
        # The base class will now find the existing order under order.id and merge in place
        super().process_order(order, update_locked_value)
