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
from typing import Awaitable, Callable, Optional

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import (
    ZERO_COSTS,
    CtrlChannel,
    Deal,
    Instrument,
    ITimeProvider,
    Order,
    TransactionCostsCalculator,
)
from qubx.core.interfaces import ISubscriptionManager
from qubx.core.utils import recognize_timeframe
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
        tcc: TransactionCostsCalculator | None = None,
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

        self._account_stats_initialized = False
        self._account_positions_initialized = False

        self.__info(f"Initialized LighterAccountProcessor for account {account_id}")

    def set_subscription_manager(self, manager: ISubscriptionManager) -> None:
        """Set the subscription manager (required by interface)"""
        self._subscription_manager = manager

    def get_total_capital(self, exchange: str | None = None) -> float:
        if not self._account_stats_initialized:
            self._async_loop.submit(self._start_subscriptions())
            self._wait_for_account_initialized()
        return super().get_total_capital(exchange)

    def start(self):
        """Start WebSocket subscriptions for account data"""
        if self._is_running:
            self.__debug("Account processor is already running")
            return

        if not self.channel or not self.channel.control.is_set():
            self.__warning("Channel not set or control not active, cannot start")
            return

        self._is_running = True

        # Start subscription tasks using AsyncThreadLoop
        self.__info("Starting Lighter account subscriptions")

        if not self._account_stats_initialized:
            # Submit connection and subscription tasks to the event loop
            self._async_loop.submit(self._start_subscriptions())
            self._wait_for_account_initialized()

        self.__info("Lighter account subscriptions started")

    def stop(self):
        """Stop all WebSocket subscriptions"""
        if not self._is_running:
            return

        self.__info("Stopping Lighter account subscriptions")

        # Cancel all subscription tasks
        for task in self._subscription_tasks:
            if not task.done():
                task.cancel()

        self._subscription_tasks.clear()
        self._is_running = False
        self.__info("Lighter account subscriptions stopped")

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
            # self.__debug(f"Migrating order: client_id={order.client_id} → server_id={order.id}")

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

    def _wait_for_account_initialized(self):
        max_wait_time = 20.0  # seconds
        elapsed = 0.0
        interval = 0.1
        while not self._account_stats_initialized or not self._account_positions_initialized:
            if elapsed >= max_wait_time:
                raise TimeoutError(f"Account was not initialized within {max_wait_time} seconds")
            time.sleep(interval)
            elapsed += interval

    async def _start_subscriptions(self):
        """Connect to WebSocket and start all subscriptions"""
        try:
            # Ensure WebSocket is connected
            if not self.ws_manager.is_connected:
                self.__info("Connecting to Lighter WebSocket...")
                await self.ws_manager.connect()
                self.__info("Connected to Lighter WebSocket")

            # Start all subscriptions
            await self._subscribe_account_all()
            await self._subscribe_account_all_orders()
            await self._subscribe_user_stats()
            await self._poller(name="sync_orders", coroutine=self._sync_orders, interval="1min")

        except Exception as e:
            self.__error(f"Failed to start subscriptions: {e}")
            self._is_running = False
            raise

    async def _subscribe_account_all(self):
        try:
            await self.ws_manager.subscribe_account_all(self._lighter_account_index, self._handle_account_all_message)
            self.__info(f"Subscribed to account_all for account {self._lighter_account_index}")
        except Exception as e:
            self.__error(f"Failed to subscribe to account_all for account {self._lighter_account_index}: {e}")
            raise

    async def _subscribe_account_all_orders(self):
        try:
            await self.ws_manager.subscribe_account_all_orders(
                self._lighter_account_index, self._handle_account_all_orders_message
            )
            self.__info(f"Subscribed to account_all_orders for account {self._lighter_account_index}")
        except Exception as e:
            self.__error(f"Failed to subscribe to account_all_orders for account {self._lighter_account_index}: {e}")
            raise

    async def _subscribe_user_stats(self):
        try:
            await self.ws_manager.subscribe_user_stats(self._lighter_account_index, self._handle_user_stats_message)
            self.__info(f"Subscribed to user_stats for account {self._lighter_account_index}")
        except Exception as e:
            self.__error(f"Failed to subscribe to user_stats for account {self._lighter_account_index}: {e}")
            raise

    async def _sync_orders(self):
        now = self.time_provider.time()
        orders = self.get_orders()
        remove_orders = []
        for order_id, order in orders.items():
            if order.status == "NEW" and order.time < now - recognize_timeframe("1min"):
                remove_orders.append(order_id)
        for order_id in remove_orders:
            self.remove_order(order_id)

    async def _poller(
        self,
        name: str,
        coroutine: Callable[[], Awaitable],
        interval: str,
        backoff: str | None = None,
    ):
        sleep_time = pd.Timedelta(interval).total_seconds()
        retries = 0

        if backoff is not None:
            sleep_time = pd.Timedelta(backoff).total_seconds()
            await asyncio.sleep(sleep_time)

        while self.channel.control.is_set():
            try:
                await coroutine()
                retries = 0  # Reset retry counter on success
            except Exception as e:
                if not self.channel.control.is_set():
                    # If the channel is closed, then ignore all exceptions and exit
                    break
                logger.error(f"Unexpected error during account polling: {e}")
                logger.exception(e)
                retries += 1
                if retries >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached. Stopping poller.")
                    break
            finally:
                if not self.channel.control.is_set():
                    break
                await asyncio.sleep(min(sleep_time * (2 ** (retries)), 60))  # Exponential backoff capped at 60s

        logger.debug(f"{name} polling task has been stopped")

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
                position.position_avg_price_funds = pos_state.avg_entry_price
                position.r_pnl = pos_state.realized_pnl

                # Update market price for unrealized PnL recalculation
                # Use the avg_entry_price as a reference if no better price available
                # (market price updates will come from market data feed)
                if position.last_update_price == 0 or np.isnan(position.last_update_price):
                    position.last_update_price = pos_state.avg_entry_price

            updated_positions_str = "\n\t".join(
                [
                    f"{instrument.symbol} --> {pos_state.quantity:+.4f} @ {pos_state.avg_entry_price:.4f}"
                    for instrument, pos_state in position_states.items()
                ]
            )
            self.__debug(f"Updated positions:\n\t{updated_positions_str}")

            if not self._account_positions_initialized:
                self._account_positions_initialized = True
                self.__info("Account positions initialized")

            # Send deals through channel for strategy notification
            # Note: process_deals is overridden to track fees without updating positions
            for instrument, deal in deals:
                # Send: (instrument, "deals", [deal], False)
                # False means not a snapshot
                self.channel.send((instrument, "deals", [deal], False))

                # self.__debug(
                #     f"Sent deal: {instrument.symbol} {deal.amount:+.4f} @ {deal.price:.4f} "
                #     f"fee={deal.fee_amount:.6f} (id={deal.id})"
                # )

            # Funding payments are handled by the data provider, so I commented them here to avoid double sending
            # Send funding payments through channel
            # for instrument, payments in funding_payments.items():
            #     for payment in payments:
            #         # Send: (instrument, DataType.FUNDING_PAYMENT, payment, False)
            #         self.channel.send((instrument, DataType.FUNDING_PAYMENT, payment, False))
            #         logger.debug(
            #             f"Sent funding payment: {instrument.symbol} rate={payment.funding_rate:.6f} at {payment.time}"
            #         )

        except Exception as e:
            self.__error(f"Error handling account_all message: {e}")
            logger.exception(e)

    async def _handle_account_all_orders_message(self, message: dict):
        try:
            orders = parse_account_all_orders_message(message, self.instrument_loader)

            _d_msg = "\n\t".join([f"{order.id} ({order.instrument.symbol})" for order in orders])
            logger.debug(f"Received {len(orders)} orders: \n\t{_d_msg}")

            for order in orders:
                self.channel.send((order.instrument, "order", order, False))

                # self.__debug(
                #     f"Sent order: {instrument.symbol} {order.side} {order.quantity:+.4f} @ {order.price:.4f} "
                #     f"[{order.status}] (order_id={order.id})"
                # )

        except Exception as e:
            self.__error(f"Error handling account_all_orders message: {e}")
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

            updated_balances_str = "\n\t".join(
                [
                    f"{currency} --> {balance.total:.2f} (free={balance.free:.2f}, locked={balance.locked:.2f})"
                    for currency, balance in balances.items()
                ]
            )
            self.__debug(f"Updated balances:\n\t{updated_balances_str}")

            if not self._account_stats_initialized:
                self._account_stats_initialized = True
                self.__info("Account stats initialized")

        except Exception as e:
            self.__error(f"Error handling user_stats message: {e}")
            logger.exception(e)

    def __info(self, msg: str):
        logger.info(self.__format(msg))

    def __debug(self, msg: str):
        logger.debug(self.__format(msg))

    def __warning(self, msg: str):
        logger.warning(self.__format(msg))

    def __error(self, msg: str):
        logger.error(self.__format(msg))

    def __format(self, msg: str):
        return f"<green>[Lighter]</green> {msg}"
