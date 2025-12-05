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
from typing import Awaitable, Callable, Optional, cast

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
    RestoredState,
    TransactionCostsCalculator,
)
from qubx.core.interfaces import IHealthMonitor, ISubscriptionManager
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
        health_monitor: IHealthMonitor,
        base_currency: str = "USDC",
        tcc: TransactionCostsCalculator | None = None,
        initial_capital: float = 100_000,
        max_retries: int = 10,
        connection_timeout: int = 30,
        restored_state: RestoredState | None = None,
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
            health_monitor=health_monitor,
            exchange="LIGHTER",
            tcc=tcc,
            initial_capital=0,  # Will be updated from WebSocket
            restored_state=restored_state,
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

        # Position sync buffering (for drift correction)
        self._position_sync_buffer: dict[Instrument, dict] = {}
        self._last_deal_time: dict[Instrument, pd.Timestamp] = {}
        self._last_position_sync_time: dict[Instrument, pd.Timestamp] = {}
        self._position_sync_threshold_sec: float = 5.0  # Apply sync after 5s of no deals

        # Margin tracking from user_stats (account-level margin usage percentage)
        self._margin_usage: float | None = None

        self.__info(f"Initialized LighterAccountProcessor for account {account_id}")

    def set_subscription_manager(self, manager: ISubscriptionManager) -> None:
        """Set the subscription manager (required by interface)"""
        self._subscription_manager = manager

    def get_total_capital(self, exchange: str | None = None) -> float:
        if not self._account_stats_initialized:
            self._async_loop.submit(self._start_subscriptions())
            self._wait_for_account_initialized()
        return super().get_total_capital(exchange)

    def get_margin_ratio(self, exchange: str | None = None) -> float:
        """
        Get margin ratio from Lighter's margin_usage.

        Lighter provides margin_usage as a percentage (0-100) of margin used.
        We convert it to margin_ratio where:
        - margin_ratio = 1 means at liquidation (100% margin used)
        - margin_ratio = 2 means 50% margin used
        - margin_ratio = 100 means very little margin used (safe)

        Formula: margin_ratio = 100 / margin_usage
        """
        if self._margin_usage is None or self._margin_usage <= 0:
            return 100.0  # No margin used, max safety
        return min(100.0, 100.0 / self._margin_usage)

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

    def process_deals(self, instrument: Instrument, deals: list[Deal]) -> None:
        self._fill_missing_fee_info(instrument, deals)
        pos = self._positions.get(instrument)

        if pos is not None:
            conversion_rate = 1
            traded_amnt, realized_pnl, deal_cost = 0, 0, 0

            for d in deals:
                _o_deals = self._processed_trades[d.order_id]

                if d.id not in _o_deals:
                    _o_deals.append(d.id)

                    r_pnl, fee_in_base = pos.update_position_by_deal(d, conversion_rate)
                    realized_pnl += r_pnl
                    deal_cost += d.amount * d.price / conversion_rate
                    traded_amnt += d.amount
                    logger.debug(
                        f"  [<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: traded {d.amount} @ {d.price} -> {realized_pnl:.2f} {self.base_currency} realized profit"
                    )

    def process_order(self, order: Order, update_locked_value: bool = True) -> None:
        """
        Override process_order to handle Lighter's server-assigned order IDs.

        Lighter assigns server IDs different from client_id. When an order
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
            await self._poller(name="sync_positions", coroutine=self._sync_positions_from_buffer, interval="3sec")

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

    def _apply_position_sync(
        self, instrument: Instrument, quantity: float, avg_entry_price: float, allocated_margin: float | None = None
    ) -> None:
        """
        Apply position sync from Lighter to local position state.

        Args:
            instrument: The instrument to sync
            quantity: Position quantity from Lighter
            avg_entry_price: Average entry price from Lighter
            allocated_margin: Margin allocated to position from Lighter (optional)
        """
        position = self.get_position(instrument)
        position.quantity = quantity
        position.position_avg_price = avg_entry_price
        position.position_avg_price_funds = avg_entry_price

        # Set maintenance margin from exchange if available
        if allocated_margin is not None and allocated_margin > 0:
            position.set_external_maint_margin(allocated_margin)

    async def _sync_positions_from_buffer(self):
        """
        Periodically check buffered position syncs and apply if needed.

        Apply position sync only if:
        1. Last deal timestamp < last position sync timestamp (deal came before sync)
        2. More than threshold seconds since last position sync (5s default)
        3. Current position differs from buffered position (drift detected)
        """
        now = cast(pd.Timestamp, pd.Timestamp(self.time_provider.time(), unit="ns"))

        for instrument, buffered_state in list(self._position_sync_buffer.items()):
            position = self.get_position(instrument)
            last_deal_time = self._last_deal_time.get(instrument, None)
            last_sync_time = self._last_position_sync_time.get(instrument, None)
            if last_sync_time is None:
                continue

            # Check conditions for applying sync
            deal_before_sync = last_deal_time is None or last_deal_time < last_sync_time
            time_since_sync = (now - last_sync_time).total_seconds()
            enough_time_passed = time_since_sync > self._position_sync_threshold_sec

            # Check if positions differ
            qty_differs = abs(position.quantity - buffered_state["quantity"]) > instrument.min_size
            price_differs = abs(position.position_avg_price - buffered_state["avg_entry_price"]) > instrument.tick_size
            position_differs = qty_differs or price_differs

            # Apply sync if all conditions met
            if deal_before_sync and enough_time_passed and position_differs:
                self.__warning(
                    f"Position drift detected for {instrument.symbol}: "
                    f"local={position.quantity:.4f}@{position.position_avg_price:.4f} "
                    f"vs sync={buffered_state['quantity']:.4f}@{buffered_state['avg_entry_price']:.4f}. "
                    f"Applying position sync."
                )

                # Apply position sync from buffer using helper
                self._apply_position_sync(
                    instrument,
                    buffered_state["quantity"],
                    buffered_state["avg_entry_price"],
                    buffered_state.get("allocated_margin"),
                )

                # Remove from buffer after applying
                self._position_sync_buffer.pop(instrument)

            # Clean up old buffers (older than 30 seconds)
            elif time_since_sync > self._position_sync_threshold_sec * 2:
                self._position_sync_buffer.pop(instrument, None)

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
        - Trades: Sent through channel for base class to process (updates position and r_pnl)
        - Positions: Buffered for drift correction (applied by poller only if needed)

        This ensures deals are always processed with correct pre-trade state, while
        Lighter's position syncs act as periodic drift correction.
        """
        try:
            # Parse message into positions dict, deals list, and funding payments
            position_states, deals, funding_payments = parse_account_all_message(
                message, self.instrument_loader, self._lighter_account_index
            )

            now = cast(pd.Timestamp, pd.Timestamp(self.time_provider.time(), unit="ns"))

            # STEP 1: Send deals through channel (base class will process them)
            # Base class process_deals() will update position quantity, avg_price, and r_pnl
            for instrument, deal in deals:
                self._last_deal_time[instrument] = now
                self.channel.send((instrument, "deals", [deal], False))

            # STEP 2: Handle position syncs
            # - During initialization: apply immediately to sync initial state
            # - During normal operation: buffer for drift correction
            if not self._account_positions_initialized:
                # Initial sync: apply positions immediately
                for instrument, pos_state in position_states.items():
                    self._apply_position_sync(
                        instrument, pos_state.quantity, pos_state.avg_entry_price, pos_state.allocated_margin
                    )

                    # Update market price for unrealized PnL
                    position = self.get_position(instrument)
                    if position.last_update_price == 0 or np.isnan(position.last_update_price):
                        position.last_update_price = pos_state.avg_entry_price

                if position_states:
                    synced_positions_str = "\n\t".join(
                        [
                            f"{instrument.symbol} --> {pos_state.quantity:+.4f} @ {pos_state.avg_entry_price:.4f}"
                            for instrument, pos_state in position_states.items()
                        ]
                    )
                    self.__info(f"Initial position sync:\n\t{synced_positions_str}")

                self._account_positions_initialized = True
                self.__info("Account positions initialized")
            else:
                # Normal operation: buffer position syncs for drift correction
                for instrument, pos_state in position_states.items():
                    self._position_sync_buffer[instrument] = {
                        "quantity": pos_state.quantity,
                        "avg_entry_price": pos_state.avg_entry_price,
                        "allocated_margin": pos_state.allocated_margin,
                        "timestamp": now,
                    }
                    self._last_position_sync_time[instrument] = now

        except Exception as e:
            self.__error(f"Error handling account_all message: {e}")
            logger.exception(e)

    async def _handle_account_all_orders_message(self, message: dict):
        try:
            orders = parse_account_all_orders_message(message, self.instrument_loader)
            for order in orders:
                self.channel.send((order.instrument, "order", order, False))

        except Exception as e:
            self.__error(f"Error handling account_all_orders message: {e}")
            logger.exception(e)

    async def _handle_user_stats_message(self, message: dict):
        """
        Handle user_stats WebSocket messages.

        Parses balance information, updates account balance, and extracts margin_usage.
        """
        try:
            balances = parse_user_stats_message(message, self.exchange)

            for currency, balance in balances.items():
                self.update_balance(currency, balance.total, balance.locked)

            # Extract margin_usage from stats (percentage 0-100 of margin used)
            stats = message.get("stats", {})
            margin_usage_str = stats.get("margin_usage")
            if margin_usage_str is not None:
                try:
                    self._margin_usage = float(margin_usage_str)
                except (ValueError, TypeError):
                    pass

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
