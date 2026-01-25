import asyncio
import time
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, cast

import numpy as np
import pandas as pd

from qubx import logger
from qubx.connectors.registry import account_processor
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import (
    CtrlChannel,
    Deal,
    Instrument,
    ITimeProvider,
    Order,
    RestoredState,
    TransactionCostsCalculator,
)
from qubx.core.interfaces import IDataProvider, IHealthMonitor, ISubscriptionManager
from qubx.core.utils import recognize_timeframe
from qubx.utils.misc import AsyncThreadLoop

from .factory import get_lighter_client, get_lighter_instrument_loader, get_lighter_ws_manager
from .parsers import (
    PositionState,
    parse_account_all_message,
    parse_account_all_orders_message,
    parse_user_stats_message,
)

if TYPE_CHECKING:
    from qubx.utils.runner.accounts import AccountConfigurationManager


@account_processor("xlighter")
class LighterAccountProcessor(BasicAccountProcessor):
    def __init__(
        self,
        exchange_name: str,
        channel: CtrlChannel,
        time_provider: ITimeProvider,
        account_manager: "AccountConfigurationManager",
        tcc: TransactionCostsCalculator,
        health_monitor: IHealthMonitor,
        data_provider: IDataProvider | None = None,
        restored_state: RestoredState | None = None,
        read_only: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        max_retries: int = 10,
        connection_timeout: int = 30,
        **kwargs,
    ):
        creds = account_manager.get_exchange_credentials(exchange_name)
        settings = account_manager.get_exchange_settings(exchange_name)

        account_index = int(creds.get_extra_field("account_index", 0) or 0)
        api_key_index = int(creds.get_extra_field("api_key_index", 0) or 0)
        account_id = str(account_index) if account_index else exchange_name
        base_currency = creds.base_currency

        self.client = get_lighter_client(
            api_key=creds.api_key,
            private_key=creds.secret,
            account_index=account_index,
            api_key_index=api_key_index,
            testnet=settings.testnet,
        )

        super().__init__(
            account_id=account_id,
            time_provider=time_provider,
            base_currency=base_currency,
            health_monitor=health_monitor,
            exchange="LIGHTER",
            tcc=tcc,
            initial_capital=0,
            restored_state=restored_state,
        )

        self.instrument_loader = get_lighter_instrument_loader()
        self.ws_manager = get_lighter_ws_manager(
            api_key=creds.api_key,
            private_key=creds.secret,
            account_index=account_index,
            api_key_index=api_key_index,
            testnet=settings.testnet,
        )
        self.channel = channel
        self.max_retries = max_retries
        self.connection_timeout = connection_timeout

        self._async_loop = AsyncThreadLoop(self.client._loop)
        self._is_running = False
        self._subscription_manager: Optional[ISubscriptionManager] = None
        self._subscription_tasks: list = []
        self._lighter_account_index = int(account_id) if account_id.isdigit() else account_index
        self._auth_token: Optional[str] = None
        self._auth_token_expiry: Optional[int] = None
        self._processed_tx_hashes: set[str] = set()
        self._account_stats_initialized = False
        self._account_positions_initialized = False
        self._last_deal_time: dict[Instrument, pd.Timestamp] = {}
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
        """
        Process deals - position already updated from position messages.

        In position-primary mode, this method only:
        - Fills missing fee info from TCC
        - Tracks commissions from real deals
        - Does NOT update position (already done from position updates)
        """
        self._fill_missing_fee_info(instrument, deals)
        pos = self._positions.get(instrument)

        if pos is not None:
            for d in deals:
                _o_deals = self._processed_trades[d.order_id or ""]

                if d.id not in _o_deals:
                    _o_deals.append(d.id)

                    # Track commissions
                    if d.fee_amount and d.fee_amount > 0:
                        pos.commissions += d.fee_amount

                    logger.debug(
                        f"  [<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: "
                        f"deal {d.id[:20]}: {d.amount:+.4f} @ {d.price:.4f}"
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
            self._async_loop.submit(self._poller(name="sync_orders", coroutine=self._sync_orders, interval="1min"))
            # self._async_loop.submit(self._poller(name="volume_quota", coroutine=self._log_volume_quota, interval="10s"))

        except Exception as e:
            self.__error(f"Failed to start subscriptions: {e}")
            self._is_running = False
            raise

    async def _log_volume_quota(self):
        self.__info(f"Volume quota remaining: {self.ws_manager.get_volume_quota_remaining()}")

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
        # TODO: add additional fetch to get orders that we could have missed
        for order_id, order in orders.items():
            if order.status in ["NEW", "PENDING"] and order.time < now - recognize_timeframe("1min"):
                remove_orders.append((order_id, order))  # Keep order reference

        for order_id, order in remove_orders:
            # Create canceled order event to notify strategy
            canceled_order = Order(
                id=order.id,
                type=order.type,
                instrument=order.instrument,
                time=now,
                quantity=order.quantity,
                price=order.price,
                side=order.side,
                status="CANCELED",
                time_in_force=order.time_in_force,
                client_id=order.client_id,
                cost=order.cost,
                options=order.options,
            )

            # Send through channel for ProcessingManager to handle
            # This will trigger strategy.on_order_update with the canceled order
            self.channel.send((order.instrument, "order", canceled_order, False))

            # Remove from internal tracking
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

    def _calculate_realized_pnl(
        self,
        prev_qty: float,
        prev_avg: float,
        new_qty: float,
        new_avg: float,
        current_price: float,
    ) -> float:
        """
        Calculate realized PnL from position change.

        Args:
            prev_qty: Previous position quantity (signed: positive=long, negative=short)
            prev_avg: Previous average entry price
            new_qty: New position quantity (signed)
            new_avg: New average entry price
            current_price: Current market price (used as exec_price for closes)

        Returns:
            Realized PnL delta (positive = profit)
        """
        if prev_qty == 0:
            return 0.0  # Opening from flat - no realized PnL

        # Determine if position flipped (long→short or short→long)
        flipped = (prev_qty > 0) != (new_qty > 0) and new_qty != 0

        # Estimate execution price
        if flipped:
            exec_price = new_avg  # New entry price = old exit price (same trade)
        else:
            exec_price = current_price

        # Determine quantity closed
        if new_qty == 0:
            qty_closed = abs(prev_qty)  # Full close
        elif abs(new_qty) < abs(prev_qty):
            qty_closed = abs(prev_qty) - abs(new_qty)  # Partial close
        elif flipped:
            qty_closed = abs(prev_qty)  # Close entire old position before flip
        else:
            return 0.0  # Position increased - no realized PnL

        # Calculate PnL based on previous direction
        if prev_qty > 0:  # Was long
            return qty_closed * (exec_price - prev_avg)
        else:  # Was short
            return qty_closed * (prev_avg - exec_price)

    def _apply_position_from_server(
        self,
        instrument: Instrument,
        state: PositionState,
    ) -> None:
        """
        Apply position state from server as authoritative source.

        Calculates realized PnL from position change, then updates position state.

        Args:
            instrument: The instrument to update
            state: PositionState from server containing authoritative data
        """
        position = self.get_position(instrument)

        # Get previous state BEFORE updating
        prev_qty = position.quantity
        prev_avg = position.position_avg_price

        # Get current market price for exec_price estimation
        current_price = position.last_update_price
        if current_price == 0 or np.isnan(current_price):
            current_price = state.avg_entry_price  # Fallback

        # Calculate realized PnL from position change
        realized_pnl_delta = self._calculate_realized_pnl(
            prev_qty, prev_avg,
            state.quantity, state.avg_entry_price,
            current_price,
        )

        # Update position state from server
        position.quantity = state.quantity
        position.position_avg_price = state.avg_entry_price
        position.position_avg_price_funds = state.avg_entry_price

        # Accumulate realized PnL
        position.r_pnl += realized_pnl_delta

        # Set external margin if available
        if state.allocated_margin > 0:
            position.set_external_maint_margin(state.allocated_margin)

        # Update market price for unrealized PnL calculation if not set
        if position.last_update_price == 0 or np.isnan(position.last_update_price):
            position.last_update_price = state.avg_entry_price

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

        Position-primary mode:
        - Positions: Applied immediately as authoritative source (qty, avg_price, r_pnl)
        - Deals: Sent through channel for logging and fee extraction only
        """
        try:
            # Parse message into positions dict, deals list, and funding payments
            position_states, deals, funding_payments = parse_account_all_message(
                message, self.instrument_loader, self._lighter_account_index
            )

            now = cast(pd.Timestamp, pd.Timestamp(self.time_provider.time(), unit="ns"))

            # STEP 1: Apply position updates from server (authoritative)
            for instrument, pos_state in position_states.items():
                self._apply_position_from_server(instrument, pos_state)

            if not self._account_positions_initialized:
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

            # STEP 2: Send real deals through channel for logging and fee extraction
            for instrument, deal in deals:
                self._last_deal_time[instrument] = now
                self.channel.send((instrument, "deals", [deal], False))

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
