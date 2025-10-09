"""
LighterAccountProcessor - IAccountProcessor implementation for Lighter exchange.

Tracks account state via WebSocket subscriptions:
- Positions via account_all channel
- Balances via user_stats channel
- Orders via account_all channel
- Fills via executed_transaction channel
"""

import asyncio
from collections import defaultdict
from decimal import Decimal
from typing import Optional

from qubx import logger
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import (
    AssetBalance,
    CtrlChannel,
    Deal,
    FundingPayment,
    Instrument,
    ITimeProvider,
    Order,
    Position,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.interfaces import ISubscriptionManager

from .client import LighterClient
from .instruments import LighterInstrumentLoader
from .utils import lighter_order_side_to_qubx
from .websocket import LighterWebSocketManager


class LighterAccountProcessor(BasicAccountProcessor):
    """
    Account processor for Lighter exchange.

    Subscribes to WebSocket channels:
    - `account_all/{account_id}` - Positions, balances, orders
    - `user_stats/{account_id}` - Account statistics
    - `executed_transaction` - Fill notifications (global, filtered by account)

    Provides real-time account state tracking compatible with Qubx framework.
    """

    def __init__(
        self,
        account_id: str,
        client: LighterClient,
        instrument_loader: LighterInstrumentLoader,
        ws_manager: LighterWebSocketManager,
        channel: CtrlChannel,
        time_provider: ITimeProvider,
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

        # State tracking
        self._is_running = False
        self._subscription_manager: Optional[ISubscriptionManager] = None
        self._subscription_tasks: list = []

        # Lighter-specific IDs
        self._lighter_account_index = int(account_id)
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

        # Start subscription tasks
        logger.info("Starting Lighter account subscriptions")

        # Subscribe to account channels using asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in an event loop, schedule coroutines
            task1 = asyncio.create_task(self._subscribe_account_all())
            task2 = asyncio.create_task(self._subscribe_user_stats())
            task3 = asyncio.create_task(self._subscribe_executed_transactions())
            self._subscription_tasks = [task1, task2, task3]
        else:
            # Not in event loop, run them synchronously
            loop.run_until_complete(
                asyncio.gather(
                    self._subscribe_account_all(),
                    self._subscribe_user_stats(),
                    self._subscribe_executed_transactions(),
                )
            )

        logger.info("Lighter account subscriptions started")

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

    async def _subscribe_account_all(self):
        """
        Subscribe to account_all channel for positions, balances, and orders.

        Message format:
        {
            "channel": "account_all/{account_id}",
            "account": {
                "positions": [...],
                "balance": "123456.78",
                "orders": [...],
                ...
            }
        }
        """
        channel = f"account_all/{self._lighter_account_index}"
        logger.info(f"Subscribing to {channel}")

        try:
            await self.ws_manager.subscribe_account_all(
                account_id=self._lighter_account_index, handler=self._handle_account_all_message
            )
            logger.info(f"Successfully subscribed to {channel}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            raise

    async def _subscribe_user_stats(self):
        """
        Subscribe to user_stats channel for account statistics.

        Message format:
        {
            "channel": "user_stats/{account_id}",
            "stats": {
                "portfolio_value": "123456.78",
                "margin_usage": "45000.00",
                "available_balance": "78456.78",
                "leverage": "2.5",
                ...
            }
        }
        """
        channel = f"user_stats/{self._lighter_account_index}"
        logger.info(f"Subscribing to {channel}")

        try:
            await self.ws_manager.subscribe_user_stats(
                account_id=self._lighter_account_index, handler=self._handle_user_stats_message
            )
            logger.info(f"Successfully subscribed to {channel}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            raise

    async def _subscribe_executed_transactions(self):
        """
        Subscribe to executed_transaction channel for fill notifications.

        Message format:
        {
            "channel": "executed_transaction",
            "txs": [
                {
                    "tx_hash": "0x...",
                    "l1_address": "0x...",
                    "account_index": 225671,
                    "trades": {
                        "0": [{...}],  # market_id -> trades
                        "1": [{...}]
                    },
                    ...
                },
                ...
            ]
        }
        """
        channel = "executed_transaction"
        logger.info(f"Subscribing to {channel}")

        try:
            await self.ws_manager.subscribe_executed_transaction(handler=self._handle_executed_transaction_message)
            logger.info(f"Successfully subscribed to {channel}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            raise

    async def _handle_account_all_message(self, message: dict):
        """
        Handle account_all WebSocket messages.

        Updates positions, balances, and orders from account snapshot.
        """
        try:
            account_data = message.get("account", {})

            # Update positions
            positions = account_data.get("positions", [])
            self._update_positions_from_lighter(positions)

            # Update balance
            balance_str = account_data.get("balance")
            if balance_str:
                total_balance = float(balance_str)
                # Lighter uses USDC as base currency
                self.update_balance(self.base_currency, total_balance, 0.0)

            # Update orders
            orders = account_data.get("orders", [])
            self._update_orders_from_lighter(orders)

        except Exception as e:
            logger.error(f"Error handling account_all message: {e}")
            logger.exception(e)

    async def _handle_user_stats_message(self, message: dict):
        """
        Handle user_stats WebSocket messages.

        Updates account statistics like equity, margin usage, available balance.
        """
        try:
            stats = message.get("stats", {})

            # Update available balance (more accurate than account_all balance)
            available_balance_str = stats.get("available_balance")
            if available_balance_str:
                available = float(available_balance_str)
                # Note: This is free capital, not total
                # Total capital = portfolio_value
                portfolio_value_str = stats.get("portfolio_value")
                if portfolio_value_str:
                    total = float(portfolio_value_str)
                    locked = total - available
                    self.update_balance(self.base_currency, total, locked)

            # Log leverage and margin usage for monitoring
            leverage_str = stats.get("leverage")
            margin_usage_str = stats.get("margin_usage")
            if leverage_str and margin_usage_str:
                logger.debug(
                    f"Account stats - Leverage: {leverage_str}, Margin usage: {margin_usage_str}, "
                    f"Available: {available_balance_str}"
                )

        except Exception as e:
            logger.error(f"Error handling user_stats message: {e}")
            logger.exception(e)

    async def _handle_executed_transaction_message(self, message: dict):
        """
        Handle executed_transaction WebSocket messages.

        Processes fill notifications for this account.
        """
        try:
            txs = message.get("txs", [])

            for tx in txs:
                # Filter by account
                if tx.get("account_index") != self._lighter_account_index:
                    continue

                # Check if already processed
                tx_hash = tx.get("tx_hash")
                if tx_hash in self._processed_tx_hashes:
                    continue

                # Process trades in this transaction
                trades_by_market = tx.get("trades", {})
                for market_id_str, trade_list in trades_by_market.items():
                    market_id = int(market_id_str)
                    instrument = self._get_instrument_for_market_id(market_id)

                    if instrument is None:
                        logger.warning(f"Unknown market_id {market_id}, skipping fills")
                        continue

                    # Convert trades to deals
                    deals = self._convert_lighter_trades_to_deals(trade_list, instrument, tx)

                    # Process deals for this instrument
                    if deals:
                        self.process_deals(instrument, deals)

                # Mark transaction as processed
                self._processed_tx_hashes.add(tx_hash)

        except Exception as e:
            logger.error(f"Error handling executed_transaction message: {e}")
            logger.exception(e)

    def _update_positions_from_lighter(self, positions: list[dict]):
        """
        Update positions from Lighter format.

        Lighter position format:
        {
            "market_index": 0,
            "symbol": "BTC-USDC",
            "position": "1.2345",
            "sign": 1,  # 1 for long, -1 for short
            "avg_entry_price": "43500.50",
            "position_value": "53703.12",
            ...
        }
        """
        for pos_data in positions:
            try:
                # Parse position data
                market_id = pos_data.get("market_index")
                position_size_str = pos_data.get("position", "0")
                sign = pos_data.get("sign", 1)
                avg_entry_str = pos_data.get("avg_entry_price", "0")

                position_size = float(position_size_str)
                if position_size == 0:
                    continue  # Skip zero positions

                # Get instrument
                instrument = self._get_instrument_for_market_id(market_id)
                if instrument is None:
                    logger.warning(f"Unknown market_id {market_id}, skipping position")
                    continue

                # Calculate signed quantity
                signed_quantity = position_size * sign
                avg_entry = float(avg_entry_str)

                # Get or create position
                position = self.get_position(instrument)

                # Update position (Lighter provides absolute position, not incremental)
                # Reset position and set new values
                position.reset()
                position.quantity = signed_quantity
                position.position_avg_price_funds = avg_entry
                position.position_avg_price_base = avg_entry  # USDC-settled, so same

                logger.debug(
                    f"Updated position for {instrument.symbol}: "
                    f"qty={signed_quantity}, avg_entry={avg_entry}"
                )

            except Exception as e:
                logger.error(f"Error updating position from Lighter: {e}")
                continue

    def _update_orders_from_lighter(self, orders: list[dict]):
        """
        Update active orders from Lighter format.

        Lighter order format:
        {
            "order_id": "123456",
            "client_order_id": "789",
            "market_index": 0,
            "is_ask": false,
            "price": "43500.00",
            "initial_base_amount": "1.0",
            "remaining_base_amount": "0.5",
            "timestamp": 1234567890,
            "order_type": 0,  # 0=limit, 1=market
            "time_in_force": 1,  # 0=IOC, 1=GTC, 2=ALO
            ...
        }
        """
        # Clear existing orders (snapshot approach)
        current_order_ids = set(self._active_orders.keys())

        # Track orders from this update
        updated_order_ids = set()

        for order_data in orders:
            try:
                order_id = str(order_data.get("order_id"))
                updated_order_ids.add(order_id)

                # Get instrument
                market_id = order_data.get("market_index")
                instrument = self._get_instrument_for_market_id(market_id)
                if instrument is None:
                    logger.warning(f"Unknown market_id {market_id}, skipping order")
                    continue

                # Parse order fields
                is_ask = order_data.get("is_ask", False)
                side = "SELL" if is_ask else "BUY"
                price_str = order_data.get("price", "0")
                initial_str = order_data.get("initial_base_amount", "0")
                remaining_str = order_data.get("remaining_base_amount", "0")
                timestamp_ms = order_data.get("timestamp", 0)

                price = float(price_str)
                initial_size = float(initial_str)
                remaining_size = float(remaining_str)
                filled_size = initial_size - remaining_size

                # Determine status
                if remaining_size == 0:
                    status = "filled"
                elif filled_size > 0:
                    status = "partially_filled"
                else:
                    status = "open"

                # Determine type
                order_type_int = order_data.get("order_type", 0)
                order_type = "limit" if order_type_int == 0 else "market"

                # Create or update Order
                order = Order(
                    instrument=instrument,
                    side=side,
                    type=order_type,
                    quantity=initial_size,
                    price=price if order_type == "limit" else None,
                    time=self.time_provider.now(),  # Use current time, Lighter timestamp is in ms
                    order_id=order_id,
                    client_id=str(order_data.get("client_order_id", "")),
                    status=status,
                    filled_quantity=filled_size,
                )

                # Add or update in active orders
                self._active_orders[order_id] = order

                logger.debug(
                    f"Updated order {order_id}: {instrument.symbol} {side} "
                    f"{initial_size} @ {price} (filled: {filled_size})"
                )

            except Exception as e:
                logger.error(f"Error updating order from Lighter: {e}")
                continue

        # Remove orders that are no longer active
        removed_order_ids = current_order_ids - updated_order_ids
        for order_id in removed_order_ids:
            self.remove_order(order_id)
            logger.debug(f"Removed order {order_id} (no longer in snapshot)")

    def _convert_lighter_trades_to_deals(
        self, trades: list[dict], instrument: Instrument, tx: dict
    ) -> list[Deal]:
        """
        Convert Lighter trade format to Qubx Deal objects.

        Lighter trade format:
        {
            "trade_id": 212690112,
            "market_id": 0,
            "size": "1.3792",
            "price": "4335.02",
            "is_maker_ask": false,
            "timestamp": 1760040869198,
            "bid_order_id": "123",
            "ask_order_id": "456",
            "bid_account_id": 225671,
            "ask_account_id": 999,
            ...
        }
        """
        deals = []

        for trade in trades:
            try:
                # Determine if we're the buyer or seller
                bid_account = trade.get("bid_account_id")
                ask_account = trade.get("ask_account_id")
                is_buyer = bid_account == self._lighter_account_index
                is_seller = ask_account == self._lighter_account_index

                if not is_buyer and not is_seller:
                    # Not our trade
                    continue

                # Parse trade fields
                trade_id = str(trade.get("trade_id"))
                size_str = trade.get("size", "0")
                price_str = trade.get("price", "0")
                timestamp_ms = trade.get("timestamp", 0)

                size = float(size_str)
                price = float(price_str)

                # Determine side (from our perspective)
                if is_buyer:
                    side = "B"  # Lighter format
                    order_id = str(trade.get("bid_order_id", ""))
                else:
                    side = "S"  # Lighter format
                    order_id = str(trade.get("ask_order_id", ""))

                # Create Deal
                deal = Deal(
                    instrument=instrument,
                    quantity=size,
                    price=price,
                    time=self.time_provider.now(),  # Use current time
                    side=lighter_order_side_to_qubx(side),
                    order_id=order_id,
                    trade_id=trade_id,
                    commission=0.0,  # Will be calculated by tcc
                    commission_asset=self.base_currency,
                )

                deals.append(deal)

                logger.debug(
                    f"Converted trade {trade_id}: {instrument.symbol} {side} {size} @ {price}"
                )

            except Exception as e:
                logger.error(f"Error converting Lighter trade to Deal: {e}")
                continue

        return deals

    def _get_instrument_for_market_id(self, market_id: int) -> Optional[Instrument]:
        """
        Get Qubx Instrument for a Lighter market_id.

        Args:
            market_id: Lighter market ID

        Returns:
            Instrument object or None if not found
        """
        ticker = self.instrument_loader.market_id_to_ticker.get(market_id)
        if ticker is None:
            return None

        # Convert to Qubx instrument lookup format
        # Format: XLIGHTER:SWAP:BTC-USDC
        instrument_key = f"XLIGHTER:SWAP:{ticker}"

        # Get from instrument loader's cache
        return self.instrument_loader.instruments_cache.get(instrument_key)
