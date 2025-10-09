"""
LighterBroker - IBroker implementation for Lighter exchange.

Handles order operations:
- Order creation (market/limit)
- Order cancellation
- Order modification
- Order tracking
"""

import asyncio
import uuid
from typing import Any, Optional

from qubx import logger
from qubx.core.basics import CtrlChannel, Instrument, ITimeProvider, Order, OrderSide
from qubx.core.errors import ErrorLevel, OrderCancellationError, OrderCreationError, create_error_event
from qubx.core.exceptions import InvalidOrderParameters, OrderNotFound
from qubx.core.interfaces import IAccountProcessor, IBroker, IDataProvider

from .client import LighterClient
from .constants import (
    ORDER_TIME_IN_FORCE_GTT,
    ORDER_TIME_IN_FORCE_IOC,
    ORDER_TIME_IN_FORCE_POST_ONLY,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
)
from .instruments import LighterInstrumentLoader
# Utils imported as needed


class LighterBroker(IBroker):
    """
    Broker for Lighter exchange.

    Supports:
    - Market and limit orders
    - Order cancellation
    - Order modification (via cancel + replace)
    - WebSocket order updates (via AccountProcessor)
    """

    def __init__(
        self,
        client: LighterClient,
        instrument_loader: LighterInstrumentLoader,
        channel: CtrlChannel,
        time_provider: ITimeProvider,
        account: IAccountProcessor,
        data_provider: IDataProvider,
        cancel_timeout: int = 30,
        cancel_retry_interval: int = 2,
        max_cancel_retries: int = 10,
    ):
        """
        Initialize Lighter broker.

        Args:
            client: LighterClient for REST/WebSocket operations
            instrument_loader: Instrument loader with market_id mappings
            channel: Control channel for sending events
            time_provider: Time provider for timestamps
            account: Account processor for tracking orders/positions
            data_provider: Data provider (not used for orders, for consistency)
            cancel_timeout: Timeout for order cancellation (seconds)
            cancel_retry_interval: Retry interval for cancellation (seconds)
            max_cancel_retries: Maximum cancellation retry attempts
        """
        self.client = client
        self.instrument_loader = instrument_loader
        self.channel = channel
        self.time_provider = time_provider
        self.account = account
        self.data_provider = data_provider
        self.cancel_timeout = cancel_timeout
        self.cancel_retry_interval = cancel_retry_interval
        self.max_cancel_retries = max_cancel_retries

        # Track client order IDs
        self._client_order_ids: dict[str, str] = {}  # client_id -> exchange_order_id

    @property
    def is_simulated_trading(self) -> bool:
        """Check if broker is in simulation mode (always False for live)"""
        return False

    def send_order(
        self,
        instrument: Instrument,
        order_side: OrderSide,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **options,
    ) -> Order:
        """
        Send order synchronously.

        Args:
            instrument: Instrument to trade
            order_side: "buy" or "sell"
            order_type: "market" or "limit"
            amount: Order amount (in base currency)
            price: Limit price (required for limit orders)
            client_id: Client-specified order ID
            time_in_force: "gtc" (default), "ioc", or "post_only"
            **options: Additional order parameters (reduce_only, etc.)

        Returns:
            Order: Created order object

        Raises:
            InvalidOrderParameters: If order parameters are invalid
        """
        # Run async order creation in sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context, schedule as task
            future = asyncio.ensure_future(
                self._create_order(
                    instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
                )
            )
            # Block until complete
            return loop.run_until_complete(future)
        else:
            # No event loop, create one
            return asyncio.run(
                self._create_order(
                    instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
                )
            )

    def send_order_async(
        self,
        instrument: Instrument,
        order_side: OrderSide,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **options,
    ) -> Any:
        """
        Send order asynchronously.

        Errors will be sent through the channel.

        Returns:
            Task/Future that will contain the order
        """

        async def _execute_order_with_channel_errors():
            try:
                order = await self._create_order(
                    instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
                )
                return order
            except Exception as error:
                self._post_order_error_to_channel(
                    error, instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
                )
                return None

        return asyncio.create_task(_execute_order_with_channel_errors())

    async def _create_order(
        self,
        instrument: Instrument,
        order_side: OrderSide,
        order_type: str,
        amount: float,
        price: float | None,
        client_id: str | None,
        time_in_force: str,
        **options,
    ) -> Order:
        """
        Create order via Lighter API.

        Args:
            instrument: Instrument to trade
            order_side: Order side
            order_type: Order type
            amount: Order amount
            price: Limit price
            client_id: Client order ID
            time_in_force: Time in force
            **options: Additional parameters

        Returns:
            Order object

        Raises:
            InvalidOrderParameters: If parameters are invalid
        """
        # Validate parameters
        if order_type not in ["market", "limit"]:
            raise InvalidOrderParameters(f"Invalid order type: {order_type}")

        if order_type == "limit" and price is None:
            raise InvalidOrderParameters("Limit orders require a price")

        # Get market_id
        market_id = self.instrument_loader.get_market_id(instrument.symbol)
        if market_id is None:
            raise InvalidOrderParameters(f"Market ID not found for {instrument.symbol}")

        # Generate client_id if not provided
        if client_id is None:
            client_id = str(uuid.uuid4())

        # Convert parameters to Lighter format
        is_buy = order_side.upper() in ["BUY", "B"]
        lighter_order_type = ORDER_TYPE_MARKET if order_type == "market" else ORDER_TYPE_LIMIT

        # Convert time_in_force
        tif_map = {
            "gtc": ORDER_TIME_IN_FORCE_GTT,
            "gtt": ORDER_TIME_IN_FORCE_GTT,
            "ioc": ORDER_TIME_IN_FORCE_IOC,
            "post_only": ORDER_TIME_IN_FORCE_POST_ONLY,
        }
        lighter_tif = tif_map.get(time_in_force.lower(), ORDER_TIME_IN_FORCE_GTT)

        # Extract additional options
        reduce_only = options.get("reduce_only", False)
        post_only = time_in_force.lower() == "post_only"

        logger.info(
            f"Creating order: {order_side} {amount} {instrument.symbol} "
            f"@ {price if price else 'MARKET'} (type={order_type}, tif={time_in_force})"
        )

        try:
            # Create order via Lighter SDK
            created_tx, resp, err = await self.client.create_order(
                market_id=market_id,
                is_buy=is_buy,
                size=amount,
                price=price,
                order_type=lighter_order_type,
                time_in_force=lighter_tif,
                reduce_only=reduce_only,
                post_only=post_only,
            )

            if err:
                raise InvalidOrderParameters(f"Lighter API error: {err}")

            # Parse response to get order ID
            # Lighter returns transaction hash, we'll use it as order ID
            order_id = resp.tx_hash if resp and hasattr(resp, "tx_hash") else str(uuid.uuid4())

            # Track client order ID
            self._client_order_ids[client_id] = order_id

            # Create Order object
            from qubx.core.basics import OrderStatus, OrderType

            order = Order(
                id=order_id,
                type=OrderType.MARKET if order_type == "market" else OrderType.LIMIT,
                instrument=instrument,
                time=self.time_provider.time(),
                quantity=amount,
                price=price if price else 0.0,
                side=order_side,
                status=OrderStatus.OPEN,  # Will be updated via WebSocket
                time_in_force=time_in_force,
                client_id=client_id,
                options={"reduce_only": reduce_only} if reduce_only else {},
            )

            logger.info(f"Order created: {order_id} ({client_id})")
            return order

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise InvalidOrderParameters(f"Order creation failed: {e}") from e

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order synchronously.

        Args:
            order_id: Order ID or client order ID to cancel

        Returns:
            True if cancellation successful

        Raises:
            OrderNotFound: If order not found
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.ensure_future(self._cancel_order(order_id))
            return loop.run_until_complete(future)
        else:
            return asyncio.run(self._cancel_order(order_id))

    def cancel_order_async(self, order_id: str) -> None:
        """
        Cancel order asynchronously.

        Args:
            order_id: Order ID or client order ID to cancel
        """

        async def _cancel_with_errors():
            try:
                await self._cancel_order(order_id)
            except Exception as error:
                self._post_cancel_error_to_channel(error, order_id)

        asyncio.create_task(_cancel_with_errors())

    async def _cancel_order(self, order_id: str) -> bool:
        """
        Cancel order via Lighter API.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful

        Raises:
            OrderNotFound: If order not found
        """
        logger.info(f"Canceling order: {order_id}")

        try:
            # Check if this is a client order ID
            if order_id in self._client_order_ids:
                exchange_order_id = self._client_order_ids[order_id]
            else:
                exchange_order_id = order_id

            # Get order details to find market_id
            # For now, we'll need to track this separately or get from account
            orders = self.account.get_orders()
            order = None
            for ord in orders.values():
                if ord.id == exchange_order_id or ord.client_id == order_id:
                    order = ord
                    break

            if order is None:
                raise OrderNotFound(f"Order not found: {order_id}")

            # Get market_id
            market_id = self.instrument_loader.get_market_id(order.instrument.symbol)
            if market_id is None:
                raise OrderNotFound(f"Market ID not found for {order.instrument.symbol}")

            # Cancel via Lighter SDK
            created_tx, resp, err = await self.client.cancel_order(order_id=int(exchange_order_id), market_id=market_id)

            if err:
                logger.error(f"Failed to cancel order {order_id}: {err}")
                return False

            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise OrderNotFound(f"Order cancellation failed: {e}") from e

    def cancel_orders(self, instrument: Instrument) -> None:
        """
        Cancel all orders for an instrument.

        Args:
            instrument: Instrument to cancel orders for
        """
        orders = self.account.get_orders(instrument=instrument)

        for order_id in orders.keys():
            try:
                self.cancel_order_async(order_id)
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")

    def update_order(self, order_id: str, price: float, amount: float) -> Order:
        """
        Update order (via cancel + replace).

        Lighter doesn't support direct order modification,
        so we cancel the old order and create a new one.

        Args:
            order_id: Order ID to update
            price: New price
            amount: New amount

        Returns:
            New order object

        Raises:
            OrderNotFound: If order not found
        """
        # Get original order
        orders = self.account.get_orders()
        order = None
        for ord in orders.values():
            if ord.id == order_id or ord.client_id == order_id:
                order = ord
                break

        if order is None:
            raise OrderNotFound(f"Order not found: {order_id}")

        # Cancel old order
        logger.info(f"Updating order {order_id}: cancel + replace")
        self.cancel_order(order_id)

        # Create new order with same parameters but new price/amount
        return self.send_order(
            instrument=order.instrument,
            order_side=order.side,
            order_type=order.type,
            amount=amount,
            price=price,
            time_in_force=order.time_in_force,
            reduce_only=order.reduce_only,
        )

    def _post_order_error_to_channel(
        self,
        error: Exception,
        instrument: Instrument,
        order_side: OrderSide,
        order_type: str,
        amount: float,
        price: float | None,
        client_id: str | None,
        time_in_force: str,
        **options,
    ):
        """Post order creation error to channel"""
        level = ErrorLevel.MEDIUM

        if "insufficient" in str(error).lower():
            level = ErrorLevel.HIGH
            logger.error(f"INSUFFICIENT FUNDS for {order_side} {amount} {instrument.symbol}")
        elif "invalid" in str(error).lower():
            level = ErrorLevel.LOW
            logger.error(f"INVALID ORDER for {order_side} {amount} {instrument.symbol}: {error}")
        else:
            logger.error(f"Order creation error: {error}")

        error_event = OrderCreationError(
            timestamp=self.time_provider.time(),
            message=f"Error: {str(error)}",
            level=level,
            instrument=instrument,
            amount=amount,
            price=price,
            order_type=order_type,
            side=order_side,
            error=error,
        )
        self.channel.send(create_error_event(error_event))

    def _post_cancel_error_to_channel(self, error: Exception, order_id: str):
        """Post order cancellation error to channel"""
        level = ErrorLevel.MEDIUM

        if "not found" in str(error).lower():
            level = ErrorLevel.LOW
            logger.error(f"Order not found for cancellation: {order_id}")
        else:
            logger.error(f"Order cancellation error: {error}")

        error_event = OrderCancellationError(
            timestamp=self.time_provider.time(),
            message=f"Failed to cancel order {order_id}: {str(error)}",
            level=level,
            order_id=order_id,
            error=error,
        )
        self.channel.send(create_error_event(error_event))
