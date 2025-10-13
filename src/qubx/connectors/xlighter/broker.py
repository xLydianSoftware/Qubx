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
from typing import Any

from qubx import logger
from qubx.core.basics import CtrlChannel, Instrument, ITimeProvider, Order, OrderSide
from qubx.core.errors import ErrorLevel, OrderCancellationError, OrderCreationError, create_error_event
from qubx.core.exceptions import InvalidOrderParameters, OrderNotFound
from qubx.core.interfaces import IAccountProcessor, IBroker, IDataProvider
from qubx.utils.misc import AsyncThreadLoop

from .client import LighterClient
from .constants import (
    DEFAULT_28_DAY_ORDER_EXPIRY,
    DEFAULT_IOC_EXPIRY,
    ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
    ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
    ORDER_TIME_IN_FORCE_POST_ONLY,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    TX_TYPE_CANCEL_ORDER,
    TX_TYPE_CREATE_ORDER,
    TX_TYPE_MODIFY_ORDER,
)
from .instruments import LighterInstrumentLoader
from .websocket import LighterWebSocketManager

# Utils imported as needed


class LighterBroker(IBroker):
    """
    Broker for Lighter exchange.

    Supports:
    - Market and limit orders
    - Order cancellation
    - Native order modification (via sign_modify_order)
    - WebSocket order updates (via AccountProcessor)
    """

    def __init__(
        self,
        client: LighterClient,
        instrument_loader: LighterInstrumentLoader,
        ws_manager: LighterWebSocketManager,
        channel: CtrlChannel,
        time_provider: ITimeProvider,
        account: IAccountProcessor,
        data_provider: IDataProvider,
        loop: asyncio.AbstractEventLoop,
        cancel_timeout: int = 30,
        cancel_retry_interval: int = 2,
        max_cancel_retries: int = 10,
    ):
        """
        Initialize Lighter broker.

        Args:
            client: LighterClient for transaction signing
            instrument_loader: Instrument loader with market_id mappings
            ws_manager: WebSocket manager for sending transactions
            channel: Control channel for sending events
            time_provider: Time provider for timestamps
            account: Account processor for tracking orders/positions
            data_provider: Data provider (not used for orders, for consistency)
            loop: Event loop for async operations (from client)
            cancel_timeout: Timeout for order cancellation (seconds)
            cancel_retry_interval: Retry interval for cancellation (seconds)
            max_cancel_retries: Maximum cancellation retry attempts
        """
        self.client = client
        self.instrument_loader = instrument_loader
        self.ws_manager = ws_manager
        self.channel = channel
        self.time_provider = time_provider
        self.account = account
        self.data_provider = data_provider
        self.cancel_timeout = cancel_timeout
        self.cancel_retry_interval = cancel_retry_interval
        self.max_cancel_retries = max_cancel_retries

        # Async thread loop for submitting tasks to client's event loop
        self._async_loop = AsyncThreadLoop(loop)

        # Track client order IDs and indices
        self._client_order_ids: dict[str, str] = {}  # client_id -> exchange_order_id
        self._client_order_indices: dict[str, int] = {}  # client_id -> client_order_index

    @property
    def is_simulated_trading(self) -> bool:
        """Check if broker is in simulation mode (always False for live)"""
        return False

    def exchange(self) -> str:
        """Return exchange name"""
        return "LIGHTER"

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
        # Submit async order creation to event loop and wait for result
        future = self._async_loop.submit(
            self._create_order(instrument, order_side, order_type, amount, price, client_id, time_in_force, **options)
        )
        return future.result()

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
        Create order via local signing + WebSocket submission.

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
        is_ask = not is_buy  # SignerClient uses is_ask
        lighter_order_type = ORDER_TYPE_MARKET if order_type == "market" else ORDER_TYPE_LIMIT

        # Convert time_in_force
        tif_map = {
            "gtc": ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
            "gtt": ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
            "ioc": ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
            "post_only": ORDER_TIME_IN_FORCE_POST_ONLY,
        }
        lighter_tif = tif_map.get(time_in_force.lower(), ORDER_TIME_IN_FORCE_GOOD_TILL_TIME)

        # Extract additional options
        reduce_only = options.get("reduce_only", False)

        # Market orders MUST use IOC (Immediate or Cancel) time in force
        # This is a requirement of Lighter's API
        if order_type == "market":
            lighter_tif = ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL
            order_expiry = DEFAULT_IOC_EXPIRY
        else:
            # Limit orders use the mapped TIF and 28-day expiry
            order_expiry = DEFAULT_28_DAY_ORDER_EXPIRY

        # Market order slippage protection
        # Lighter requires a price even for market orders as a slippage bound
        if order_type == "market" and price is None:
            # Get max slippage from options (default 5%)
            max_slippage = options.get("max_slippage", 0.05)

            # Get current mid price from data provider
            try:
                quote = self.data_provider.get_quote(instrument)
                if quote is None:
                    raise InvalidOrderParameters(
                        f"Cannot get quote for {instrument.symbol} - no market data available for market order"
                    )

                mid_price = quote.mid_price()
                if mid_price is None or mid_price <= 0:
                    raise InvalidOrderParameters(
                        f"Invalid mid price {mid_price} for {instrument.symbol} - cannot calculate market order bound"
                    )

                # Calculate protected price based on order side
                # For BUY: willing to pay up to mid + slippage
                # For SELL: willing to accept down to mid - slippage
                if is_buy:
                    price = mid_price * (1 + max_slippage)
                else:
                    price = mid_price * (1 - max_slippage)

                logger.debug(
                    f"Market order slippage protection: mid={mid_price:.4f}, "
                    f"slippage={max_slippage * 100:.1f}%, protected_price={price:.4f}"
                )

            except Exception as e:
                raise InvalidOrderParameters(
                    f"Failed to calculate market order price for {instrument.symbol}: {e}"
                ) from e

        # Convert amounts to Lighter's integer format (scaled by market-specific decimals)
        # Lighter markets have different decimal precision for price and size
        # Use instrument's built-in precision properties
        base_amount_int = int(amount * (10**instrument.size_precision))
        price_int = int(price * (10**instrument.price_precision)) if price is not None else 0

        # Use client_id hash as client_order_index
        client_order_index = abs(hash(client_id)) % (10**9)  # Keep it within reasonable bounds
        client_id = str(client_order_index)

        logger.info(
            f"Creating order: {order_side} {amount} {instrument.symbol} "
            f"@ {price if price else 'MARKET'} (type={order_type}, tif={time_in_force})"
        )
        # logger.debug(
        #     f"Decimal conversion: amount={amount} → {base_amount_int} (10^{instrument.size_precision}), "
        #     f"price={price} → {price_int} (10^{instrument.price_precision})"
        # )

        try:
            # Step 1: Sign transaction locally
            signer = self.client.signer_client
            tx_info, error = signer.sign_create_order(
                market_index=market_id,
                client_order_index=client_order_index,
                base_amount=base_amount_int,
                price=price_int,
                is_ask=is_ask,
                order_type=lighter_order_type,
                time_in_force=lighter_tif,
                reduce_only=int(reduce_only),
                trigger_price=0,  # Not using trigger orders
                order_expiry=order_expiry,
            )

            if error or tx_info is None:
                raise InvalidOrderParameters(f"Order signing failed: {error}")

            # Step 2: Submit via WebSocket
            response = await self.ws_manager.send_tx(tx_type=TX_TYPE_CREATE_ORDER, tx_info=tx_info, tx_id=client_id)

            # Use the transaction ID from response as order ID
            order_id = response.get("tx_id", client_id)

            # Track client order ID and index
            self._client_order_ids[client_id] = order_id
            self._client_order_indices[client_id] = client_order_index

            # Create Order object
            order = Order(
                id=order_id,
                type="MARKET" if order_type == "market" else "LIMIT",
                instrument=instrument,
                time=self.time_provider.time(),
                quantity=amount,
                price=price if price else 0.0,
                side=order_side,
                status="NEW",  # Will be updated to OPEN via WebSocket when confirmed
                time_in_force=time_in_force,
                client_id=client_id,
                options={"reduce_only": reduce_only} if reduce_only else {},
            )

            # Register order with account processor immediately
            # This makes it available for cancellation before WebSocket updates arrive
            self.account.process_order(order)

            logger.info(f"Order submitted via WebSocket: {order_id} ({client_id})")
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
        # Submit async cancellation to event loop and wait for result
        future = self._async_loop.submit(self._cancel_order(order_id))
        return future.result()

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
        Cancel order via local signing + WebSocket submission.

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
                client_id = order_id
            else:
                exchange_order_id = order_id
                client_id = None

            # Get order details to find market_id and client_id
            orders = self.account.get_orders()
            order = None
            for ord in orders.values():
                if ord.id == exchange_order_id or ord.client_id == order_id:
                    order = ord
                    if client_id is None and ord.client_id:
                        client_id = ord.client_id
                    break

            if order is None:
                raise OrderNotFound(f"Order not found: {order_id}")

            # Get market_id
            market_id = self.instrument_loader.get_market_id(order.instrument.symbol)
            if market_id is None:
                raise OrderNotFound(f"Market ID not found for {order.instrument.symbol}")

            # Get the client_order_index we used during creation
            # If not available, compute it the same way as during creation
            if client_id and client_id in self._client_order_indices:
                order_index = self._client_order_indices[client_id]
            elif client_id:
                # Fallback: compute using same algorithm as creation
                order_index = abs(hash(client_id)) % (10**9)
            elif exchange_order_id.isdigit():
                order_index = int(exchange_order_id)
            else:
                # Last resort: hash the exchange_order_id but constrain to 56-bit limit
                order_index = abs(hash(exchange_order_id)) % (2**56)

            # Step 1: Sign cancellation transaction locally
            signer = self.client.signer_client
            tx_info, error = signer.sign_cancel_order(market_index=market_id, order_index=order_index)

            if error or tx_info is None:
                logger.error(f"Order cancellation signing failed: {error}")
                return False

            # Step 2: Submit via WebSocket
            await self.ws_manager.send_tx(tx_type=TX_TYPE_CANCEL_ORDER, tx_info=tx_info, tx_id=f"cancel_{order_id}")

            logger.info(f"Order cancellation submitted via WebSocket: {order_id}")
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
        Update order via native order modification.

        Uses Lighter's sign_modify_order for atomic order updates.

        Args:
            order_id: Order ID to update
            price: New price
            amount: New amount

        Returns:
            Updated order object

        Raises:
            OrderNotFound: If order not found
        """
        # Submit async modification to event loop and wait for result
        future = self._async_loop.submit(self._modify_order(order_id, price, amount))
        return future.result()

    async def _modify_order(self, order_id: str, price: float, amount: float) -> Order:
        """
        Modify order via local signing + WebSocket submission.

        Args:
            order_id: Order ID to modify
            price: New price
            amount: New amount

        Returns:
            Updated order object

        Raises:
            OrderNotFound: If order not found
        """
        try:
            # Check if this is a client order ID
            if order_id in self._client_order_ids:
                exchange_order_id = self._client_order_ids[order_id]
                client_id = order_id
            else:
                exchange_order_id = order_id
                client_id = None

            # Get order details
            orders = self.account.get_orders()
            order = None
            for ord in orders.values():
                if ord.id == exchange_order_id or ord.client_id == order_id:
                    order = ord
                    if client_id is None and ord.client_id:
                        client_id = ord.client_id
                    break

            if order is None:
                raise OrderNotFound(f"Order not found: {order_id}")

            # Get market_id
            market_id = self.instrument_loader.get_market_id(order.instrument.symbol)
            if market_id is None:
                raise OrderNotFound(f"Market ID not found for {order.instrument.symbol}")

            # Get the order_index
            if client_id and client_id in self._client_order_indices:
                order_index = self._client_order_indices[client_id]
            elif order.id.isdigit():
                order_index = order.id
            else:
                raise OrderNotFound(f"Order index not found for {order_id}")

            # Convert price and amount to Lighter's integer format
            instrument = order.instrument
            base_amount_int = int(amount * (10**instrument.size_precision))
            price_int = int(price * (10**instrument.price_precision))

            logger.debug(f"Modify order {order_id}: amount={order.quantity} → {amount}, price={order.price} → {price}")

            # Step 1: Sign modification transaction locally
            signer = self.client.signer_client
            tx_info, error = signer.sign_modify_order(
                market_index=market_id,
                order_index=order_index,
                base_amount=base_amount_int,
                price=price_int,
                trigger_price=0,  # Not using trigger orders
            )

            if error or tx_info is None:
                raise OrderNotFound(f"Order modification signing failed: {error}")

            # Step 2: Submit via WebSocket
            await self.ws_manager.send_tx(tx_type=TX_TYPE_MODIFY_ORDER, tx_info=tx_info, tx_id=f"modify_{order_id}")

            # Create updated Order object
            updated_order = Order(
                id=order.id,
                type=order.type,
                instrument=order.instrument,
                time=self.time_provider.time(),
                quantity=amount,
                price=price,
                side=order.side,
                status="OPEN",  # Will be updated via WebSocket
                time_in_force=order.time_in_force,
                client_id=order.client_id,
                options=order.options,
            )

            logger.info(f"Order modification submitted via WebSocket: {order_id}")
            return updated_order

        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            raise OrderNotFound(f"Order modification failed: {e}") from e

    async def send_orders_batch(
        self,
        orders: list[dict],
    ) -> list[Order]:
        """
        Send multiple orders in a single batch via WebSocket.

        This is useful for HFT applications where you need to submit multiple
        orders atomically (e.g., spread orders, hedging, multi-leg strategies).

        Args:
            orders: List of order dicts, each with keys:
                - instrument: Instrument
                - order_side: OrderSide
                - order_type: str
                - amount: float
                - price: float | None
                - client_id: str | None (optional)
                - time_in_force: str (optional, default="gtc")
                - **options: dict (optional, e.g., reduce_only)

        Returns:
            List of Order objects

        Raises:
            InvalidOrderParameters: If any order parameters are invalid

        Example:
            >>> orders = [
            ...     {"instrument": btc, "order_side": "BUY", "order_type": "limit",
            ...      "amount": 0.1, "price": 40000},
            ...     {"instrument": eth, "order_side": "SELL", "order_type": "limit",
            ...      "amount": 1.0, "price": 3000},
            ... ]
            >>> created_orders = await broker.send_orders_batch(orders)
        """
        if not orders:
            raise InvalidOrderParameters("Cannot send empty batch")

        if len(orders) > 50:
            raise InvalidOrderParameters(f"Batch size cannot exceed 50 orders, got {len(orders)}")

        tx_types = []
        tx_infos = []
        order_objects = []

        logger.info(f"Creating order batch: {len(orders)} orders")

        try:
            # Sign all orders locally
            for order_params in orders:
                instrument = order_params["instrument"]
                order_side = order_params["order_side"]
                order_type = order_params["order_type"]
                amount = order_params["amount"]
                price = order_params.get("price")
                client_id = order_params.get("client_id") or str(uuid.uuid4())
                time_in_force = order_params.get("time_in_force", "gtc")
                options = order_params.get("options", {})

                # Validate
                if order_type not in ["market", "limit"]:
                    raise InvalidOrderParameters(f"Invalid order type: {order_type}")
                if order_type == "limit" and price is None:
                    raise InvalidOrderParameters("Limit orders require a price")

                # Get market_id
                market_id = self.instrument_loader.get_market_id(instrument.symbol)
                if market_id is None:
                    raise InvalidOrderParameters(f"Market ID not found for {instrument.symbol}")

                # Convert parameters
                is_buy = order_side.upper() in ["BUY", "B"]
                is_ask = not is_buy
                lighter_order_type = ORDER_TYPE_MARKET if order_type == "market" else ORDER_TYPE_LIMIT

                tif_map = {
                    "gtc": ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                    "gtt": ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                    "ioc": ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
                    "post_only": ORDER_TIME_IN_FORCE_POST_ONLY,
                }
                lighter_tif = tif_map.get(time_in_force.lower(), ORDER_TIME_IN_FORCE_GOOD_TILL_TIME)

                reduce_only = options.get("reduce_only", False)

                # Market orders MUST use IOC (Immediate or Cancel) time in force
                # This is a requirement of Lighter's API
                if order_type == "market":
                    lighter_tif = ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL
                    order_expiry = DEFAULT_IOC_EXPIRY
                else:
                    # Limit orders use the mapped TIF and 28-day expiry
                    order_expiry = DEFAULT_28_DAY_ORDER_EXPIRY

                # Market order slippage protection
                # Lighter requires a price even for market orders as a slippage bound
                if order_type == "market" and price is None:
                    # Get max slippage from options (default 5%)
                    max_slippage = options.get("max_slippage", 0.05)

                    # Get current mid price from data provider
                    try:
                        quote = self.data_provider.get_quote(instrument)
                        if quote is None:
                            raise InvalidOrderParameters(
                                f"Cannot get quote for {instrument.symbol} - no market data available for market order"
                            )

                        mid_price = quote.mid_price()
                        if mid_price is None or mid_price <= 0:
                            raise InvalidOrderParameters(
                                f"Invalid mid price {mid_price} for {instrument.symbol} - cannot calculate market order bound"
                            )

                        # Calculate protected price based on order side
                        # For BUY: willing to pay up to mid + slippage
                        # For SELL: willing to accept down to mid - slippage
                        if is_buy:
                            price = mid_price * (1 + max_slippage)
                        else:
                            price = mid_price * (1 - max_slippage)

                        logger.debug(
                            f"Market order slippage protection (batch): mid={mid_price:.4f}, "
                            f"slippage={max_slippage * 100:.1f}%, protected_price={price:.4f}"
                        )

                    except Exception as e:
                        raise InvalidOrderParameters(
                            f"Failed to calculate market order price for {instrument.symbol}: {e}"
                        ) from e

                # Convert amounts using market-specific decimals
                base_amount_int = int(amount * (10**instrument.size_precision))
                price_int = int(price * (10**instrument.price_precision)) if price is not None else 0
                client_order_index = abs(hash(client_id)) % (10**9)

                # Sign transaction
                signer = self.client.signer_client
                tx_info, error = signer.sign_create_order(
                    market_index=market_id,
                    client_order_index=client_order_index,
                    base_amount=base_amount_int,
                    price=price_int,
                    is_ask=is_ask,
                    order_type=lighter_order_type,
                    time_in_force=lighter_tif,
                    reduce_only=int(reduce_only),
                    trigger_price=0,
                    order_expiry=order_expiry,
                )

                if error or tx_info is None:
                    raise InvalidOrderParameters(f"Order signing failed for {instrument.symbol}: {error}")

                tx_types.append(TX_TYPE_CREATE_ORDER)
                tx_infos.append(tx_info)

                # Track client order ID and index
                self._client_order_ids[client_id] = client_id  # Will be updated when tx is confirmed
                self._client_order_indices[client_id] = client_order_index

                # Create Order object
                order = Order(
                    id=client_id,  # Will be updated when tx is confirmed
                    type="MARKET" if order_type == "market" else "LIMIT",
                    instrument=instrument,
                    time=self.time_provider.time(),
                    quantity=amount,
                    price=price if price else 0.0,
                    side=order_side,
                    status="PENDING",
                    time_in_force=time_in_force,
                    client_id=client_id,
                    options={"reduce_only": reduce_only} if reduce_only else {},
                )
                order_objects.append(order)

            # Submit batch via WebSocket
            response = await self.ws_manager.send_batch_tx(tx_types=tx_types, tx_infos=tx_infos)

            logger.info(f"Order batch submitted via WebSocket: {response.get('count')} orders")
            return order_objects

        except Exception as e:
            logger.error(f"Failed to create order batch: {e}")
            raise InvalidOrderParameters(f"Order batch creation failed: {e}") from e

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
