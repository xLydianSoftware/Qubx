import asyncio
import uuid

from qubx import logger
from qubx.core.basics import CtrlChannel, Instrument, ITimeProvider, Order, OrderRequest, OrderSide
from qubx.core.errors import BaseErrorEvent, ErrorLevel, OrderCreationError, create_error_event
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
from .extensions import LighterExchangeAPI
from .utils import get_market_id
from .websocket import LighterWebSocketManager


class LighterBroker(IBroker):
    def __init__(
        self,
        client: LighterClient,
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
        self.ws_manager = ws_manager
        self.channel = channel
        self.time_provider = time_provider
        self.account = account
        self.data_provider = data_provider
        self.cancel_timeout = cancel_timeout
        self.cancel_retry_interval = cancel_retry_interval
        self.max_cancel_retries = max_cancel_retries
        self._async_loop = AsyncThreadLoop(loop)
        self._client_order_ids: dict[str, str] = {}  # client_id -> exchange_order_id
        self._extensions = LighterExchangeAPI(client=self.client, broker=self)

    @property
    def is_simulated_trading(self) -> bool:
        return False

    def exchange(self) -> str:
        return "LIGHTER"

    @property
    def extensions(self) -> LighterExchangeAPI:
        return self._extensions

    def make_client_id(self, client_id: str) -> str:
        return str(abs(hash(client_id)) % (10**9))

    def send_order(self, request: OrderRequest) -> Order:
        instrument = request.instrument
        order_side = request.side
        order_type = request.order_type.lower()
        amount = request.quantity
        price = request.price
        client_id = request.client_id
        time_in_force = request.time_in_force
        options = request.options

        future = self._async_loop.submit(
            self._create_order_rest(
                instrument=instrument,
                order_side=order_side,
                order_type=order_type,
                amount=amount,
                price=price,
                client_id=client_id,
                time_in_force=time_in_force,
                **options,
            )
        )
        order = future.result()
        if order is None:
            raise InvalidOrderParameters("Order creation failed - no order returned")
        return order

    def send_order_async(self, request: OrderRequest) -> None:
        instrument = request.instrument
        order_side = request.side
        order_type = request.order_type.lower()
        amount = request.quantity
        price = request.price
        client_id = request.client_id
        time_in_force = request.time_in_force
        options = request.options

        async def _execute_order_with_channel_errors():
            try:
                await self._create_order_ws(
                    instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
                )
            except Exception as error:
                self._post_order_error_to_channel(
                    error, instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
                )

        self._async_loop.submit(_execute_order_with_channel_errors())

    def cancel_order(self, order_id: str) -> bool:
        order = self._find_order(order_id)
        if order is None:
            raise OrderNotFound(f"Order not found: {order_id}")
        return self._async_loop.submit(self._cancel_order(order)).result()

    def cancel_order_async(self, order_id: str) -> None:
        order = self._find_order(order_id)
        if order is None:
            self._post_cancel_error_to_channel(OrderNotFound(f"Order not found: {order_id}"), order_id)
            return

        async def _cancel_with_errors():
            try:
                await self._cancel_order(order)
            except Exception as error:
                self._post_cancel_error_to_channel(error, order_id)

        return self._async_loop.submit(_cancel_with_errors()).result()

    def cancel_orders(self, instrument: Instrument) -> None:
        orders = self.account.get_orders(instrument=instrument)

        for order in orders.values():
            try:
                self.cancel_order_async(order.id)
            except Exception as e:
                logger.error(f"Failed to cancel order {order.id}: {e}")

    def update_order(self, order_id: str, price: float, amount: float) -> Order:
        order = self._find_order(order_id)
        if order is None:
            raise OrderNotFound(f"Order not found: {order_id}")
        future = self._async_loop.submit(self._modify_order(order, price, amount))
        return future.result()

    async def _create_order_rest(
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
        Create order using REST API (synchronous submission).
        """
        if order_type not in ["market", "limit"]:
            raise InvalidOrderParameters(f"Invalid order type: {order_type}")

        if order_type == "limit" and price is None:
            raise InvalidOrderParameters("Limit orders require a price")

        # Get market_id
        try:
            market_id = get_market_id(instrument)
        except ValueError as e:
            raise InvalidOrderParameters(str(e)) from e

        # Generate client_id if not provided
        # Convert to numeric string (what Lighter expects)
        if client_id is None:
            generated_id = str(uuid.uuid4())
            client_id = str(abs(hash(generated_id)) % (10**9))

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
        order_sign = +1 if order_side == "BUY" else -1
        reduce_only = options.get("reduce_only", None)
        if reduce_only is None:
            if self._is_position_reducing(instrument, amount * order_sign):
                reduce_only = True
            else:
                reduce_only = False

        # Market orders MUST use IOC (Immediate or Cancel) time in force
        if order_type == "market":
            lighter_tif = ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL
            order_expiry = DEFAULT_IOC_EXPIRY
        else:
            # Limit orders use the mapped TIF and 28-day expiry
            order_expiry = DEFAULT_28_DAY_ORDER_EXPIRY

        # Market order slippage protection
        if order_type == "market" and price is None:
            max_slippage = options.get("max_slippage", 0.05)
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
                if is_buy:
                    price = mid_price * (1 + max_slippage)
                else:
                    price = mid_price * (1 - max_slippage)
            except Exception as e:
                raise InvalidOrderParameters(
                    f"Failed to calculate market order price for {instrument.symbol}: {e}"
                ) from e

        # Convert amounts to Lighter's integer format
        base_amount_int = int(amount * (10**instrument.size_precision))
        price_int = int(price * (10**instrument.price_precision)) if price is not None else 0

        logger.debug(
            f"Creating order (REST): {order_side} {amount} {instrument.symbol} "
            f"@ {price if price else 'MARKET'} (type={order_type}, tif={time_in_force}, reduce_only={reduce_only})"
        )

        try:
            # Use signer_client.create_order which signs and submits via HTTP
            signer = self.client.signer_client
            nonce = await self.client.next_nonce()

            tx, tx_hash, error = await signer.create_order(
                market_index=market_id,
                client_order_index=int(client_id),
                base_amount=base_amount_int,
                price=price_int,
                is_ask=is_ask,
                order_type=lighter_order_type,
                time_in_force=lighter_tif,
                reduce_only=reduce_only,
                trigger_price=0,
                order_expiry=order_expiry,
                nonce=nonce,
                api_key_index=self.client.api_key_index,
            )

            if error is not None:
                raise InvalidOrderParameters(f"Order creation failed: {error}")

            # Use tx_hash as order_id
            order_id = client_id

            # Track client order ID
            self._client_order_ids[client_id] = order_id

            # Construct and return Order object
            order = Order(
                id=order_id,
                type="MARKET" if order_type == "market" else "LIMIT",
                instrument=instrument,
                time=self.time_provider.time(),
                quantity=amount,
                price=price if price else 0.0,
                side=order_side,
                status="OPEN",
                time_in_force=time_in_force.upper(),
                client_id=client_id,
                options={"reduce_only": reduce_only} if reduce_only else {},
            )

            logger.debug(f"Order created via REST: {order}")
            return order

        except Exception as e:
            logger.error(f"Failed to create order via REST: {e}")
            raise InvalidOrderParameters(f"Order creation failed: {e}") from e

    async def _create_order_ws(
        self,
        instrument: Instrument,
        order_side: OrderSide,
        order_type: str,
        amount: float,
        price: float | None,
        client_id: str | None,
        time_in_force: str,
        **options,
    ) -> None:
        if order_type not in ["market", "limit"]:
            raise InvalidOrderParameters(f"Invalid order type: {order_type}")

        if order_type == "limit" and price is None:
            raise InvalidOrderParameters("Limit orders require a price")

        # Get market_id
        try:
            market_id = get_market_id(instrument)
        except ValueError as e:
            raise InvalidOrderParameters(str(e)) from e

        # Generate client_id if not provided
        # Convert to numeric string (what Lighter expects)
        if client_id is None:
            generated_id = str(uuid.uuid4())
            client_id = str(abs(hash(generated_id)) % (10**9))

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
        order_sign = +1 if order_side == "BUY" else -1
        reduce_only = options.get("reduce_only", None)
        if reduce_only is None:
            if self._is_position_reducing(instrument, amount * order_sign):
                reduce_only = True
            else:
                reduce_only = False

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

            except Exception as e:
                raise InvalidOrderParameters(
                    f"Failed to calculate market order price for {instrument.symbol}: {e}"
                ) from e

        # Convert amounts to Lighter's integer format (scaled by market-specific decimals)
        # Lighter markets have different decimal precision for price and size
        # Use instrument's built-in precision properties
        base_amount_int = int(amount * (10**instrument.size_precision))
        price_int = int(price * (10**instrument.price_precision)) if price is not None else 0

        logger.debug(
            f"Creating order: {order_side} {amount} {instrument.symbol} "
            f"@ {price if price else 'MARKET'} (type={order_type}, tif={time_in_force}, reduce_only={reduce_only})"
        )

        try:
            # Step 1: Sign transaction locally
            signer = self.client.signer_client
            tx_info, error = signer.sign_create_order(
                market_index=market_id,
                client_order_index=int(client_id),
                base_amount=base_amount_int,
                price=price_int,
                is_ask=is_ask,
                order_type=lighter_order_type,
                time_in_force=lighter_tif,
                reduce_only=int(reduce_only),
                trigger_price=0,  # Not using trigger orders
                order_expiry=order_expiry,
                nonce=await self.ws_manager.next_nonce(),
            )

            if error or tx_info is None:
                raise InvalidOrderParameters(f"Order signing failed: {error}")

            # Step 2: Submit via WebSocket
            response = await self.ws_manager.send_tx(tx_type=TX_TYPE_CREATE_ORDER, tx_info=tx_info, tx_id=client_id)

            # Use the transaction ID from response as order ID
            order_id = response.get("tx_id", client_id)

            # Track client order ID and index
            self._client_order_ids[client_id] = order_id

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise InvalidOrderParameters(f"Order creation failed: {e}") from e

    def _find_order(self, order_id: str) -> Order | None:
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

        return order

    def _find_order_index(self, order: Order) -> int:
        if order.id.isdigit():
            return int(order.id)
        else:
            return abs(hash(order.id)) % (2**56)

    async def _cancel_order(self, order: Order) -> bool:
        logger.debug(f"[{order.instrument}] Canceling order @ {order.price} {order.side} {order.quantity} [{order.id}]")

        try:
            market_id = get_market_id(order.instrument)
            order_index = self._find_order_index(order)
            signer = self.client.signer_client
            tx_info, error = signer.sign_cancel_order(
                market_index=market_id, order_index=order_index, nonce=await self.ws_manager.next_nonce()
            )

            if error or tx_info is None:
                logger.error(f"Order cancellation signing failed: {error}")
                return False

            await self.ws_manager.send_tx(tx_type=TX_TYPE_CANCEL_ORDER, tx_info=tx_info, tx_id=f"cancel_{order.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order.id}: {e}")
            raise OrderNotFound(f"Order cancellation failed: {e}") from e

    async def _modify_order(self, order: Order, price: float, amount: float) -> Order:
        try:
            market_id = get_market_id(order.instrument)
            order_index = self._find_order_index(order)

            # Convert price and amount to Lighter's integer format
            instrument = order.instrument
            base_amount_int = int(amount * (10**instrument.size_precision))
            price_int = int(price * (10**instrument.price_precision))

            logger.debug(
                f"[{order.instrument.symbol}] :: Modifying order {order.id}: amount={order.quantity} → {amount}, price={order.price} → {price}"
            )

            # Step 1: Sign modification transaction locally
            signer = self.client.signer_client
            tx_info, error = signer.sign_modify_order(
                market_index=market_id,
                order_index=order_index,
                base_amount=base_amount_int,
                price=price_int,
                trigger_price=0,  # Not using trigger orders
                nonce=await self.ws_manager.next_nonce(),
            )

            if error or tx_info is None:
                raise OrderNotFound(f"Order modification signing failed: {error}")

            # Step 2: Submit via WebSocket
            await self.ws_manager.send_tx(tx_type=TX_TYPE_MODIFY_ORDER, tx_info=tx_info, tx_id=f"modify_{order.id}")

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

            return updated_order

        except Exception as e:
            logger.error(f"Failed to modify order {order.id}: {e}")
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

        logger.debug(f"Creating order batch: {len(orders)} orders")

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
                try:
                    market_id = get_market_id(instrument)
                except ValueError as e:
                    raise InvalidOrderParameters(str(e)) from e

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

                        # logger.debug(
                        #     f"Market order slippage protection (batch): mid={mid_price:.4f}, "
                        #     f"slippage={max_slippage * 100:.1f}%, protected_price={price:.4f}"
                        # )

                    except Exception as e:
                        raise InvalidOrderParameters(
                            f"Failed to calculate market order price for {instrument.symbol}: {e}"
                        ) from e

                # Convert amounts using market-specific decimals
                base_amount_int = int(amount * (10**instrument.size_precision))
                price_int = int(price * (10**instrument.price_precision)) if price is not None else 0

                # Sign transaction
                signer = self.client.signer_client
                tx_info, error = signer.sign_create_order(
                    market_index=market_id,
                    client_order_index=int(client_id),
                    base_amount=base_amount_int,
                    price=price_int,
                    is_ask=is_ask,
                    order_type=lighter_order_type,
                    time_in_force=lighter_tif,
                    reduce_only=int(reduce_only),
                    trigger_price=0,
                    order_expiry=order_expiry,
                    nonce=await self.ws_manager.next_nonce(),
                )

                if error or tx_info is None:
                    raise InvalidOrderParameters(f"Order signing failed for {instrument.symbol}: {error}")

                tx_types.append(TX_TYPE_CREATE_ORDER)
                tx_infos.append(tx_info)

                # Track client order ID and index
                self._client_order_ids[client_id] = client_id  # Will be updated when tx is confirmed

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

            logger.debug(f"Order batch submitted via WebSocket: {response.get('count')} orders")
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
        level = ErrorLevel.MEDIUM

        if "not found" in str(error).lower():
            level = ErrorLevel.LOW
            logger.error(f"Order not found for cancellation: {order_id}")
        else:
            logger.error(f"Order cancellation error: {error}")

        error_event = BaseErrorEvent(
            timestamp=self.time_provider.time(),
            message=f"Failed to cancel order {order_id}: {str(error)}",
            level=level,
            error=error,
        )
        self.channel.send(create_error_event(error_event))

    def _is_position_reducing(self, instrument: Instrument, signed_amount: float) -> bool:
        current_position = self.account.get_position(instrument)
        return (
            current_position.quantity > 0 and signed_amount < 0 and abs(signed_amount) <= abs(current_position.quantity)
        ) or (
            current_position.quantity < 0 and signed_amount > 0 and abs(signed_amount) <= abs(current_position.quantity)
        )
