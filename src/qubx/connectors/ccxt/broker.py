import asyncio
import traceback
from typing import Any

import pandas as pd

import ccxt
import ccxt.pro as cxp
from ccxt.base.errors import ExchangeError
from qubx import logger
from qubx.core.basics import (
    CtrlChannel,
    Instrument,
    Order,
    OrderSide,
)
from qubx.core.errors import ErrorLevel, OrderCancellationError, OrderCreationError, create_error_event
from qubx.core.exceptions import BadRequest, InvalidOrderParameters
from qubx.core.interfaces import (
    IAccountProcessor,
    IBroker,
    IDataProvider,
    ITimeProvider,
)
from qubx.utils.misc import AsyncThreadLoop

from .utils import ccxt_convert_order_info, instrument_to_ccxt_symbol


class CcxtBroker(IBroker):
    _exchange: cxp.Exchange
    _loop: AsyncThreadLoop

    def __init__(
        self,
        exchange: cxp.Exchange,
        channel: CtrlChannel,
        time_provider: ITimeProvider,
        account: IAccountProcessor,
        data_provider: IDataProvider,
        cancel_timeout: int = 30,
        cancel_retry_interval: int = 2,
        max_cancel_retries: int = 10,
        enable_create_order_ws: bool = False,
        enable_cancel_order_ws: bool = False,
    ):
        self._exchange = exchange
        self.ccxt_exchange_id = str(exchange.name)
        self.channel = channel
        self.time_provider = time_provider
        self.account = account
        self.data_provider = data_provider
        self._loop = AsyncThreadLoop(exchange.asyncio_loop)
        self.cancel_timeout = cancel_timeout
        self.cancel_retry_interval = cancel_retry_interval
        self.max_cancel_retries = max_cancel_retries
        self.enable_create_order_ws = enable_create_order_ws
        self.enable_cancel_order_ws = enable_cancel_order_ws

    @property
    def is_simulated_trading(self) -> bool:
        return False

    def _post_order_error_to_databus(
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
        level = ErrorLevel.LOW
        match error:
            case ccxt.InsufficientFunds():
                level = ErrorLevel.HIGH
                logger.error(
                    f"(::create_order) INSUFFICIENT FUNDS for {order_side} {amount} {order_type} for {instrument.symbol} : {error}"
                )
            case ccxt.OrderNotFillable():
                level = ErrorLevel.LOW
                logger.error(
                    f"(::create_order) ORDER NOT FILLEABLE for {order_side} {amount} {order_type} for [{instrument.symbol}] : {error}"
                )
            case ccxt.InvalidOrder():
                level = ErrorLevel.LOW
                logger.error(
                    f"(::create_order) INVALID ORDER for {order_side} {amount} {order_type} for {instrument.symbol} : {error}"
                )
            case ccxt.BadRequest():
                level = ErrorLevel.LOW
                logger.error(
                    f"(::create_order) BAD REQUEST for {order_side} {amount} {order_type} for {instrument.symbol} : {error}"
                )
            case _:
                level = ErrorLevel.MEDIUM
                logger.error(f"(::create_order) Unexpected error: {error}")

        error_event = OrderCreationError(
            timestamp=self.time_provider.time(),
            message=f"Error message: {str(error)}",
            level=level,
            instrument=instrument,
            amount=amount,
            price=price,
            order_type=order_type,
            side=order_side,
            error=error,
        )
        self.channel.send(create_error_event(error_event))

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
    ) -> Any:  # Return type as Any to avoid Future/Task typing issues
        """
        Submit an order asynchronously. Errors will be sent through the channel.

        Returns:
            Future-like object that will eventually contain the result
        """

        async def _execute_order_with_channel_errors():
            try:
                order, error = await self._create_order(
                    instrument=instrument,
                    order_side=order_side,
                    order_type=order_type,
                    amount=amount,
                    price=price,
                    client_id=client_id,
                    time_in_force=time_in_force,
                    **options,
                )

                if error:
                    self._post_order_error_to_databus(
                        error, instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
                    )
                    order = None

                return order

            except Exception as err:
                # Catch any unexpected errors and send them through the channel as well
                logger.error(f"{self.__class__.__name__} :: Unexpected error in async order creation: {err}")
                logger.error(traceback.format_exc())
                self._post_order_error_to_databus(
                    err, instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
                )
                return None

        # Submit the task to the async loop
        return self._loop.submit(_execute_order_with_channel_errors())

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
    ) -> Order | None:
        """
        Submit an order and wait for the result. Exceptions will be raised on errors.

        Returns:
            Order: The created order object

        Raises:
            Various exceptions based on the error that occurred
        """
        try:
            # Create a task that executes the order creation
            future = self._loop.submit(
                self._create_order(
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

            # Wait for the result
            order, error = future.result()

            # If there was an error, raise it
            if error:
                raise error

            # If there was no error but also no order, something went wrong
            if not order and not self.enable_create_order_ws:
                raise ExchangeError(f"{self.__class__.__name__} :: Order creation failed with no specific error")

            return order

        except Exception as err:
            # This will catch any errors from future.result() or if we explicitly raise an error
            self._post_order_error_to_databus(
                err, instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
            )
            return None

    def cancel_order(self, order_id: str) -> Order | None:
        orders = self.account.get_orders()
        if order_id not in orders:
            logger.warning(f"Order {order_id} not found in active orders")
            return None

        order = orders[order_id]
        logger.info(f"Canceling order {order_id} ...")

        # Submit the cancellation task to the async loop without waiting for the result
        self._loop.submit(self._cancel_order_with_retry(order_id, order.instrument))

        # Always return None as requested
        return None

    async def _create_order(
        self,
        instrument: Instrument,
        order_side: OrderSide,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **options,
    ) -> tuple[Order | None, Exception | None]:
        """
        Asynchronously create an order with the exchange.

        Returns:
            tuple: (Order object if successful, Exception if failed)
        """
        try:
            payload = self._prepare_order_payload(
                instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
            )
            if self.enable_create_order_ws:
                r = await self._exchange.create_order_ws(**payload)
            else:
                r = await self._exchange.create_order(**payload)

            if r is None:
                msg = "(::_create_order) No response from exchange"
                logger.error(msg)
                return None, ExchangeError(msg)

            if r["id"] is None:
                return None, None

            order = ccxt_convert_order_info(instrument, r)
            logger.info(f"New order {order}")
            return order, None

        except Exception as err:
            return None, err

    def _prepare_order_payload(
        self,
        instrument: Instrument,
        order_side: OrderSide,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **options,
    ) -> dict[str, Any]:
        params = {}
        _is_trigger_order = order_type.startswith("stop_")

        quote = self.data_provider.get_quote(instrument)
        if quote is None:
            logger.warning(f"[<y>{instrument.symbol}</y>] :: Quote is not available for order creation.")
            raise BadRequest(f"Quote is not available for order creation for {instrument.symbol}")

        # TODO: think about automatically setting reduce only when needed
        if not options.get("reduceOnly", False):
            min_notional = instrument.min_notional
            if min_notional > 0 and abs(amount) * quote.mid_price() < min_notional:
                raise InvalidOrderParameters(
                    f"[{instrument.symbol}] Order amount {amount} is too small. Minimum notional is {min_notional}"
                )

        # - handle trigger (stop) orders
        if _is_trigger_order:
            params["triggerPrice"] = price
            order_type = order_type.split("_")[1]

        if client_id:
            params["newClientOrderId"] = client_id

        if instrument.is_futures():
            params["type"] = "swap"

        ccxt_symbol = instrument_to_ccxt_symbol(instrument)

        if order_type == "limit" or _is_trigger_order:
            time_in_force = time_in_force.upper()
            params["timeInForce"] = time_in_force
            if price is None:
                raise InvalidOrderParameters(f"Price must be specified for '{order_type}' order")
            if order_side == "BUY" and time_in_force == "GTX" and price >= quote.ask:
                logger.info(
                    f"[{instrument.symbol}] :: GTX BUY order price {price} is greater than ask price {quote.ask}. Setting 1 tick below ask."
                )
                price = quote.ask - instrument.tick_size
            elif order_side == "SELL" and time_in_force == "GTX" and price <= quote.bid:
                logger.info(
                    f"[{instrument.symbol}] :: GTX SELL order price {price} is less than bid price {quote.bid}. Setting 1 tick above bid."
                )
                price = quote.bid + instrument.tick_size

        return {
            "symbol": ccxt_symbol,
            "type": order_type.lower(),
            "side": order_side.lower(),
            "amount": amount,
            "price": price,
            "params": params,
        }

    async def _cancel_order_with_retry(self, order_id: str, instrument: Instrument) -> bool:
        """
        Attempts to cancel an order with retries.

        Args:
            order_id: The ID of the order to cancel
            symbol: The symbol of the instrument

        Returns:
            bool: True if cancellation was successful, False otherwise
        """
        start_time = self.time_provider.time()
        timeout_delta = self.cancel_timeout
        retries = 0

        while True:
            try:
                if self.enable_cancel_order_ws:
                    await self._exchange.cancel_order_ws(order_id, symbol=instrument_to_ccxt_symbol(instrument))
                else:
                    await self._exchange.cancel_order(order_id, symbol=instrument_to_ccxt_symbol(instrument))
                return True
            except ccxt.OperationRejected as err:
                err_msg = str(err).lower()
                # Check if the error is about an unknown order or non-existent order
                if "unknown order" in err_msg or "order does not exist" in err_msg or "order not found" in err_msg:
                    # These errors might be temporary if the order is still being processed, so retry
                    logger.debug(f"[{order_id}] Order not found for cancellation, might retry: {err}")
                    # Continue with the retry logic instead of returning immediately
                else:
                    # For other operation rejected errors, don't retry
                    logger.debug(f"[{order_id}] Could not cancel order: {err}")
                    return False
            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.ExchangeNotAvailable) as e:
                logger.warning(f"[{order_id}] Network or exchange error while cancelling: {e}")
                # Continue with retry logic
            except Exception as err:
                logger.error(f"Unexpected error canceling order {order_id}: {err}")
                return False

            # Common retry logic for all retryable errors
            current_time = self.time_provider.time()
            elapsed_seconds = pd.Timedelta(current_time - start_time).total_seconds()
            retries += 1

            if elapsed_seconds >= timeout_delta or retries >= self.max_cancel_retries:
                logger.error(f"Timeout reached for canceling order {order_id}")
                self.channel.send(
                    create_error_event(
                        OrderCancellationError(
                            timestamp=self.time_provider.time(),
                            order_id=order_id,
                            message=f"Timeout reached for canceling order {order_id}",
                            instrument=instrument,
                            level=ErrorLevel.LOW,
                            error=None,
                        )
                    )
                )
                return False

            # Wait before retrying with exponential backoff
            backoff_time = min(self.cancel_retry_interval * (2 ** (retries - 1)), 30)
            logger.debug(f"Retrying order cancellation for {order_id} in {backoff_time} seconds (retry {retries})")
            await asyncio.sleep(backoff_time)

        # This should never be reached due to the return statements above,
        # but it's here to satisfy the type checker
        return False

    def cancel_orders(self, instrument: Instrument) -> None:
        orders = self.account.get_orders()
        instrument_orders = [order_id for order_id, order in orders.items() if order.instrument == instrument]

        # Submit all cancellations without waiting for results
        for order_id in instrument_orders:
            self.cancel_order(order_id)

    def update_order(self, order_id: str, price: float | None = None, amount: float | None = None) -> Order:
        raise NotImplementedError("Not implemented yet")

    def exchange(self) -> str:
        """
        Return the name of the exchange this broker is connected to.
        """
        return self.ccxt_exchange_id.upper()
