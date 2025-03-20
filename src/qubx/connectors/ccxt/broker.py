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
)
from qubx.core.exceptions import InvalidOrderParameters
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
        price_match_pct: float = 0.05,
        cancel_timeout: int = 30,
        cancel_retry_interval: int = 2,
        max_cancel_retries: int = 10,
    ):
        self._exchange = exchange
        self.ccxt_exchange_id = str(exchange.name)
        self.channel = channel
        self.time_provider = time_provider
        self.account = account
        self.data_provider = data_provider
        self.price_match_fraction = price_match_pct / 100.0
        self._loop = AsyncThreadLoop(exchange.asyncio_loop)
        self.cancel_timeout = cancel_timeout
        self.cancel_retry_interval = cancel_retry_interval
        self.max_cancel_retries = max_cancel_retries

    @property
    def is_simulated_trading(self) -> bool:
        return False

    def send_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **options,
    ) -> Order:
        params = {}
        _is_trigger_order = order_type.startswith("stop_")

        if order_type == "limit" or _is_trigger_order:
            params["timeInForce"] = time_in_force.upper()
            if price is None:
                raise InvalidOrderParameters(f"Price must be specified for '{order_type}' order")

        quote = self.data_provider.get_quote(instrument)

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

        if "priceMatch" in options:
            params["priceMatch"] = options["priceMatch"]

        if instrument.is_futures():
            params["type"] = "swap"

            if time_in_force == "gtx" and price is not None:
                reference_price = quote.bid if order_side == "buy" else quote.ask
                if (
                    self.price_match_fraction > 0
                    and abs(price - reference_price) / reference_price < self.price_match_fraction
                ):
                    logger.debug(
                        f"[<y>{self.__class__.__name__}</y>] [{instrument.symbol}] :: Price {price} is close to reference price {reference_price}."
                        f" Setting price match to QUEUE."
                    )
                    params["priceMatch"] = "QUEUE"

        ccxt_symbol = instrument_to_ccxt_symbol(instrument)

        r: dict[str, Any] | None = None
        try:
            r = self._loop.submit(
                self._exchange.create_order(
                    symbol=ccxt_symbol,
                    type=order_type,  # type: ignore
                    side=order_side,  # type: ignore
                    amount=amount,
                    price=price,
                    params=params,
                )
            ).result()
        except ccxt.OrderNotFillable as exc:
            logger.error(
                f"(::send_order) [{instrument.symbol}] ORDER NOT FILLEABLE for {order_side} {amount} {order_type} : {exc}"
            )
            exc_msg = str(exc)
            if "priceMatch" not in params and "-5022" in exc_msg or "Post Only order will be rejected" in exc_msg:
                logger.debug(f"(::send_order) [{instrument.symbol}] Trying again with price match ...")
                return self.send_order(
                    instrument=instrument,
                    order_side=order_side,
                    order_type=order_type,
                    amount=amount,
                    price=price,
                    client_id=client_id,
                    time_in_force=time_in_force,
                    priceMatch="QUEUE",
                    **options,
                )
            raise exc
        except ccxt.BadRequest as exc:
            logger.error(
                f"(::send_order) BAD REQUEST for {order_side} {amount} {order_type} for {instrument.symbol} : {exc}"
            )
            raise exc
        except Exception as err:
            logger.error(f"(::send_order) {order_side} {amount} {order_type} for {instrument.symbol} exception : {err}")
            logger.error(traceback.format_exc())
            raise err

        if r is None:
            msg = "(::send_order) No response from exchange"
            logger.error(msg)
            raise ExchangeError(msg)

        order = ccxt_convert_order_info(instrument, r)
        logger.info(f"New order {order}")
        return order

    def cancel_order(self, order_id: str) -> Order | None:
        orders = self.account.get_orders()
        if order_id not in orders:
            logger.warning(f"Order {order_id} not found in active orders")
            return None

        order = orders[order_id]
        logger.info(f"Canceling order {order_id} ...")

        # Submit the cancellation task to the async loop without waiting for the result
        self._loop.submit(self._cancel_order_with_retry(order_id, instrument_to_ccxt_symbol(order.instrument)))

        # Always return None as requested
        return None

    async def _cancel_order_with_retry(self, order_id: str, symbol: str) -> bool:
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
                await self._exchange.cancel_order(order_id, symbol=symbol)
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
                logger.debug(f"[{order_id}] Network or exchange error while cancelling: {e}")
                # Continue with retry logic
            except Exception as err:
                logger.error(f"Unexpected error canceling order {order_id}: {err}")
                logger.error(traceback.format_exc())
                return False

            # Common retry logic for all retryable errors
            current_time = self.time_provider.time()
            elapsed_seconds = pd.Timedelta(current_time - start_time).total_seconds()

            if elapsed_seconds >= timeout_delta:
                logger.error(f"Timeout reached for canceling order {order_id}")
                return False

            retries += 1
            if retries >= self.max_cancel_retries:
                logger.error(f"Max retries ({self.max_cancel_retries}) reached for canceling order {order_id}")
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
