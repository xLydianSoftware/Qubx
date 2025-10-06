from typing import Any

import pandas as pd

from qubx import logger
from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.connectors.ccxt.utils import ccxt_convert_order_info, instrument_to_ccxt_symbol
from qubx.core.basics import (
    Instrument,
    Order,
    OrderSide,
)
from qubx.core.exceptions import BadRequest


class HyperliquidCcxtBroker(CcxtBroker):
    """
    HyperLiquid-specific broker that handles market order slippage requirements.

    HyperLiquid requires a price even for market orders to calculate max slippage.
    This broker automatically calculates slippage-protected prices for market orders.
    """

    def __init__(
        self,
        *args,
        market_order_slippage: float = 0.05,  # 5% default slippage
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.market_order_slippage = market_order_slippage

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
        Override _create_order to fill missing order details from request.

        HyperLiquid returns minimal order information (only order ID),
        so we need to reconstruct the full order from the request parameters.
        """
        try:
            payload = self._prepare_order_payload(
                instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
            )
            if self.enable_create_order_ws:
                r = await self._exchange_manager.exchange.create_order_ws(**payload)
            else:
                r = await self._exchange_manager.exchange.create_order(**payload)

            if r is None:
                msg = "(::_create_order) No response from exchange"
                logger.error(msg)
                from ccxt.base.errors import ExchangeError

                return None, ExchangeError(msg)

            if r["id"] is None:
                return None, None

            # Fill in missing fields from the request parameters
            # HyperLiquid only returns order ID, so we need to add the rest
            if not r.get("symbol"):
                r["symbol"] = payload["symbol"]
            if not r.get("type"):
                r["type"] = payload["type"]
            if not r.get("side"):
                r["side"] = payload["side"]
            if not r.get("amount") or r.get("amount") == 0:
                r["amount"] = payload["amount"]
            if not r.get("price") or r.get("price") == 0:
                if payload["price"] is not None:
                    r["price"] = payload["price"]
            if not r.get("timestamp"):
                r["timestamp"] = pd.Timestamp.now(tz="UTC").value // 1_000_000  # Convert to milliseconds
            if not r.get("status"):
                r["status"] = "open"
            if not r.get("timeInForce"):
                r["timeInForce"] = payload["params"].get("timeInForce")
            if not r.get("clientOrderId") and client_id:
                r["clientOrderId"] = client_id
            if not r.get("cost"):
                r["cost"] = 0.0

            # Ensure reduceOnly is propagated
            if payload["params"].get("reduceOnly"):
                r["reduceOnly"] = True

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
        # Handle market orders that need slippage-protected prices
        if order_type.lower() == "market" and price is None:
            quote = self.data_provider.get_quote(instrument)
            if quote is None:
                logger.warning(
                    f"[<y>{instrument.symbol}</y>] :: Quote is not available for market order slippage calculation."
                )
                raise BadRequest(
                    f"Quote is not available for market order slippage calculation for {instrument.symbol}"
                )

            # Get slippage from options or use default
            slippage = options.get("slippage", self.market_order_slippage)

            # Calculate slippage-protected price
            if order_side.upper() == "BUY":
                # For buy orders, add slippage to ask price to ensure execution
                price = quote.ask * (1 + slippage)
                logger.debug(
                    f"[<y>{instrument.symbol}</y>] :: Market BUY order: using slippage-protected price {price:.6f} (ask: {quote.ask:.6f}, slippage: {slippage:.1%})"
                )
            else:  # SELL
                # For sell orders, subtract slippage from bid price to ensure execution
                price = quote.bid * (1 - slippage)
                logger.debug(
                    f"[<y>{instrument.symbol}</y>] :: Market SELL order: using slippage-protected price {price:.6f} (bid: {quote.bid:.6f}, slippage: {slippage:.1%})"
                )

        # Call parent implementation with calculated price
        payload = super()._prepare_order_payload(
            instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
        )

        # Add slippage parameter to params if specified in options
        params = payload.get("params", {})
        if "slippage" in options:
            # HyperLiquid accepts slippage as a percentage (e.g., 0.05 for 5%)
            params["px"] = price  # Explicit price for slippage calculation

        payload["params"] = params
        return payload

    async def _edit_order_async(
        self, order_id: str, existing_order: Order, price: float, amount: float
    ) -> tuple[Order | None, Exception | None]:
        """Hyperliquid requires also params to match the original order."""
        try:
            ccxt_symbol = instrument_to_ccxt_symbol(existing_order.instrument)
            ccxt_side = "buy" if existing_order.side == "BUY" else "sell"

            params = {}
            if existing_order.time_in_force:
                params["timeInForce"] = existing_order.time_in_force
            if existing_order.options.get("reduceOnly", False):
                params["reduceOnly"] = True

            result = await self._exchange_manager.exchange.edit_order(
                id=order_id, symbol=ccxt_symbol, type="limit", side=ccxt_side, amount=amount, price=price, params=params
            )

            # Convert the result back to our Order format
            updated_order = ccxt_convert_order_info(existing_order.instrument, result)
            return updated_order, None

        except Exception as err:
            logger.error(f"Async edit order failed for {order_id}: {err}")
            return None, err
