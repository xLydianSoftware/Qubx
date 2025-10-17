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

    def _enrich_order_response(
        self,
        response: dict[str, Any],
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fill in missing fields in HyperLiquid order response.

        HyperLiquid often returns minimal order information (sometimes only order ID),
        so we need to reconstruct the full order from the request parameters.

        Args:
            response: The raw response from HyperLiquid
            symbol: CCXT symbol (e.g., "BTC/USDC:USDC")
            order_type: Order type ("limit" or "market")
            side: Order side ("buy" or "sell")
            amount: Order amount
            price: Order price (optional for market orders)
            client_id: Client order ID (optional)
            params: Additional params (timeInForce, reduceOnly, etc.)

        Returns:
            Enriched response dictionary ready for ccxt_convert_order_info
        """
        if params is None:
            params = {}

        # Fill in missing fields from the request parameters
        if not response.get("symbol"):
            response["symbol"] = symbol
        if not response.get("type"):
            response["type"] = order_type
        # Always set side from request params (CCXT can't parse from resting/filled structures)
        if not response.get("side") or response.get("side") == "unknown":
            response["side"] = side
        if not response.get("amount") or response.get("amount") == 0:
            response["amount"] = amount
        if not response.get("price") or response.get("price") == 0:
            if price is not None:
                response["price"] = price
        if not response.get("timestamp"):
            response["timestamp"] = pd.Timestamp.now(tz="UTC").value // 1_000_000  # Convert to milliseconds
        if not response.get("status"):
            response["status"] = "open"
        if not response.get("timeInForce"):
            response["timeInForce"] = params.get("timeInForce")
        if not response.get("clientOrderId") and client_id:
            response["clientOrderId"] = client_id
        if not response.get("cost"):
            response["cost"] = 0.0

        # Ensure reduceOnly is propagated
        if params.get("reduceOnly"):
            response["reduceOnly"] = True

        return response

    def _translate_time_in_force(self, tif: str | None) -> str | None:
        """
        Translate standard timeInForce values to HyperLiquid format.

        HyperLiquid uses:
        - "Alo" for Add Liquidity Only (post-only/maker)
        - "Ioc" for Immediate or Cancel
        - "Gtc" for Good til Cancelled

        This method translates common aliases:
        - GTX (Binance post-only) → Alo
        - FOK (Fill or Kill) → Ioc
        - GTC, IOC, ALO (any case) → proper HyperLiquid format
        """
        if tif is None:
            return None

        # If already in correct HyperLiquid format, return as-is
        if tif in ["Alo", "Ioc", "Gtc"]:
            return tif

        # Translation map for timeInForce values
        tif_map = {
            "GTX": "Alo",  # Binance post-only → Hyperliquid post-only
            "FOK": "Ioc",  # Fill or Kill → Immediate or Cancel
            "GTC": "Gtc",  # Good til Cancelled (normalize case)
            "IOC": "Ioc",  # Immediate or Cancel (normalize case)
            "ALO": "Alo",  # Add Liquidity Only (normalize case)
        }

        # Normalize to uppercase for lookup
        tif_upper = tif.upper()
        return tif_map.get(tif_upper, tif)  # Return translated or original if not in map

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
            r = self._enrich_order_response(
                response=r,
                symbol=payload["symbol"],
                order_type=payload["type"],
                side=payload["side"],
                amount=payload["amount"],
                price=payload["price"],
                client_id=client_id,
                params=payload["params"],
            )

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

        # Translate timeInForce to HyperLiquid format (GTX → Alo, etc.)
        params = payload.get("params", {})
        if "timeInForce" in params:
            params["timeInForce"] = self._translate_time_in_force(params["timeInForce"])

        # Add slippage parameter to params if specified in options
        if "slippage" in options:
            # HyperLiquid accepts slippage as a percentage (e.g., 0.05 for 5%)
            params["px"] = price  # Explicit price for slippage calculation

        payload["params"] = params
        return payload

    async def _edit_order_async(
        self, order_id: str, existing_order: Order, price: float, amount: float
    ) -> tuple[Order | None, Exception | None]:
        """
        Override _edit_order_async with WebSocket support and response enrichment.

        HyperLiquid requires params to match the original order and may return
        minimal order information, so we fill missing fields from the request.
        """
        try:
            ccxt_symbol = instrument_to_ccxt_symbol(existing_order.instrument)
            ccxt_side = "buy" if existing_order.side == "BUY" else "sell"

            params = {}
            if existing_order.time_in_force:
                # Translate TIF to HyperLiquid format (GTX → Alo, etc.)
                params["timeInForce"] = self._translate_time_in_force(existing_order.time_in_force)
            else:
                # If no TIF is set on the existing order, default to Alo (post-only) for limit orders
                # This is the safest default for market making
                params["timeInForce"] = "Alo"
            if existing_order.options.get("reduceOnly", False):
                params["reduceOnly"] = True

            # Use WebSocket if enabled, otherwise use REST API
            if self.enable_edit_order_ws:
                result = await self._exchange_manager.exchange.edit_order_ws(
                    id=order_id,
                    symbol=ccxt_symbol,
                    type="limit",
                    side=ccxt_side,
                    amount=amount,
                    price=price,
                    params=params,
                )
            else:
                result = await self._exchange_manager.exchange.edit_order(
                    id=order_id,
                    symbol=ccxt_symbol,
                    type="limit",
                    side=ccxt_side,
                    amount=amount,
                    price=price,
                    params=params,
                )

            # Fill in missing fields from the request parameters
            result = self._enrich_order_response(
                response=result,
                symbol=ccxt_symbol,
                order_type="limit",
                side=ccxt_side,
                amount=amount,
                price=price,
                client_id=existing_order.client_id,
                params=params,
            )

            # Convert the result back to our Order format
            updated_order = ccxt_convert_order_info(existing_order.instrument, result)
            return updated_order, None

        except Exception as err:
            logger.error(f"Async edit order failed for {order_id}: {err}")
            return None, err
