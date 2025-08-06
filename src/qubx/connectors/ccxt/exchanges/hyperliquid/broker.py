from typing import Any

from qubx import logger
from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.core.basics import Instrument, OrderSide
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
                logger.warning(f"[<y>{instrument.symbol}</y>] :: Quote is not available for market order slippage calculation.")
                raise BadRequest(f"Quote is not available for market order slippage calculation for {instrument.symbol}")

            # Get slippage from options or use default
            slippage = options.get("slippage", self.market_order_slippage)
            
            # Calculate slippage-protected price
            if order_side.upper() == "BUY":
                # For buy orders, add slippage to ask price to ensure execution
                price = quote.ask * (1 + slippage)
                logger.debug(f"[<y>{instrument.symbol}</y>] :: Market BUY order: using slippage-protected price {price:.6f} (ask: {quote.ask:.6f}, slippage: {slippage:.1%})")
            else:  # SELL
                # For sell orders, subtract slippage from bid price to ensure execution
                price = quote.bid * (1 - slippage)
                logger.debug(f"[<y>{instrument.symbol}</y>] :: Market SELL order: using slippage-protected price {price:.6f} (bid: {quote.bid:.6f}, slippage: {slippage:.1%})")

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