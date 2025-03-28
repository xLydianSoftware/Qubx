from typing import Any

from qubx import logger
from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.core.basics import Instrument, OrderSide
from qubx.core.exceptions import BadRequest


class BinanceCcxtBroker(CcxtBroker):
    def __init__(
        self,
        *args,
        enable_price_match: bool = False,
        price_match_ticks: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_price_match = enable_price_match
        self.price_match_ticks = price_match_ticks

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
        payload = super()._prepare_order_payload(
            instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
        )
        params = payload.get("params", {})
        if "priceMatch" in options:
            params["priceMatch"] = options["priceMatch"]

        quote = self.data_provider.get_quote(instrument)
        if quote is None:
            logger.warning(f"[<y>{instrument.symbol}</y>] :: Quote is not available for order creation.")
            raise BadRequest(f"Quote is not available for price match for {instrument.symbol}")

        if time_in_force == "gtx" and price is not None and self.enable_price_match:
            if (order_side == "BUY" and quote.bid - price < self.price_match_ticks * instrument.tick_size) or (
                order_side == "SELL" and price - quote.ask < self.price_match_ticks * instrument.tick_size
            ):
                params["priceMatch"] = "QUEUE"
                logger.debug(f"[<y>{instrument.symbol}</y>] :: Price match is set to QUEUE. Price will be ignored.")

        if "priceMatch" in params:
            # - if price match is set, we don't need to specify the price
            payload["price"] = None

        payload["params"] = params
        return payload
