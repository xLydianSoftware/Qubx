import asyncio
from dataclasses import asdict
from typing import Dict, List

import ccxt.pro as cxp
from bfxapi import REST_HOST
from bfxapi import Client as BfxClient
from bfxapi.types import Order as BfxOrder
from ccxt.async_support.base.ws.cache import ArrayCache, ArrayCacheByTimestamp
from ccxt.async_support.base.ws.client import Client
from ccxt.base.errors import ArgumentsRequired, BadRequest, NotSupported
from ccxt.base.precise import Precise
from ccxt.base.types import (
    Any,
    Balances,
    Int,
    Market,
    Num,
    Order,
    OrderSide,
    OrderType,
    Str,
    Strings,
    Tickers,
)

from qubx import logger


class BitfinexF(cxp.bitfinex):
    """
    Extended binance exchange to provide quote asset volumes support
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # we are adding here the official bitfinex client to extend missing functionality
        self.bfx = BfxClient(rest_host=REST_HOST, api_key=self.apiKey, api_secret=self.secret)

        @self.bfx.wss.on("authenticated")
        def on_authenticated(data: dict[str, Any]):
            logger.info(f"Successful login for user {data['userId']}.")

        asyncio.run_coroutine_threadsafe(self.bfx.wss.start(), self.asyncio_loop)

    def describe(self):
        """
        Overriding watchTrades to use aggTrade instead of trade.
        """
        return self.deep_extend(
            super().describe(),
            {
                "has": {
                    "watchBidsAsks": True,
                    "createOrderWs": True,
                    "cancelOrderWs": True,
                }
            },
        )

    def watch_bids_asks(self, symbol: str, params: dict = {}):
        return self.watch_order_book(symbol, None, params)

    def un_watch_bids_asks(self, symbol: str, params: dict = {}):
        if hasattr(self, "un_watch_order_book"):
            return self.un_watch_order_book(symbol, params)

    async def fetch_balance(self, params={}) -> Balances:
        params["type"] = "margin"
        return await super().fetch_balance(params)

    async def create_order(
        self, symbol: str, type: OrderType, side: OrderSide, amount: float, price: Num = None, params={}
    ):
        params.pop("type", None)
        if "timeInForce" in params and params["timeInForce"] == "GTX":
            # GTX is not supported by bitfinex, so we need to convert it to PO
            params["timeInForce"] = "PO"
            params["postOnly"] = True

        if "lev" not in params:
            params["lev"] = 2

        response = await super().create_order(symbol, type, side, amount, price, params)
        return response

    async def create_order_ws(
        self, symbol: str, type: OrderType, side: OrderSide, amount: float, price: Num = None, params={}
    ) -> Order:
        params.pop("type", None)
        if "timeInForce" in params and params["timeInForce"] == "GTX":
            # GTX is not supported by bitfinex, so we need to convert it to PO
            params["timeInForce"] = "PO"
            params["postOnly"] = True

        if "lev" not in params:
            params["lev"] = 2

        await self.load_markets()
        market = self.market(symbol)
        request = self.create_order_request(symbol, type, side, amount, price, params)

        # if "newClientOrderId" in request:
        #     request["cid"] = request["newClientOrderId"]
        #     del request["newClientOrderId"]

        _params = {
            "type": request["type"],
            "symbol": request["symbol"],
            "amount": float(request["amount"]),
            "lev": request["lev"],
        }

        if "price" in request:
            _params["price"] = float(request["price"])
        else:
            _params["price"] = None

        if "flags" in request:
            _params["flags"] = request["flags"]

        await self.bfx.wss.inputs.submit_order(**_params)
        return self.safe_order({"info": {}}, market)  # type: ignore

    async def cancel_order_ws(self, id: str, symbol: Str = None, params={}) -> Order | None:
        await self.bfx.wss.inputs.cancel_order(id=int(id))
        return None

    async def watch_orders(self, symbol: Str = None, since: Int = None, limit: Int = None, params={}) -> List[Order]:
        response = await super().watch_orders(symbol, since, limit, params)
        return response

    def parse_ws_order_status(self, status):
        statuses: dict = {
            "ACTIVE": "open",
            "CANCELED": "canceled",
            "EXECUTED": "closed",
            "PARTIALLY": "open",
            "POSTONLY": "canceled",  # this status means that a postonly order got canceled on submission
        }
        return self.safe_string(statuses, status, status)

    def parse_order_status(self, status: Str):
        if status is None:
            return status
        parts = status.split(" ")
        state = self.safe_string(parts, 0)
        statuses: dict = {
            "ACTIVE": "open",
            "PARTIALLY": "open",
            "EXECUTED": "closed",
            "CANCELED": "canceled",
            "INSUFFICIENT": "canceled",
            "POSTONLY CANCELED": "canceled",
            "POSTONLY": "canceled",
            "RSN_DUST": "rejected",
            "RSN_PAUSE": "rejected",
            "IOC CANCELED": "canceled",
            "FILLORKILL CANCELED": "canceled",
        }
        return self.safe_string(statuses, state, status)

    def parse_position(self, position: dict, market: Market = None):
        #
        #    [
        #        "tBTCUSD",                    # SYMBOL
        #        "ACTIVE",                     # STATUS
        #        0.0195,                       # AMOUNT
        #        8565.0267019,                 # BASE_PRICE
        #        0,                            # MARGIN_FUNDING
        #        0,                            # MARGIN_FUNDING_TYPE
        #        -0.33455568705000516,         # PL
        #        -0.0003117550117425625,       # PL_PERC
        #        7045.876419249083,            # PRICE_LIQ
        #        3.0673001895895604,           # LEVERAGE
        #        null,                         # _PLACEHOLDER
        #        142355652,                    # POSITION_ID
        #        1574002216000,                # MTS_CREATE
        #        1574002216000,                # MTS_UPDATE
        #        null,                         # _PLACEHOLDER
        #        0,                            # TYPE
        #        null,                         # _PLACEHOLDER
        #        0,                            # COLLATERAL
        #        0,                            # COLLATERAL_MIN
        #        # META
        #        {
        #            "reason": "TRADE",
        #            "order_id": 34271018124,
        #            "liq_stage": null,
        #            "trade_price": "8565.0267019",
        #            "trade_amount": "0.0195",
        #            "order_id_oppo": 34277498022
        #        }
        #    ]
        #
        positionList = self.safe_list(position, "result")
        marketId = self.safe_string(positionList, 0)
        amount = self.safe_string(positionList, 2)
        timestamp = self.safe_integer(positionList, 12)
        meta = self.safe_string(positionList, 19)
        tradePrice = self.safe_string(meta, "trade_price")
        tradeAmount = self.safe_string(meta, "trade_amount")
        return self.safe_position(
            {
                "info": positionList,
                "id": self.safe_string(positionList, 11),
                "symbol": self.safe_symbol(marketId, market),
                "notional": None,
                "marginMode": "isolated",  # derivatives use isolated, margin uses cross, https://support.bitfinex.com/hc/en-us/articles/360035475374-Derivatives-Trading-on-Bitfinex
                "liquidationPrice": self.safe_number(positionList, 8),
                "entryPrice": self.safe_number(positionList, 3),
                "unrealizedPnl": self.safe_number(positionList, 6),
                "percentage": self.safe_number(positionList, 7),
                "contracts": self.parse_number(amount),
                "contractSize": None,
                "markPrice": None,
                "lastPrice": None,
                "side": "long" if Precise.string_gt(amount, "0") else "short",
                "hedged": None,
                "timestamp": timestamp,
                "datetime": self.iso8601(timestamp),
                "lastUpdateTimestamp": self.safe_integer(positionList, 13),
                "maintenanceMargin": self.safe_number(positionList, 18),
                "maintenanceMarginPercentage": None,
                "collateral": self.safe_number(positionList, 17),
                "initialMargin": self.parse_number(Precise.string_mul(tradeAmount, tradePrice)),
                "initialMarginPercentage": None,
                "leverage": self.safe_number(positionList, 9),
                "marginRatio": None,
                "stopLossPrice": None,
                "takeProfitPrice": None,
            }
        )

    def _bfx_order_to_ccxt_order(self, bfx_order: BfxOrder, market: Market = None) -> Order:
        flags = self.parse_order_flags(bfx_order.flags)
        postOnly = False
        if flags is not None:
            for i in range(0, len(flags)):
                if flags[i] == "postOnly":
                    postOnly = True

        side = "sell" if Precise.string_lt(bfx_order.amount, "0") else "buy"
        timeInForce = self.parse_time_in_force(bfx_order.order_type)
        type = self.safe_string(self.safe_value(self.options, "exchangeTypes"), bfx_order.order_type)

        return self.safe_order(
            {
                "info": asdict(bfx_order),
                "id": str(bfx_order.id),
                "clientOrderId": str(bfx_order.cid),
                "timestamp": bfx_order.mts_create,
                "datetime": self.iso8601(bfx_order.mts_create),
                "lastTradeTimestamp": None,
                "symbol": self.safe_symbol(bfx_order.symbol, market),
                "type": type,
                "timeInForce": timeInForce,
                "postOnly": postOnly,
                "side": side,
                "price": bfx_order.price,
                "triggerPrice": None,
                "amount": bfx_order.amount,
                "cost": None,
                "average": bfx_order.price_avg,
                "filled": None,
                "remaining": None,
                "status": None,
                "fee": None,
                "trades": None,
            },
            market,
        )  # type: ignore
