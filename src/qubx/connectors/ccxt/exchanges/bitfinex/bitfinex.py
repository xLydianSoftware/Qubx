from typing import Dict, List

import ccxt.pro as cxp
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


class BitfinexF(cxp.bitfinex):
    """
    Extended binance exchange to provide quote asset volumes support
    """

    def describe(self):
        """
        Overriding watchTrades to use aggTrade instead of trade.
        """
        return self.deep_extend(
            super().describe(),
            {
                "has": {
                    "watchBidsAsks": True,
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
        response = await super().create_order(symbol, type, side, amount, price, params)
        return response

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
