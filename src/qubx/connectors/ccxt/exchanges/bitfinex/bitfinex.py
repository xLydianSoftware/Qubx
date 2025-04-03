from typing import Dict, List

import ccxt.pro as cxp
from ccxt.async_support.base.ws.cache import ArrayCache, ArrayCacheByTimestamp
from ccxt.async_support.base.ws.client import Client
from ccxt.base.errors import ArgumentsRequired, BadRequest, NotSupported
from ccxt.base.precise import Precise
from ccxt.base.types import (
    Any,
    Balances,
    Num,
    Order,
    OrderSide,
    OrderType,
    Strings,
    Tickers,
)


class MyBitfinex(cxp.bitfinex):
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
