"""
This module contains the CCXT connectors for the exchanges.
"""

from functools import partial

import ccxt.pro as cxp

from ..broker import CcxtBroker
from .binance.broker import BinanceCcxtBroker
from .binance.exchange import BINANCE_UM_MM, BinancePortfolioMargin, BinanceQV, BinanceQVUSDM
from .bitfinex.bitfinex import BitfinexF
from .bitfinex.bitfinex_account import BitfinexAccountProcessor
from .kraken.kraken import CustomKrakenFutures

EXCHANGE_ALIASES = {
    "binance": "binanceqv",
    "binance.um": "binanceqv_usdm",
    "binance.cm": "binancecoinm",
    "binance.pm": "binancepm",
    "kraken.f": "custom_krakenfutures",
    "binance.um.mm": "binance_um_mm",
    "bitfinex.f": "bitfinex_f",
}

CUSTOM_BROKERS = {
    "binance": partial(BinanceCcxtBroker, enable_create_order_ws=True, enable_cancel_order_ws=False),
    "binance.um": partial(BinanceCcxtBroker, enable_create_order_ws=True, enable_cancel_order_ws=True),
    "binance.cm": partial(BinanceCcxtBroker, enable_create_order_ws=True, enable_cancel_order_ws=False),
    "binance.pm": partial(BinanceCcxtBroker, enable_create_order_ws=False, enable_cancel_order_ws=False),
    "bitfinex.f": partial(CcxtBroker, enable_create_order_ws=True, enable_cancel_order_ws=True),
}

CUSTOM_ACCOUNTS = {
    "bitfinex.f": BitfinexAccountProcessor,
}

cxp.binanceqv = BinanceQV  # type: ignore
cxp.binanceqv_usdm = BinanceQVUSDM  # type: ignore
cxp.binancepm = BinancePortfolioMargin  # type: ignore
cxp.binance_um_mm = BINANCE_UM_MM  # type: ignore
cxp.custom_krakenfutures = CustomKrakenFutures  # type: ignore
cxp.bitfinex_f = BitfinexF  # type: ignore

cxp.exchanges.append("binanceqv")
cxp.exchanges.append("binanceqv_usdm")
cxp.exchanges.append("binancepm")
cxp.exchanges.append("binancepm_usdm")
cxp.exchanges.append("binance_um_mm")
cxp.exchanges.append("custom_krakenfutures")
cxp.exchanges.append("bitfinex_f")
