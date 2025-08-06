"""
This module contains the CCXT connectors for the exchanges.
"""

from dataclasses import dataclass
from functools import partial

import ccxt.pro as cxp

from ..broker import CcxtBroker
from .binance.broker import BinanceCcxtBroker
from .binance.exchange import BINANCE_UM_MM, BinancePortfolioMargin, BinanceQV, BinanceQVUSDM
from .bitfinex.bitfinex import BitfinexF
from .bitfinex.bitfinex_account import BitfinexAccountProcessor
from .hyperliquid.broker import HyperliquidCcxtBroker
from .hyperliquid.hyperliquid import Hyperliquid, HyperliquidF
from .kraken.kraken import CustomKrakenFutures


@dataclass
class ReaderCapabilities:
    """Configuration for exchange-specific reader capabilities."""

    supports_bulk_funding: bool = True
    supports_bulk_ohlcv: bool = True
    max_symbols_per_request: int = 1000
    default_funding_interval_hours: float = 8.0  # Default for most exchanges (Binance, etc.)


EXCHANGE_ALIASES = {
    "binance": "binanceqv",
    "binance.um": "binanceqv_usdm",
    "binance.cm": "binancecoinm",
    "binance.pm": "binancepm",
    "kraken.f": "custom_krakenfutures",
    "binance.um.mm": "binance_um_mm",
    "bitfinex.f": "bitfinex_f",
    "hyperliquid": "hyperliquid",
    "hyperliquid.f": "hyperliquid_f",
}

CUSTOM_BROKERS = {
    "binance": partial(BinanceCcxtBroker, enable_create_order_ws=True, enable_cancel_order_ws=False),
    "binance.um": partial(BinanceCcxtBroker, enable_create_order_ws=True, enable_cancel_order_ws=True),
    "binance.cm": partial(BinanceCcxtBroker, enable_create_order_ws=True, enable_cancel_order_ws=False),
    "binance.pm": partial(BinanceCcxtBroker, enable_create_order_ws=False, enable_cancel_order_ws=False),
    "bitfinex.f": partial(CcxtBroker, enable_create_order_ws=True, enable_cancel_order_ws=True),
    "hyperliquid": partial(HyperliquidCcxtBroker, enable_create_order_ws=True, enable_cancel_order_ws=False),
    "hyperliquid.f": partial(HyperliquidCcxtBroker, enable_create_order_ws=True, enable_cancel_order_ws=False),
}

CUSTOM_ACCOUNTS = {
    "bitfinex.f": BitfinexAccountProcessor,
}

READER_CAPABILITIES = {
    "hyperliquid": ReaderCapabilities(
        supports_bulk_funding=False,
        default_funding_interval_hours=1.0  # Hyperliquid uses 1-hour funding
    ),
    "hyperliquid.f": ReaderCapabilities(
        supports_bulk_funding=False,
        default_funding_interval_hours=1.0  # Hyperliquid uses 1-hour funding
    ),
}

cxp.binanceqv = BinanceQV  # type: ignore
cxp.binanceqv_usdm = BinanceQVUSDM  # type: ignore
cxp.binancepm = BinancePortfolioMargin  # type: ignore
cxp.binance_um_mm = BINANCE_UM_MM  # type: ignore
cxp.custom_krakenfutures = CustomKrakenFutures  # type: ignore
cxp.bitfinex_f = BitfinexF  # type: ignore
cxp.hyperliquid = Hyperliquid  # type: ignore
cxp.hyperliquid_f = HyperliquidF  # type: ignore

cxp.exchanges.append("binanceqv")
cxp.exchanges.append("binanceqv_usdm")
cxp.exchanges.append("binancepm")
cxp.exchanges.append("binancepm_usdm")
cxp.exchanges.append("binance_um_mm")
cxp.exchanges.append("custom_krakenfutures")
cxp.exchanges.append("bitfinex_f")
cxp.exchanges.append("hyperliquid")
cxp.exchanges.append("hyperliquid_f")
