"""
This module contains the CCXT connectors for the exchanges.
"""

from dataclasses import dataclass, field

import ccxt.pro as cxp

from ..broker import CcxtBroker
from .binance.broker import BinanceCcxtBroker
from .binance.exchange import BINANCE_UM_MM, BinancePortfolioMargin, BinanceQV, BinanceQVUSDM
from .gateio.gateio import GateioFutures
from .hyperliquid.account import HyperliquidAccountProcessor
from .hyperliquid.broker import HyperliquidCcxtBroker
from .hyperliquid.hyperliquid import Hyperliquid, HyperliquidF
from .kraken.kraken import CustomKrakenFutures
from .okx.account import OkxAccountProcessor
from .okx.broker import OkxCcxtBroker
from .okx.okx import OkxFutures

# Bitfinex requires optional qubx-bitfinex-api package
try:
    from .bitfinex.bitfinex import BitfinexF
    from .bitfinex.bitfinex_account import BitfinexAccountProcessor

    _HAS_BITFINEX = True
except ImportError:
    _HAS_BITFINEX = False


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
    "bybit.f": "bybit",
    "gateio.f": "gateio_futures",
    "okx.f": "okx_futures",
}

@dataclass(frozen=True)
class BrokerConfig:
    """Exchange-specific broker class and default kwargs."""

    cls: type
    kwargs: dict = field(default_factory=dict)


CUSTOM_BROKERS: dict[str, BrokerConfig] = {
    "binance": BrokerConfig(BinanceCcxtBroker, {"enable_create_order_ws": True, "enable_cancel_order_ws": False}),
    "binance.um": BrokerConfig(BinanceCcxtBroker, {"enable_create_order_ws": True, "enable_cancel_order_ws": True}),
    "binance.cm": BrokerConfig(BinanceCcxtBroker, {"enable_create_order_ws": True, "enable_cancel_order_ws": False}),
    "binance.pm": BrokerConfig(BinanceCcxtBroker, {"enable_create_order_ws": False, "enable_cancel_order_ws": False}),
    **({"bitfinex.f": BrokerConfig(CcxtBroker, {"enable_create_order_ws": True, "enable_cancel_order_ws": True})} if _HAS_BITFINEX else {}),
    "hyperliquid": BrokerConfig(
        HyperliquidCcxtBroker,
        {"enable_create_order_ws": True, "enable_cancel_order_ws": False, "enable_edit_order_ws": True},
    ),
    "hyperliquid.f": BrokerConfig(
        HyperliquidCcxtBroker,
        {"enable_create_order_ws": True, "enable_cancel_order_ws": False, "enable_edit_order_ws": True},
    ),
    "okx.f": BrokerConfig(OkxCcxtBroker, {}),
}

CUSTOM_ACCOUNTS = {
    **({"bitfinex.f": BitfinexAccountProcessor} if _HAS_BITFINEX else {}),
    "hyperliquid": HyperliquidAccountProcessor,
    "hyperliquid.f": HyperliquidAccountProcessor,
    "okx.f": OkxAccountProcessor,
}

READER_CAPABILITIES = {
    "hyperliquid": ReaderCapabilities(
        supports_bulk_funding=False,
        default_funding_interval_hours=1.0,  # Hyperliquid uses 1-hour funding
    ),
    "hyperliquid.f": ReaderCapabilities(
        supports_bulk_funding=False,
        default_funding_interval_hours=1.0,  # Hyperliquid uses 1-hour funding
    ),
    "gateio.f": ReaderCapabilities(
        supports_bulk_funding=False,
        default_funding_interval_hours=8.0,
    ),
}

cxp.binanceqv = BinanceQV  # type: ignore
cxp.binanceqv_usdm = BinanceQVUSDM  # type: ignore
cxp.binancepm = BinancePortfolioMargin  # type: ignore
cxp.binance_um_mm = BINANCE_UM_MM  # type: ignore
cxp.custom_krakenfutures = CustomKrakenFutures  # type: ignore
if _HAS_BITFINEX:
    cxp.bitfinex_f = BitfinexF  # type: ignore
cxp.hyperliquid = Hyperliquid  # type: ignore
cxp.hyperliquid_f = HyperliquidF  # type: ignore
cxp.gateio_futures = GateioFutures  # type: ignore
cxp.okx_futures = OkxFutures  # type: ignore

cxp.exchanges.append("binanceqv")
cxp.exchanges.append("binanceqv_usdm")
cxp.exchanges.append("binancepm")
cxp.exchanges.append("binancepm_usdm")
cxp.exchanges.append("binance_um_mm")
cxp.exchanges.append("custom_krakenfutures")
if _HAS_BITFINEX:
    cxp.exchanges.append("bitfinex_f")
cxp.exchanges.append("hyperliquid")
cxp.exchanges.append("hyperliquid_f")
cxp.exchanges.append("gateio_futures")
cxp.exchanges.append("okx_futures")
