"""
This module contains the CCXT connectors for the exchanges.
"""

from dataclasses import dataclass

import ccxt.pro as cxp

from ..connector import CcxtConnector
from .binance.exchange import BINANCE_UM_MM, BinancePortfolioMargin, BinanceQV, BinanceQVUSDM
from .bitfinex.connector import BitfinexCcxtConnector
from .gateio.gateio import GateioFutures
from .hyperliquid.hyperliquid import Hyperliquid, HyperliquidF
from .kraken.kraken import CustomKrakenFutures
from .okx.connector import OkxCcxtConnector
from .okx.okx import OkxFutures

# Bitfinex requires optional qubx-bitfinex-api package
try:
    from .bitfinex.bitfinex import BitfinexF

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

# CcxtConnector subclasses per exchange (commit 4b). Unlisted exchanges (Binance,
# Hyperliquid, ...) use the base CcxtConnector via the factory fallback. OKX/Bitfinex
# need the split orders/fills streams; OKX additionally overrides balance/clOrdId.
# Keyed by the same normalized exchange names the broker/account registries use, plus
# the bare ``okx``/``bitfinex`` aliases the factory may receive. Bitfinex's connector
# subclass has NO dependency on the optional qubx-bitfinex-api package (it only needs
# the base connector + shared mixin), so it is registered unconditionally.
CUSTOM_CONNECTORS: dict[str, type[CcxtConnector]] = {
    "okx": OkxCcxtConnector,
    "okx.f": OkxCcxtConnector,
    "bitfinex": BitfinexCcxtConnector,
    "bitfinex.f": BitfinexCcxtConnector,
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
