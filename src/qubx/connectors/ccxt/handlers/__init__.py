"""
Data type handlers for CCXT data provider.

This package contains specialized handlers for different types of market data,
providing clean separation of concerns and easy extensibility.
"""

from .base import BaseDataTypeHandler, IDataTypeHandler
from .factory import DataTypeHandlerFactory
from .funding_rate import FundingRateDataHandler
from .liquidation import LiquidationDataHandler
from .ohlc import OhlcDataHandler
from .open_interest import OpenInterestDataHandler
from .orderbook import OrderBookDataHandler
from .quote import QuoteDataHandler
from .trade import TradeDataHandler

__all__ = [
    "IDataTypeHandler",
    "BaseDataTypeHandler",
    "DataTypeHandlerFactory",
    "OhlcDataHandler",
    "TradeDataHandler",
    "OrderBookDataHandler",
    "QuoteDataHandler",
    "LiquidationDataHandler",
    "FundingRateDataHandler",
    "OpenInterestDataHandler",
]
