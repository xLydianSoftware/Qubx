"""
Factory for creating data type handlers in CCXT data provider.

This module provides a centralized way to create and manage data type handlers,
allowing for easy extension and testing of different data types.
"""

from typing import Dict, Type

from ccxt.pro import Exchange

from .base import IDataTypeHandler
from .funding_rate import FundingRateDataHandler
from .liquidation import LiquidationDataHandler
from .ohlc import OhlcDataHandler
from .open_interest import OpenInterestDataHandler
from .orderbook import OrderBookDataHandler
from .quote import QuoteDataHandler
from .trade import TradeDataHandler


class DataTypeHandlerFactory:
    """
    Factory for creating data type handlers.

    Provides a centralized registry of available handlers and creates instances
    on demand. Supports both built-in handlers and custom handler registration.
    """

    # Registry of available handler classes
    _handler_registry: Dict[str, Type[IDataTypeHandler]] = {
        "ohlc": OhlcDataHandler,
        "trade": TradeDataHandler,
        "orderbook": OrderBookDataHandler,
        "quote": QuoteDataHandler,
        "liquidation": LiquidationDataHandler,
        "funding_rate": FundingRateDataHandler,
        "open_interest": OpenInterestDataHandler,
    }

    def __init__(self, data_provider, exchange: Exchange, exchange_id: str):
        """
        Initialize the factory with references to the data provider and exchange.

        Args:
            data_provider: Reference to the CcxtDataProvider instance
            exchange: CCXT exchange object
            exchange_id: Exchange identifier for logging
        """
        self._data_provider = data_provider
        self._exchange = exchange
        self._exchange_id = exchange_id
        self._handler_instances: Dict[str, IDataTypeHandler] = {}

    def get_handler(self, data_type: str) -> IDataTypeHandler | None:
        """
        Get or create a handler for the specified data type.

        Args:
            data_type: Data type identifier (e.g., "ohlc", "trade")

        Returns:
            Handler instance if available, None if not supported
        """
        # Return cached instance if already created
        if data_type in self._handler_instances:
            return self._handler_instances[data_type]

        # Check if handler class is registered
        handler_class = self._handler_registry.get(data_type)
        if handler_class is None:
            return None

        # Create and cache new instance
        handler = handler_class(self._data_provider, self._exchange, self._exchange_id)
        self._handler_instances[data_type] = handler
        return handler

    def has_handler(self, data_type: str) -> bool:
        """
        Check if a handler is available for the specified data type.

        Args:
            data_type: Data type identifier (e.g., "ohlc", "trade")

        Returns:
            True if handler is available, False otherwise
        """
        return data_type in self._handler_registry

    def get_supported_data_types(self) -> list[str]:
        """
        Get list of all supported data types.

        Returns:
            List of supported data type identifiers
        """
        return list(self._handler_registry.keys())

    @classmethod
    def register_handler(cls, data_type: str, handler_class: Type[IDataTypeHandler]) -> None:
        """
        Register a custom handler class for a data type.

        Args:
            data_type: Data type identifier
            handler_class: Handler class implementing IDataTypeHandler
        """
        cls._handler_registry[data_type] = handler_class

    @classmethod
    def unregister_handler(cls, data_type: str) -> None:
        """
        Unregister a handler for a data type.

        Args:
            data_type: Data type identifier to remove
        """
        cls._handler_registry.pop(data_type, None)

    def clear_cache(self) -> None:
        """Clear all cached handler instances."""
        self._handler_instances.clear()
