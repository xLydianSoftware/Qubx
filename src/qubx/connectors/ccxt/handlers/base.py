"""
Data type handler interfaces and implementations for CCXT data provider.

This module provides a clean abstraction for handling different types of market data
(OHLC, trades, orderbooks, etc.) with dedicated handlers for each type.
"""

from abc import ABC, abstractmethod
from typing import Set

from ccxt.pro import Exchange
from qubx.core.basics import CtrlChannel, Instrument
from ..subscription_config import SubscriptionConfiguration


class IDataTypeHandler(ABC):
    """
    Interface for handling specific data types in CCXT data provider.

    Each data type (OHLC, trades, orderbooks, etc.) has its own handler that:
    - Knows how to subscribe to that data type via CCXT
    - Handles data processing and conversion
    - Manages warmup data fetching for backtesting
    """

    @abstractmethod
    def prepare_subscription(
        self, name: str, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **params
    ) -> SubscriptionConfiguration:
        """
        Prepare subscription configuration for this data type.

        Args:
            name: Stream name for this subscription
            sub_type: Parsed subscription type (e.g., "ohlc", "trade")
            channel: Control channel for managing subscription lifecycle
            instruments: Set of instruments to subscribe to
            **params: Additional parameters specific to data type
            
        Returns:
            SubscriptionConfiguration with subscriber and unsubscriber functions
        """
        pass

    @abstractmethod
    async def warmup(self, instruments: Set[Instrument], **params) -> None:
        """
        Fetch historical data for warmup during backtesting.

        Args:
            instruments: Set of instruments to warm up
            **params: Additional parameters specific to data type
        """
        pass

    @property
    @abstractmethod
    def data_type(self) -> str:
        """Return the data type this handler supports (e.g., "ohlc", "trade")."""
        pass


class BaseDataTypeHandler(IDataTypeHandler):
    """
    Base implementation providing common functionality for data type handlers.

    Handles common CCXT operations and provides helper methods for data conversion.
    """

    def __init__(self, data_provider, exchange: Exchange, exchange_id: str):
        """
        Initialize the handler with references to the data provider and exchange.

        Args:
            data_provider: Reference to the CcxtDataProvider instance
            exchange: CCXT exchange object
            exchange_id: Exchange identifier for logging
        """
        self._data_provider = data_provider
        self._exchange = exchange
        self._exchange_id = exchange_id

    def _get_ccxt_symbols(self, instruments: Set[Instrument]) -> list[str]:
        """Convert instruments to CCXT symbols."""
        from ..utils import instrument_to_ccxt_symbol

        return [instrument_to_ccxt_symbol(instr) for instr in instruments]

    def _find_instrument_by_symbol(self, symbol: str) -> Instrument | None:
        """Find instrument by CCXT symbol."""
        from ..utils import ccxt_find_instrument

        return ccxt_find_instrument(symbol, self._data_provider._instrument_list)
