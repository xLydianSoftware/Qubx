"""
Unit tests for DataTypeHandlerFactory.

Tests handler registration, retrieval, caching, and error handling.
"""

import pytest
from unittest.mock import MagicMock

from qubx.connectors.ccxt.handlers import (
    DataTypeHandlerFactory,
    OhlcDataHandler,
    TradeDataHandler,
    OrderBookDataHandler,
    QuoteDataHandler,
    LiquidationDataHandler,
    FundingRateDataHandler,
    OpenInterestDataHandler,
)
from qubx.core.basics import CtrlChannel


class TestDataTypeHandlerFactory:
    """Test suite for DataTypeHandlerFactory."""

    @pytest.fixture
    def mock_data_provider(self):
        """Create a mock data provider for factory testing."""
        data_provider = MagicMock()
        data_provider._exchange_id = "test_exchange"
        data_provider._last_quotes = {}
        data_provider.channel = MagicMock(spec=CtrlChannel)
        data_provider.time_provider = MagicMock()
        return data_provider

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock CCXT exchange for testing."""
        exchange = MagicMock()
        exchange.name = "test_exchange"
        return exchange

    @pytest.fixture
    def handler_factory(self, mock_data_provider, mock_exchange):
        """Create a DataTypeHandlerFactory instance for testing."""
        return DataTypeHandlerFactory(
            data_provider=mock_data_provider,
            exchange=mock_exchange,
            exchange_id="test_exchange"
        )

    def test_initialization(self, handler_factory, mock_data_provider, mock_exchange):
        """Test that factory initializes with correct dependencies."""
        assert handler_factory._data_provider == mock_data_provider
        assert handler_factory._exchange == mock_exchange
        assert handler_factory._exchange_id == "test_exchange"
        assert handler_factory._handler_instances == {}

    def test_get_ohlc_handler(self, handler_factory):
        """Test getting OHLC handler."""
        handler = handler_factory.get_handler("ohlc")
        
        assert isinstance(handler, OhlcDataHandler)
        assert handler._data_provider == handler_factory._data_provider
        assert handler._exchange == handler_factory._exchange

    def test_get_trade_handler(self, handler_factory):
        """Test getting trade handler."""
        handler = handler_factory.get_handler("trade")
        
        assert isinstance(handler, TradeDataHandler)
        assert handler._data_provider == handler_factory._data_provider
        assert handler._exchange == handler_factory._exchange

    def test_get_orderbook_handler(self, handler_factory):
        """Test getting orderbook handler."""
        handler = handler_factory.get_handler("orderbook")
        
        assert isinstance(handler, OrderBookDataHandler)
        assert handler._data_provider == handler_factory._data_provider
        assert handler._exchange == handler_factory._exchange

    def test_get_quote_handler(self, handler_factory):
        """Test getting quote handler."""
        handler = handler_factory.get_handler("quote")
        
        assert isinstance(handler, QuoteDataHandler)
        assert handler._data_provider == handler_factory._data_provider
        assert handler._exchange == handler_factory._exchange

    def test_get_liquidation_handler(self, handler_factory):
        """Test getting liquidation handler."""
        handler = handler_factory.get_handler("liquidation")
        
        assert isinstance(handler, LiquidationDataHandler)
        assert handler._data_provider == handler_factory._data_provider
        assert handler._exchange == handler_factory._exchange

    def test_get_funding_rate_handler(self, handler_factory):
        """Test getting funding rate handler."""
        handler = handler_factory.get_handler("funding_rate")
        
        assert isinstance(handler, FundingRateDataHandler)
        assert handler._data_provider == handler_factory._data_provider
        assert handler._exchange == handler_factory._exchange

    def test_get_open_interest_handler(self, handler_factory):
        """Test getting open interest handler."""
        handler = handler_factory.get_handler("open_interest")
        
        assert isinstance(handler, OpenInterestDataHandler)
        assert handler._data_provider == handler_factory._data_provider
        assert handler._exchange == handler_factory._exchange

    def test_handler_caching(self, handler_factory):
        """Test that handlers are cached after first creation."""
        # Get handler first time
        handler1 = handler_factory.get_handler("ohlc")
        
        # Verify it's cached
        assert "ohlc" in handler_factory._handler_instances
        assert handler_factory._handler_instances["ohlc"] is handler1
        
        # Get handler second time
        handler2 = handler_factory.get_handler("ohlc")
        
        # Should return the same cached instance
        assert handler2 is handler1

    def test_multiple_handler_types_cached(self, handler_factory):
        """Test that different handler types are cached independently."""
        ohlc_handler = handler_factory.get_handler("ohlc")
        trade_handler = handler_factory.get_handler("trade")
        quote_handler = handler_factory.get_handler("quote")
        
        # All should be cached
        assert len(handler_factory._handler_instances) == 3
        assert handler_factory._handler_instances["ohlc"] is ohlc_handler
        assert handler_factory._handler_instances["trade"] is trade_handler
        assert handler_factory._handler_instances["quote"] is quote_handler
        
        # All should be different instances
        assert ohlc_handler is not trade_handler
        assert trade_handler is not quote_handler
        assert ohlc_handler is not quote_handler

    def test_get_unknown_handler(self, handler_factory):
        """Test getting handler for unknown data type."""
        handler = handler_factory.get_handler("unknown_type")
        
        assert handler is None

    def test_get_handler_case_sensitivity(self, handler_factory):
        """Test that handler retrieval is case sensitive."""
        handler1 = handler_factory.get_handler("ohlc")
        handler2 = handler_factory.get_handler("OHLC")  # Different case
        
        assert isinstance(handler1, OhlcDataHandler)
        assert handler2 is None  # Should not match

    def test_get_all_supported_handlers(self, handler_factory):
        """Test that all expected data types have handlers."""
        expected_types = [
            "ohlc",
            "trade", 
            "orderbook",
            "quote",
            "liquidation",
            "funding_rate",
            "open_interest",
        ]
        
        for data_type in expected_types:
            handler = handler_factory.get_handler(data_type)
            assert handler is not None, f"No handler found for {data_type}"

    def test_handler_factory_isolation(self, mock_data_provider, mock_exchange):
        """Test that different factory instances don't share caches."""
        factory1 = DataTypeHandlerFactory(mock_data_provider, mock_exchange, "exchange1")
        factory2 = DataTypeHandlerFactory(mock_data_provider, mock_exchange, "exchange2")
        
        # Get handlers from both factories
        handler1 = factory1.get_handler("ohlc")
        handler2 = factory2.get_handler("ohlc")
        
        # Should be different instances
        assert handler1 is not handler2
        
        # Should have separate caches
        assert factory1._handler_instances is not factory2._handler_instances
        assert len(factory1._handler_instances) == 1
        assert len(factory2._handler_instances) == 1

    def test_handler_dependencies_passed_correctly(self, handler_factory, mock_data_provider, mock_exchange):
        """Test that handlers receive correct dependencies from factory."""
        handler = handler_factory.get_handler("trade")
        
        # Verify handler has access to the same dependencies
        assert handler._data_provider is mock_data_provider
        assert handler._exchange is mock_exchange

    def test_factory_with_none_exchange(self, mock_data_provider):
        """Test factory behavior with None exchange.""" 
        factory = DataTypeHandlerFactory(mock_data_provider, None, "test_exchange")
        
        # Should still be able to create handlers
        handler = factory.get_handler("ohlc")
        assert isinstance(handler, OhlcDataHandler)
        assert handler._exchange is None

    def test_factory_with_none_data_provider(self, mock_exchange):
        """Test factory behavior with None data provider."""
        factory = DataTypeHandlerFactory(None, mock_exchange, "test_exchange")
        
        # Should still be able to create handlers  
        handler = factory.get_handler("trade")
        assert isinstance(handler, TradeDataHandler)
        assert handler._data_provider is None

    def test_get_handler_empty_string(self, handler_factory):
        """Test getting handler with empty string."""
        handler = handler_factory.get_handler("")
        assert handler is None

    def test_get_handler_none(self, handler_factory):
        """Test getting handler with None data type."""
        handler = handler_factory.get_handler(None)
        assert handler is None

    def test_handler_registry_completeness(self, handler_factory):
        """Test that the handler registry maps to correct handler classes."""
        expected_mapping = {
            "ohlc": OhlcDataHandler,
            "trade": TradeDataHandler,
            "orderbook": OrderBookDataHandler,
            "quote": QuoteDataHandler,
            "liquidation": LiquidationDataHandler,
            "funding_rate": FundingRateDataHandler,
            "open_interest": OpenInterestDataHandler,
        }
        
        for data_type, expected_class in expected_mapping.items():
            handler = handler_factory.get_handler(data_type)
            assert isinstance(handler, expected_class), f"Expected {expected_class} for {data_type}, got {type(handler)}"