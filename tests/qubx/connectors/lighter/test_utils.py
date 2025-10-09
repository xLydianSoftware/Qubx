"""Tests for Lighter utility functions"""

import numpy as np
import pytest

from qubx.core.basics import Instrument, AssetType, MarketType, OrderSide
from qubx.connectors.lighter.utils import (
    lighter_symbol_to_qubx,
    qubx_symbol_to_lighter,
    lighter_order_side_to_qubx,
    qubx_order_side_to_lighter,
    lighter_price_to_float,
    float_to_lighter_price,
    lighter_size_to_float,
    float_to_lighter_size,
    convert_lighter_quote,
)
from qubx.connectors.lighter.constants import LighterOrderSide


class TestSymbolConversion:
    """Test symbol conversion functions"""

    def test_lighter_to_qubx(self):
        """Test converting Lighter symbol to Qubx format"""
        assert lighter_symbol_to_qubx("BTC-USDC") == "BTC/USDC:USDC"
        assert lighter_symbol_to_qubx("ETH-USDC") == "ETH/USDC:USDC"
        assert lighter_symbol_to_qubx("SOL-USDC") == "SOL/USDC:USDC"

    def test_qubx_to_lighter(self):
        """Test converting Qubx symbol to Lighter format"""
        assert qubx_symbol_to_lighter("BTC/USDC:USDC") == "BTC-USDC"
        assert qubx_symbol_to_lighter("ETH/USDC:USDC") == "ETH-USDC"
        assert qubx_symbol_to_lighter("BTC/USDC") == "BTC-USDC"  # Without settle

    def test_round_trip_conversion(self):
        """Test that symbol conversion is reversible"""
        lighter_symbol = "BTC-USDC"
        qubx_symbol = lighter_symbol_to_qubx(lighter_symbol)
        result = qubx_symbol_to_lighter(qubx_symbol)
        assert result == lighter_symbol


class TestOrderSideConversion:
    """Test order side conversion functions"""

    def test_lighter_to_qubx(self):
        """Test converting Lighter side to Qubx OrderSide"""
        assert lighter_order_side_to_qubx("B") == "BUY"
        assert lighter_order_side_to_qubx("S") == "SELL"

    def test_lighter_to_qubx_invalid(self):
        """Test that invalid Lighter side raises error"""
        with pytest.raises(ValueError):
            lighter_order_side_to_qubx("X")

    def test_qubx_to_lighter(self):
        """Test converting Qubx OrderSide to Lighter format"""
        assert qubx_order_side_to_lighter("BUY") == "B"
        assert qubx_order_side_to_lighter("SELL") == "S"

    def test_round_trip_conversion(self):
        """Test that side conversion is reversible"""
        for side_str in ["B", "S"]:
            qubx_side = lighter_order_side_to_qubx(side_str)
            result = qubx_order_side_to_lighter(qubx_side)
            assert result == side_str


class TestPriceConversion:
    """Test price conversion functions"""

    def test_lighter_price_to_float(self):
        """Test converting Lighter integer price to float"""
        # With 2 decimals: 10000 = 100.00
        assert lighter_price_to_float("10000", 2) == 100.00
        # With 6 decimals: 1000000 = 1.000000
        assert lighter_price_to_float("1000000", 6) == 1.0
        # With 0 decimals: 100 = 100
        assert lighter_price_to_float("100", 0) == 100.0

    def test_float_to_lighter_price(self):
        """Test converting float price to Lighter integer format"""
        assert float_to_lighter_price(100.00, 2) == "10000"
        assert float_to_lighter_price(1.0, 6) == "1000000"
        assert float_to_lighter_price(100.0, 0) == "100"

    def test_round_trip_price_conversion(self):
        """Test that price conversion is reversible"""
        decimals = 2
        original_str = "12345"
        float_price = lighter_price_to_float(original_str, decimals)
        result = float_to_lighter_price(float_price, decimals)
        assert result == original_str


class TestSizeConversion:
    """Test size conversion functions"""

    def test_lighter_size_to_float(self):
        """Test converting Lighter integer size to float"""
        assert lighter_size_to_float("1000", 3) == 1.0
        assert lighter_size_to_float("500", 3) == 0.5
        assert lighter_size_to_float("1", 3) == 0.001

    def test_float_to_lighter_size(self):
        """Test converting float size to Lighter integer format"""
        assert float_to_lighter_size(1.0, 3) == "1000"
        assert float_to_lighter_size(0.5, 3) == "500"
        assert float_to_lighter_size(0.001, 3) == "1"

    def test_round_trip_size_conversion(self):
        """Test that size conversion is reversible"""
        decimals = 3
        original_str = "1234"
        float_size = lighter_size_to_float(original_str, decimals)
        result = float_to_lighter_size(float_size, decimals)
        assert result == original_str


class TestQuoteConversion:
    """Test converting Lighter orderbook to Quote"""

    def test_convert_lighter_quote(self):
        """Test converting orderbook data to Quote"""
        orderbook_data = {
            "asks": [{"price": "100.50", "size": "1.5"}, {"price": "100.60", "size": "2.0"}],
            "bids": [{"price": "100.40", "size": "1.0"}, {"price": "100.30", "size": "0.5"}],
        }

        timestamp_ns = 1234567890000000000
        quote = convert_lighter_quote(orderbook_data, timestamp_ns)

        # Quote.time is stored as integer nanoseconds
        assert quote.time == timestamp_ns
        # Quote has 'ask' and 'bid' attributes, not 'ask_price' and 'bid_price'
        assert quote.ask == 100.50
        assert quote.ask_size == 1.5
        assert quote.bid == 100.40
        assert quote.bid_size == 1.0

    def test_convert_lighter_quote_empty(self):
        """Test converting empty orderbook"""
        orderbook_data = {"asks": [], "bids": []}

        timestamp_ns = 1234567890000000000
        quote = convert_lighter_quote(orderbook_data, timestamp_ns)

        # When orderbook is empty, prices should be NaN
        assert np.isnan(quote.ask)
        assert quote.ask_size == 0.0
        assert np.isnan(quote.bid)
        assert quote.bid_size == 0.0
