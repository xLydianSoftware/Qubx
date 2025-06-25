"""
Unit tests for the Slack message formatter.
"""

import numpy as np
import pytest

from qubx.core.basics import AssetType, Instrument, MarketType, Signal, TargetPosition
from qubx.exporters.formatters import SlackMessageFormatter

# MockAccountViewer is now imported from conftest.py via the fixture


@pytest.fixture
def instrument():
    """Fixture for a test instrument."""
    return Instrument(
        symbol="BTC-USDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SPOT,
        exchange="BINANCE",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.00001,
        min_size=0.0001,
    )


@pytest.fixture
def signal(instrument):
    """Fixture for a test signal."""
    return Signal("", instrument, 1.0, reference_price=50000.0, group="test_group", options={"test_key": "test_value"})


@pytest.fixture
def target_position(instrument, signal):
    """Fixture for a test target position."""
    time_now = np.datetime64("now")
    # Create a target position with reference price
    return TargetPosition(time_now, signal.instrument, 0.1)


class TestSlackMessageFormatter:
    """Unit tests for the SlackMessageFormatter."""

    def test_format_signal(self, account_viewer, signal):
        """Test formatting a signal."""
        formatter = SlackMessageFormatter()
        current_time = np.datetime64("now")

        result = formatter.format_signal(current_time, signal, account_viewer)

        # Check that the result is a dictionary with blocks
        assert isinstance(result, dict)
        assert "blocks" in result

        blocks = result["blocks"]
        assert len(blocks) >= 2

        # Check header block
        assert blocks[0]["type"] == "header"
        assert "New Signal" in blocks[0]["text"]["text"]

        # Check section block
        assert blocks[1]["type"] == "section"
        section_text = blocks[1]["text"]["text"]
        assert "BUY" in section_text
        assert "BTC-USDT" in section_text
        assert "BINANCE" in section_text
        assert "50000.0" in section_text
        assert "test_group" in section_text
        assert "test_key" in section_text
        assert "test_value" in section_text

        # Check account info block
        assert len(blocks) > 2
        assert blocks[2]["type"] == "divider"
        assert blocks[3]["type"] == "section"
        assert "Account Info" in blocks[3]["text"]["text"]
        assert "12000.00" in blocks[3]["text"]["text"]

    def test_format_signal_without_account_info(self, account_viewer, signal):
        """Test formatting a signal without account info."""
        formatter = SlackMessageFormatter(include_account_info=False)
        current_time = np.datetime64("now")

        result = formatter.format_signal(current_time, signal, account_viewer)

        # Check that the result is a dictionary with blocks
        assert isinstance(result, dict)
        assert "blocks" in result

        blocks = result["blocks"]
        assert len(blocks) == 2  # Only header and section, no account info

        # Check header block
        assert blocks[0]["type"] == "header"
        assert "New Signal" in blocks[0]["text"]["text"]

        # Check section block
        assert blocks[1]["type"] == "section"
        section_text = blocks[1]["text"]["text"]
        assert "BUY" in section_text

    def test_format_target_position(self, account_viewer, target_position):
        """Test formatting a target position."""
        formatter = SlackMessageFormatter()
        current_time = np.datetime64("now")

        result = formatter.format_target_position(current_time, target_position, account_viewer)

        # Check that the result is a dictionary with blocks
        assert isinstance(result, dict)
        assert "blocks" in result

        blocks = result["blocks"]
        assert len(blocks) >= 2

        # Check header block
        assert blocks[0]["type"] == "header"
        assert "Target Position" in blocks[0]["text"]["text"]

        # Check section block
        assert blocks[1]["type"] == "section"
        section_text = blocks[1]["text"]["text"]
        assert "BTC-USDT" in section_text
        assert "BINANCE" in section_text
        assert "0.1" in section_text
        # We don't check for price since it might be None or empty

    def test_format_position_change(self, account_viewer, instrument):
        """Test formatting a position change."""
        formatter = SlackMessageFormatter()
        current_time = np.datetime64("now")
        price = 50000.0

        result = formatter.format_position_change(current_time, instrument, price, account_viewer)

        # Check that the result is a dictionary with blocks
        assert isinstance(result, dict)
        assert "blocks" in result

        blocks = result["blocks"]
        assert len(blocks) >= 2

        # Check header block
        assert blocks[0]["type"] == "header"
        assert "Position Change" in blocks[0]["text"]["text"]

        # Check section block
        assert blocks[1]["type"] == "section"
        section_text = blocks[1]["text"]["text"]
        assert "BTC-USDT" in section_text
        assert "BINANCE" in section_text
        assert "50000.0" in section_text
        assert "Current Quantity" in section_text

    def test_custom_emoji(self, account_viewer, signal):
        """Test using a custom emoji."""
        formatter = SlackMessageFormatter(strategy_emoji=":rocket:")
        current_time = np.datetime64("now")

        result = formatter.format_signal(current_time, signal, account_viewer)

        # Check header block contains custom emoji
        blocks = result["blocks"]
        header_text = blocks[0]["text"]["text"]
        assert ":rocket:" in header_text
