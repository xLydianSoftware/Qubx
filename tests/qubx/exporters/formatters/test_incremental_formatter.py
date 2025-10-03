"""
Unit tests for the IncrementalFormatter.
"""

import numpy as np
import pytest

from qubx.core.basics import AssetType, Instrument, MarketType, dt_64
from qubx.core.interfaces import IAccountViewer
from qubx.exporters.formatters import IncrementalFormatter
from tests.qubx.exporters.utils.mocks import MockAccountViewer


@pytest.fixture
def instrument() -> Instrument:
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
def formatter() -> IncrementalFormatter:
    """Fixture for a test formatter."""
    return IncrementalFormatter(alert_name="test_alert")


@pytest.fixture
def formatter_with_mapping() -> IncrementalFormatter:
    """Fixture for a test formatter with exchange mapping."""
    exchange_mapping = {"BINANCE": "BINANCE_FUTURES"}
    return IncrementalFormatter(alert_name="test_alert", exchange_mapping=exchange_mapping)


@pytest.fixture
def timestamp() -> dt_64:
    """Fixture for a test timestamp."""
    return np.datetime64("2023-01-01T12:00:00")


class TestIncrementalFormatter:
    """Unit tests for the IncrementalFormatter."""

    def test_no_leverage_change(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: IncrementalFormatter,
        timestamp: dt_64,
    ):
        """Test that no message is generated when there is no leverage change."""
        # Set initial leverage
        account_viewer.set_leverage(instrument, 1.0)

        # First call to establish the baseline
        formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        # Second call with the same leverage should return empty dict
        result = formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        assert result == {}

    def test_position_increase_same_side(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: IncrementalFormatter,
        timestamp: dt_64,
    ):
        """Test position increase on the same side (long to longer long)."""
        # Set initial leverage
        account_viewer.set_leverage(instrument, 1.0)

        # First call to establish the baseline
        formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        # Increase leverage
        account_viewer.set_leverage(instrument, 2.0)

        # Second call should generate an ENTRY message
        result = formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        assert result["type"] == "ENTRY"
        data = result["data"]
        assert "action':'ENTRY" in data
        assert "exchange':'BINANCE" in data
        assert "alertName':'test_alert" in data
        assert "symbol':'BTCUSDT" in data
        assert "side':'BUY" in data
        assert "leverage':1.0" in data  # Leverage change is 1.0 (from 1.0 to 2.0)
        assert "entryPrice':50000.0" in data

    def test_position_increase_same_side_short(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: IncrementalFormatter,
        timestamp: dt_64,
    ):
        """Test position increase on the same side (short to shorter short)."""
        # Set initial leverage
        account_viewer.set_leverage(instrument, -1.0)

        # First call to establish the baseline
        formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        # Increase leverage (more negative)
        account_viewer.set_leverage(instrument, -2.0)

        # Second call should generate an ENTRY message
        result = formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        assert result["type"] == "ENTRY"
        data = result["data"]
        assert "action':'ENTRY" in data
        assert "exchange':'BINANCE" in data
        assert "alertName':'test_alert" in data
        assert "symbol':'BTCUSDT" in data
        assert "side':'SELL" in data
        assert "leverage':1.0" in data  # Leverage change is 1.0 (from -1.0 to -2.0)
        assert "entryPrice':50000.0" in data

    def test_position_decrease(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: IncrementalFormatter,
        timestamp: dt_64,
    ):
        """Test position decrease (partial exit)."""
        # Set initial leverage
        account_viewer.set_leverage(instrument, 2.0)

        # First call to establish the baseline
        formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        # Decrease leverage
        account_viewer.set_leverage(instrument, 1.0)

        # Second call should generate an EXIT message
        result = formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        assert result["type"] == "EXIT"
        data = result["data"]
        assert "action':'EXIT" in data
        assert "exchange':'BINANCE" in data
        assert "alertName':'test_alert" in data
        assert "symbol':'BTCUSDT" in data
        assert "exitFraction':0.5" in data  # Exit fraction is 0.5 (from 2.0 to 1.0)
        assert "exitPrice':50000.0" in data

    def test_position_full_exit(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: IncrementalFormatter,
        timestamp: dt_64,
    ):
        """Test full position exit."""
        # Set initial leverage
        account_viewer.set_leverage(instrument, 2.0)

        # First call to establish the baseline
        formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        # Full exit (leverage to 0)
        account_viewer.set_leverage(instrument, 0.0)

        # Second call should generate an EXIT message
        result = formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        assert result["type"] == "EXIT"
        data = result["data"]
        assert "action':'EXIT" in data
        assert "exchange':'BINANCE" in data
        assert "alertName':'test_alert" in data
        assert "symbol':'BTCUSDT" in data
        assert "exitFraction':1.0" in data  # Exit fraction is 1.0 (full exit)
        assert "exitPrice':50000.0" in data

    def test_position_side_change(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: IncrementalFormatter,
        timestamp: dt_64,
    ):
        """Test position side change (long to short)."""
        # Set initial leverage
        account_viewer.set_leverage(instrument, 1.0)

        # First call to establish the baseline
        formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        # Change side from long to short
        account_viewer.set_leverage(instrument, -1.0)

        # Second call should generate an ENTRY message for the full short position
        result = formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        assert result["type"] == "ENTRY"
        data = result["data"]
        assert "action':'ENTRY" in data
        assert "exchange':'BINANCE" in data
        assert "alertName':'test_alert" in data
        assert "symbol':'BTCUSDT" in data
        assert "side':'SELL" in data
        assert "leverage':1.0" in data  # Full leverage of the short position
        assert "entryPrice':50000.0" in data

    def test_exchange_mapping(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter_with_mapping: IncrementalFormatter,
        timestamp: dt_64,
    ):
        """Test that exchange mapping is used correctly."""
        # Set initial leverage
        account_viewer.set_leverage(instrument, 1.0)

        # First call to establish the baseline
        formatter_with_mapping.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        # Increase leverage
        account_viewer.set_leverage(instrument, 2.0)

        # Second call should generate an ENTRY message with mapped exchange
        result = formatter_with_mapping.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        assert result["type"] == "ENTRY"
        data = result["data"]
        assert "exchange':'BINANCE_FUTURES" in data  # Mapped exchange name

    def test_new_position(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: IncrementalFormatter,
        timestamp: dt_64,
    ):
        """Test opening a new position from zero."""
        # First call with zero leverage
        account_viewer.set_leverage(instrument, 0.0)
        formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        # Open a new position
        account_viewer.set_leverage(instrument, 1.0)

        # Should generate an ENTRY message
        result = formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)

        assert result["type"] == "ENTRY"
        data = result["data"]
        assert "action':'ENTRY" in data
        assert "side':'BUY" in data
        assert "leverage':1.0" in data  # Full leverage of the new position
    
    def test_position_increase_after_restart(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        timestamp: dt_64,
    ):
        """Test position increase after initial position."""
        account_viewer.set_leverage(instrument, 1)
        exchange_mapping = {"BINANCE": "BINANCE_FUTURES"}
        formatter = IncrementalFormatter(alert_name="test_alert", 
                                         exchange_mapping=exchange_mapping, 
                                         account=account_viewer)
        
        # Increase leverage
        account_viewer.set_leverage(instrument, 2.0)

        result = formatter.format_position_change(timestamp, instrument, 50000.0, account_viewer)
        assert result["type"] == "ENTRY"
        data = result["data"]
        assert "action':'ENTRY" in data
        assert "leverage':1.0" in data
        
