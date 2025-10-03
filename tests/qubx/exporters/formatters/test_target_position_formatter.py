"""
Unit tests for the TargetPositionFormatter.
"""

import numpy as np
import pytest

from qubx.core.basics import AssetType, Instrument, MarketType, Signal, TargetPosition, dt_64
from qubx.exporters.formatters import TargetPositionFormatter
from tests.qubx.exporters.utils.mocks import MockAccountViewer


@pytest.fixture
def instrument() -> Instrument:
    """Fixture for a test instrument."""
    return Instrument(
        symbol="BTC-USDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="BINANCE.UM",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


@pytest.fixture
def formatter() -> TargetPositionFormatter:
    """Fixture for a test formatter."""
    return TargetPositionFormatter(alert_name="test_alert")


@pytest.fixture
def formatter_with_mapping() -> TargetPositionFormatter:
    """Fixture for a test formatter with exchange mapping."""
    exchange_mapping = {"BINANCE.UM": "BINANCE_USDT"}
    return TargetPositionFormatter(alert_name="quarta", exchange_mapping=exchange_mapping)


@pytest.fixture
def timestamp() -> dt_64:
    """Fixture for a test timestamp."""
    return np.datetime64("2023-01-01T12:00:00")


@pytest.fixture
def signal(instrument, timestamp) -> Signal:
    """Fixture for a test signal."""
    return Signal(time=timestamp, instrument=instrument, signal=1.0, reference_price=50000.0)


class TestTargetPositionFormatter:
    """Unit tests for the TargetPositionFormatter."""

    def test_format_long_position_with_entry_price(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: TargetPositionFormatter,
        timestamp: dt_64,
        signal: Signal,
    ):
        """Test formatting a long target position with entry_price."""
        # Create target position with entry_price
        target = TargetPosition(
            time=timestamp,
            instrument=instrument,
            target_position_size=0.5,  # 0.5 BTC
            entry_price=50000.0,
        )

        # Format the target position
        result = formatter.format_target_position(timestamp, target, account_viewer)

        # Verify the result
        assert result["action"] == "TARGET_POSITION"
        assert result["alertName"] == "test_alert"
        assert result["exchange"] == "BINANCE.UM"
        assert result["symbol"] == "BTCUSDT"
        assert result["side"] == "BUY"
        # Leverage = (0.5 * 50000) / 12000 = 25000 / 12000 ≈ 2.083
        assert abs(result["leverage"] - 2.0833333333333335) < 0.001

    def test_format_short_position_with_entry_price(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: TargetPositionFormatter,
        timestamp: dt_64,
        signal: Signal,
    ):
        """Test formatting a short target position with entry_price."""
        # Create target position with entry_price
        target = TargetPosition(
            time=timestamp,
            instrument=instrument,
            target_position_size=-0.3,  # -0.3 BTC (short)
            entry_price=50000.0,
        )

        # Format the target position
        result = formatter.format_target_position(timestamp, target, account_viewer)

        # Verify the result
        assert result["action"] == "TARGET_POSITION"
        assert result["side"] == "SELL"
        # Leverage = abs(-0.3 * 50000) / 12000 = 15000 / 12000 = 1.25
        assert abs(result["leverage"] - 1.25) < 0.001

    def test_format_position_without_entry_price(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: TargetPositionFormatter,
        timestamp: dt_64,
        signal: Signal,
    ):
        """Test formatting a target position without entry_price, using position last_update_price."""
        # Set the position's last_update_price
        account_viewer.set_position_price(instrument, 51000.0)

        # Create target position without entry_price
        target = TargetPosition(
            time=timestamp,
            instrument=instrument,
            target_position_size=0.4,  # 0.4 BTC
            entry_price=None,
        )

        # Format the target position
        result = formatter.format_target_position(timestamp, target, account_viewer)

        # Verify the result
        assert result["action"] == "TARGET_POSITION"
        assert result["side"] == "BUY"
        # Leverage = (0.4 * 51000) / 12000 = 20400 / 12000 = 1.7
        assert abs(result["leverage"] - 1.7) < 0.001

    def test_format_position_without_price_returns_empty(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: TargetPositionFormatter,
        timestamp: dt_64,
        signal: Signal,
    ):
        """Test that formatting without any price returns empty dict."""
        # Create target position without entry_price and no position price set
        target = TargetPosition(
            time=timestamp,
            instrument=instrument,
            target_position_size=0.4,
            entry_price=None,
        )

        # Format the target position - should return empty dict
        result = formatter.format_target_position(timestamp, target, account_viewer)

        assert result == {}

    def test_exchange_mapping(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter_with_mapping: TargetPositionFormatter,
        timestamp: dt_64,
        signal: Signal,
    ):
        """Test that exchange mapping is used correctly."""
        # Create target position
        target = TargetPosition(
            time=timestamp,
            instrument=instrument,
            target_position_size=0.5,
            entry_price=50000.0,
        )

        # Format the target position
        result = formatter_with_mapping.format_target_position(timestamp, target, account_viewer)

        # Verify the mapped exchange name
        assert result["exchange"] == "BINANCE_USDT"
        assert result["alertName"] == "quarta"

    def test_zero_position_size(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: TargetPositionFormatter,
        timestamp: dt_64,
        signal: Signal,
    ):
        """Test formatting a zero position size."""
        # Create target position with zero size
        target = TargetPosition(
            time=timestamp,
            instrument=instrument,
            target_position_size=0.0,
            entry_price=50000.0,
        )

        # Format the target position
        result = formatter.format_target_position(timestamp, target, account_viewer)

        # Should still format, but leverage will be 0
        assert result["leverage"] == 0.0
        # Side is determined by sign, but for 0 it should be BUY (positive or zero)
        assert result["side"] == "SELL"  # 0 is not > 0, so SELL

    def test_leverage_calculation_precision(
        self,
        account_viewer: MockAccountViewer,
        instrument: Instrument,
        formatter: TargetPositionFormatter,
        timestamp: dt_64,
        signal: Signal,
    ):
        """Test precise leverage calculation."""
        # Create target position
        # Notional = 1.5 * 60000 = 90000
        # Total capital = 12000
        # Leverage = 90000 / 12000 = 7.5
        target = TargetPosition(
            time=timestamp,
            instrument=instrument,
            target_position_size=1.5,
            entry_price=60000.0,
        )

        # Format the target position
        result = formatter.format_target_position(timestamp, target, account_viewer)

        # Verify precise leverage
        assert result["leverage"] == 7.5
