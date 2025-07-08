from unittest.mock import Mock

import pandas as pd
import pytest

from qubx.core.basics import AssetType, FundingPayment, Instrument, MarketType, Position, dt_64


class TestPositionFunding:
    """Test suite for position funding payment functionality."""

    @pytest.fixture
    def mock_instrument(self):
        """Create a mock instrument for testing."""
        return Instrument(
            symbol="BTCUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            exchange="binance",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTCUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
            contract_size=1.0,
        )

    @pytest.fixture
    def sample_position(self, mock_instrument):
        """Create a sample position for testing."""
        return Position(
            instrument=mock_instrument,
            quantity=1.0,  # Long 1 BTC
            pos_average_price=50000.0,
            r_pnl=0.0,
        )

    @pytest.fixture
    def sample_funding_payment(self):
        """Create a sample funding payment."""
        return FundingPayment(
            time=pd.Timestamp("2025-01-08 00:00:00").asm8,
            symbol="BTCUSDT",
            funding_rate=0.0001,  # 0.01% funding rate
            funding_interval_hours=8,
        )

    def test_position_apply_funding_payment_long_position_pays(self, sample_position, sample_funding_payment):
        """Test applying funding payment to a long position (pays funding)."""
        # Positive funding rate means longs pay shorts
        # With 1 BTC position at $50,000 and 0.01% rate:
        # Funding payment = 1 * 50000 * 0.0001 = $5

        initial_r_pnl = sample_position.r_pnl

        # Apply funding payment
        funding_amount = sample_position.apply_funding_payment(sample_funding_payment, mark_price=50000.0)

        # Verify funding amount (negative because position pays)
        assert funding_amount == -5.0

        # Verify realized PnL decreased by funding payment
        assert sample_position.r_pnl == initial_r_pnl - 5.0

        # Verify cumulative funding tracking
        assert sample_position.cumulative_funding == -5.0

        # Verify funding payment history
        assert len(sample_position.funding_payments) == 1
        assert sample_position.funding_payments[0] == sample_funding_payment

        # Verify last funding time
        assert sample_position.last_funding_time == sample_funding_payment.time

    def test_position_apply_funding_payment_short_position_receives(self, sample_position, sample_funding_payment):
        """Test applying funding payment to a short position (receives funding)."""
        # Make position short
        sample_position.quantity = -1.0  # Short 1 BTC
        initial_r_pnl = sample_position.r_pnl

        # Apply funding payment
        funding_amount = sample_position.apply_funding_payment(sample_funding_payment, mark_price=50000.0)

        # Verify funding amount (positive because position receives)
        assert funding_amount == 5.0

        # Verify realized PnL increased by funding payment
        assert sample_position.r_pnl == initial_r_pnl + 5.0

        # Verify cumulative funding tracking
        assert sample_position.cumulative_funding == 5.0

    def test_position_apply_funding_payment_negative_rate(self, sample_position):
        """Test funding payment with negative funding rate."""
        # Negative funding rate means shorts pay longs
        funding_payment = FundingPayment(
            time=pd.Timestamp("2025-01-08 08:00:00").asm8,
            symbol="BTCUSDT",
            funding_rate=-0.0002,  # -0.02% funding rate
            funding_interval_hours=8,
        )

        # Long position receives funding when rate is negative
        funding_amount = sample_position.apply_funding_payment(funding_payment, mark_price=50000.0)

        # Verify funding amount (positive because long receives)
        assert funding_amount == 10.0  # 1 * 50000 * 0.0002
        assert sample_position.r_pnl == 10.0
        assert sample_position.cumulative_funding == 10.0

    def test_position_apply_multiple_funding_payments(self, sample_position):
        """Test applying multiple funding payments over time."""
        funding_payments = [
            FundingPayment(
                time=pd.Timestamp(f"2025-01-08 {h:02d}:00:00").asm8,
                symbol="BTCUSDT",
                funding_rate=0.0001 if h % 16 == 0 else -0.0001,
                funding_interval_hours=8,
            )
            for h in [0, 8, 16]  # 3 funding payments
        ]

        total_funding = 0.0
        mark_price = 50000.0

        for fp in funding_payments:
            funding_amount = sample_position.apply_funding_payment(fp, mark_price)
            total_funding += funding_amount

        # Verify cumulative funding
        assert sample_position.cumulative_funding == total_funding

        # Verify all payments tracked
        assert len(sample_position.funding_payments) == 3

        # Verify last funding time
        assert sample_position.last_funding_time == funding_payments[-1].time

    def test_position_apply_funding_payment_zero_position(self, mock_instrument, sample_funding_payment):
        """Test funding payment on zero position."""
        # Create position with zero quantity
        position = Position(
            instrument=mock_instrument,
            quantity=0.0,
            pos_average_price=0.0,
            r_pnl=0.0,
        )

        # Apply funding payment
        funding_amount = position.apply_funding_payment(sample_funding_payment, mark_price=50000.0)

        # Verify no funding payment on zero position
        assert funding_amount == 0.0
        assert position.cumulative_funding == 0.0
        assert position.r_pnl == 0.0

    def test_position_funding_pnl_calculation(self, sample_position):
        """Test funding PnL calculation methods."""
        # Apply some funding payments
        funding_payments = [
            (FundingPayment(pd.Timestamp("2025-01-08 00:00:00").asm8, "BTCUSDT", 0.0001, 8), 50000.0),
            (FundingPayment(pd.Timestamp("2025-01-08 08:00:00").asm8, "BTCUSDT", -0.0002, 8), 51000.0),
            (FundingPayment(pd.Timestamp("2025-01-08 16:00:00").asm8, "BTCUSDT", 0.0003, 8), 49000.0),
        ]

        for fp, mark_price in funding_payments:
            sample_position.apply_funding_payment(fp, mark_price)

        # Test get_funding_pnl method
        funding_pnl = sample_position.get_funding_pnl()
        assert funding_pnl == sample_position.cumulative_funding

        # Test total PnL includes funding
        # Since funding is already included in pnl, just verify the method works
        total_pnl_with_funding = sample_position.get_total_pnl_with_funding()
        assert total_pnl_with_funding == sample_position.pnl

    def test_position_funding_payment_different_mark_prices(self, sample_position):
        """Test funding payments with different mark prices."""
        funding_payment = FundingPayment(
            time=pd.Timestamp("2025-01-08 00:00:00").asm8,
            symbol="BTCUSDT",
            funding_rate=0.0001,
            funding_interval_hours=8,
        )

        # Test with different mark prices
        mark_prices = [45000.0, 50000.0, 55000.0]
        expected_payments = [-4.5, -5.0, -5.5]  # Long pays more as price increases

        for mark_price, expected in zip(mark_prices, expected_payments):
            # Reset position for each test
            position = Position(
                instrument=sample_position.instrument,
                quantity=1.0,
                pos_average_price=50000.0,
                r_pnl=0.0,
            )

            funding_amount = position.apply_funding_payment(funding_payment, mark_price)
            assert funding_amount == expected

    def test_position_funding_payment_with_fractional_position(self, mock_instrument):
        """Test funding payment with fractional position size."""
        position = Position(
            instrument=mock_instrument,
            quantity=0.12345,  # Fractional BTC
            pos_average_price=50000.0,
            r_pnl=0.0,
        )

        funding_payment = FundingPayment(
            time=pd.Timestamp("2025-01-08 00:00:00").asm8,
            symbol="BTCUSDT",
            funding_rate=0.0001,
            funding_interval_hours=8,
        )

        # Apply funding payment
        funding_amount = position.apply_funding_payment(funding_payment, mark_price=50000.0)

        # Verify funding amount
        expected = -0.12345 * 50000.0 * 0.0001  # -0.61725
        assert pytest.approx(funding_amount, rel=1e-6) == expected

    def test_position_funding_payment_history_limit(self, sample_position):
        """Test that funding payment history has a reasonable limit."""
        # Apply many funding payments
        for i in range(1000):
            fp = FundingPayment(
                time=pd.Timestamp(f"2025-01-{8 + i // 96:02d} {(i * 15) % 24:02d}:00:00").asm8,
                symbol="BTCUSDT",
                funding_rate=0.0001,
                funding_interval_hours=8,
            )
            sample_position.apply_funding_payment(fp, mark_price=50000.0)

        # Verify history is limited (e.g., last 100 payments)
        # This assumes a max_funding_history limit of 100
        assert len(sample_position.funding_payments) <= 100

        # Verify cumulative funding is still accurate
        assert sample_position.cumulative_funding == -5.0 * 1000  # All payments tracked

    def test_position_funding_payment_initialization(self, mock_instrument):
        """Test that new positions have proper funding payment fields initialized."""
        position = Position(
            instrument=mock_instrument,
            quantity=1.0,
            pos_average_price=50000.0,
            r_pnl=0.0,
        )

        # Verify funding fields are initialized
        assert hasattr(position, "cumulative_funding")
        assert position.cumulative_funding == 0.0
        assert hasattr(position, "funding_payments")
        assert position.funding_payments == []
        assert hasattr(position, "last_funding_time")
        assert pd.isna(position.last_funding_time) or position.last_funding_time == pd.NaT

    def test_position_funding_payment_edge_cases(self, sample_position):
        """Test edge cases for funding payments."""
        # Test with zero funding rate
        zero_funding = FundingPayment(
            time=pd.Timestamp("2025-01-08 00:00:00").asm8, symbol="BTCUSDT", funding_rate=0.0, funding_interval_hours=8
        )
        funding_amount = sample_position.apply_funding_payment(zero_funding, mark_price=50000.0)
        assert funding_amount == 0.0

        # Test with very small funding rate
        tiny_funding = FundingPayment(
            time=pd.Timestamp("2025-01-08 08:00:00").asm8,
            symbol="BTCUSDT",
            funding_rate=1e-10,
            funding_interval_hours=8,
        )
        funding_amount = sample_position.apply_funding_payment(tiny_funding, mark_price=50000.0)
        assert abs(funding_amount) < 1e-5  # Very small payment

        # Test with large funding rate (0.5%)
        large_funding = FundingPayment(
            time=pd.Timestamp("2025-01-08 16:00:00").asm8,
            symbol="BTCUSDT",
            funding_rate=0.005,
            funding_interval_hours=8,
        )
        funding_amount = sample_position.apply_funding_payment(large_funding, mark_price=50000.0)
        assert funding_amount == -250.0  # 1 * 50000 * 0.005
