"""
Unit tests for funding payment handler functionality in CCXT connector.

Tests the funding rate handler's ability to emit funding payment events
when funding intervals elapse, using the correct rate from the previous period.
"""

import pandas as pd
import pytest
from unittest.mock import Mock

from qubx.core.basics import AssetType, FundingRate, MarketType, Instrument
from qubx.connectors.ccxt.handlers.funding_rate import FundingRateDataHandler


class TestFundingPaymentHandler:
    """Test funding payment emission logic in FundingRateDataHandler."""

    @pytest.fixture
    def mock_instrument(self):
        """Create a mock futures instrument for testing."""
        return Instrument(
            symbol="BTCUSDT",
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
    def handler(self):
        """Create a funding rate handler for testing."""
        mock_data_provider = Mock()
        mock_exchange = Mock()
        exchange_id = "test_exchange"
        
        handler = FundingRateDataHandler(mock_data_provider, mock_exchange, exchange_id)
        return handler

    def test_first_funding_rate_no_payment(self, handler, mock_instrument):
        """Test that first funding rate update doesn't emit payment."""
        funding_rate = FundingRate(
            time=pd.Timestamp("2025-01-15 07:30:00").asm8,
            rate=0.0001,
            interval="8h",
            next_funding_time=pd.Timestamp("2025-01-15 08:00:00").asm8
        )
        
        should_emit = handler._should_emit_payment(mock_instrument, funding_rate)
        assert should_emit is False
        
        # Verify rate is stored
        assert mock_instrument.symbol in handler._pending_funding_rates
        stored_info = handler._pending_funding_rates[mock_instrument.symbol]
        assert stored_info['rate'] == funding_rate
        assert stored_info['payment_time'] == funding_rate.next_funding_time

    def test_same_period_no_payment(self, handler, mock_instrument):
        """Test that multiple updates in same funding period don't emit payment."""
        # First update
        funding_rate1 = FundingRate(
            time=pd.Timestamp("2025-01-15 07:30:00").asm8,
            rate=0.0001,
            interval="8h",
            next_funding_time=pd.Timestamp("2025-01-15 08:00:00").asm8
        )
        handler._should_emit_payment(mock_instrument, funding_rate1)
        
        # Second update in same period
        funding_rate2 = FundingRate(
            time=pd.Timestamp("2025-01-15 07:45:00").asm8,
            rate=0.0001,
            interval="8h",
            next_funding_time=pd.Timestamp("2025-01-15 08:00:00").asm8  # Same next funding time
        )
        
        should_emit = handler._should_emit_payment(mock_instrument, funding_rate2)
        assert should_emit is False

    def test_new_funding_period_emits_payment(self, handler, mock_instrument):
        """Test that new funding period emits payment with previous rate."""
        # First update - establish baseline
        funding_rate1 = FundingRate(
            time=pd.Timestamp("2025-01-15 07:30:00").asm8,
            rate=0.0001,  # This rate should be used for payment
            interval="8h",
            next_funding_time=pd.Timestamp("2025-01-15 08:00:00").asm8
        )
        handler._should_emit_payment(mock_instrument, funding_rate1)
        
        # Second update - new funding period
        funding_rate2 = FundingRate(
            time=pd.Timestamp("2025-01-15 08:01:00").asm8,
            rate=0.0002,  # New rate, but payment should use 0.0001
            interval="8h",
            next_funding_time=pd.Timestamp("2025-01-15 16:00:00").asm8  # Advanced funding time
        )
        
        should_emit = handler._should_emit_payment(mock_instrument, funding_rate2)
        assert should_emit is True
        
        # Verify payment info uses previous rate
        payment_key = f"{mock_instrument.symbol}_payment"
        assert payment_key in handler._pending_funding_rates
        payment_info = handler._pending_funding_rates[payment_key]
        assert payment_info['rate'] == 0.0001  # Previous rate, not current
        assert payment_info['time'] == pd.Timestamp("2025-01-15 08:00:00").asm8

    def test_multiple_funding_periods(self, handler, mock_instrument):
        """Test multiple funding period transitions."""
        funding_updates = [
            # (time, rate, next_funding_time, should_emit)
            ("2025-01-15 07:30:00", 0.0001, "2025-01-15 08:00:00", False),  # First update
            ("2025-01-15 07:45:00", 0.0001, "2025-01-15 08:00:00", False),  # Same period
            ("2025-01-15 08:01:00", 0.0002, "2025-01-15 16:00:00", True),   # New period
            ("2025-01-15 12:00:00", 0.0002, "2025-01-15 16:00:00", False),  # Same period
            ("2025-01-15 16:05:00", 0.0003, "2025-01-16 00:00:00", True),   # Another new period
        ]
        
        expected_payment_rates = [0.0001, 0.0002]  # Rates that should be used for payments
        payment_count = 0
        
        for time_str, rate, next_funding_str, expected_emit in funding_updates:
            funding_rate = FundingRate(
                time=pd.Timestamp(time_str).asm8,
                rate=rate,
                interval="8h",
                next_funding_time=pd.Timestamp(next_funding_str).asm8
            )
            
            should_emit = handler._should_emit_payment(mock_instrument, funding_rate)
            assert should_emit == expected_emit
            
            if expected_emit:
                payment_key = f"{mock_instrument.symbol}_payment"
                payment_info = handler._pending_funding_rates[payment_key]
                assert payment_info['rate'] == expected_payment_rates[payment_count]
                payment_count += 1

    def test_create_funding_payment(self, handler, mock_instrument):
        """Test funding payment creation using stored payment info."""
        # Set up payment info
        payment_time = pd.Timestamp("2025-01-15 08:00:00").asm8
        payment_key = f"{mock_instrument.symbol}_payment"
        handler._pending_funding_rates[payment_key] = {
            'rate': 0.0001,
            'time': payment_time,
            'interval_hours': 8
        }
        
        # Create payment
        payment = handler._create_funding_payment(mock_instrument)
        
        assert payment.time == payment_time
        assert payment.funding_rate == 0.0001
        assert payment.funding_interval_hours == 8

    def test_extract_interval_hours(self, handler):
        """Test interval hours extraction from different formats."""
        test_cases = [
            ("8h", 8),
            ("4h", 4),
            ("1h", 1),
            ("8", 8),
            ("4", 4),
            ("unknown", 8),  # Default fallback
            ("", 8),  # Default fallback
        ]
        
        for interval_str, expected_hours in test_cases:
            result = handler._extract_interval_hours(interval_str)
            assert result == expected_hours

    def test_create_funding_payment_fallback(self, handler, mock_instrument):
        """Test funding payment creation fallback when no payment info stored."""
        # Store current rate info but no payment info
        current_rate = FundingRate(
            time=pd.Timestamp("2025-01-15 08:00:00").asm8,
            rate=0.0001,
            interval="8h",
            next_funding_time=pd.Timestamp("2025-01-15 16:00:00").asm8
        )
        handler._pending_funding_rates[mock_instrument.symbol] = {
            'rate': current_rate,
            'payment_time': current_rate.next_funding_time,
            'stored_at': current_rate.time
        }
        
        # Create payment (should use current rate as fallback)
        payment = handler._create_funding_payment(mock_instrument)
        
        assert payment.time == current_rate.time
        assert payment.funding_rate == current_rate.rate
        assert payment.funding_interval_hours == 8

    def test_create_funding_payment_no_data_error(self, handler, mock_instrument):
        """Test that creating payment without any data raises error."""
        with pytest.raises(ValueError, match="No funding rate data available"):
            handler._create_funding_payment(mock_instrument)

    def test_pending_funding_rates_storage(self, handler):
        """Test that handler properly stores pending funding rates."""
        # Initially empty
        assert len(handler._pending_funding_rates) == 0
        
        # After storing funding rate info
        test_key = "BTCUSDT"
        test_info = {
            'rate': Mock(),
            'payment_time': pd.Timestamp('2025-01-15 08:00:00').asm8,
            'stored_at': pd.Timestamp('2025-01-15 07:30:00').asm8
        }
        
        handler._pending_funding_rates[test_key] = test_info
        assert test_key in handler._pending_funding_rates
        assert handler._pending_funding_rates[test_key] == test_info
        assert len(handler._pending_funding_rates) == 1