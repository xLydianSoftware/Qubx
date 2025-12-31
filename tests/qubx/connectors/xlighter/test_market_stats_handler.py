"""
Unit tests for Lighter market stats handler.

Tests interval-based emission logic for FundingRate and OpenInterest,
ensuring no lookahead bias by buffering data and emitting at interval boundaries.
"""

import pytest

from qubx.connectors.xlighter.handlers.stats import MarketStatsHandler
from qubx.core.basics import FundingPayment, FundingRate, Instrument, MarketType, OpenInterest, dt_64


@pytest.fixture
def mock_instrument_loader():
    """Create a mock instrument loader with test instruments."""

    class MockInstrumentLoader:
        def __init__(self):
            # Create test instruments for market_id=0 (BTC) and market_id=1 (ETH)
            self.btc_instrument = Instrument(
                symbol="BTCUSDC",
                asset_type="CRYPTO",
                market_type=MarketType.SWAP,
                exchange="LIGHTER",
                base="BTC",
                quote="USDC",
                settle="USDC",
                exchange_symbol="BTC",
                tick_size=0.01,
                lot_size=0.001,
                min_size=0.001,
                min_notional=5.0,
            )

            self.eth_instrument = Instrument(
                symbol="ETHUSDC",
                asset_type="CRYPTO",
                market_type=MarketType.SWAP,
                exchange="LIGHTER",
                base="ETH",
                quote="USDC",
                settle="USDC",
                exchange_symbol="ETH",
                tick_size=0.01,
                lot_size=0.01,
                min_size=0.01,
                min_notional=5.0,
            )

            self.market_id_to_instrument = {
                0: self.btc_instrument,
                1: self.eth_instrument,
            }

        def get_instrument_by_market_id(self, market_id: int):
            return self.market_id_to_instrument.get(market_id)

    return MockInstrumentLoader()


@pytest.fixture
def handler(mock_instrument_loader):
    """Create market stats handler with mock instrument loader."""
    return MarketStatsHandler(instrument_loader=mock_instrument_loader)


def create_market_stats_message(market_id: int = 0, timestamp: int | None = None, **stats) -> dict:
    """
    Helper to create market stats message.

    Args:
        market_id: Market ID
        timestamp: Message timestamp in milliseconds (optional)
        **stats: Market stats fields (funding_rate, open_interest, etc.)
    """
    message = {
        "channel": "market_stats:all",
        "type": "update/market_stats",
        "market_stats": {
            str(market_id): {
                "market_id": market_id,
                "index_price": stats.get("index_price", "4000.00"),
                "mark_price": stats.get("mark_price", "4000.00"),
                "open_interest": stats.get("open_interest", "1000000.0"),
                "current_funding_rate": stats.get("current_funding_rate", "0.0001"),
                "funding_rate": stats.get("funding_rate", "0.0001"),
                "funding_timestamp": stats.get("funding_timestamp", 1760972400000),
                "last_trade_price": "4000.00",
            }
        },
    }

    if timestamp is not None:
        message["timestamp"] = timestamp

    return message


class TestMarketStatsHandlerBasics:
    """Test basic handler functionality."""

    def test_can_handle_market_stats_message(self, handler):
        """Test that handler recognizes market stats messages."""
        message = create_market_stats_message()
        assert handler.can_handle(message)

    def test_cannot_handle_other_messages(self, handler):
        """Test that handler rejects non-market-stats messages."""
        message = {"channel": "trade:0", "type": "update/trade", "data": []}
        assert not handler.can_handle(message)


class TestIntervalBasedEmissionFundingRate:
    """Test interval-based emission for FundingRate (1min intervals)."""

    def test_first_message_buffers_no_emission(self, handler):
        """Test that first message is buffered but not emitted."""
        # Message at 18:00:10
        message = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 0 * 60 + 10),  # 18:00:10 in ms
            current_funding_rate="0.0005",
        )

        result = handler.handle(message)

        # Should return None or empty dict (no emission yet)
        assert result is None or len(result) == 0

        # But buffer should be populated
        assert 0 in handler._funding_rate_buffer
        assert handler._funding_rate_buffer[0]["data"]["rate"] == 5e-06  # 0.0005 / 100

    def test_same_interval_updates_buffer_no_emission(self, handler):
        """Test that messages in same interval update buffer without emission."""

        # Message 1 at 18:00:10
        msg1 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 0 * 60 + 10),
            current_funding_rate="0.0005",
        )
        result1 = handler.handle(msg1)
        assert result1 is None or len(result1) == 0

        # Message 2 at 18:00:45 (same 1-min interval: 18:00-18:01)
        msg2 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 0 * 60 + 45),
            current_funding_rate="0.0006",
        )
        result2 = handler.handle(msg2)
        assert result2 is None or len(result2) == 0

        # Buffer should have latest value
        assert handler._funding_rate_buffer[0]["data"]["rate"] == pytest.approx(6e-06)  # 0.0006 / 100

    def test_boundary_crossing_emits_buffered_data(self, handler):
        """Test that crossing interval boundary emits buffered data."""
        btc = handler.instrument_loader.get_instrument_by_market_id(0)

        # Message 1 at 18:00:45 (buffers, no emission)
        msg1 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 0 * 60 + 45),
            current_funding_rate="0.0005",
            mark_price="4000.00",
            index_price="4001.00",
            funding_timestamp=1760972400000,
        )
        result1 = handler.handle(msg1)
        assert result1 is None or len(result1) == 0

        # Message 2 at 18:01:10 (crosses boundary, emits msg1 data)
        msg2 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 1 * 60 + 10),
            current_funding_rate="0.0006",
            mark_price="4010.00",
            index_price="4011.00",
            funding_timestamp=1760972400000,
        )
        result2 = handler.handle(msg2)

        # Should emit msg1 data with 18:01:00 timestamp
        assert result2 is not None
        assert btc in result2
        funding_rates = [obj for obj in result2[btc] if isinstance(obj, FundingRate)]
        assert len(funding_rates) == 1

        fr = funding_rates[0]
        # Check that emitted data is from msg1 (0.0005 / 100, not 0.0006 / 100)
        assert fr.rate == 5e-06  # 0.0005 / 100
        assert fr.mark_price == 4000.00
        assert fr.index_price == 4001.00

        # Check that timestamp is floored to 18:01:00
        expected_time = dt_64(int(1000 * (18 * 3600 + 1 * 60 + 0) * 1_000_000), "ns")
        assert fr.time == expected_time

    def test_multiple_boundaries_only_emit_latest(self, handler):
        """Test that skipping intervals only emits for current boundary."""
        btc = handler.instrument_loader.get_instrument_by_market_id(0)

        # Message at 18:00:30
        msg1 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 0 * 60 + 30),
            current_funding_rate="0.0005",
        )
        handler.handle(msg1)

        # Skip to 18:05:10 (skipped 18:01, 18:02, 18:03, 18:04)
        msg2 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 5 * 60 + 10),
            current_funding_rate="0.0010",
        )
        result = handler.handle(msg2)

        # Should emit msg1 data with 18:05:00 timestamp (current boundary)
        # Gaps at 18:01, 18:02, 18:03, 18:04 are left empty
        assert btc in result
        funding_rates = [obj for obj in result[btc] if isinstance(obj, FundingRate)]
        assert len(funding_rates) == 1
        assert funding_rates[0].rate == 5e-06  # 0.0005 / 100

        # Timestamp should be 18:05:00 (current boundary, not 18:01:00)
        expected_time = dt_64(int(1000 * (18 * 3600 + 5 * 60 + 0) * 1_000_000), "ns")
        assert funding_rates[0].time == expected_time


class TestIntervalBasedEmissionOpenInterest:
    """Test interval-based emission for OpenInterest (5min intervals)."""

    def test_first_message_buffers_no_emission(self, handler):
        """Test that first message is buffered but not emitted."""
        message = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 0 * 60 + 10),
            open_interest="1000000.0",
            mark_price="4000.00",
        )

        result = handler.handle(message)
        assert result is None or len(result) == 0

        # Buffer should be populated
        assert 0 in handler._open_interest_buffer
        assert handler._open_interest_buffer[0]["data"]["open_interest"] == 1000000.0

    def test_same_interval_updates_buffer_no_emission(self, handler):
        """Test messages in same 5min interval update buffer without emission of OpenInterest."""
        btc = handler.instrument_loader.get_instrument_by_market_id(0)

        # Message 1 at 18:00:10
        msg1 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 0 * 60 + 10),
            open_interest="1000000.0",
            mark_price="4000.00",
        )
        handler.handle(msg1)

        # Message 2 at 18:00:45 (same 5-min interval: 18:00-18:05, and same 1-min for funding)
        msg2 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 0 * 60 + 45),
            open_interest="1100000.0",
            mark_price="4010.00",
        )
        result2 = handler.handle(msg2)

        # No OpenInterest should be emitted (still in first 5-min interval)
        if result2 and btc in result2:
            ois = [obj for obj in result2[btc] if isinstance(obj, OpenInterest)]
            assert len(ois) == 0

        # Buffer should have latest value
        assert handler._open_interest_buffer[0]["data"]["open_interest"] == 1100000.0

    def test_boundary_crossing_emits_buffered_data(self, handler):
        """Test crossing 5min boundary emits buffered data."""
        btc = handler.instrument_loader.get_instrument_by_market_id(0)

        # Message at 18:04:55
        msg1 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 4 * 60 + 55),
            open_interest="1200000.0",
            mark_price="4020.00",
        )
        handler.handle(msg1)

        # Message at 18:05:30 (crosses 18:05 boundary)
        msg2 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 5 * 60 + 30),
            open_interest="1250000.0",
            mark_price="4030.00",
        )
        result = handler.handle(msg2)

        # Should emit msg1 data with 18:05:00 timestamp
        assert btc in result
        ois = [obj for obj in result[btc] if isinstance(obj, OpenInterest)]
        assert len(ois) == 1

        oi = ois[0]
        assert oi.open_interest == 1200000.0
        assert oi.open_interest_usd == 1200000.0 * 4020.00  # Uses msg1 mark_price
        assert oi.symbol == "BTCUSDC"

        # Timestamp should be 18:05:00
        expected_time = dt_64(int(1000 * (18 * 3600 + 5 * 60 + 0) * 1_000_000), "ns")
        assert oi.time == expected_time


class TestFundingPaymentEmission:
    """Test funding payment emission based on funding_timestamp changes."""

    def test_no_payment_on_first_message(self, handler):
        """Test that no payment is emitted on first message."""
        message = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 0 * 60 + 10),
            funding_rate="0.0002",
            funding_timestamp=1760972400000,
        )

        result = handler.handle(message)
        # No payment should be emitted
        if result:
            payments = [obj for obj in result.get(handler.instrument_loader.btc_instrument, []) if isinstance(obj, FundingPayment)]
            assert len(payments) == 0

    def test_payment_emitted_when_timestamp_changes(self, handler):
        """Test that payment is emitted when funding_timestamp advances."""
        btc = handler.instrument_loader.get_instrument_by_market_id(0)

        # First message at hour 1
        msg1 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 0 * 60 + 10),
            funding_rate="0.0002",
            funding_timestamp=1760972400000,
        )
        handler.handle(msg1)

        # Second message at hour 2 (funding_timestamp advanced)
        msg2 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (19 * 3600 + 0 * 60 + 10),
            funding_rate="0.0003",
            funding_timestamp=1760976000000,  # 1 hour later
        )
        result = handler.handle(msg2)

        # Should emit payment
        assert btc in result
        payments = [obj for obj in result[btc] if isinstance(obj, FundingPayment)]
        assert len(payments) == 1

        payment = payments[0]
        assert payment.funding_rate == pytest.approx(3e-06)  # 0.0003 / 100 (Rate from msg2)
        assert payment.funding_interval_hours == 1

        # Timestamp should be the new funding_timestamp
        expected_time = dt_64(int(1760976000000 * 1_000_000), "ns")
        assert payment.time == expected_time

    def test_no_payment_when_timestamp_unchanged(self, handler):
        """Test that no payment when funding_timestamp doesn't change."""
        # Message 1
        msg1 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 0 * 60 + 10),
            funding_rate="0.0002",
            funding_timestamp=1760972400000,
        )
        handler.handle(msg1)

        # Message 2 with same funding_timestamp
        msg2 = create_market_stats_message(
            market_id=0,
            timestamp=1000 * (18 * 3600 + 30 * 60 + 10),
            funding_rate="0.0002",
            funding_timestamp=1760972400000,  # Same as msg1
        )
        result = handler.handle(msg2)

        # No payment should be emitted
        if result:
            btc = handler.instrument_loader.get_instrument_by_market_id(0)
            payments = [obj for obj in result.get(btc, []) if isinstance(obj, FundingPayment)]
            assert len(payments) == 0


class TestMultipleInstruments:
    """Test handler with multiple instruments in same message."""

    def test_multiple_instruments_processed_independently(self, handler):
        """Test that multiple instruments are processed with separate state."""
        # Message with both BTC and ETH
        message = {
            "channel": "market_stats:all",
            "type": "update/market_stats",
            "timestamp": 1000 * (18 * 3600 + 0 * 60 + 10),
            "market_stats": {
                "0": {  # BTC
                    "market_id": 0,
                    "open_interest": "1000000.0",
                    "mark_price": "4000.00",
                    "current_funding_rate": "0.0001",
                    "funding_rate": "0.0001",
                    "funding_timestamp": 1760972400000,
                    "index_price": "4000.00",
                },
                "1": {  # ETH
                    "market_id": 1,
                    "open_interest": "500000.0",
                    "mark_price": "200.00",
                    "current_funding_rate": "0.0002",
                    "funding_rate": "0.0002",
                    "funding_timestamp": 1760972400000,
                    "index_price": "200.00",
                },
            },
        }

        result = handler.handle(message)
        # First message - no emission
        assert result is None or len(result) == 0

        # Verify both instruments buffered separately
        assert 0 in handler._open_interest_buffer
        assert 1 in handler._open_interest_buffer
        assert handler._open_interest_buffer[0]["data"]["open_interest"] == 1000000.0
        assert handler._open_interest_buffer[1]["data"]["open_interest"] == 500000.0
