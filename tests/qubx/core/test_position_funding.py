import pandas as pd
import pytest

from qubx.core.basics import Instrument, MarketType, Position


class TestPositionFunding:
    """Position.apply_funding_payment books a settled cash delta (time, amount)."""

    @pytest.fixture
    def instrument(self):
        return Instrument(
            symbol="BTCUSDT",
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
    def position(self, instrument):
        return Position(instrument=instrument, quantity=1.0, pos_average_price=50_000.0)

    def test_books_amount_and_stamps_time(self, position):
        t = pd.Timestamp("2025-01-08 00:00:00").asm8
        amount = position.apply_funding_payment(t, -5.0)
        assert amount == -5.0
        assert position.cumulative_funding == -5.0
        assert position.r_pnl == -5.0
        assert position.pnl == -5.0
        assert position.last_funding_time == t

    def test_multiple_payments_accumulate(self, position):
        times = [pd.Timestamp(f"2025-01-08 {h:02d}:00:00").asm8 for h in (0, 8, 16)]
        for t, amount in zip(times, (-5.0, 10.0, -2.5)):
            position.apply_funding_payment(t, amount)
        assert position.cumulative_funding == pytest.approx(2.5)
        assert position.r_pnl == pytest.approx(2.5)
        assert position.last_funding_time == times[-1]

    def test_books_on_flat_position(self, instrument):
        # a settlement is account truth even at qty=0 (closed between settle and delivery)
        position = Position(instrument=instrument)
        assert position.apply_funding_payment(pd.Timestamp("2025-01-08").asm8, -2.0) == -2.0
        assert position.cumulative_funding == -2.0

    def test_initialization(self, instrument):
        position = Position(instrument=instrument, quantity=1.0, pos_average_price=50_000.0)
        assert position.cumulative_funding == 0.0
        assert pd.isna(position.last_funding_time)
