# tests/qubx/core/test_position_adl.py
"""Tests for Position.adl_level field (added for Hyperliquid ADL exposure)."""

from qubx.core.basics import Instrument, MarketType, Position


def _make_instrument() -> Instrument:
    """Minimal instrument for Position construction."""
    return Instrument(
        symbol="ETHUSDT",
        market_type=MarketType.SWAP,
        exchange="BINANCE.UM",
        base="ETH",
        quote="USDT",
        settle="USDT",
        exchange_symbol="ETHUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def test_position_adl_level_default_none():
    """Position constructed without ADL info should have adl_level=None."""
    instr = _make_instrument()
    pos = Position(instrument=instr, quantity=0.0)
    assert pos.adl_level is None


def test_position_adl_level_can_be_set():
    """Exchange-driven account processors set adl_level on existing positions."""
    instr = _make_instrument()
    pos = Position(instrument=instr, quantity=1.0, pos_average_price=2000.0)
    pos.adl_level = 2
    assert pos.adl_level == 2


def test_position_reset_clears_adl_level():
    """reset() should restore adl_level to None alongside other fields."""
    instr = _make_instrument()
    pos = Position(instrument=instr, quantity=1.0, pos_average_price=2000.0)
    pos.adl_level = 3
    pos.reset()
    assert pos.adl_level is None
    assert pos.quantity == 0.0  # sanity: reset still resets the rest
