# tests/qubx/core/test_position_initial_margin.py
"""Tests for Position.initial_margin + _initial_margin_external machinery.

Mirrors the existing maint_margin / _maint_margin_external pattern so live
account processors can write the exchange-reported value and the framework
won't recompute it on price updates.
"""

from qubx.core.basics import Instrument, MarketType, Position


def _make_instrument() -> Instrument:
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


def test_position_initial_margin_default_zero():
    pos = Position(instrument=_make_instrument(), quantity=0.0)
    assert pos.initial_margin == 0.0
    assert pos._initial_margin_external is False


def test_set_external_initial_margin_marks_external_and_stores_value():
    pos = Position(instrument=_make_instrument(), quantity=1.0, pos_average_price=2000.0)
    pos.set_external_initial_margin(123.45)
    assert pos.initial_margin == 123.45
    assert pos._initial_margin_external is True


def test_external_initial_margin_survives_price_update():
    """When set externally, _update_initial_margin must NOT overwrite."""
    pos = Position(instrument=_make_instrument(), quantity=1.0, pos_average_price=2000.0)
    pos.set_external_initial_margin(50.0)
    pos._update_initial_margin()  # exercise the recompute path directly
    assert pos.initial_margin == 50.0


def test_internal_initial_margin_recomputes_when_not_external():
    """Without an external value, _update_initial_margin can populate from
    instrument metadata + position size.  Default impl yields 0.0 today
    (Instrument.initial_margin is 0.0 unless populated by metadata storage).
    """
    pos = Position(instrument=_make_instrument(), quantity=1.0, pos_average_price=2000.0)
    pos.last_update_price = 2000.0
    pos._update_initial_margin()
    # No external value, no instrument-level initial_margin → stays 0.0
    assert pos.initial_margin == 0.0
    assert pos._initial_margin_external is False


def test_position_reset_clears_initial_margin_and_external_flag():
    pos = Position(instrument=_make_instrument(), quantity=1.0, pos_average_price=2000.0)
    pos.set_external_initial_margin(75.0)
    pos.reset()
    assert pos.initial_margin == 0.0
    assert pos._initial_margin_external is False


def test_reset_by_position_copies_initial_margin_state():
    src = Position(instrument=_make_instrument(), quantity=1.0, pos_average_price=2000.0)
    src.set_external_initial_margin(99.9)
    dst = Position(instrument=_make_instrument())
    dst.reset_by_position(src)
    assert dst.initial_margin == 99.9
    assert dst._initial_margin_external is True
