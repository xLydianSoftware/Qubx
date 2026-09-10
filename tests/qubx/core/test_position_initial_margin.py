# tests/qubx/core/test_position_initial_margin.py
"""Tests for Position.initial_margin + _initial_margin_external machinery.

Mirrors the existing maint_margin / _maint_margin_external pattern so live
account processors can write the exchange-reported value and the framework
won't recompute it on price updates.
"""

import numpy as np
import pytest

from qubx.core.basics import DEFAULT_MAINTENANCE_MARGIN, Instrument, MarketType, Position

T0 = np.datetime64("2026-05-28T00:00:00", "ns")
T1 = np.datetime64("2026-05-28T00:01:00", "ns")
T2 = np.datetime64("2026-05-28T00:02:00", "ns")


def _make_instrument(initial_margin: float = 0.0) -> Instrument:
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
        initial_margin=initial_margin,
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


# --------------------------------------------------------------------------- #
# Venue-reported margins must not outlive the position.
#
# Venues report margins for OPEN positions only (Hyperliquid clearinghouseState,
# Binance positionRisk both omit flat ones), so once the deal ledger books a
# position to zero there is no snapshot left to refresh a value stamped via
# set_external_*_margin. The close itself has to drop the value AND the sticky
# flag, or AccountState.total_maint_margin()/margin_ratio() keep reading a
# maintenance requirement for risk that no longer exists.
# --------------------------------------------------------------------------- #


def test_full_close_clears_venue_reported_margins():
    pos = Position(instrument=_make_instrument())
    pos.update_position(T0, 100.0, 2000.0)
    pos.set_external_maint_margin(15.35693)
    pos.set_external_initial_margin(26.85)

    pos.update_position(T1, 0.0, 2010.0)

    assert pos.is_open() is False
    assert pos.maint_margin == 0.0
    assert pos._maint_margin_external is False
    assert pos.initial_margin == 0.0
    assert pos._initial_margin_external is False


def test_partial_close_keeps_venue_reported_margins():
    # the guard must fire only on going flat — a reduced position still carries the
    # venue value until the next snapshot re-reports it
    pos = Position(instrument=_make_instrument())
    pos.update_position(T0, 100.0, 2000.0)
    pos.set_external_maint_margin(15.35693)
    pos.set_external_initial_margin(26.85)

    pos.update_position(T1, 40.0, 2010.0)

    assert pos.is_open() is True
    assert pos.maint_margin == 15.35693
    assert pos._maint_margin_external is True
    assert pos.initial_margin == 26.85
    assert pos._initial_margin_external is True


def test_reopen_after_close_derives_margins_until_venue_reports_again():
    pos = Position(instrument=_make_instrument(initial_margin=0.1))
    pos.update_position(T0, 100.0, 2000.0)
    pos.set_external_maint_margin(15.35693)
    pos.set_external_initial_margin(26.85)
    pos.update_position(T1, 0.0, 2010.0)

    # reopen with no venue snapshot in between: framework-derived, not the stale value, not zero
    pos.update_position(T2, 2.0, 2020.0)
    assert pos.maint_margin == pytest.approx(DEFAULT_MAINTENANCE_MARGIN * 2.0 * 2020.0)
    assert pos._maint_margin_external is False
    assert pos.initial_margin == pytest.approx(0.1 * 2.0 * 2020.0)
    assert pos._initial_margin_external is False

    # the next snapshot carrying the instrument takes over again
    pos.set_external_maint_margin(14.9)
    pos.set_external_initial_margin(29.8)
    assert pos.maint_margin == 14.9
    assert pos._maint_margin_external is True
    assert pos.initial_margin == 29.8
    assert pos._initial_margin_external is True


def test_flatten_clears_external_flags_so_reopen_derives_margins():
    # flatten() (settle_position / missed-close recovery) zeroes the values; it must drop the
    # flags too, otherwise a reopen with no quote in between is an OPEN position reading
    # zero margin — an account that looks maximally safe with real risk on.
    pos = Position(instrument=_make_instrument(initial_margin=0.1))
    pos.update_position(T0, 100.0, 2000.0)
    pos.set_external_maint_margin(15.35693)
    pos.set_external_initial_margin(26.85)

    pos.flatten()
    assert pos.maint_margin == 0.0
    assert pos._maint_margin_external is False
    assert pos.initial_margin == 0.0
    assert pos._initial_margin_external is False

    pos.update_position(T1, 2.0, 2020.0)
    assert pos.is_open() is True
    assert pos.maint_margin == pytest.approx(DEFAULT_MAINTENANCE_MARGIN * 2.0 * 2020.0)
    assert pos.maint_margin > 0.0
    assert pos.initial_margin == pytest.approx(0.1 * 2.0 * 2020.0)
    assert pos.initial_margin > 0.0


def test_derived_margins_open_close_cycle_without_venue_values():
    # simulated / backtester path: the external flags are never set, so margins are derived
    # while open and zero once flat — the flat guard changes nothing here
    pos = Position(instrument=_make_instrument(initial_margin=0.1))

    pos.update_position(T0, 100.0, 2000.0)
    assert pos.maint_margin == pytest.approx(DEFAULT_MAINTENANCE_MARGIN * 100.0 * 2000.0)
    assert pos.initial_margin == pytest.approx(0.1 * 100.0 * 2000.0)

    pos.update_position(T1, 40.0, 2010.0)
    assert pos.maint_margin == pytest.approx(DEFAULT_MAINTENANCE_MARGIN * 40.0 * 2010.0)
    assert pos.initial_margin == pytest.approx(0.1 * 40.0 * 2010.0)

    pos.update_position(T2, 0.0, 2020.0)
    assert pos.maint_margin == 0.0
    assert pos._maint_margin_external is False
    assert pos.initial_margin == 0.0
    assert pos._initial_margin_external is False
