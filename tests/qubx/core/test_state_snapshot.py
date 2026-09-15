"""``position_entry`` — the per-position dict the 5s state snapshot writes to Redis.

The snapshot is serialized with a plain ``json.dumps``, and the platform reads it with a Go
decoder that rejects ``Infinity``/``NaN`` — one of those blanks the whole bot state, not just
the field — so every case here asserts the entry survives ``allow_nan=False``.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.core.basics import Instrument, MarketType, Position
from qubx.core.state_snapshot import finite, position_entry

_NOW = np.datetime64("2026-09-15T00:00:00")


def _instrument(symbol: str = "BTCUSDT") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange="BINANCE.UM",
        base=symbol.replace("USDT", ""),
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def _position(instrument: Instrument, quantity: float, avg_price: float, mark: float) -> Position:
    pos = Position(instrument=instrument, quantity=quantity, pos_average_price=avg_price)
    pos.update_market_price(_NOW, mark, 1.0)
    return pos


def _account(
    *,
    leverage: float = 0.0,
    instrument_leverage: float | None = 3.0,
    max_instrument_leverage: float | None = 125.0,
    max_notional: float = 1_000_000.0,
) -> MagicMock:
    account = MagicMock()
    account.get_leverage.return_value = leverage
    account.get_instrument_leverage.return_value = instrument_leverage
    account.get_max_instrument_leverage.return_value = max_instrument_leverage
    account.get_max_instrument_notional.return_value = max_notional
    return account


def _dumps(entry: dict) -> str:
    return json.dumps(entry, allow_nan=False)


class TestFinite:
    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
    def test_non_finite_is_none(self, value):
        assert finite(value) is None

    def test_finite_passes_through_as_float(self):
        assert isinstance(finite(3), float)
        assert finite(3) == 3.0
        assert finite(-2.5) == -2.5


class TestPositionEntry:
    def test_an_open_short_reports_a_negative_notional(self):
        instrument = _instrument()
        position = _position(instrument, -2.0, 50_000.0, 51_000.0)
        entry = position_entry(_account(leverage=-5.1), instrument, position)

        assert entry["quantity"] == -2.0
        assert entry["avg_price"] == 50_000.0
        assert entry["current_price"] == 51_000.0
        assert entry["leverage"] == -5.1
        assert entry["notional"] == -102_000.0
        assert entry["instrument_leverage"] == 3.0
        assert entry["max_instrument_leverage"] == 125.0
        assert entry["max_notional"] == 1_000_000.0
        _dumps(entry)

    def test_a_flat_instrument_still_reports_the_venue_settings(self):
        instrument = _instrument()
        position = _position(instrument, 0.0, 0.0, 51_000.0)
        entry = position_entry(_account(), instrument, position)

        assert entry["quantity"] == 0.0
        assert entry["notional"] == 0.0
        assert entry["instrument_leverage"] == 3.0
        assert entry["max_instrument_leverage"] == 125.0
        _dumps(entry)

    def test_an_uncapped_venue_reports_no_max_notional(self):
        """The account manager answers inf when the venue publishes no per-asset cap."""
        instrument = _instrument()
        position = _position(instrument, 1.0, 50_000.0, 50_000.0)
        entry = position_entry(_account(max_notional=float("inf")), instrument, position)

        assert entry["max_notional"] is None
        _dumps(entry)

    def test_an_unmarked_position_serializes(self):
        """Every `set_universe` seats a flat position per new instrument, and each marks at NaN
        until its first quote — so this is the shape the whole document is written in right
        after a universe rotation, not an edge case."""
        instrument = _instrument()
        position = Position(instrument=instrument, quantity=1.0, pos_average_price=50_000.0)
        assert np.isnan(position.notional_value)
        assert np.isnan(position.last_update_price)
        entry = position_entry(_account(), instrument, position)

        assert entry["notional"] is None
        assert entry["current_price"] is None
        _dumps(entry)

    def test_a_connector_with_nothing_cached_reports_no_venue_settings(self):
        """None is the read side's "not populated yet" — it must not become a number here."""
        instrument = _instrument()
        position = _position(instrument, 1.0, 50_000.0, 50_000.0)
        entry = position_entry(_account(instrument_leverage=None, max_instrument_leverage=None), instrument, position)

        assert entry["instrument_leverage"] is None
        assert entry["max_instrument_leverage"] is None
        _dumps(entry)

    def test_the_keys_the_platform_reads_are_all_present(self):
        instrument = _instrument()
        entry = position_entry(_account(), instrument, _position(instrument, 1.0, 50_000.0, 50_000.0))
        assert list(entry) == [
            "quantity",
            "avg_price",
            "market_value",
            "unrealized_pnl",
            "current_price",
            "leverage",
            "notional",
            "instrument_leverage",
            "max_instrument_leverage",
            "max_notional",
        ]
