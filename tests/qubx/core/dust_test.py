"""Dust predicates: ``Instrument.tradeable_size`` / ``is_dust`` and the latched ``Position.is_dust``."""

import pickle

import numpy as np
import pytest

from qubx.core.basics import Instrument, MarketType, Position
from qubx.core.lookups import _InstrumentEncoder, _InstrumentMapper

_NOW = np.datetime64("2026-10-23T00:00:00")


def _instrument(market_type: MarketType = MarketType.SPOT, **kw) -> Instrument:
    spec = {"tick_size": 0.0001, "lot_size": 0.1, "min_size": 0.1, "min_notional": 5.0} | kw
    exchange = "BINANCE" if market_type == MarketType.SPOT else "BINANCE.UM"
    return Instrument("INITUSDT", market_type, exchange, "INIT", "USDT", "USDT", "INITUSDT", **spec)


def _position(instrument: Instrument, quantity: float, price: float | None) -> Position:
    pos = Position(instrument=instrument, quantity=quantity, pos_average_price=1.0)
    if price is not None:
        pos.update_market_price(_NOW, price, 1.0)
    return pos


class TestMinNotionalOnReduce:
    @pytest.mark.parametrize(
        "market_type, expected",
        [(MarketType.SPOT, True), (MarketType.MARGIN, True), (MarketType.SWAP, False), (MarketType.FUTURE, False)],
    )
    def test_default_follows_the_market_type(self, market_type, expected):
        assert _instrument(market_type).min_notional_on_reduce is expected

    def test_override_and_reset(self):
        i = _instrument(MarketType.SWAP)
        i.min_notional_on_reduce = True
        assert i.min_notional_on_reduce is True
        i.min_notional_on_reduce = None
        assert i.min_notional_on_reduce is False

    def test_override_survives_pickle(self):
        i = _instrument(MarketType.SWAP)
        i.min_notional_on_reduce = True
        assert pickle.loads(pickle.dumps(i)).min_notional_on_reduce is True

    def test_lookup_cache_format_is_unchanged(self):
        import json

        i = _instrument()
        d = json.loads(json.dumps(i, cls=_InstrumentEncoder))
        assert "min_notional_on_reduce" not in d
        assert _InstrumentMapper.from_dict(d).min_notional_on_reduce is True


class TestTradeableSize:
    @pytest.mark.parametrize(
        "market_type, qty, price, expected",
        [
            # lot clause
            (MarketType.SPOT, 332.964, 1.0, 332.9),
            (MarketType.SPOT, 0.064, None, 0.0),
            (MarketType.SWAP, 0.064, None, 0.0),
            # notional clause: spot only, on the floored size
            (MarketType.SPOT, 4.0, 1.0, 0.0),
            (MarketType.SPOT, 5.05, 1.0, 5.0),
            (MarketType.SPOT, 5.04, 0.999, 0.0),
            (MarketType.SWAP, 4.0, 1.0, 4.0),
            # unknown price skips notional
            (MarketType.SPOT, 4.0, None, 4.0),
            (MarketType.SPOT, 4.0, float("nan"), 4.0),
            # signed
            (MarketType.SPOT, -332.964, 1.0, -332.9),
            (MarketType.SPOT, -4.0, 1.0, 0.0),
            (MarketType.SWAP, -4.0, 1.0, -4.0),
            (MarketType.SPOT, 0.0, 1.0, 0.0),
        ],
    )
    def test_clauses(self, market_type, qty, price, expected):
        assert _instrument(market_type).tradeable_size(qty, price) == pytest.approx(expected)

    @pytest.mark.parametrize("qty, expected", [(0.45, 0.0), (0.5, 0.5), (-0.57, -0.5)])
    def test_min_size_clause(self, qty, expected):
        i = _instrument(MarketType.SWAP, lot_size=0.1, min_size=0.5)
        assert i.tradeable_size(qty) == pytest.approx(expected)

    def test_overridden_derivative_applies_notional(self):
        i = _instrument(MarketType.SWAP)
        i.min_notional_on_reduce = True
        assert i.tradeable_size(4.0, 1.0) == 0.0

    def test_notional_uses_contract_size(self):
        i = _instrument(MarketType.SWAP, lot_size=1.0, min_size=1.0, contract_size=10.0)
        i.min_notional_on_reduce = True
        assert i.tradeable_size(1.0, 0.6) == 1.0
        assert i.tradeable_size(1.0, 0.4) == 0.0

    @pytest.mark.parametrize(
        "market_type, qty, price, expected",
        [
            (MarketType.SPOT, 0.0, 1.0, False),
            (MarketType.SPOT, 0.064, None, True),
            (MarketType.SPOT, 4.0, 1.0, True),
            (MarketType.SPOT, 4.0, None, False),
            (MarketType.SWAP, 4.0, 1.0, False),
            (MarketType.SPOT, -332.964, 1.0, False),
        ],
    )
    def test_is_dust(self, market_type, qty, price, expected):
        assert _instrument(market_type).is_dust(qty, price) is expected


class TestPositionDust:
    def test_tradeable_quantity_uses_the_last_price(self):
        assert _position(_instrument(), 332.964, 1.0).tradeable_quantity == pytest.approx(332.9)
        assert _position(_instrument(), 4.0, 1.0).tradeable_quantity == 0.0
        assert _position(_instrument(), 4.0, None).tradeable_quantity == 4.0

    def test_flat_is_not_dust(self):
        assert not _position(_instrument(), 0.0, 1.0).is_dust()

    def test_sub_lot_is_dust_without_a_price(self):
        assert _position(_instrument(), 0.064, None).is_dust()

    def test_unknown_price_is_not_notional_dust(self):
        assert not _position(_instrument(), 4.0, None).is_dust()

    def test_derivative_ignores_notional(self):
        assert not _position(_instrument(MarketType.SWAP), 4.0, 1.0).is_dust()

    def test_latch_enters_below_min_notional_and_leaves_at_the_band(self):
        pos = _position(_instrument(), 4.0, 1.30)  # 5.2
        assert not pos.is_dust()
        for price, dust in [(1.20, True), (1.40, True), (1.55, True), (1.5625, False), (1.30, False), (1.24, True)]:
            pos.update_market_price(_NOW, price, 1.0)
            assert pos.is_dust() is dust, price

    def test_configurable_band(self):
        pos = _position(_instrument(), 4.0, 1.20)
        pos.dust_band = 1.1
        assert pos.is_dust()
        pos.update_market_price(_NOW, 1.38, 1.0)  # 5.52 >= 5.5
        assert not pos.is_dust()

    def test_size_change_re_evaluates_from_scratch(self):
        pos = _position(_instrument(), 4.0, 1.20)  # 4.8
        assert pos.is_dust()
        pos.update_market_price(_NOW, 1.30, 1.0)  # 5.2: inside the band
        assert pos.is_dust()
        pos.change_position_by(_NOW, 0.1, 1.30)  # 4.1 * 1.3 = 5.33, fresh: not dust
        assert not pos.is_dust()

    def test_short_spot_dust(self):
        pos = _position(_instrument(), -4.0, 1.20)
        assert pos.is_dust()
        pos.update_market_price(_NOW, 1.30, 1.0)
        assert pos.is_dust()

    def test_lot_clause_is_not_latched(self):
        pos = _position(_instrument(), 0.064, 1000.0)
        assert pos.is_dust()

    def test_is_open_is_unchanged(self):
        pos = _position(_instrument(), 4.0, 1.0)
        assert pos.is_dust() and pos.is_open()
