"""Tests for IAccountViewer.get_total_initial_margin / get_total_maint_margin.

Mirrors the existing get_total_required_margin pattern (which sums maint
margin across positions) but splits initial vs. maintenance for clarity:
- initial: "can I open more?"
- maintenance: "how close am I to liquidation?"
"""

from qubx.core.basics import Instrument, MarketType, Position


def _make_instrument(symbol: str = "ETHUSDT") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange="BINANCE.UM",
        base="ETH",
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def test_iaccountviewer_has_total_initial_and_maint_margin_methods():
    """Both methods must be declared on the IAccountViewer interface."""
    from qubx.core.interfaces import IAccountViewer

    assert hasattr(IAccountViewer, "get_total_initial_margin"), (
        "IAccountViewer must declare get_total_initial_margin"
    )
    assert hasattr(IAccountViewer, "get_total_maint_margin"), (
        "IAccountViewer must declare get_total_maint_margin"
    )


def test_get_total_initial_margin_sums_position_initial_margins():
    """Default impl in BasicAccountProcessor sums Position.initial_margin across positions."""
    instr_a = _make_instrument("ETHUSDT")
    instr_b = _make_instrument("BTCUSDT")
    pos_a = Position(instrument=instr_a, quantity=1.0, pos_average_price=2000.0)
    pos_b = Position(instrument=instr_b, quantity=0.1, pos_average_price=50000.0)
    pos_a.set_external_initial_margin(100.0)
    pos_b.set_external_initial_margin(250.0)

    class _Stub:
        _positions = {instr_a: pos_a, instr_b: pos_b}

        def get_total_initial_margin(self, exchange=None):
            return sum(p.initial_margin for p in self._positions.values())

    stub = _Stub()
    assert stub.get_total_initial_margin() == 350.0


def test_get_total_maint_margin_sums_position_maint_margins():
    instr_a = _make_instrument("ETHUSDT")
    instr_b = _make_instrument("BTCUSDT")
    pos_a = Position(instrument=instr_a, quantity=1.0, pos_average_price=2000.0)
    pos_b = Position(instrument=instr_b, quantity=0.1, pos_average_price=50000.0)
    pos_a.set_external_maint_margin(50.0)
    pos_b.set_external_maint_margin(125.0)

    class _Stub:
        _positions = {instr_a: pos_a, instr_b: pos_b}

        def get_total_maint_margin(self, exchange=None):
            return sum(p.maint_margin for p in self._positions.values())

    stub = _Stub()
    assert stub.get_total_maint_margin() == 175.0


def test_basic_account_processor_get_total_initial_margin_returns_sum():
    """End-to-end: BasicAccountProcessor's default impl produces the right sum."""
    from unittest.mock import MagicMock

    from qubx.core.account import BasicAccountProcessor

    acc = BasicAccountProcessor.__new__(BasicAccountProcessor)
    instr = _make_instrument()
    pos = Position(instrument=instr, quantity=1.0, pos_average_price=2000.0)
    pos.set_external_initial_margin(42.0)
    acc._positions = {instr: pos}
    acc._exchange = "BINANCE.UM"

    assert acc.get_total_initial_margin() == 42.0


def test_basic_account_processor_get_total_maint_margin_returns_sum():
    from qubx.core.account import BasicAccountProcessor

    acc = BasicAccountProcessor.__new__(BasicAccountProcessor)
    instr = _make_instrument()
    pos = Position(instrument=instr, quantity=1.0, pos_average_price=2000.0)
    pos.set_external_maint_margin(17.0)
    acc._positions = {instr: pos}
    acc._exchange = "BINANCE.UM"

    assert acc.get_total_maint_margin() == 17.0
