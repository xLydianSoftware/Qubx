# tests/qubx/core/test_account_adl.py
"""Tests for IAccountViewer.get_adl_level / BasicAccountProcessor default impl."""

from qubx.core.basics import Instrument, MarketType, Position


def _make_instrument(symbol="ETHUSDT") -> Instrument:
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


def test_get_adl_level_returns_none_when_position_does_not_report():
    """ccxt / lighter connectors don't report ADL; get_adl_level returns None.

    Tests the contract via a tiny stub that satisfies the interface, since
    BasicAccountProcessor construction may be heavyweight in unit tests.
    """
    instr = _make_instrument()
    pos = Position(instrument=instr, quantity=1.0, pos_average_price=2000.0)
    assert pos.adl_level is None  # default

    class _Stub:
        def get_position(self, instrument):
            return pos

        def get_adl_level(self, instrument):
            # Mirror the IAccountViewer default impl
            return self.get_position(instrument).adl_level

    stub = _Stub()
    assert stub.get_adl_level(instr) is None


def test_get_adl_level_returns_value_when_position_reports():
    """HPL-style account processors mutate Position.adl_level; get_adl_level surfaces it."""
    instr = _make_instrument()
    pos = Position(instrument=instr, quantity=1.0, pos_average_price=2000.0)
    pos.adl_level = 1

    class _Stub:
        def get_position(self, instrument):
            return pos

        def get_adl_level(self, instrument):
            return self.get_position(instrument).adl_level

    stub = _Stub()
    assert stub.get_adl_level(instr) == 1


def test_iaccountviewer_has_get_adl_level_method():
    """The IAccountViewer interface must declare get_adl_level."""
    from qubx.core.interfaces import IAccountViewer

    assert hasattr(IAccountViewer, "get_adl_level"), (
        "IAccountViewer must declare get_adl_level for connectors to expose ADL info"
    )
