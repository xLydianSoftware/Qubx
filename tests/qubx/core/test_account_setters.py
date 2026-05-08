"""Tests for IAccountProcessor.set_instrument_leverage / set_margin_mode.

Both are signed actions on the venue but conceptually account-config changes,
not trades — they live on the account processor, not the broker.

Defaults raise NotImplementedError; live exchange processors override.
"""

import pytest

from qubx.core.basics import Instrument, MarketType


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


def test_iaccountprocessor_declares_setter_methods():
    from qubx.core.interfaces import IAccountProcessor

    for name in ("set_instrument_leverage", "set_margin_mode"):
        assert hasattr(IAccountProcessor, name), f"IAccountProcessor must declare {name}"


def test_default_set_instrument_leverage_raises_not_implemented():
    from qubx.core.account import BasicAccountProcessor

    acc = BasicAccountProcessor.__new__(BasicAccountProcessor)
    acc._exchange = "BINANCE.UM"

    with pytest.raises(NotImplementedError):
        acc.set_instrument_leverage(_make_instrument(), 5.0)


def test_default_set_margin_mode_raises_not_implemented():
    from qubx.core.account import BasicAccountProcessor

    acc = BasicAccountProcessor.__new__(BasicAccountProcessor)
    acc._exchange = "BINANCE.UM"

    with pytest.raises(NotImplementedError):
        acc.set_margin_mode(_make_instrument(), "isolated")


def test_set_instrument_leverage_accepts_float():
    """Subclasses overriding the method must accept float (HPL allows 1.5, etc.)."""
    from qubx.core.account import BasicAccountProcessor

    captured = {}

    class _Subclass(BasicAccountProcessor):
        def __init__(self):
            self._exchange = "TEST"

        def set_instrument_leverage(self, instrument, leverage):
            captured["leverage"] = leverage
            return True

    sub = _Subclass()
    assert sub.set_instrument_leverage(_make_instrument(), 2.5) is True
    assert captured["leverage"] == 2.5


def test_set_margin_mode_accepts_cross_or_isolated():
    from qubx.core.account import BasicAccountProcessor

    captured = {}

    class _Subclass(BasicAccountProcessor):
        def __init__(self):
            self._exchange = "TEST"

        def set_margin_mode(self, instrument, mode):
            captured["mode"] = mode
            return True

    sub = _Subclass()
    assert sub.set_margin_mode(_make_instrument(), "cross") is True
    assert captured["mode"] == "cross"
    sub.set_margin_mode(_make_instrument(), "isolated")
    assert captured["mode"] == "isolated"
