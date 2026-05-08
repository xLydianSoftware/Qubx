"""Tests for IAccountViewer's instrument-leverage / margin-mode read surface.

Distinguishes the exchange-side per-(account, instrument) settings from
the observed/computed leverage already on get_leverage:

- get_leverage(instrument)              -> float            (observed = notional/equity)
- get_instrument_leverage(instrument)   -> float | None     (exchange setting)
- get_max_instrument_leverage(...)      -> float | None     (venue cap)
- get_max_instrument_notional(...)      -> float            (venue cap, inf if none)
- get_margin_mode(instrument)           -> "cross"|"isolated"|None

Note: the previous `_*_cache` injection tests (which exercised the
removed `getattr(self, "_*_cache", None)` lookup on the base) were
deleted alongside that implementation; the soft-default contract is
covered in test_account_settings_defaults.py and live overrides are
exercised by per-connector tests in the exchanges repo.
"""

import math

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


def test_iaccountviewer_declares_new_methods():
    from qubx.core.interfaces import IAccountViewer

    for name in (
        "get_instrument_leverage",
        "get_max_instrument_leverage",
        "get_max_instrument_notional",
        "get_margin_mode",
    ):
        assert hasattr(IAccountViewer, name), f"IAccountViewer must declare {name}"


def test_default_impls_return_none_or_inf():
    """BasicAccountProcessor without venue data returns None for unknowns,
    inf for max_instrument_notional (no per-asset cap by default)."""
    from qubx.core.account import BasicAccountProcessor

    acc = BasicAccountProcessor.__new__(BasicAccountProcessor)
    acc._exchange = "BINANCE.UM"
    instr = _make_instrument()

    assert acc.get_instrument_leverage(instr) is None
    assert acc.get_max_instrument_leverage(instr) is None
    assert acc.get_margin_mode(instr) is None
    assert math.isinf(acc.get_max_instrument_notional(instr))
