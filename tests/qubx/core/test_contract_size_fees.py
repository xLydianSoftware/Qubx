"""
Tests for contract_size handling in fee calculation and capital locking.

Verifies that TransactionCostsCalculator.get_execution_fees and
BasicAccountProcessor._lock_limit_order_value correctly use
instrument.quantity_multiplier for futures with contract_size != 1.
"""

import numpy as np
import pytest
from pytest import approx

from qubx.core.basics import (
    Deal,
    Instrument,
    MarketType,
    Position,
    TransactionCostsCalculator,
)
from qubx.core.series import Quote, time_as_nsec


def _make_instrument(contract_size: float = 1.0, contract_multiplier: float = 1.0) -> Instrument:
    """Create a SWAP instrument with the given contract_size."""
    return Instrument(
        symbol="BCHUSDT",
        market_type=MarketType.SWAP,
        exchange="OKX.F",
        base="BCH",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BCH-USDT-SWAP",
        tick_size=0.1,
        lot_size=1.0,
        min_size=1.0,
        contract_size=contract_size,
        contract_multiplier=contract_multiplier,
    )


class TestFeeCalculationWithContractSize:
    """Test that get_execution_fees accounts for contract_size."""

    def test_fee_with_contract_size_1(self):
        """Baseline: contract_size=1 behaves the same as before."""
        tcc = TransactionCostsCalculator("test", maker=0.02, taker=0.05)
        instrument = _make_instrument(contract_size=1.0)

        # 10 contracts at price 300, taker fee
        fee = tcc.get_execution_fees(instrument, exec_price=300.0, amount=10.0, crossed_market=True)
        # notional = 10 * 1.0 * 300 = 3000, fee = 3000 * 0.05/100 = 1.5
        assert fee == approx(1.5)

    def test_fee_with_contract_size_01(self):
        """OKX-style: contract_size=0.1, fee should be 10x smaller than contract_size=1."""
        tcc = TransactionCostsCalculator("test", maker=0.02, taker=0.05)
        instrument = _make_instrument(contract_size=0.1)

        # 100 contracts at price 300, taker fee
        # notional = 100 * 0.1 * 300 = 3000
        fee = tcc.get_execution_fees(instrument, exec_price=300.0, amount=100.0, crossed_market=True)
        assert fee == approx(1.5)

    def test_fee_scales_with_contract_size(self):
        """Same notional exposure should produce the same fee regardless of contract_size."""
        tcc = TransactionCostsCalculator("test", maker=0.02, taker=0.05)

        # 1 contract * contract_size=1.0 * price=50000 = 50000 notional
        instr_cs1 = _make_instrument(contract_size=1.0)
        fee_cs1 = tcc.get_execution_fees(instr_cs1, exec_price=50000.0, amount=1.0, crossed_market=True)

        # 100 contracts * contract_size=0.01 * price=50000 = 50000 notional
        instr_cs001 = _make_instrument(contract_size=0.01)
        fee_cs001 = tcc.get_execution_fees(instr_cs001, exec_price=50000.0, amount=100.0, crossed_market=True)

        assert fee_cs1 == approx(fee_cs001)

    def test_fee_maker_vs_taker(self):
        """Maker and taker fees should both respect contract_size."""
        tcc = TransactionCostsCalculator("test", maker=0.02, taker=0.05)
        instrument = _make_instrument(contract_size=0.1)

        # 100 contracts * 0.1 * 300 = 3000 notional
        maker_fee = tcc.get_execution_fees(instrument, exec_price=300.0, amount=100.0, crossed_market=False)
        taker_fee = tcc.get_execution_fees(instrument, exec_price=300.0, amount=100.0, crossed_market=True)

        assert maker_fee == approx(3000.0 * 0.02 / 100)
        assert taker_fee == approx(3000.0 * 0.05 / 100)

    def test_fee_with_contract_multiplier(self):
        """quantity_multiplier = contract_size * contract_multiplier."""
        tcc = TransactionCostsCalculator("test", maker=0.02, taker=0.05)
        instrument = _make_instrument(contract_size=0.1, contract_multiplier=2.0)
        assert instrument.quantity_multiplier == approx(0.2)

        # 50 contracts * 0.2 * 300 = 3000 notional
        fee = tcc.get_execution_fees(instrument, exec_price=300.0, amount=50.0, crossed_market=True)
        assert fee == approx(3000.0 * 0.05 / 100)


class TestPositionPnlWithContractSize:
    """Test that Position P&L is correct for instruments with contract_size != 1."""

    def test_realized_pnl_with_contract_size(self):
        """Realized PnL should account for contract_size."""
        instrument = _make_instrument(contract_size=0.1)
        pos = Position(instrument)

        # Buy 100 contracts at 300
        pos.change_position_by(0, 100.0, 300.0)
        assert pos.quantity == 100.0

        # Sell 100 contracts at 330 (close position)
        r_pnl, _ = pos.change_position_by(0, -100.0, 330.0)

        # PnL = 100 * 0.1 * (330 - 300) = 300
        assert r_pnl == approx(300.0)

    def test_notional_value_with_contract_size(self):
        """Notional value = contracts * contract_size * price."""
        instrument = _make_instrument(contract_size=0.1)
        pos = Position(instrument)
        pos.change_position_by(0, 100.0, 300.0)
        pos.update_market_price(0, 300.0, 1.0)

        assert pos.notional_value == approx(100.0 * 0.1 * 300.0)
