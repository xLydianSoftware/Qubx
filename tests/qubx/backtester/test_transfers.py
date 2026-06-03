from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.backtester.transfers import SimulationTransferManager
from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.basics import Balance, ITimeProvider


class _T(ITimeProvider):
    def time(self) -> np.datetime64:
        return np.datetime64("2026-05-28T00:00:00", "ns")


def _am():
    am = SimulatedAccountManager(
        connectors={"E1": MagicMock(), "E2": MagicMock()},
        strategy=MagicMock(),
        time=_T(),
    )
    am.get_state("E1").update_balance("USDT", Balance(exchange="E1", currency="USDT", total=1000.0, free=1000.0))
    return am


def test_transfer_moves_balance_between_exchanges():
    am = _am()
    tm = SimulationTransferManager(am, _T())

    txid = tm.transfer_funds("E1", "E2", "USDT", 300.0)

    assert am.get_balance("USDT", exchange="E1").total == 700.0
    assert am.get_balance("USDT", exchange="E1").free == 700.0
    assert am.get_balance("USDT", exchange="E2").total == 300.0
    assert am.get_balance("USDT", exchange="E2").free == 300.0
    status = tm.get_transfer_status(txid)
    assert status["amount"] == 300.0
    assert status["from_exchange"] == "E1"
    assert status["to_exchange"] == "E2"
    assert status["status"] == "completed"
    assert txid in tm.get_transfers()


def test_transfer_preserves_destination_balance_identity():
    am = _am()
    am.get_state("E2").update_balance("USDT", Balance(exchange="E2", currency="USDT", total=50.0, free=50.0))
    dest_ref = am.get_balance("USDT", exchange="E2")
    tm = SimulationTransferManager(am, _T())

    tm.transfer_funds("E1", "E2", "USDT", 100.0)

    assert am.get_balance("USDT", exchange="E2") is dest_ref  # same object, updated in place
    assert dest_ref.total == 150.0


def test_insufficient_funds_raises_and_leaves_balances_untouched():
    am = _am()
    tm = SimulationTransferManager(am, _T())
    with pytest.raises(ValueError, match="Insufficient funds"):
        tm.transfer_funds("E1", "E2", "USDT", 5000.0)
    assert am.get_balance("USDT", exchange="E1").total == 1000.0


def test_unknown_currency_raises():
    am = _am()
    tm = SimulationTransferManager(am, _T())
    with pytest.raises(ValueError, match="not found"):
        tm.transfer_funds("E1", "E2", "DOGE", 1.0)


def test_unknown_transfer_status_raises():
    am = _am()
    tm = SimulationTransferManager(am, _T())
    with pytest.raises(ValueError, match="Transfer not found"):
        tm.get_transfer_status("nope")
