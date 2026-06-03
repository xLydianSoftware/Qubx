import uuid
from typing import Any

from qubx import logger
from qubx.core.account_manager import AccountManager
from qubx.core.basics import Balance, ITimeProvider
from qubx.core.interfaces import ITransferManager


class SimulationTransferManager(ITransferManager):
    """Instant, in-memory fund transfers between exchanges for simulation.

    Moves balances directly on the AccountManager's per-exchange state (debit source,
    credit destination) and records each transfer for status lookup / export.
    """

    def __init__(self, account_manager: AccountManager, time_provider: ITimeProvider):
        self._account = account_manager
        self._time = time_provider
        self._transfers: dict[str, dict[str, Any]] = {}

    def transfer_funds(self, from_exchange: str, to_exchange: str, currency: str, amount: float) -> str:
        from_balance = self._account.get_balance(currency, exchange=from_exchange)
        if from_balance is None:
            raise ValueError(f"Currency '{currency}' not found in {from_exchange}")
        if from_balance.free < amount:
            raise ValueError(
                f"Insufficient funds in {from_exchange}: "
                f"{from_balance.free:.8f} {currency} available, {amount:.8f} requested"
            )

        # Instant transfer: debit the source and credit the destination. update_balance is
        # identity-preserving, so holders of either Balance keep a live reference.
        from_balance.total -= amount
        from_balance.free -= amount
        self._account.get_state(from_exchange).update_balance(currency, from_balance)

        to_balance = self._account.get_balance(currency, exchange=to_exchange)
        if to_balance is None:
            to_balance = Balance(exchange=to_exchange, currency=currency)
        to_balance.total += amount
        to_balance.free += amount
        self._account.get_state(to_exchange).update_balance(currency, to_balance)

        transaction_id = f"sim_{uuid.uuid4().hex[:12]}"
        self._transfers[transaction_id] = {
            "transaction_id": transaction_id,
            "timestamp": self._time.time(),
            "from_exchange": from_exchange,
            "to_exchange": to_exchange,
            "currency": currency,
            "amount": amount,
            "status": "completed",  # transfers are instant in simulation
        }
        logger.debug(f"[SimTransfer] {amount:.8f} {currency} {from_exchange} -> {to_exchange} ({transaction_id})")
        return transaction_id

    def get_transfer_status(self, transaction_id: str) -> dict[str, Any]:
        record = self._transfers.get(transaction_id)
        if record is None:
            raise ValueError(f"Transfer not found: {transaction_id}")
        return dict(record)

    def get_transfers(self) -> dict[str, dict[str, Any]]:
        return {tid: dict(rec) for tid, rec in self._transfers.items()}
