import uuid
from typing import Any

from qubx import logger
from qubx.core.account_manager import AccountManager
from qubx.core.basics import ITimeProvider
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
        if amount <= 0:
            raise ValueError(f"Transfer amount must be positive, got {amount}")
        from_balance = self._account.get_balance(currency, exchange=from_exchange)
        if from_balance.free < amount:
            raise ValueError(
                f"Insufficient funds in {from_exchange}: "
                f"{from_balance.free:.8f} {currency} available, {amount:.8f} requested"
            )

        # adjust_balance mutates the held Balance in place (holders keep live references) and creates missing ones
        self._account.adjust_balance(from_exchange, currency, -amount)
        # both-stables transfers credit the destination's base currency — mirrors the live service's route asset
        to_base = self._account.get_base_currency(to_exchange)
        credit_currency = to_base if {currency, to_base} <= {"USDT", "USDC"} else currency
        self._account.adjust_balance(to_exchange, credit_currency, amount)

        transaction_id = f"sim_{uuid.uuid4().hex[:12]}"
        self._transfers[transaction_id] = {
            "transaction_id": transaction_id,
            "timestamp": self._time.time(),
            "from_exchange": from_exchange,
            "to_exchange": to_exchange,
            "currency": currency,
            "credited_currency": credit_currency,
            "amount": amount,
            "status": "completed",
        }
        logger.debug(
            f"[SimTransfer] {amount:.8f} {currency} {from_exchange} -> {to_exchange} "
            f"(credited as {credit_currency}, {transaction_id})"
        )
        return transaction_id

    def get_transfer_status(self, transaction_id: str) -> dict[str, Any]:
        record = self._transfers.get(transaction_id)
        if record is None:
            raise ValueError(f"Transfer not found: {transaction_id}")
        return dict(record)

    def get_transfers(self) -> dict[str, dict[str, Any]]:
        return {tid: dict(rec) for tid, rec in self._transfers.items()}
