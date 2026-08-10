import uuid

from qubx import logger
from qubx.core.account_manager import AccountManager
from qubx.core.basics import ITimeProvider, Transfer, TransferStatus
from qubx.core.interfaces import ITransferManager


class SimulationTransferManager(ITransferManager):
    """Instant, in-memory fund transfers between exchanges for simulation.

    Moves balances directly on the AccountManager's per-exchange state (debit source,
    credit destination) and records each transfer for status lookup / export.
    """

    def __init__(self, account_manager: AccountManager, time_provider: ITimeProvider):
        self._account = account_manager
        self._time = time_provider
        self._transfers: dict[str, Transfer] = {}

    def transfer_funds(self, from_exchange: str, to_exchange: str, currency: str, amount: float) -> str:
        if amount <= 0:
            raise ValueError(f"Transfer amount must be positive, got {amount}")
        to_currency = self._account.get_base_currency(to_exchange)
        # Venues settle in their own currency: a withdrawal leaves in `currency` and arrives as
        # `to_currency`. Both legs must be priced — an unpriced asset has no transferable value.
        from_rate = self._account.get_state(from_exchange).conversion_rate_to_base(currency)
        to_rate = self._account.get_state(to_exchange).conversion_rate_to_base(to_currency)
        if from_rate is None or to_rate is None:
            raise ValueError(f"Cannot transfer {currency} to {to_exchange} {to_currency}: no conversion rate")
        to_amount = amount * from_rate / to_rate

        from_balance = self._account.get_balance(currency, exchange=from_exchange)
        if from_balance.free < amount:
            raise ValueError(
                f"Insufficient funds in {from_exchange}: "
                f"{from_balance.free:.8f} {currency} available, {amount:.8f} requested"
            )

        self._account.adjust_balance(from_exchange, currency, -amount)
        self._account.adjust_balance(to_exchange, to_currency, to_amount)

        transaction_id = f"sim_{uuid.uuid4().hex[:12]}"
        self._transfers[transaction_id] = Transfer(
            transaction_id=transaction_id,
            from_exchange=from_exchange,
            to_exchange=to_exchange,
            currency=currency,
            amount=amount,
            status=TransferStatus.COMPLETED,  # transfers are instant in simulation
            timestamp=self._time.time(),
            to_currency=to_currency,
            to_amount=to_amount,
        )
        logger.debug(
            f"[SimTransfer] {amount:.8f} {currency} {from_exchange} -> "
            f"{to_amount:.8f} {to_currency} {to_exchange} ({transaction_id})"
        )
        return transaction_id

    def get_transfer_status(self, transaction_id: str) -> Transfer:
        transfer = self._transfers.get(transaction_id)
        if transfer is None:
            raise ValueError(f"Transfer not found: {transaction_id}")
        return transfer

    def get_transfers(self) -> dict[str, Transfer]:
        return dict(self._transfers)
