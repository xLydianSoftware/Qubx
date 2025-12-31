import uuid
from typing import Any

import pandas as pd

from qubx import logger
from qubx.core.account import CompositeAccountProcessor
from qubx.core.basics import ITimeProvider
from qubx.core.interfaces import ITransferManager


class SimulationTransferManager(ITransferManager):
    """
    Transfer manager for simulation mode.

    Handles fund transfers between exchanges by directly manipulating account balances.
    All transfers are instant and tracked in a DataFrame for export to results.
    """

    _account: CompositeAccountProcessor
    _time: ITimeProvider
    _transfers: list[dict[str, Any]]

    def __init__(self, account_processor: CompositeAccountProcessor, time_provider: ITimeProvider):
        """
        Initialize simulation transfer manager.

        Args:
            account_processor: Account processor (typically CompositeAccountProcessor)
            time_provider: Time provider for timestamping transfers
        """
        self._account = account_processor
        self._time = time_provider
        self._transfers = []

    def transfer_funds(self, from_exchange: str, to_exchange: str, currency: str, amount: float) -> str:
        """
        Transfer funds between exchanges (instant in simulation).

        Args:
            from_exchange: Source exchange identifier
            to_exchange: Destination exchange identifier
            currency: Currency to transfer
            amount: Amount to transfer

        Returns:
            str: Transaction ID (UUID)

        Raises:
            ValueError: If exchanges not found or insufficient funds
        """
        # Generate transaction ID
        transaction_id = f"sim_{uuid.uuid4().hex[:12]}"

        # Get timestamp
        timestamp = self._time.time()

        # Get individual processors
        try:
            from_processor = self._account.get_account_processor(from_exchange)
            to_processor = self._account.get_account_processor(to_exchange)
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Exchange not found: {e}")

        # Validate sufficient funds
        from_balances_list = from_processor.get_balances()
        from_balance = next((b for b in from_balances_list if b.currency == currency), None)

        if from_balance is None:
            raise ValueError(f"Currency '{currency}' not found in {from_exchange}")

        available = from_balance.free
        if available < amount:
            raise ValueError(
                f"Insufficient funds in {from_exchange}: "
                f"{available:.8f} {currency} available, {amount:.8f} {currency} requested"
            )

        # Execute transfer (instant balance manipulation)
        from_balance.total -= amount
        from_balance.free -= amount

        to_balances_list = to_processor.get_balances()
        to_balance = next((b for b in to_balances_list if b.currency == currency), None)

        total_amount = to_balance.total + amount if to_balance is not None else amount
        locked_amount = to_balance.locked if to_balance is not None else 0
        to_processor.update_balance(currency, total_amount, locked_amount, to_exchange)

        # Record transfer
        transfer_record = {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "from_exchange": from_exchange,
            "to_exchange": to_exchange,
            "currency": currency,
            "amount": amount,
            "status": "completed",  # Always completed in simulation
        }
        self._transfers.append(transfer_record)

        logger.debug(f"[SimTransfer] {amount:.8f} {currency} {from_exchange} → {to_exchange} (ID: {transaction_id})")

        return transaction_id

    def get_transfer_status(self, transaction_id: str) -> dict[str, Any]:
        """
        Get the status of a transfer.

        Args:
            transaction_id: Transaction ID

        Returns:
            dict[str, Any]: Transfer status information
        """
        # Find transfer
        for transfer in self._transfers:
            if transfer["transaction_id"] == transaction_id:
                return transfer.copy()

        # Not found
        return {
            "transaction_id": transaction_id,
            "status": "not_found",
            "error": f"Transaction {transaction_id} not found",
        }

    def get_transfers(self) -> dict[str, dict[str, Any]]:
        """
        Get all transfers as a dictionary.

        Returns:
            dict[str, dict[str, Any]]: Dictionary mapping transaction IDs to transfer info
        """
        return {t["transaction_id"]: t for t in self._transfers}

    def get_transfers_dataframe(self) -> pd.DataFrame:
        """
        Get all transfers as a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns [transaction_id, timestamp, from_exchange,
                         to_exchange, currency, amount, status]
        """
        if not self._transfers:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(
                columns=["transaction_id", "from_exchange", "to_exchange", "currency", "amount", "status"]  # type: ignore
            )

        return pd.DataFrame(self._transfers).set_index("timestamp")
