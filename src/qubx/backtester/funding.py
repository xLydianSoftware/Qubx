"""Funding booking for simulated accounts (backtest + paper)."""

from qubx.core.basics import FundingPayment, Instrument
from qubx.core.events import FundingPaymentEvent
from qubx.core.interfaces import IAccountViewer, IFundingBooker


class SimFundingBooker(IFundingBooker):
    """Turns universe-scoped market funding tuples into account-scoped FundingPaymentEvents.

    Wired by the runner only when the account is simulated: the framework is the venue, so
    this is the one non-connector FundingPaymentEvent emitter. Emits only for instruments
    where our position is open — account-scoped by construction; booking is computed
    (amount=None) since a sim has no venue-reported cash truth.
    """

    def __init__(self, account: IAccountViewer) -> None:
        self._account = account

    def on_market_funding(self, instrument: Instrument, payment: FundingPayment) -> FundingPaymentEvent | None:
        pos = self._account.get_position(instrument)
        if pos is None or abs(pos.quantity) < instrument.min_size:
            return None
        return FundingPaymentEvent(instrument=instrument, payment=payment, amount=None)
