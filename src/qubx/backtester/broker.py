from qubx.backtester.ome import OmeReport
from qubx.core.basics import (
    CtrlChannel,
    Instrument,
    Order,
)
from qubx.core.interfaces import IBroker

from .account import SimulatedAccountProcessor


class SimulatedBroker(IBroker):
    channel: CtrlChannel

    _account: SimulatedAccountProcessor

    def __init__(
        self,
        channel: CtrlChannel,
        account: SimulatedAccountProcessor,
        exchange_id: str = "simulated",
    ) -> None:
        self.channel = channel
        self._account = account
        self._exchange_id = exchange_id

    @property
    def is_simulated_trading(self) -> bool:
        return True

    def send_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **options,
    ) -> Order:
        ome = self._account.ome.get(instrument)
        if ome is None:
            raise ValueError(f"ExchangeService:send_order :: No OME configured for '{instrument.symbol}'!")

        # - try to place order in OME
        report = ome.place_order(
            order_side.upper(),  # type: ignore
            order_type.upper(),  # type: ignore
            amount,
            price,
            client_id,
            time_in_force,
            **options,
        )

        self._send_exec_report(instrument, report)
        return report.order

    def send_order_async(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **optional,
    ) -> None:
        self.send_order(instrument, order_side, order_type, amount, price, client_id, time_in_force, **optional)

    def cancel_order(self, order_id: str) -> Order | None:
        instrument = self._account.order_to_instrument.get(order_id)
        if instrument is None:
            raise ValueError(f"ExchangeService:cancel_order :: can't find order with id = '{order_id}'!")

        ome = self._account.ome.get(instrument)
        if ome is None:
            raise ValueError(f"ExchangeService:send_order :: No OME configured for '{instrument}'!")

        # - cancel order in OME and remove from the map to free memory
        order_update = ome.cancel_order(order_id)
        if order_update is None:
            return None

        self._send_exec_report(instrument, order_update)

        return order_update.order

    def cancel_orders(self, instrument: Instrument) -> None:
        raise NotImplementedError("Not implemented yet")

    def update_order(self, order_id: str, price: float | None = None, amount: float | None = None) -> Order:
        raise NotImplementedError("Not implemented yet")

    def _send_exec_report(self, instrument: Instrument, report: OmeReport):
        self.channel.send((instrument, "order", report.order, False))
        if report.exec is not None:
            self.channel.send((instrument, "deals", [report.exec], False))

    def exchange(self) -> str:
        return self._exchange_id.upper()
