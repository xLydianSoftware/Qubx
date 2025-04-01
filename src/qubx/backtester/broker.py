from qubx.backtester.ome import SimulatedExecutionReport
from qubx.backtester.simulated_exchange import ISimulatedExchange
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
    _exchange: ISimulatedExchange

    def __init__(
        self,
        channel: CtrlChannel,
        account: SimulatedAccountProcessor,
        simulated_exchange: ISimulatedExchange,
    ) -> None:
        self.channel = channel
        self._account = account
        self._exchange = simulated_exchange

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
        # - place order at exchange and send exec report to data channel
        self._send_execution_report(
            report := self._exchange.place_order(
                instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
            )
        )
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
        self._send_execution_report(order_update := self._exchange.cancel_order(order_id))
        return order_update.order if order_update is not None else None

    def cancel_orders(self, instrument: Instrument) -> None:
        raise NotImplementedError("Not implemented yet")

    def update_order(self, order_id: str, price: float | None = None, amount: float | None = None) -> Order:
        raise NotImplementedError("Not implemented yet")

    def _send_execution_report(self, report: SimulatedExecutionReport | None):
        if report is None:
            return

        self.channel.send((report.instrument, "order", report.order, False))
        if report.exec is not None:
            self.channel.send((report.instrument, "deals", [report.exec], False))

    def exchange(self) -> str:
        return self._exchange.exchange_id
