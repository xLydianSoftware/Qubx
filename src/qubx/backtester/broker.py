from qubx import logger
from qubx.backtester.ome import SimulatedExecutionReport
from qubx.backtester.simulated_exchange import ISimulatedExchange
from qubx.core.basics import (
    CtrlChannel,
    Instrument,
    Order,
)
from qubx.core.exceptions import BadRequest, OrderNotFound
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

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order synchronously and return success status."""
        try:
            self._send_execution_report(order_update := self._exchange.cancel_order(order_id))
            return order_update is not None
        except OrderNotFound:
            # Order was already cancelled or doesn't exist
            logger.debug(f"Order {order_id} not found")
            return False

    def cancel_order_async(self, order_id: str) -> None:
        """Cancel an order asynchronously (fire-and-forget)."""
        # For simulation, async is same as sync since it's fast
        self.cancel_order(order_id)

    def cancel_orders(self, instrument: Instrument) -> None:
        raise NotImplementedError("Not implemented yet")

    def update_order(self, order_id: str, price: float, amount: float) -> Order:
        """Update an existing limit order using cancel+recreate strategy.

        Args:
            order_id: The ID of the order to update
            price: New price for the order
            amount: New amount for the order

        Returns:
            Order: The updated (newly created) order object

        Raises:
            OrderNotFound: If the order is not found
            BadRequest: If the order is not a limit order
        """
        # Get the existing order from account
        active_orders = self._account.get_orders()
        existing_order = active_orders.get(order_id)
        if not existing_order:
            raise OrderNotFound(f"Order {order_id} not found")

        # Validate that it's a limit order
        if existing_order.type.lower() != "limit":
            raise BadRequest(
                f"Order {order_id} is not a limit order (type: {existing_order.type}). Only limit orders can be updated."
            )

        # Cancel the existing order first
        self.cancel_order(order_id)

        # Create a new order with updated parameters, preserving original properties
        updated_order = self.send_order(
            instrument=existing_order.instrument,
            order_side=existing_order.side,
            order_type="limit",
            amount=abs(amount),
            price=price,
            client_id=existing_order.client_id,  # Preserve original client_id for tracking
            time_in_force=existing_order.time_in_force or "gtc",
        )

        return updated_order

    def _send_execution_report(self, report: SimulatedExecutionReport | None):
        if report is None:
            return

        self.channel.send((report.instrument, "order", report.order, False))
        if report.exec is not None:
            self.channel.send((report.instrument, "deals", [report.exec], False))

    def exchange(self) -> str:
        return self._exchange.exchange_id
