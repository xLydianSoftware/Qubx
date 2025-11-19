from qubx import logger
from qubx.backtester.ome import SimulatedExecutionReport
from qubx.backtester.simulated_exchange import ISimulatedExchange
from qubx.core.basics import (
    CtrlChannel,
    Instrument,
    Order,
    OrderRequest,
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

    def send_order(self, request: OrderRequest) -> Order:
        """Submit order synchronously in simulation."""
        instrument = request.instrument
        order_side = request.side
        order_type = request.order_type
        amount = request.quantity
        price = request.price
        client_id = request.client_id
        time_in_force = request.time_in_force
        options = request.options

        self._send_execution_report(
            report := self._exchange.place_order(
                instrument, order_side, order_type, amount, price, client_id, time_in_force, **options
            )
        )
        return report.order

    def send_order_async(self, request: OrderRequest) -> None:
        """Submit order asynchronously (same as sync in simulation)."""
        self.send_order(request)

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
        if existing_order.type != "LIMIT":
            raise BadRequest(
                f"Order {order_id} is not a limit order (type: {existing_order.type}). Only limit orders can be updated."
            )

        self.cancel_order(order_id)

        request = OrderRequest(
            instrument=existing_order.instrument,
            quantity=abs(amount),
            price=price,
            order_type="LIMIT",
            side=existing_order.side,
            time_in_force=existing_order.time_in_force or "gtc",
            options={},
        )

        updated_order = self.send_order(request)

        return updated_order

    def _send_execution_report(self, report: SimulatedExecutionReport | None):
        if report is None:
            return

        self.channel.send((report.instrument, "order", report.order, False))
        if report.exec is not None:
            self.channel.send((report.instrument, "deals", [report.exec], False))

    def exchange(self) -> str:
        return self._exchange.exchange_id
