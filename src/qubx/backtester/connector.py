import asyncio

from qubx.backtester.ome import SimulatedExecutionReport
from qubx.backtester.simulated_exchange import ISimulatedExchange
from qubx.core.basics import CtrlChannel, Instrument, ITimeProvider, OrderRequest
from qubx.core.events import (
    AccountSnapshot,
    AccountSnapshotEvent,
    ChannelMessage,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.exceptions import OrderNotFound
from qubx.core.series import OrderBook, Quote, Trade, TradeArray


class SimulatedConnector:
    """IConnector implementation wrapping the simulator's OME via ISimulatedExchange.

    submit/cancel/update route through the exchange and translate the resulting
    SimulatedExecutionReport into typed AccountMessage events on the channel.
    Channel dispatch is synchronous in backtest, so a submit that crosses the
    book emits both Accepted and Filled within the same submit_order call.
    """

    channel: CtrlChannel
    exchange_name: str

    def __init__(
        self,
        *,
        channel: CtrlChannel,
        exchange: ISimulatedExchange,
        time_provider: ITimeProvider,
        # Accepted for IConnector construction parity with live connectors; simulation has no event loop.
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.channel = channel
        self.exchange_name = exchange.exchange_id
        self._exchange = exchange
        self._time = time_provider

    def send(self, event: ChannelMessage) -> None:
        self.channel.send(event)

    def submit_order(self, request: OrderRequest) -> None:
        report = self._exchange.place_order(
            instrument=request.instrument,
            order_side=request.side,
            order_type=request.order_type,
            amount=request.quantity,
            price=request.price,
            client_id=request.client_id,
            time_in_force=request.time_in_force,
        )
        self._emit_from_report(report)

    def cancel_order(self, *, client_order_id: str | None = None, venue_order_id: str | None = None) -> None:
        oid = venue_order_id or client_order_id
        if oid is None:
            raise ValueError("cancel_order: missing identifier")
        try:
            report = self._exchange.cancel_order(oid)
        except OrderNotFound:
            report = None
        if report is not None:
            self._emit_from_report(report)

    def update_order(
        self,
        *,
        client_order_id: str | None = None,
        venue_order_id: str | None = None,
        price: float | None = None,
        quantity: float | None = None,
    ) -> None:
        oid = venue_order_id or client_order_id
        try:
            old = self._exchange.cancel_order(oid)  # type: ignore[arg-type]
        except OrderNotFound:
            old = None
        if old is None:
            self.send(
                OrderUpdateRejectedEvent(
                    instrument=None,
                    client_order_id=client_order_id or oid,  # type: ignore[arg-type]
                    reason="update_order: order not found",
                )
            )
            return
        old_order = old.order
        # cancel+recreate keeps the original client_order_id, so the strategy sees a
        # single OrderUpdatedEvent rather than a Canceled+Accepted pair.
        new = self._exchange.place_order(
            instrument=old_order.instrument,
            order_side=old_order.side,
            order_type=old_order.type,
            amount=quantity if quantity is not None else old_order.quantity,
            price=price if price is not None else old_order.price,
            client_id=old_order.client_id,
            time_in_force=old_order.time_in_force,
        )
        self.send(
            OrderUpdatedEvent(
                instrument=new.order.instrument,
                client_order_id=new.order.client_id,
                venue_order_id=new.order.id,
                new_price=price,
                new_quantity=quantity,
            )
        )

    def request_order_status(self, *, client_order_id: str | None = None, venue_order_id: str | None = None) -> None:
        oid = venue_order_id or client_order_id
        for order in self._exchange.get_open_orders().values():
            if order.id == oid or order.client_id == oid:
                self.send(
                    OrderAcceptedEvent(
                        instrument=order.instrument,
                        client_order_id=order.client_id,
                        venue_order_id=order.id,
                        accepted_at=self._time.time(),
                    )
                )
                return
        self.send(
            OrderRejectedEvent(
                instrument=None,
                client_order_id=client_order_id or oid,  # type: ignore[arg-type]
                reason="reconcile: order not present at venue",
            )
        )

    def request_snapshot(self) -> None:
        open_orders = list(self._exchange.get_open_orders().values())
        snapshot = AccountSnapshot(
            exchange=self.exchange_name,
            as_of=self._time.time(),
            open_orders=open_orders,
        )
        self.send(AccountSnapshotEvent(instrument=None, snapshot=snapshot))

    def process_market_data(self, instrument: Instrument, data: Quote | OrderBook | Trade | TradeArray) -> None:
        for report in self._exchange.process_market_data(instrument, data):
            self._emit_from_report(report)

    def _emit_from_report(self, report: SimulatedExecutionReport) -> None:
        order = report.order
        status = order.status
        # Raw UPPERCASE status strings from the OME (not the lowercase OrderStatus enum used elsewhere
        # in core). The OME emits "OPEN", "CLOSED", and "CANCELED" in practice; "NEW" and "FILLED"
        # are accepted defensively for forward-compatibility with other exchange adapters.
        if status in ("OPEN", "NEW"):
            self.send(
                OrderAcceptedEvent(
                    instrument=report.instrument,
                    client_order_id=order.client_id,
                    venue_order_id=order.id,
                    accepted_at=report.timestamp,
                )
            )
        elif status == "CANCELED":
            self.send(
                OrderCanceledEvent(
                    instrument=report.instrument,
                    client_order_id=order.client_id,
                    venue_order_id=order.id,
                )
            )
        if report.exec is not None:
            # OME marks a fully-filled order CLOSED; a still-OPEN order with an exec
            # is a partial fill (resting remainder stays in the book).
            if status in ("CLOSED", "FILLED"):
                self.send(
                    OrderFilledEvent(
                        instrument=report.instrument,
                        client_order_id=order.client_id,
                        venue_order_id=order.id,
                        fill=report.exec,
                    )
                )
            else:
                # The current OME fills atomically (always CLOSED on fill), so this branch is
                # presently unreachable. It is forward-compatibility scaffolding for exchanges
                # that emit partial fills with a resting remainder.
                self.send(
                    OrderPartiallyFilledEvent(
                        instrument=report.instrument,
                        client_order_id=order.client_id,
                        venue_order_id=order.id,
                        fill=report.exec,
                    )
                )

    def is_ws_ready(self) -> bool:
        return True

    def force_ws_reconnect_sync(self) -> bool:
        return True

    def connect(self) -> None:
        self.request_snapshot()

    def disconnect(self) -> None:
        pass

    def make_client_id(self, suggested: str) -> str:
        return suggested

    @property
    def is_simulated_trading(self) -> bool:
        return True

    @property
    def read_only(self) -> bool:
        return False

    def set_instrument_leverage(self, instrument: Instrument, leverage: float) -> bool:
        return True

    def set_margin_mode(self, instrument: Instrument, mode: str) -> bool:
        return True
