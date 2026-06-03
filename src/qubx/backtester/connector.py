from qubx.backtester.ome import SimulatedExecutionReport
from qubx.backtester.simulated_exchange import ISimulatedExchange
from qubx.core.basics import CtrlChannel, Instrument, ITimeProvider, OrderRequest, OrderStatus, Timestamped
from qubx.core.connector import ChannelEmitter
from qubx.core.events import (
    AccountSnapshot,
    AccountSnapshotEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.exceptions import ExchangeError, InvalidOrder, InvalidOrderParameters, OrderNotFound
from qubx.core.series import OrderBook, Quote, Trade, TradeArray


class SimulatedConnector(ChannelEmitter):
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
        # Absorb extra construction kwargs (e.g. live connectors' loop=...) for parity;
        # simulation has no event loop, so they're ignored.
        **_kwargs,
    ):
        self.channel = channel
        self.exchange_name = exchange.exchange_id
        self._exchange = exchange
        self._time = time_provider

    def submit_order(self, request: OrderRequest) -> None:
        try:
            report = self._exchange.place_order(
                instrument=request.instrument,
                order_side=request.side,
                order_type=request.order_type,
                amount=request.quantity,
                price=request.price,
                client_id=request.client_id,
                time_in_force=request.time_in_force,
                **(request.options or {}),
            )
        except (InvalidOrder, InvalidOrderParameters):
            # Framework-side rejection (bad side/type/amount/price/TIF) — surface
            # synchronously so the caller fixes it; never rides the channel.
            raise
        except ExchangeError as e:
            # Venue verdict (e.g. a stop that would trigger immediately) — rides the
            # channel as OrderRejectedEvent rather than raising. This is the connector
            # rejection boundary: the same logical failure must take the same path
            # everywhere, so venue refusals are always async events, not exceptions.
            self.send(
                OrderRejectedEvent(
                    instrument=request.instrument, client_order_id=request.client_id, reason=str(e)
                )
            )
            return
        self._emit_from_report(report)

    def cancel_order(self, client_order_id: str | None = None, venue_order_id: str | None = None) -> None:
        # Prefer the venue id (the OME keys orders by it); fall back to the cid before the ack.
        oid = venue_order_id or client_order_id
        if not oid:
            raise ValueError("cancel_order: client_order_id or venue_order_id is required")
        try:
            report = self._exchange.cancel_order(oid)
        except OrderNotFound:
            return  # already gone at the venue (filled/canceled) — nothing to emit
        self._emit_from_report(report)

    def update_order(
        self,
        client_order_id: str | None = None,
        venue_order_id: str | None = None,
        price: float | None = None,
        quantity: float | None = None,
    ) -> None:
        oid = venue_order_id or client_order_id
        if not oid:
            raise ValueError("update_order: client_order_id or venue_order_id is required")
        try:
            old = self._exchange.cancel_order(oid)
        except OrderNotFound:
            self.send(
                OrderUpdateRejectedEvent(
                    instrument=None,
                    client_order_id=client_order_id or oid,  # cid if known, else the venue id
                    venue_order_id=venue_order_id,
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
            **(old_order.options or {}),
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

    def request_order_status(self, client_order_id: str | None = None, venue_order_id: str | None = None) -> None:
        oid = venue_order_id or client_order_id
        if not oid:
            raise ValueError("request_order_status: client_order_id or venue_order_id is required")
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
                client_order_id=client_order_id or oid,  # cid if known, else the venue id
                venue_order_id=venue_order_id,
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

    def process_market_data(self, instrument: Instrument, data: Timestamped) -> None:
        # Single entry point that drives the OME (runner._feed_ome in backtest,
        # _feed_simulated_connector in paper) with whatever a subscription produces. The OME
        # matches resting orders against Quote/OrderBook/Trade/TradeArray, so those pass
        # through unchanged — full book depth / trade-array fidelity. Anything else (an OHLC
        # Bar, a bare price) isn't matchable, so emulate a tradeable quote from it. Note we
        # can't just always emulate: emulate_quote_from_data has no TradeArray case and
        # collapses an OrderBook to a single quote.
        if isinstance(data, (Quote, OrderBook, Trade, TradeArray)):
            feed: Quote | OrderBook | Trade | TradeArray | None = data
        else:
            feed = self._exchange.emulate_quote_from_data(instrument, self._time.time(), data)
        if feed is None:
            return
        for report in self._exchange.process_market_data(instrument, feed):
            self._emit_from_report(report)

    def _emit_from_report(self, report: SimulatedExecutionReport) -> None:
        order = report.order
        status = order.status
        # The OME emits ACCEPTED (resting / just-placed) and CANCELED on the order; a fill
        # carries the deal in report.exec with status FILLED.
        if status in (OrderStatus.ACCEPTED, OrderStatus.SUBMITTED):
            self.send(
                OrderAcceptedEvent(
                    instrument=report.instrument,
                    client_order_id=order.client_id,
                    venue_order_id=order.id,
                    accepted_at=report.timestamp,
                )
            )
        elif status == OrderStatus.CANCELED:
            self.send(
                OrderCanceledEvent(
                    instrument=report.instrument,
                    client_order_id=order.client_id,
                    venue_order_id=order.id,
                )
            )
        if report.exec is not None:
            # A FILLED order is fully filled; a still-live order with an exec is a partial
            # fill (resting remainder stays in the book).
            if status == OrderStatus.FILLED:
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

    def reconnect(self) -> bool:
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
