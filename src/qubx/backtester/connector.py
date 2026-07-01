from qubx.backtester.ome import SimulatedExecutionReport
from qubx.backtester.simulated_exchange import ISimulatedExchange, emulate_quote_from_data, get_simulated_exchange
from qubx.core.basics import (
    ZERO_COSTS,
    CtrlChannel,
    Instrument,
    ITimeProvider,
    Order,
    OrderRequest,
    OrderStatus,
    Timestamped,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.connector import ChannelEmitter
from qubx.core.events import (
    AccountSnapshot,
    AccountSnapshotEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
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
    _ome: ISimulatedExchange

    def __init__(
        self,
        *,
        channel: CtrlChannel,
        exchange_name: str,
        time_provider: ITimeProvider,
        tcc: TransactionCostsCalculator = ZERO_COSTS,
        accurate_stop_orders_execution: bool = False,
        # Absorb extra construction kwargs (e.g. live connectors' loop=...) for parity;
        # simulation has no event loop, so they're ignored.
        **_kwargs,
    ):
        self.channel = channel
        # The OME-backed exchange is private to the connector: nothing else holds it, so all
        # order matching and market-data feeding routes through this connector.
        self._ome = get_simulated_exchange(exchange_name, time_provider, tcc, accurate_stop_orders_execution)
        self.exchange_name = self._ome.exchange_id
        self._time = time_provider

    def submit_order(self, request: OrderRequest) -> None:
        try:
            report = self._ome.place_order(
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
                OrderRejectedEvent(instrument=request.instrument, client_order_id=request.client_id, reason=str(e))
            )
            return
        self._emit_from_report(report)

    def cancel_order(self, order: Order) -> None:
        # Prefer the venue id (the OME keys orders by it); fall back to the cid before the ack.
        oid = order.venue_order_id or order.client_order_id
        try:
            report = self._ome.cancel_order(oid)
        except OrderNotFound:
            self.send(
                OrderCancelRejectedEvent(
                    instrument=order.instrument,
                    client_order_id=order.client_order_id,
                    venue_order_id=order.venue_order_id,
                    reason=f"cancel_order: order not found for {oid}",
                )
            )
            return
        self._emit_from_report(report)

    def update_order(self, order: Order, *, price: float | None = None, quantity: float | None = None) -> None:
        oid = order.venue_order_id or order.client_order_id
        try:
            old = self._ome.cancel_order(oid)
        except OrderNotFound:
            self.send(
                OrderUpdateRejectedEvent(
                    instrument=order.instrument,
                    client_order_id=order.client_order_id,
                    venue_order_id=order.venue_order_id,
                    reason="update_order: order not found",
                )
            )
            return
        old_order = old.order
        # cancel+recreate keeps the original client_order_id, so the strategy sees a
        # single OrderUpdatedEvent rather than a Canceled+Accepted pair.
        try:
            new = self._ome.place_order(
                instrument=old_order.instrument,
                order_side=old_order.side,
                order_type=old_order.type,
                amount=quantity if quantity is not None else old_order.quantity,
                price=price if price is not None else old_order.price,
                client_id=old_order.client_order_id,
                time_in_force=old_order.time_in_force,
                **(old_order.options or {}),
            )
        except ExchangeError as e:
            # The re-place was rejected AFTER the original was canceled in the OME, so the
            # synchronous-raise contract no longer applies (even for InvalidOrder, a subclass):
            # TM's rollback would revert the order to ACCEPTED while the venue holds nothing,
            # stranding it forever (no sweeps in sim). Converge the AM with OME truth instead —
            # update rejected, then the original canceled (the live cancel+recreate failure
            # contract: the strategy learns the update failed AND the order is gone).
            self.send(
                OrderUpdateRejectedEvent(
                    instrument=old_order.instrument,
                    client_order_id=old_order.client_order_id,
                    venue_order_id=old_order.venue_order_id,
                    reason=str(e),
                )
            )
            self.send(
                OrderCanceledEvent(
                    instrument=old_order.instrument,
                    client_order_id=old_order.client_order_id,
                    venue_order_id=old_order.venue_order_id,
                )
            )
            return
        self.send(
            OrderUpdatedEvent(
                instrument=new.order.instrument,
                client_order_id=new.order.client_order_id,
                venue_order_id=new.order.venue_order_id,
                new_price=price,
                new_quantity=quantity,
            )
        )
        # If the re-placed (modified) order immediately crossed the book, the OME's report
        # carries a Deal — surface it so AM applies the fill (it otherwise only re-ACCEPTs on
        # the Updated, diverging order/position state). We emit ONLY the fill, not a full
        # _emit_from_report(new): the OrderUpdatedEvent above already covers the resting
        # (non-crossed) case, so feeding the report through would emit a redundant
        # OrderAcceptedEvent on the already-ACCEPTED order.
        self._emit_exec(new)

    def request_order_status(self, order: Order) -> None:
        # The simulated OME resolves orders by id alone (live venues need the symbol off
        # the order for the REST fetch).
        oid = order.venue_order_id or order.client_order_id
        for resting in self._ome.get_open_orders().values():
            if resting.venue_order_id == oid or resting.client_order_id == oid:
                self.send(
                    OrderAcceptedEvent(
                        instrument=resting.instrument,
                        client_order_id=resting.client_order_id,
                        venue_order_id=resting.venue_order_id,
                        accepted_at=self._time.time(),
                    )
                )
                return
        self.send(
            OrderRejectedEvent(
                instrument=order.instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.venue_order_id,
                reason="reconcile: order not present at venue",
            )
        )

    def request_hist_deals(self, instrument: Instrument, since: dt_64) -> None:
        # No-op: the simulated OME books every fill as a deal at match time, so there are
        # never executions to recover (and sim snapshots carry no positions, so the
        # position-confirm task that would request these never fires).
        pass

    def request_snapshot(self) -> None:
        open_orders = list(self._ome.get_open_orders().values())
        snapshot = AccountSnapshot(
            exchange=self.exchange_name,
            as_of=self._time.time(),
            open_orders=open_orders,
        )
        self.send(AccountSnapshotEvent(instrument=None, snapshot=snapshot))

    def process_market_data(self, instrument: Instrument, data: Timestamped) -> None:
        # Single entry point that drives the OME (the SimulationRunner per tick in backtest,
        # _feed_simulated_connector in paper) with whatever a subscription produces. The OME
        # matches resting orders against Quote/OrderBook/Trade/TradeArray, so those pass
        # through unchanged — full book depth / trade-array fidelity. Anything else (an OHLC
        # Bar, a bare price) isn't matchable, so emulate a tradeable quote from it. Note we
        # can't just always emulate: emulate_quote_from_data has no TradeArray case and
        # collapses an OrderBook to a single quote.
        if isinstance(data, (Quote, OrderBook, Trade, TradeArray)):
            feed: Quote | OrderBook | Trade | TradeArray | None = data
        else:
            feed = emulate_quote_from_data(instrument, self._time.time(), data)
        if feed is None:
            return
        for report in self._ome.process_market_data(instrument, feed):
            self._emit_from_report(report)

    def on_subscribe(self, instrument: Instrument) -> None:
        # OME lifecycle: reset the per-instrument book so a re-subscribe starts with no stale BBO.
        self._ome.on_subscribe(instrument)

    def on_unsubscribe(self, instrument: Instrument) -> None:
        self._ome.on_unsubscribe(instrument)

    def _emit_from_report(self, report: SimulatedExecutionReport) -> None:
        order = report.order
        status = order.status
        # The OME emits ACCEPTED (resting / just-placed) and CANCELED on the order; a fill
        # carries the deal in report.exec with status FILLED.
        if status in (OrderStatus.ACCEPTED, OrderStatus.SUBMITTED):
            self.send(
                OrderAcceptedEvent(
                    instrument=report.instrument,
                    client_order_id=order.client_order_id,
                    venue_order_id=order.venue_order_id,
                    accepted_at=report.timestamp,
                )
            )
        elif status == OrderStatus.CANCELED:
            self.send(
                OrderCanceledEvent(
                    instrument=report.instrument,
                    client_order_id=order.client_order_id,
                    venue_order_id=order.venue_order_id,
                )
            )
        self._emit_exec(report)

    def _emit_exec(self, report: SimulatedExecutionReport) -> None:
        """Emit the fill leg of a report: a FILLED order is fully filled; a still-live order
        with an exec is a partial fill (resting remainder stays in the book — the current OME
        fills atomically, so that branch is forward-compatibility scaffolding). No-op when
        the report carries no exec (resting/canceled)."""
        if report.exec is None:
            return
        order = report.order
        event_type = OrderFilledEvent if order.status == OrderStatus.FILLED else OrderPartiallyFilledEvent
        self.send(
            event_type(
                instrument=report.instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.venue_order_id,
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

    def set_max_instrument_leverage(self, instrument: Instrument, leverage: float) -> bool:
        return True

    def set_margin_mode(self, instrument: Instrument, mode: str) -> bool:
        return True

    # Simulation does not model venue margin: the config getters read neutral.
    def get_max_instrument_leverage(self, instrument: Instrument) -> float | None:
        return None

    def get_max_instrument_notional(self, instrument: Instrument) -> float:
        return float("inf")

    def get_margin_mode(self, instrument: Instrument) -> str | None:
        return None

    def get_adl_level(self, instrument: Instrument) -> int | None:
        return None
