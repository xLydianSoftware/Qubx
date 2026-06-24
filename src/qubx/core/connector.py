from typing import Protocol, runtime_checkable

from qubx.core.basics import CtrlChannel, Instrument, Order, OrderRequest, Timestamped
from qubx.core.events import ChannelMessage


class ChannelEmitter:
    """Gives a connector its ``send`` helper for emitting events on the channel.

    ``send`` is NOT part of the IConnector contract — the framework never calls it; the
    connector emits its own lifecycle / market-data events. Concrete connectors set
    ``channel`` in their constructor and inherit ``send`` here instead of each redefining
    it.
    """

    channel: CtrlChannel

    def send(self, event: ChannelMessage) -> None:
        self.channel.send(event)


@runtime_checkable
class IMarketDataSink(Protocol):
    """Connector that consumes market data locally — a simulated venue feeding its OME.

    Not part of the IConnector contract: live connectors execute at the venue and never
    see market data. The processing manager narrows to this protocol before feeding a
    paper-trading connector.
    """

    def process_market_data(self, instrument: Instrument, data: Timestamped) -> None: ...


@runtime_checkable
class IConnector(Protocol):
    exchange_name: str

    def submit_order(self, request: OrderRequest) -> None: ...

    # cancel/update/request_order_status take the whole ``Order`` the AccountManager already
    # holds (it is the single source of order state). The connector reads everything the venue
    # call needs straight off it — both ids (it prefers ``venue_order_id`` when present, the
    # venue's own id, and falls back to ``client_order_id``, the only id known before the ack),
    # the ``instrument`` (HL asset index, ccxt symbol), and ``side``/``type``/``time_in_force``
    # that an edit-as-cancel-and-replace needs. This keeps every connector a stateless adapter:
    # no per-order cache. The ``Order`` is READ-ONLY — the connector must extract what it needs
    # SYNCHRONOUSLY (before scheduling any async venue call) and never mutate it, so the async
    # path never races the AM mutating the live object. Resulting events carry both ids, so the
    # AM routes by either.
    def cancel_order(self, order: Order) -> None: ...

    def update_order(self, order: Order, *, price: float | None = None, quantity: float | None = None) -> None: ...

    def request_order_status(self, order: Order) -> None: ...

    def request_snapshot(self) -> None: ...

    def is_ws_ready(self) -> bool: ...
    def reconnect(self) -> bool: ...  # synchronous WS reconnect; returns success
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def make_client_id(self, suggested: str) -> str: ...

    @property
    def is_simulated_trading(self) -> bool: ...

    def set_instrument_leverage(self, instrument: Instrument, leverage: float) -> bool: ...
    def set_margin_mode(self, instrument: Instrument, mode: str) -> bool: ...
