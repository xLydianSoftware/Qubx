from typing import Protocol, runtime_checkable

from qubx.core.basics import CtrlChannel, Instrument, OrderRequest
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
class IConnector(Protocol):
    exchange_name: str

    def submit_order(self, request: OrderRequest) -> None: ...

    # cancel_order / update_order are addressed by BOTH ids of the SAME order — not either/or.
    # client_order_id is always known (synthesized as ``ext:<venue_id>`` for external orders)
    # so it is required; venue_order_id is added once the venue acks and is PREFERRED by the
    # connector when present (it falls back to the client id before the ack). The caller passes
    # whatever it has and the connector picks the id the venue accepts — so the id choice stays
    # in the connector (which knows the venue), not the caller.
    def cancel_order(self, client_order_id: str,
                     venue_order_id: str | None = None) -> None: ...

    def update_order(self, client_order_id: str,
                     venue_order_id: str | None = None,
                     price: float | None = None,
                     quantity: float | None = None) -> None: ...

    def request_order_status(self, client_order_id: str,
                             venue_order_id: str | None = None) -> None: ...

    def request_snapshot(self) -> None: ...

    def is_ws_ready(self) -> bool: ...
    def reconnect(self) -> bool: ...  # synchronous WS reconnect; returns success
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def make_client_id(self, suggested: str) -> str: ...

    @property
    def is_simulated_trading(self) -> bool: ...

    @property
    def read_only(self) -> bool: ...

    def set_instrument_leverage(self, instrument: Instrument,
                                 leverage: float) -> bool: ...
    def set_margin_mode(self, instrument: Instrument, mode: str) -> bool: ...
