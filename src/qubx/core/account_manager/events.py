from qubx.core.basics import AssetBalance, Deal, Instrument, Order, Position

from ..events import ChannelMessage, msg


@msg
class AccountMessage(ChannelMessage):
    pass


@msg
class OrderEvent(AccountMessage):
    """Base class for order events."""

    client_order_id: str
    venue_order_id: str | None = None


@msg
class OrderAcceptedEvent(OrderEvent):
    pass


@msg
class OrderRejectedEvent(OrderEvent):
    reason: str
    code: str | None = None


@msg
class OrderPartiallyFilledEvent(OrderEvent):
    fill: Deal | None = None


@msg
class OrderFilledEvent(OrderEvent):
    fill: Deal | None = None


@msg
class DealEvent(OrderEvent):
    deal: Deal


@msg
class OrderCanceledEvent(OrderEvent):
    pass


@msg
class OrderExpiredEvent(OrderEvent):
    pass


@msg
class OrderUpdatedEvent(OrderEvent):
    new_price: float | None
    new_quantity: float | None


@msg
class OrderCancelRejectedEvent(OrderEvent):
    reason: str
    code: str | None = None


@msg
class OrderUpdateRejectedEvent(OrderEvent):
    reason: str
    code: str | None = None


@msg
class PositionUpdateEvent(AccountMessage):
    position: Position


@msg
class BalanceUpdateEvent(AccountMessage):
    balance: AssetBalance


@msg
class AccountSnapshotEvent(AccountMessage):
    exchange: str
    open_orders: dict[Instrument, list[Order]]
    positions: dict[Instrument, Position]
    balances: list[AssetBalance]
