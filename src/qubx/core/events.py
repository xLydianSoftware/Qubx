from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qubx.core.basics import (
    Balance,
    Deal,
    FundingPayment,
    Instrument,
    Order,
    OrderBook,
    Position,
    Quote,
    Trade,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelMessage:
    instrument: Instrument | None
    is_historical: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class AccountMessage(ChannelMessage):
    """Anything that mutates AccountState. AM.apply() accepts only these."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketDataMessage(ChannelMessage):
    """Hot path — never reaches AM."""


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderAcceptedEvent(AccountMessage):
    client_order_id: str
    venue_order_id: str
    accepted_at: np.datetime64


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderRejectedEvent(AccountMessage):
    client_order_id: str
    reason: str
    code: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderPartiallyFilledEvent(AccountMessage):
    client_order_id: str
    venue_order_id: str | None
    fill: Deal


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderFilledEvent(AccountMessage):
    client_order_id: str
    venue_order_id: str | None
    fill: Deal


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderCanceledEvent(AccountMessage):
    client_order_id: str
    venue_order_id: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderExpiredEvent(AccountMessage):
    client_order_id: str
    venue_order_id: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderUpdatedEvent(AccountMessage):
    client_order_id: str
    venue_order_id: str
    new_price: float | None
    new_quantity: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderCancelRejectedEvent(AccountMessage):
    client_order_id: str
    reason: str
    code: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderUpdateRejectedEvent(AccountMessage):
    client_order_id: str
    reason: str
    code: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionUpdateEvent(AccountMessage):
    position: Position


@dataclass(frozen=True, slots=True, kw_only=True)
class BalanceUpdateEvent(AccountMessage):
    balance: Balance


@dataclass(frozen=True, slots=True, kw_only=True)
class FundingPaymentEvent(AccountMessage):
    payment: FundingPayment


@dataclass(frozen=True, slots=True, kw_only=True)
class AccountSnapshot:
    exchange: str
    as_of: np.datetime64
    open_orders: list[Order] | None = None
    positions: list[Position] | None = None
    balances: list[Balance] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AccountSnapshotEvent(AccountMessage):
    snapshot: AccountSnapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class ReconcileDiff:
    exchange: str
    as_of: np.datetime64
    orders_newly_terminal: list[Order] = field(default_factory=list)
    orders_materialized: list[Order] = field(default_factory=list)
    orders_updated: list[Order] = field(default_factory=list)
    positions_updated: list[Position] = field(default_factory=list)
    balances_updated: list[Balance] = field(default_factory=list)


@dataclass(frozen=True, slots=True, kw_only=True)
class QuoteEvent(MarketDataMessage):
    quote: Quote


@dataclass(frozen=True, slots=True, kw_only=True)
class TradeEvent(MarketDataMessage):
    trade: Trade


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderBookEvent(MarketDataMessage):
    orderbook: OrderBook


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorEvent(ChannelMessage):
    error: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class CustomEvent(ChannelMessage):
    name: str
    payload: Any
