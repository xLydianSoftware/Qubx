from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qubx.core.basics import (
    Balance,
    Bar,
    DataType,
    Deal,
    FundingPayment,
    FundingRate,
    Instrument,
    Liquidation,
    OpenInterest,
    Order,
    OrderBook,
    Position,
    Quote,
    Trade,
)


# kw_only=True is REQUIRED, not stylistic: the base carries a defaulted field
# (is_historical) and subclasses add required fields, so positional ordering
# would raise "non-default argument follows default argument" at class
# definition. kw_only sidesteps the ordering entirely.
@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelMessage:
    instrument: Instrument | None
    is_historical: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class AccountMessage(ChannelMessage):
    """Marker base class (no fields of its own) for anything that mutates
    AccountState. AM.apply() is typed to accept only these, and ProcessingManager
    routes by isinstance(event, AccountMessage) — market data can't be misrouted
    into the state machine."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketDataMessage(ChannelMessage):
    """Marker base class (no fields of its own) for the hot path: ProcessingManager
    dispatches these straight to strategy callbacks via isinstance-based dispatch;
    they never reach AM."""


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
class OhlcEvent(MarketDataMessage):
    bar: Bar
    # The timeframe lives in the data-type string (ohlc(1h)), not on the Bar, and
    # the cache keys its OHLC series by it — so it travels on the event.
    timeframe: str


@dataclass(frozen=True, slots=True, kw_only=True)
class FundingRateEvent(MarketDataMessage):
    funding_rate: FundingRate


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenInterestEvent(MarketDataMessage):
    open_interest: OpenInterest


@dataclass(frozen=True, slots=True, kw_only=True)
class LiquidationEvent(MarketDataMessage):
    liquidation: Liquidation


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduledEvent(ChannelMessage):
    """Scheduled/control trigger off the channel (time, fit, delisting_check,
    stale_data_check, state_snapshot, or a custom scheduled method). `kind` is the
    scheduled event identifier the dispatcher routes on; `payload` carries the
    trigger's (prev, exec) time tuple or custom data."""

    kind: str
    payload: Any = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorEvent(ChannelMessage):
    error: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class CustomEvent(ChannelMessage):
    name: str
    payload: Any


# Base DataTypes that map to a typed market-data event (the set event_for_data_type
# converts). Producers and the process_data adapter share this constant to agree on
# what becomes a typed event vs. what stays on the tuple path.
MARKET_DATA_TYPES: frozenset = frozenset(
    {
        DataType.QUOTE,
        DataType.TRADE,
        DataType.ORDERBOOK,
        DataType.OHLC,
        DataType.FUNDING_RATE,
        DataType.OPEN_INTEREST,
        DataType.LIQUIDATION,
        DataType.FUNDING_PAYMENT,
    }
)


def event_for_data_type(
    data_type: str | DataType,
    *,
    instrument: Instrument | None,
    payload: Any,
    is_historical: bool = False,
) -> ChannelMessage:
    """Wrap an arriving market-data payload in its typed event — the single source
    of truth used by data producers and the `process_data` tuple adapter.
    `data_type_for_event` is the inverse the typed dispatch uses to recover the
    (parameterized) data-type string the cache keys on. `funding_payment` maps to
    the `AccountMessage` `FundingPaymentEvent` (it both mutates balances and feeds
    the market-data path — the dispatch handles that hybrid)."""
    base, params = DataType.from_str(data_type)
    match base:
        case DataType.QUOTE:
            return QuoteEvent(instrument=instrument, is_historical=is_historical, quote=payload)
        case DataType.TRADE:
            return TradeEvent(instrument=instrument, is_historical=is_historical, trade=payload)
        case DataType.ORDERBOOK:
            return OrderBookEvent(instrument=instrument, is_historical=is_historical, orderbook=payload)
        case DataType.OHLC:
            return OhlcEvent(
                instrument=instrument, is_historical=is_historical, bar=payload, timeframe=params["timeframe"]
            )
        case DataType.FUNDING_RATE:
            return FundingRateEvent(instrument=instrument, is_historical=is_historical, funding_rate=payload)
        case DataType.OPEN_INTEREST:
            return OpenInterestEvent(instrument=instrument, is_historical=is_historical, open_interest=payload)
        case DataType.LIQUIDATION:
            return LiquidationEvent(instrument=instrument, is_historical=is_historical, liquidation=payload)
        case DataType.FUNDING_PAYMENT:
            return FundingPaymentEvent(instrument=instrument, is_historical=is_historical, payment=payload)
        case _:
            raise ValueError(f"no typed market-data event for data type: {data_type}")


def data_type_for_event(event: ChannelMessage) -> str:
    """Inverse of `event_for_data_type`: the (parameterized) data-type string for a
    typed market-data event."""
    match event:
        case QuoteEvent():
            return str(DataType.QUOTE)
        case TradeEvent():
            return str(DataType.TRADE)
        case OrderBookEvent():
            return str(DataType.ORDERBOOK)
        case OhlcEvent():
            return DataType.OHLC[event.timeframe]
        case FundingRateEvent():
            return str(DataType.FUNDING_RATE)
        case OpenInterestEvent():
            return str(DataType.OPEN_INTEREST)
        case LiquidationEvent():
            return str(DataType.LIQUIDATION)
        case FundingPaymentEvent():
            return str(DataType.FUNDING_PAYMENT)
        case _:
            raise ValueError(f"no data type for event: {type(event).__name__}")
