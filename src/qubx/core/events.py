from dataclasses import dataclass
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
class OrderEvent(AccountMessage):
    """Base for order-lifecycle events, addressed by ``client_order_id`` (always present —
    synthesized ``ext:<venue_id>`` for external orders). ``venue_order_id`` is None until
    the venue acks (and stays None on reject events that never reached the venue)."""

    client_order_id: str
    venue_order_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderAcceptedEvent(OrderEvent):
    accepted_at: np.datetime64


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderRejectedEvent(OrderEvent):
    reason: str
    code: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderPartiallyFilledEvent(OrderEvent):
    fill: Deal


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderFilledEvent(OrderEvent):
    fill: Deal


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderCanceledEvent(OrderEvent):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderExpiredEvent(OrderEvent):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderUpdatedEvent(OrderEvent):
    new_price: float | None
    new_quantity: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderCancelRejectedEvent(OrderEvent):
    reason: str
    code: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderUpdateRejectedEvent(OrderEvent):
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
    scheduled event identifier the dispatcher routes on.

    `payload` is intentionally `Any`: its shape is a tagged union keyed by `kind` — a
    (prev, exec) time tuple for the built-in triggers, a dict for backtester signal
    injection, or arbitrary data for user-registered custom schedules. The set of kinds
    is open (strategies register their own), so one explicit type would either misrepresent
    some kinds or force a class-per-kind that custom schedules couldn't extend."""

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

# Typed market-data event class -> the attribute holding its payload. A single O(1)
# table, co-located with the event classes, that the dispatch reads instead of a
# per-tick match/case (see payload_for_event).
_PAYLOAD_ATTR: dict[type, str] = {
    QuoteEvent: "quote",
    TradeEvent: "trade",
    OrderBookEvent: "orderbook",
    OhlcEvent: "bar",
    FundingRateEvent: "funding_rate",
    OpenInterestEvent: "open_interest",
    LiquidationEvent: "liquidation",
    FundingPaymentEvent: "payment",
}

# Event class -> its (non-parameterized) data-type string. OHLC is omitted on purpose:
# its data type carries the timeframe and is built per-event in data_type_for_event.
_EVENT_DATA_TYPE: dict[type, str] = {
    QuoteEvent: str(DataType.QUOTE),
    TradeEvent: str(DataType.TRADE),
    OrderBookEvent: str(DataType.ORDERBOOK),
    FundingRateEvent: str(DataType.FUNDING_RATE),
    OpenInterestEvent: str(DataType.OPEN_INTEREST),
    LiquidationEvent: str(DataType.LIQUIDATION),
    FundingPaymentEvent: str(DataType.FUNDING_PAYMENT),
}


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
    """Inverse of `event_for_data_type`: the (parameterized) data-type string for a typed
    market-data event, used by the typed dispatch to key the cache/throttler. O(1) table
    lookup; OHLC is special-cased because its data type carries the timeframe."""
    if isinstance(event, OhlcEvent):
        return DataType.OHLC[event.timeframe]
    dtype = _EVENT_DATA_TYPE.get(type(event))
    if dtype is None:
        raise ValueError(f"no data type for event: {type(event).__name__}")
    return dtype


def payload_for_event(event: ChannelMessage) -> Any:
    """The wrapped payload (quote/trade/bar/…) carried by a typed market-data event.
    O(1) table lookup, the counterpart to data_type_for_event for the dispatch hot path."""
    attr = _PAYLOAD_ATTR.get(type(event))
    if attr is None:
        raise ValueError(f"no payload for market-data event: {type(event).__name__}")
    return getattr(event, attr)
