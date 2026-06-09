from dataclasses import dataclass

import numpy as np

from qubx.core.basics import (
    Balance,
    Bar,
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
    """Marker base class (no fields of its own) for market-data events. DORMANT: market data
    currently rides (instrument, d_type, data, is_historical) tuples through
    ProcessingManager.process_data, so these typed classes are declared but unused. They are
    kept as the foundation for a future, separate market-data typing refactor."""


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
    # Venue-reported account figures (see VenueAccountFigures): None when the venue
    # payload lacks them. Sim never sets them, so backtests always derive.
    equity: float | None = None
    available_margin: float | None = None
    margin_ratio: float | None = None


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
