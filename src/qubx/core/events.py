from dataclasses import dataclass
from typing import dataclass_transform

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


# kw_only is REQUIRED, not stylistic: the base carries a defaulted field
# (is_historical) and subclasses add required fields, so positional ordering
# would raise "non-default argument follows default argument" at class
# definition. kw_only sidesteps the ordering entirely.
@dataclass_transform(frozen_default=True, kw_only_default=True)
def msg[T](cls: type[T]) -> type[T]:
    return dataclass(frozen=True, slots=True, kw_only=True)(cls)


@msg
class ChannelMessage:
    instrument: Instrument | None
    is_historical: bool = False


@msg
class AccountMessage(ChannelMessage):
    """Marker base class (no fields of its own) for anything that mutates
    AccountState. AM.apply() is typed to accept only these, and ProcessingManager
    routes by isinstance(event, AccountMessage) — market data can't be misrouted
    into the state machine."""


@msg
class MarketDataMessage(ChannelMessage):
    """Marker base class (no fields of its own) for market-data events. DORMANT: market data
    currently rides (instrument, d_type, data, is_historical) tuples through
    ProcessingManager.process_data, so these typed classes are declared but unused. They are
    kept as the foundation for a future, separate market-data typing refactor."""


@msg
class OrderEvent(AccountMessage):
    """Base for order-lifecycle events, addressed by ``client_order_id`` (always present —
    synthesized ``ext:<venue_id>`` for external orders). ``venue_order_id`` is None until
    the venue acks (and stays None on reject events that never reached the venue)."""

    client_order_id: str
    venue_order_id: str | None = None


@msg
class OrderAcceptedEvent(OrderEvent):
    accepted_at: np.datetime64


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
    """A trade (execution) addressed to an order — the ledger leg of the hybrid event model.

    Order status events drive the lifecycle; a DealEvent drives the ledger. Combined-stream
    venues (Binance) deliver status+deal together, so the deal rides embedded on the fill
    events above. Split-stream venues (OKX/Bitfinex) deliver executions on a separate
    stream — each trade arrives as one DealEvent and the AccountManager correlates it to
    the order by id, deduped by ``deal.trade_id``. Never changes order status."""

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
    balance: Balance


@msg
class FundingPaymentEvent(AccountMessage):
    payment: FundingPayment


@msg
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


@msg
class AccountSnapshotEvent(AccountMessage):
    snapshot: AccountSnapshot


@msg
class QuoteEvent(MarketDataMessage):
    quote: Quote


@msg
class TradeEvent(MarketDataMessage):
    trade: Trade


@msg
class OrderBookEvent(MarketDataMessage):
    orderbook: OrderBook


@msg
class OhlcEvent(MarketDataMessage):
    bar: Bar
    # The timeframe lives in the data-type string (ohlc(1h)), not on the Bar, and
    # the cache keys its OHLC series by it — so it travels on the event.
    timeframe: str


@msg
class FundingRateEvent(MarketDataMessage):
    funding_rate: FundingRate


@msg
class OpenInterestEvent(MarketDataMessage):
    open_interest: OpenInterest


@msg
class LiquidationEvent(MarketDataMessage):
    liquidation: Liquidation
