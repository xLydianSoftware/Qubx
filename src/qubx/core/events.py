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


# kw_only is REQUIRED, not stylistic: bases carry defaulted fields (e.g.
# OrderEvent.venue_order_id) while subclasses add required ones, so positional
# ordering would raise "non-default argument follows default argument" at class
# definition. kw_only sidesteps the ordering entirely.
@dataclass_transform(frozen_default=True, kw_only_default=True)
def msg[T](cls: type[T]) -> type[T]:
    return dataclass(frozen=True, slots=True, kw_only=True)(cls)


@msg
class ChannelMessage:
    instrument: Instrument | None


@msg
class AccountMessage(ChannelMessage):
    """Marker base class (no fields of its own) for anything that mutates
    AccountState. AM.apply() is typed to accept only these, and ProcessingManager
    routes by isinstance(event, AccountMessage) — market data can't be misrouted
    into the state machine."""


@msg
class MarketDataMessage(ChannelMessage):
    """Reserved for a future typed market-data path. Market data deliberately rides
    (instrument, d_type, data, is_historical) tuples today (see 4ac821d1: only account
    events were typed), so this hierarchy is declared but unused."""


@msg
class OrderEvent(AccountMessage):
    """Base for order-lifecycle events, addressed by ``client_order_id`` when known.
    ``client_order_id`` is None when the connector only has the venue's id (e.g. a deal
    or reject observed before the cid index is seeded, or an external order) — the event
    is then addressed by ``venue_order_id`` alone and the AccountManager resolves (or
    materializes ``ext:<venue_id>``) from it. ``venue_order_id`` is None until the venue
    acks (and stays None on reject events that never reached the venue)."""

    client_order_id: str | None
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
    """Absolute venue position push (e.g. Binance ACCOUNT_UPDATE). ``as_of`` is the
    venue event time — same clock domain as ``Deal.time`` — used by the reducer's
    stale-push ratchet and drift detection."""

    position: Position
    as_of: np.datetime64


@msg
class BalanceUpdateEvent(AccountMessage):
    """Absolute venue balance push. ``as_of`` is the venue event time (same clock
    domain as ``Deal.time``), driving the per-currency ratchet and covered-delta
    guards; ``reason`` is the venue's change reason when reported (Binance ``a.m``:
    ORDER / FUNDING_FEE / ...). Fires NO strategy callback by design — balances are
    read via ctx."""

    balance: Balance
    as_of: np.datetime64
    reason: str | None = None


@msg
class FundingPaymentEvent(AccountMessage):
    payment: FundingPayment


@msg
class AccountSnapshot:
    """Venue-truth capture produced by ``IConnector.request_snapshot``. A None field
    means that leg was not observed (failed fetch / not applicable), never "empty".

    Producer contract: ``open_orders`` carry a producer-classified ``Order.origin`` —
    only the connector knows the framework-cid prefix its venue echoes back (OKX
    strips the underscore from ``qubx_``), so reconcile trusts the assigned origin
    instead of re-classifying.
    """

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
