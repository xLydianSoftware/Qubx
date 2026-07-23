import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from functools import cache
from queue import Empty, Queue
from threading import Event
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd

from qubx.core.exceptions import QueueTimeout
from qubx.core.series import Bar, OrderBook, Quote, Trade, time_as_nsec
from qubx.core.utils import prec_ceil, prec_floor, time_delta_to_str, time_to_str
from qubx.utils.misc import Stopwatch
from qubx.utils.ntp import start_ntp_thread, time_now
from qubx.utils.time import to_timedelta

dt_64 = np.datetime64
td_64 = np.timedelta64


def _as_dt64_or_nat(value: Any) -> "dt_64":
    """Normalize an arbitrary timestamp-like value (str / int / datetime / datetime64 / None) to a
    nanosecond ``datetime64``. ``None`` and unparseable / NaT-like values become ``NaT``."""
    if value is None:
        return np.datetime64("NaT")
    if isinstance(value, np.datetime64):
        return value if not np.isnat(value) else np.datetime64("NaT")
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return np.datetime64("NaT")
    return np.datetime64("NaT") if pd.isna(ts) else ts.asm8


OPTION_FILL_AT_SIGNAL_PRICE = "fill_at_signal_price"
OPTION_SIGNAL_PRICE = "signal_price"
OPTION_SKIP_PRICE_CROSS_CONTROL = "skip_price_cross_control"
OPTION_AVOID_STOP_ORDER_PRICE_VALIDATION = "avoid_stop_order_price_validation"

SW = Stopwatch()


@dataclass
class Liquidation:
    time: dt_64
    quantity: float
    price: float
    side: int

    def __repr__(self):
        return f"[{time_to_str(self.time, 'ns')}]\t {self.quantity} @ {self.price} | {self.side}"  # type: ignore


@dataclass
class AggregatedLiquidations:
    time: dt_64
    avg_buy_price: float
    last_buy_price: float
    buy_amount: float
    buy_count: int
    buy_notional: float

    avg_sell_price: float
    last_sell_price: float
    sell_amount: float
    sell_count: int
    sell_notional: float

    def __repr__(self):
        return f"[{time_to_str(self.time, 'ns')}]\t B:{self.buy_amount} @ {self.avg_buy_price} | S:{self.sell_amount} @ {self.avg_sell_price}"  # type: ignore


@dataclass
class FundingRate:
    time: dt_64
    rate: float
    interval: str
    next_funding_time: dt_64
    mark_price: float | None = None
    index_price: float | None = None

    def __repr__(self):
        return f"[{time_to_str(self.time, 'ns')}]\t {self.rate:.5f} ({self.interval})"  # type: ignore


@dataclass
class FundingPayment:
    """
    Represents a funding payment for a perpetual swap position.

    Based on QuestDB schema: timestamp, symbol, funding_rate, funding_interval_hours
    """

    time: int  # - nanosecond epoch timestamp, consistent with other Timestamped types
    funding_rate: float
    funding_interval_hours: int

    def __post_init__(self):
        if abs(self.funding_rate) > 1.0:
            raise ValueError(f"Invalid funding rate: {self.funding_rate} (must be between -1.0 and 1.0)")

        if self.funding_interval_hours <= 0:
            raise ValueError(f"Invalid funding interval: {self.funding_interval_hours} (must be positive)")

    @property
    def funding_rate_apr(self) -> float:
        return self.funding_rate * 365 * 24 / self.funding_interval_hours * 100

    def __repr__(self):
        return f"[{time_to_str(self.time, 'ns')}]\t {self.funding_rate:.5f} ({self.funding_interval_hours}H)"  # type: ignore


@dataclass
class OpenInterest:
    """
    Represents open interest data for a perpetual swap contract.

    Based on QuestDB schema: timestamp, symbol, open_interest, open_interest_usd
    """

    time: dt_64
    symbol: str
    open_interest: float  # Open interest in base asset units
    open_interest_usd: float  # Open interest in USD value

    def __repr__(self):
        return f"[{time_to_str(self.time, 'ns')}]\t {self.symbol} | {self.open_interest:.2f} ({self.open_interest_usd:.4f})"  # type: ignore


@dataclass
class TimestampedDict:
    """
    Generic class for representing arbitrary data (as dict) with timestamp

    TODO: probably we need to have generic interface for classes like Quote, Bar, .... etc
    """

    time: dt_64
    data: dict[str, Any]

    def __getitem__(self, k: str):
        return self.data[k]

    def __repr__(self):
        return f"[{time_to_str(self.time, 'ns')}]\t {str(self.data)}"  # type: ignore


class ITimeProvider:
    """
    Generic interface for providing current time
    """

    def time(self) -> dt_64:
        """
        Returns current time
        """
        ...


# Alias for timestamped data types used in Qubx
Timestamped: TypeAlias = (
    Quote
    | Trade
    | Bar
    | OrderBook
    | TimestampedDict
    | FundingRate
    | Liquidation
    | FundingPayment
    | AggregatedLiquidations
)


@dataclass
class TargetPosition:
    """
    Class for presenting target position calculated from signal
    """

    time: dt_64 | str  # time when position was created
    instrument: "Instrument"
    target_position_size: float  # actual position size after processing in sizer
    entry_price: float | None = None
    stop_price: float | None = None
    take_price: float | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def price(self) -> float | None:
        return self.entry_price

    @property
    def stop(self) -> float | None:
        return self.stop_price

    @property
    def take(self) -> float | None:
        return self.take_price

    def __str__(self) -> str:
        _d = f"{pd.Timestamp(self.time).strftime('%Y-%m-%d %H:%M:%S.%f')}"
        _p = f" @ {self.entry_price}" if self.entry_price is not None else ""
        _s = f" stop: {self.stop_price}" if self.stop_price is not None else ""
        _t = f" take: {self.take_price}" if self.take_price is not None else ""
        return f"[{_d}] TARGET {self.target_position_size:+f} {self.instrument.base}{_p}{_s}{_t} for {self.instrument}"


@dataclass
class Signal:
    """
    Class for presenting signals generated by strategy

    Attributes:
        reference_price: float - aux market price when signal was generated
        is_service: bool - when we need this signal only for informative purposes (post-factum risk management etc)

        Options:
            - allow_override: bool - if True, and there is another signal for the same instrument, then override current.
            - group: str - group name for signal
            - comment: str - comment for signal
            - options: dict[str, Any] - additional options for signal
    """

    time: dt_64 | str  # time when signal was generated
    instrument: "Instrument"
    signal: float
    price: float | None = None
    stop: float | None = None
    take: float | None = None
    reference_price: float | None = None
    group: str = ""
    comment: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    is_service: bool = False  # when we need this signal only for informative purposes (post-factum risk management etc)

    def target_for_amount(self, amount: float, **kwargs) -> TargetPosition:
        assert not self.is_service, "Service signals can't be converted to target positions !"
        return self.instrument.target(
            self.time,
            self.instrument.round_size_down(amount),
            entry_price=self.price,
            stop_price=self.stop,
            take_price=self.take,
            options=self.options,
            **kwargs,
        )

    def __str__(self) -> str:
        _d = f"{pd.Timestamp(self.time).strftime('%Y-%m-%d %H:%M:%S.%f')}"
        _p = f" @ {self.price}" if self.price is not None else ""
        _s = f" stop: {self.stop}" if self.stop is not None else ""
        _t = f" take: {self.take}" if self.take is not None else ""
        _r = f" {self.reference_price:.2f}" if self.reference_price is not None else ""
        _c = f" ({self.comment})" if self.comment else ""
        _i = "SERVICE ::" if self.is_service else ""

        return f"[{_d}] {_i}{self.group}{_r} {self.signal:+.2f} {self.instrument}{_p}{_s}{_t}{_c}"

    def copy(self) -> "Signal":
        """
        Return a copy of the original signal
        """
        return Signal(
            self.time,
            self.instrument,
            self.signal,
            self.price,
            self.stop,
            self.take,
            self.reference_price,
            self.group,
            self.comment,
            dict(self.options),
            self.is_service,
        )


@dataclass
class InitializingSignal(Signal):
    """
    Special signal type for post-warmup initialization
    """

    use_limit_order: bool = False  # if True, then use limit order for post-warmup initialization

    def __str__(self) -> str:
        _d = f"{pd.Timestamp(self.time).strftime('%Y-%m-%d %H:%M:%S.%f')}"
        _p = f" @ {self.price}" if self.price is not None else ""
        _s = f" stop: {self.stop}" if self.stop is not None else ""
        _t = f" take: {self.take}" if self.take is not None else ""
        _r = f" {self.reference_price:.2f}" if self.reference_price is not None else ""
        _c = f" ({self.comment})" if self.comment else ""

        return f"[{_d}] POST-WARMUP-INIT ::{self.group}{_r} {self.signal:+.2f} {self.instrument}{_p}{_s}{_t}{_c}"


class MarketType(StrEnum):
    # - spot/cash markets
    SPOT = "SPOT"
    MARGIN = "MARGIN"
    STOCK = "STOCK"
    FOREX = "FOREX"
    BOND = "BOND"

    # - derivatives
    SWAP = "SWAP"
    FUTURE = "FUTURE"
    OPTION = "OPTION"
    CFD = "CFD"

    # - reference (non-tradable)
    INDEX = "INDEX"


@dataclass(order=True)
class Instrument:
    """
    Instrument class.

     - 2025-06-11: Important change for FUTURE type: now instrument's symbol contains delivery date in format YYYYMMDD.
        So now for let's say september's BTCUSDT future, symbol would be BTCUSD.20250914
        and full id is `BINANCE.UM:FUTURE:BTCUSD.20250914`
    """

    symbol: str
    market_type: MarketType
    exchange: str
    base: str
    quote: str
    settle: str
    exchange_symbol: str  # symbol used by the exchange
    tick_size: float  # minimal price step
    lot_size: float  # minimal position size
    min_size: float  # minimal allowed position size
    min_notional: float = 0.0  # minimal notional value
    initial_margin: float = 0.0  # initial margin
    maint_margin: float = 0.0  # maintenance margin
    liquidation_fee: float = 0.0  # liquidation fee
    contract_size: float = 1.0  # contract size (tokens per contract)
    contract_multiplier: float = 1.0  # contract multiplier (additional multiplier, always 1 for crypto)
    onboard_date: datetime | None = None  # date when instrument was listed on the exchange
    delivery_date: datetime | None = None  # date when instrument is delivered
    delist_date: datetime | None = None  # date when instrument is delisted
    inverse: bool = False  # if true, then the future is inverse

    def __post_init__(self):
        # define how ordering works
        object.__setattr__(self, "sort_index", f"{self.exchange}:{self.market_type}:{self.symbol}")

    @property
    def quantity_multiplier(self) -> float:
        """Combined multiplier: contract_size * contract_multiplier. Multiply contracts by this to get token quantity."""
        return self.contract_size * self.contract_multiplier

    @property
    def price_precision(self):
        if not hasattr(self, "_price_precision"):
            self._price_precision = int(abs(np.log10(self.tick_size)))
        return self._price_precision

    @property
    def size_precision(self):
        if not hasattr(self, "_size_precision"):
            self._size_precision = int(abs(np.log10(self.lot_size)))
        return self._size_precision

    @property
    def asset(self) -> str:
        if self.base.startswith("1000"):
            return self.base.replace("1000", "")
        elif self.base.startswith("1000000"):
            return self.base.replace("1000000", "")
        else:
            return self.base

    def is_futures(self) -> bool:
        return self.market_type in [MarketType.FUTURE, MarketType.SWAP]

    def is_spot(self) -> bool:
        # TODO: handle margin better
        return self.market_type in [MarketType.SPOT, MarketType.MARGIN]

    def round_size_down(self, size: float) -> float:
        """
        Round down size to specified precision

        i.size_precision == 3
        i.round_size_up(0.1234) -> 0.123
        """
        return prec_floor(size, self.size_precision)

    def round_size_up(self, size: float) -> float:
        """
        Round up size to specified precision

        i.size_precision == 3
        i.round_size_up(0.1234) -> 0.124
        """
        return prec_ceil(size, self.size_precision)

    def round_price_down(self, price: float) -> float:
        """
        Round down price to specified precision

        i.price_precision == 3
        i.round_price_down(1.234999, 3) -> 1.234
        """
        return prec_floor(price, self.price_precision)

    def round_price_up(self, price: float) -> float:
        """
        Round up price to specified precision

        i.price_precision == 3
        i.round_price_up(1.234999) -> 1.235
        """
        return prec_ceil(price, self.price_precision)

    def service_signal(
        self,
        time: dt_64 | str | ITimeProvider,
        signal: float,
        price: float | None = None,
        stop: float | None = None,
        take: float | None = None,
        group: str = "",
        comment: str = "",
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> Signal:
        """
        Create service signal for the instrument
        """
        return self.signal(time, signal, price, stop, take, group, comment, options, is_service=True, **kwargs)

    def signal(
        self,
        time: dt_64 | str | ITimeProvider,
        signal: float,
        price: float | None = None,
        stop: float | None = None,
        take: float | None = None,
        group: str = "",
        comment: str = "",
        options: dict[str, Any] | None = None,
        is_service: bool = False,
        **kwargs,
    ) -> Signal:
        """
        Create signal for the instrument
        """
        return Signal(
            time=time.time() if isinstance(time, ITimeProvider) else time,
            instrument=self,
            signal=signal,
            price=price,
            stop=stop,
            take=take,
            group=group,
            comment=comment,
            options=(options or {}) | kwargs,
            is_service=is_service,
        )

    def target(
        self,
        time: dt_64 | str | ITimeProvider,
        amount: float,
        entry_price: float | None = None,
        stop_price: float | None = None,
        take_price: float | None = None,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> TargetPosition:
        """
        Create target position for the instrument
        """
        return TargetPosition(
            time=time.time() if isinstance(time, ITimeProvider) else time,
            instrument=self,
            target_position_size=self.round_size_down(amount),
            entry_price=entry_price,
            stop_price=stop_price,
            take_price=take_price,
            options=(options or {}) | kwargs,
        )

    def __hash__(self) -> int:
        return hash((self.symbol, self.exchange, self.market_type))

    def __eq__(self, other: Any) -> bool:
        if other is None or not isinstance(other, Instrument):
            return False
        return str(self) == str(other)

    def __str__(self) -> str:
        return ":".join([self.exchange, self.market_type, self.symbol])

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def parse_notation(notation: str) -> tuple[str | None, MarketType | None, str]:
        """
        Parse instrument notation string into (exchange, market_type, symbol).

        Supports:
            "BTCUSDT"                    -> (None, None, "BTCUSDT")
            "BINANCE.UM:BTCUSDT"        -> ("BINANCE.UM", None, "BTCUSDT")
            "BINANCE.UM:SWAP:BTCUSDT"   -> ("BINANCE.UM", MarketType.SWAP, "BTCUSDT")
        """
        parts = notation.split(":")
        match len(parts):
            case 1:
                return None, None, parts[0]
            case 2:
                return parts[0], None, parts[1]
            case 3:
                mid = parts[1].upper()
                _valid = {mt.value for mt in MarketType}
                if mid not in _valid:
                    raise ValueError(
                        f"Invalid market type '{parts[1]}' in notation '{notation}'. "
                        f"Valid types: {', '.join(sorted(_valid))}"
                    )
                return parts[0], MarketType(mid), parts[2]
            case _:
                raise ValueError(
                    f"Invalid instrument notation: '{notation}'. "
                    f"Expected SYMBOL, EXCHANGE:SYMBOL, or EXCHANGE:MARKET_TYPE:SYMBOL"
                )

    def info(self):
        info_str = f"""
┌─────────────────────────────┐
│ Instrument Information      │
└─────────────────────────────┘
  Exchange:          {self.exchange}
  Symbol:            {self.symbol}
  Market Type:       {self.market_type}
  Base:              {self.base}
  Quote:             {self.quote}
  Exchange Symbol:   {self.exchange_symbol.upper()}
  Tick Size:         {self.tick_size}
  Lot Size:          {self.lot_size}
  Min Size:          {self.min_size}
  Min Notional:      {self.min_notional}
  Contract Size:     {self.contract_size}
  Contract Mult:     {self.contract_multiplier}
  Initial Margin:    {self.initial_margin}
  Maint. Margin:     {self.maint_margin}
  Onboard Date:      {self.onboard_date}
  Delist Date:       {self.delist_date}
"""
        print(info_str)


class TransactionCostsCalculator:
    """
    A class for calculating transaction costs for a trading strategy.
    Attributes
    ----------
    name : str
        The name of the transaction costs calculator.
    maker : float
        The maker fee, as a percentage of the transaction value.
    taker : float
        The taker fee, as a percentage of the transaction value.

    """

    name: str
    maker: float
    taker: float

    def __init__(self, name: str, maker: float, taker: float):
        self.name = name
        self.maker = maker / 100.0
        self.taker = taker / 100.0

    def get_execution_fees(
        self, instrument: Instrument, exec_price: float, amount: float, crossed_market=False, conversion_rate=1.0
    ):
        notional = abs(amount * instrument.quantity_multiplier * exec_price)
        if crossed_market:
            return notional * self.taker / conversion_rate
        else:
            return notional * self.maker / conversion_rate

    def get_overnight_fees(self, instrument: Instrument, amount: float):
        return 0.0

    def get_funding_rates_fees(self, instrument: Instrument, amount: float):
        return 0.0

    def get_maker_fee_rate(self) -> float:
        """Get maker fee rate as decimal (e.g., 0.0002 for 0.02%)."""
        return self.maker

    def get_taker_fee_rate(self) -> float:
        """Get taker fee rate as decimal (e.g., 0.0004 for 0.04%)."""
        return self.taker

    def __repr__(self):
        return f"<{self.name}: {self.maker * 100:.4f} / {self.taker * 100:.4f}>"


ZERO_COSTS = TransactionCostsCalculator("Zero", 0.0, 0.0)


@dataclass
class TriggerEvent:
    """
    Event data for strategy trigger
    """

    time: dt_64
    type: str
    instrument: Instrument | None
    data: Any | None


@dataclass
class MarketEvent:
    """
    Market data update.
    """

    time: dt_64
    type: str
    instrument: Instrument | None
    data: Any
    is_trigger: bool = False

    def to_trigger(self) -> TriggerEvent:
        return TriggerEvent(self.time, self.type, self.instrument, self.data)

    def __repr__(self):
        _items = [
            f"time={self.time}",
            f"type={self.type}",
        ]
        if self.instrument is not None:
            _items.append(f"instrument={self.instrument}")
        _items.append(f"data={self.data}")
        return f"MarketEvent({', '.join(_items)})"


@dataclass
class Deal:
    trade_id: str  # trade id
    order_id: str  # VENUE (exchange-assigned) order id, not the client_order_id
    time: dt_64  # time of trade
    amount: float  # signed traded amount: positive for buy and negative for selling
    price: float
    aggressive: bool
    fee_amount: float | None = None
    fee_currency: str | None = None


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    INITIALIZED = "INITIALIZED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    PENDING_CANCEL = "PENDING_CANCEL"
    PENDING_UPDATE = "PENDING_UPDATE"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    # Reconciler give-up terminal: an order we could not confirm at the venue after the
    # status-fetch budget was exhausted (neither a fill/cancel/reject arrived nor a snapshot).
    LOST = "LOST"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_ORDER_STATUSES

    @property
    def is_inflight(self) -> bool:
        return self in _INFLIGHT_ORDER_STATUSES

    @property
    def is_pending(self) -> bool:
        return self in _PENDING_ORDER_STATUSES


_TERMINAL_ORDER_STATUSES = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
        OrderStatus.LOST,
    }
)
_INFLIGHT_ORDER_STATUSES = frozenset(
    {
        OrderStatus.SUBMITTED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.PENDING_UPDATE,
    }
)
_PENDING_ORDER_STATUSES = frozenset(
    {
        OrderStatus.PENDING_CANCEL,
        OrderStatus.PENDING_UPDATE,
    }
)


# Client-id prefixes shared by every producer/classifier of order client ids.
# FRAMEWORK_CID_PREFIX marks an order as framework-originated (ClientIdStore._create_id,
# connector make_client_id); EXTERNAL_CID_PREFIX marks a cid synthesized for an order the
# framework discovered at the venue but never placed.
FRAMEWORK_CID_PREFIX = "qubx_"
EXTERNAL_CID_PREFIX = "ext:"


class OrderOrigin(StrEnum):
    FRAMEWORK = "FRAMEWORK"
    RECOVERED = "RECOVERED"
    EXTERNAL = "EXTERNAL"


def classify_origin(client_order_id: str, *, framework_prefix: str = FRAMEWORK_CID_PREFIX) -> OrderOrigin:
    """Classify an order observed in venue data by its client id: the framework cid
    prefix marks a framework order seen back from the venue (RECOVERED), anything
    else is EXTERNAL. Orders the framework places itself are FRAMEWORK at creation
    and never pass through here.

    ``framework_prefix`` is for connectors whose venue cid charset mangles
    ``FRAMEWORK_CID_PREFIX`` (OKX bans ``_``): they classify with the prefix the venue
    actually echoes back, derived from the same sanitizer their ``make_client_id`` uses.
    """
    if client_order_id.startswith(framework_prefix):
        return OrderOrigin.RECOVERED
    return OrderOrigin.EXTERNAL


def resolve_reduce_only(options: dict[str, Any]) -> bool | None:
    """Resolve the reduceOnly flag from order options, accepting either spelling — camelCase
    ``reduceOnly`` or snake_case ``reduce_only`` (``reduceOnly`` wins if both are present).

    Returns ``None`` when the caller specified neither, so callers can distinguish 'unset' from
    an explicit ``False`` (e.g. to auto-resolve reduceOnly from the current position). Single
    source of truth for the alias so the Order field and the connector payload never disagree.
    """
    if "reduceOnly" in options:
        return bool(options["reduceOnly"])
    if "reduce_only" in options:
        return bool(options["reduce_only"])
    return None


class OrderChange(StrEnum):
    """What happened to an order, paired with it on ApplyResult. Covers the cases
    order.status can't express on its own: UPDATED (status unchanged), CANCEL_REJECTED/
    UPDATE_REJECTED (status reverts)."""

    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    UPDATED = "UPDATED"
    CANCEL_REJECTED = "CANCEL_REJECTED"
    UPDATE_REJECTED = "UPDATE_REJECTED"
    LOST = "LOST"


@dataclass
class OrderRequest:
    """
    Represents an order submission request (order intent).

    This is created by TradingManager and enriched by brokers with exchange-specific
    metadata in the options dict. The client_id is never mutated and is used for
    order tracking and health monitoring correlation.

    Attributes:
        instrument: The trading instrument
        quantity: Order quantity (positive for buy, negative for sell in some contexts)
        price: Limit price (None for market orders)
        order_type: "MARKET" or "LIMIT"
        side: "BUY" or "SELL"
        client_id: Unique identifier
        time_in_force: Order duration ("gtc", "ioc", "fok", etc.)
        options: Exchange-specific metadata (e.g., lighter_client_order_index)
    """

    instrument: Instrument
    quantity: float
    price: float | None = None
    order_type: OrderType = OrderType.LIMIT
    side: OrderSide = OrderSide.BUY
    time_in_force: str = "gtc"
    client_id: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrderTransition:
    """One entry in an Order's status-transition audit trail (see Order.transitions)."""

    time: dt_64
    from_status: OrderStatus
    to_status: OrderStatus


@dataclass(slots=True, kw_only=True)
class Order:
    client_order_id: str
    type: OrderType
    instrument: Instrument
    quantity: float
    side: OrderSide
    time_in_force: str
    status: OrderStatus = OrderStatus.INITIALIZED
    venue_order_id: str | None = None
    price: float | None = None  # None for market orders (no limit price)
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None
    # submission timestamp; grace-window reconcile measures order age from this field
    submitted_at: dt_64 | None = None
    accepted_at: dt_64 | None = None
    # Last update timestamp: the VENUE's update time when we have it (snapshot order
    # `lastUpdateTimestamp`/`updateTime`, or the venue event's own ts), else our local
    # processing time as a fallback. The single canonical update clock — drives the Differ
    # grace gate, terminal eviction, and the reconciler's monotonic guard.
    last_update_time: dt_64 | None = None
    rejected_reason: str | None = None
    # venue/connector error code accompanying a reject (e.g. the ccxt error class name);
    # None when the reject path carries no code (synthetic reconcile rejects).
    error_code: str | None = None
    reduce_only: bool = False
    post_only: bool = False
    # Defaults to FRAMEWORK (the common case); the snapshot/external materialization
    # paths set EXTERNAL / RECOVERED explicitly.
    origin: OrderOrigin = OrderOrigin.FRAMEWORK
    options: dict[str, Any] = field(default_factory=dict)

    def require_venue_id(self) -> str:
        if self.venue_order_id is None:
            raise ValueError(f"Order {self.client_order_id} has no venue_order_id (status={self.status})")
        return self.venue_order_id

    def record_fill(self, quantity: float, price: float) -> None:
        """Accumulate one fill into filled_quantity and the running average fill price.

        ``quantity`` is the fill size (sign-agnostic — absolute size is used for the
        weighted average). filled_quantity mirrors real, irreversible fills and so is only
        ever increased. The caller owns dedup (apply a given fill at most once) and any
        position/balance side effects; this only maintains the order's own fill totals.
        """
        qty = abs(quantity)
        if self.avg_fill_price is None:
            self.avg_fill_price = price
        else:
            self.avg_fill_price = (self.avg_fill_price * self.filled_quantity + price * qty) / (
                self.filled_quantity + qty
            )
        self.filled_quantity += qty

    def __str__(self) -> str:
        _id = (self.venue_order_id or "????") + (f" :: {self.client_order_id}" if self.client_order_id else "")
        return f"[{_id}] {self.type} {self.side} {self.quantity} of {self.instrument} {('@ ' + str(self.price)) if self.price else ''} ({self.time_in_force}) [{self.status}]"


@dataclass
class Balance:
    exchange: str
    currency: str
    free: float = 0.0
    locked: float = 0.0
    total: float = 0.0
    # Venue update timestamp (venue clock). Available only from the WS push (event time `E`);
    # the REST balance snapshot carries none on Binance UM (account updateTime=0). See the
    # reconciliation design doc.
    last_update_time: dt_64 | None = None

    def __str__(self) -> str:
        return f"{self.exchange}:{self.currency} free={self.free:.2f} locked={self.locked:.2f} total={self.total:.2f}"

    def reset_by_balance(self, balance: "Balance") -> None:
        # In-place value copy (exchange/currency are identity) so holders of this
        # Balance keep a live reference. Mirrors Position.reset_by_position.
        self.free = balance.free
        self.locked = balance.locked
        self.total = balance.total
        self.last_update_time = balance.last_update_time


class TransferStatus(StrEnum):
    """Normalized status of an inter-exchange fund transfer.

    Live services expose a venue-native status (kept on ``Transfer.raw_status``); this is the
    3-state vocabulary the framework and strategies reason about. Values are lowercase to match
    the transfers-log column content and the historical status strings.
    """

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (TransferStatus.COMPLETED, TransferStatus.FAILED)


@dataclass(frozen=True)
class Transfer:
    """A single inter-exchange fund transfer tracked by an ``ITransferManager``.

    Immutable: managers replace the tracked instance on each refresh rather than mutating it,
    so a value handed to a caller never changes underfoot (the live manager is read from both
    the strategy and the control-server threads).
    """

    transaction_id: str
    from_exchange: str
    to_exchange: str
    currency: str
    amount: float
    status: TransferStatus
    timestamp: dt_64
    raw_status: str | None = None  # venue-native status; None for simulation
    failure_reason: str | None = None  # populated when status is FAILED

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping (datetime64 -> str, status -> its value) for wire/reporting."""
        return {
            "transaction_id": self.transaction_id,
            "timestamp": str(self.timestamp),
            "from_exchange": self.from_exchange,
            "to_exchange": self.to_exchange,
            "currency": self.currency,
            "amount": self.amount,
            "status": str(self.status),
            "raw_status": self.raw_status,
            "failure_reason": self.failure_reason,
        }


DEFAULT_MAINTENANCE_MARGIN = 0.05


class Position:
    instrument: Instrument  # instrument for this position
    quantity: float = 0.0  # quantity positive for long and negative for short
    pnl: float = 0.0  # total cumulative position PnL in portfolio basic funds currency
    r_pnl: float = 0.0  # realized cumulative position PnL in portfolio basic funds currency
    market_value: float = 0.0  # position's market value in quote currency
    market_value_funds: float = 0.0  # position market value in portfolio funded currency
    position_avg_price: float = 0.0  # average position price
    position_avg_price_funds: float = 0.0  # average position price
    commissions: float = 0.0  # cumulative commissions paid for this position

    last_update_time: int = np.nan  # when price updated or position changed    # type: ignore
    last_update_price: float = np.nan  # last update price (actually instrument's price) in quoted currency
    last_update_conversion_rate: float = np.nan  # last update conversion rate

    # margin requirements
    initial_margin: float = 0.0
    _initial_margin_external: bool = False  # If True, initial_margin is managed by exchange (skip recalculation)
    maint_margin: float = 0.0
    _maint_margin_external: bool = False  # If True, maint_margin is managed by exchange (skip recalculation)

    # ADL queue position from the exchange (None if not reported).
    # Lower values = more likely to be auto-deleveraged.
    adl_level: int | None = None

    # Per-instrument venue settings reported by the exchange (None if not reported).
    # ``leverage`` is the configured/initial leverage tier (distinct from the observed
    # notional/equity ratio); ``max_notional`` is the venue's notional cap at that leverage.
    leverage: float | None = None
    margin_mode: Literal["cross", "isolated"] | None = None
    max_notional: float | None = None

    # funding payment tracking
    cumulative_funding: float = 0.0  # cumulative funding paid (negative) or received (positive)
    last_funding_time: dt_64 = np.datetime64("NaT")  # last funding payment time

    # episode tracking - baselines stamped at the last flat->open transition.
    # An episode is the span from one flat->open transition to the next return to flat.
    # These are a view over the lifetime accumulators (r_pnl / commissions / cumulative_funding):
    # with zero baselines the episode accessors equal the lifetime values (legacy degradation).
    episode_start_time: dt_64 = np.datetime64("NaT")  # stamp of the opening deal
    r_pnl_at_open: float = 0.0
    commissions_at_open: float = 0.0
    cumulative_funding_at_open: float = 0.0

    # - helpers for position processing
    _qty_multiplier: float = 1.0
    __pos_incr_qty: float = 0

    def __init__(
        self,
        instrument: Instrument,
        quantity: float = 0.0,
        pos_average_price: float = 0.0,
        r_pnl: float = 0.0,
        cumulative_funding: float = 0.0,
        commissions: float = 0.0,
        episode_start_time: dt_64 | str | int | float | None = None,
        r_pnl_at_open: float | None = None,
        commissions_at_open: float | None = None,
        cumulative_funding_at_open: float | None = None,
    ) -> None:
        self.instrument = instrument

        self.reset()

        self.r_pnl = r_pnl
        self.cumulative_funding = cumulative_funding
        self.commissions = commissions

        if quantity != 0.0 and pos_average_price > 0.0:
            self.quantity = quantity
            self.position_avg_price = pos_average_price
            self.__pos_incr_qty = abs(quantity)

        # - episode baselines
        # Round-trip path (restorer): the three at-open baselines are supplied -> assign verbatim
        # (episode_start_time may legitimately be NaT/None, so we key off the baselines, not the time).
        # Episode-at-init path: constructed open with baselines absent (legacy row or bare open
        # construction) -> stamp the episode from the supplied lifetime accumulators. `episode_start_time`
        # is the restore time when provided, else NaT ("opening never observed"). Flat construction keeps
        # the zero defaults set by reset().
        _baselines_provided = (
            r_pnl_at_open is not None and commissions_at_open is not None and cumulative_funding_at_open is not None
        )
        if _baselines_provided:
            self.episode_start_time = _as_dt64_or_nat(episode_start_time)
            self.r_pnl_at_open = r_pnl_at_open
            self.commissions_at_open = commissions_at_open
            self.cumulative_funding_at_open = cumulative_funding_at_open
        elif quantity != 0.0:
            self.episode_start_time = _as_dt64_or_nat(episode_start_time)
            self.r_pnl_at_open = self.r_pnl
            self.commissions_at_open = self.commissions
            self.cumulative_funding_at_open = self.cumulative_funding

    def reset(self) -> None:
        """
        Reset position to zero
        """
        self.quantity = 0.0
        self.pnl = 0.0
        self.r_pnl = 0.0
        self.market_value = 0.0
        self.market_value_funds = 0.0
        self.position_avg_price = 0.0
        self.position_avg_price_funds = 0.0
        self.commissions = 0.0
        self.last_update_time = np.nan  # type: ignore
        self.last_update_price = np.nan
        self.last_update_conversion_rate = np.nan
        self.initial_margin = 0.0
        self._initial_margin_external = False
        self.maint_margin = 0.0
        self._maint_margin_external = False
        self.adl_level = None
        self.leverage = None
        self.margin_mode = None
        self.max_notional = None
        self.cumulative_funding = 0.0
        self.last_funding_time = np.datetime64("NaT")  # type: ignore
        self.episode_start_time = np.datetime64("NaT")  # type: ignore
        self.r_pnl_at_open = 0.0
        self.commissions_at_open = 0.0
        self.cumulative_funding_at_open = 0.0
        self.__pos_incr_qty = 0
        self._qty_multiplier = self.instrument.quantity_multiplier

    def flatten(self) -> None:
        """
        Mark the position flat WITHOUT trading: zero quantity and the derived
        market values / margins, while KEEPING realized PnL, average price,
        commissions and funding history.

        For an instrument whose market has been delisted/removed from the
        exchange (already cash-settled) we cannot place a closing order, so we
        reconcile the in-memory position to flat. Unlike reset(), this preserves
        r_pnl so the record stays identical to a normally-closed position.

        Episodes: no explicit change here. Going flat ends the current episode via the
        canonical flat predicate (``abs(quantity) < lot_size``); the episode baselines are
        left untouched so the closed episode's final values stay readable via the accessors.
        """
        self.quantity = 0.0
        self.market_value = 0.0
        self.market_value_funds = 0.0
        self.initial_margin = 0.0
        self.maint_margin = 0.0
        self.pnl = self.r_pnl  # unrealized PnL is zero at zero quantity
        self.__pos_incr_qty = 0

    def reset_by_position(self, pos: "Position") -> None:
        self.quantity = pos.quantity
        self.pnl = pos.pnl
        self.r_pnl = pos.r_pnl
        self.market_value = pos.market_value
        self.market_value_funds = pos.market_value_funds
        self.position_avg_price = pos.position_avg_price
        self.position_avg_price_funds = pos.position_avg_price_funds
        self.commissions = pos.commissions
        self.last_update_time = pos.last_update_time
        self.last_update_price = pos.last_update_price
        self.last_update_conversion_rate = pos.last_update_conversion_rate
        self.initial_margin = pos.initial_margin
        self._initial_margin_external = pos._initial_margin_external
        self.maint_margin = pos.maint_margin
        self._maint_margin_external = pos._maint_margin_external
        self.adl_level = pos.adl_level
        self.leverage = pos.leverage
        self.margin_mode = pos.margin_mode
        self.max_notional = pos.max_notional
        self.cumulative_funding = pos.cumulative_funding
        self.last_funding_time = pos.last_funding_time if hasattr(pos, "last_funding_time") else np.datetime64("NaT")
        self.episode_start_time = getattr(pos, "episode_start_time", np.datetime64("NaT"))
        self.r_pnl_at_open = getattr(pos, "r_pnl_at_open", 0.0)
        self.commissions_at_open = getattr(pos, "commissions_at_open", 0.0)
        self.cumulative_funding_at_open = getattr(pos, "cumulative_funding_at_open", 0.0)
        self.__pos_incr_qty = pos.__pos_incr_qty

    def reconcile_size(self, quantity: float, avg_price: float, *, timestamp: dt_64 | None = None) -> None:
        """Authoritative size/avg-price correction (venue snapshot reconcile). Touches
        sizing fields only — accumulated accounting (r_pnl, commissions, funding) is
        locally owned and must survive a snapshot.

        Episodes: if this takes a flat position to open (first-connect recovery of a
        pre-existing position whose true opening was never observed), stamp an episode
        from the *current* accumulators — the honest lower bound, "episode starts now".
        ``timestamp`` is the venue snapshot's time; falls back to ``last_update_time``.
        """
        was_open = self.is_open()
        self.quantity = quantity
        self.position_avg_price = avg_price
        self.position_avg_price_funds = avg_price  # conversion seam is fixed at 1.0 (AccountState.conversion_rate)
        self.__pos_incr_qty = abs(quantity)
        if not was_open and self.is_open():
            self.episode_start_time = timestamp if timestamp is not None else self.last_update_time
            self.r_pnl_at_open = self.r_pnl
            self.commissions_at_open = self.commissions
            self.cumulative_funding_at_open = self.cumulative_funding

    @property
    def notional_value(self) -> float:
        return self.quantity * self._qty_multiplier * self.last_update_price / self.last_update_conversion_rate

    def _price(self, update: Quote | Trade) -> float:
        if isinstance(update, Quote):
            return update.bid if np.sign(self.quantity) > 0 else update.ask
        elif isinstance(update, Trade):
            return update.price
        raise ValueError(f"Unknown update type: {type(update)}")

    def change_position_by(
        self,
        timestamp: dt_64,
        amount: float,
        exec_price: float,
        fee_amount: float = 0,
        conversion_rate: float = 1,
        *,
        realize_only: bool = False,
    ) -> tuple[float, float]:
        return self.update_position(
            timestamp,
            self.instrument.round_size_down(self.quantity + amount),
            exec_price,
            fee_amount,
            conversion_rate=conversion_rate,
            realize_only=realize_only,
        )

    def update_position(
        self,
        timestamp: dt_64,
        position: float,
        exec_price: float,
        fee_amount: float = 0,
        conversion_rate: float = 1,
        *,
        realize_only: bool = False,
    ) -> tuple[float, float]:
        # - realize_only=True books the closing pnl + fee but leaves size/avg/balance to a snapshot
        #   reconcile (situation II: recovering deals already in an authoritative venue size)
        deal_pnl = 0
        quantity = self.quantity
        comms = 0

        if quantity != position:
            # - whether the position is open *before* this deal (drives episode stamping below)
            was_open = self.is_open()

            pos_change = position - quantity
            direction = np.sign(pos_change)
            prev_direction = np.sign(quantity)

            # how many shares are closed/open
            qty_closing = min(abs(self.quantity), abs(pos_change)) * direction if prev_direction != direction else 0
            qty_opening = pos_change if prev_direction == direction else pos_change - qty_closing

            if not np.isclose(qty_closing, 0.0):
                deal_pnl = qty_closing * self._qty_multiplier * (self.position_avg_price - exec_price)

            # - update only realized pnl (in case we need to update by staled deals)
            if realize_only:
                self.r_pnl += deal_pnl / conversion_rate
                comms = fee_amount / conversion_rate
                self.commissions += comms
                return deal_pnl, comms

            # - use the same flatness predicate the halves below use, so episode stamping / fee
            #   attribution never disagree with the size mutation about which halves are present
            has_closing = not np.isclose(qty_closing, 0.0)
            has_opening = not np.isclose(qty_opening, 0.0)

            # - fee attribution: when a single deal both closes and opens (a sign flip) the fee is split
            #   pro-rata by |closing| : |opening|; otherwise the whole fee goes to the single side present.
            abs_c, abs_o = abs(qty_closing), abs(qty_opening)
            if has_closing and has_opening:
                fee_closing = fee_amount * abs_c / (abs_c + abs_o)
            elif has_closing:
                fee_closing = fee_amount
            else:
                fee_closing = 0.0
            fee_opening = fee_amount - fee_closing

            # - apply the realized close to the size
            if has_closing:
                self.__pos_incr_qty -= abs(qty_closing)
                quantity += qty_closing

                # - reset average price to 0 if position is fully closed
                # Use the rounded target position to avoid floating-point false positives
                if abs(position) < self.instrument.lot_size:
                    quantity = 0.0
                    self.position_avg_price = 0.0
                    self.__pos_incr_qty = 0

            # - if it has something to add to position let's update price and cost
            if has_opening:
                _abs_qty_open = abs(qty_opening)

                pos_avg_price_raw = (_abs_qty_open * exec_price + self.__pos_incr_qty * self.position_avg_price) / (
                    self.__pos_incr_qty + _abs_qty_open
                )

                # - round position average price to be in line with how it's calculated by broker
                self.position_avg_price = self.instrument.round_price_down(pos_avg_price_raw)
                self.__pos_incr_qty += _abs_qty_open

            # - update position and position's price
            self.position_avg_price_funds = self.position_avg_price / conversion_rate
            self.quantity = position

            # - the closing realization and its share of the fee belong to the OLD episode: book them
            #   into the lifetime accumulators *before* stamping the new baselines.
            self.r_pnl += deal_pnl / conversion_rate
            self.commissions += fee_closing / conversion_rate

            # - episode stamping: a flat->open transition opens an episode; a sign flip always re-stamps
            #   (its closing half was just attributed to the old episode above). The opening deal's own
            #   fee (booked below) then lands INSIDE the new episode.
            opens_episode = (not was_open and has_opening) or (has_closing and has_opening)
            if opens_episode:
                self.episode_start_time = _as_dt64_or_nat(time_as_nsec(timestamp))
                self.r_pnl_at_open = self.r_pnl
                self.commissions_at_open = self.commissions
                self.cumulative_funding_at_open = self.cumulative_funding

            # - the opening half's fee lands INSIDE the (possibly new) episode
            self.commissions += fee_opening / conversion_rate

            # - update pnl
            self.update_market_price(time_as_nsec(timestamp), exec_price, conversion_rate)

            # - total transaction cost of this deal in funds currency (return contract: whole-deal fee)
            comms = fee_amount / conversion_rate

        return deal_pnl, comms

    def update_market_price_by_tick(self, tick: Quote | Trade, conversion_rate: float = 1) -> float:
        return self.update_market_price(tick.time, self._price(tick), conversion_rate, stamp_update_time=False)

    def update_position_by_deal(
        self, deal: Deal, conversion_rate: float = 1, *, realize_only: bool = False
    ) -> tuple[float, float]:
        time = deal.time.as_unit("ns").asm8 if isinstance(deal.time, pd.Timestamp) else deal.time
        return self.change_position_by(
            timestamp=time,
            amount=deal.amount,
            exec_price=deal.price,
            fee_amount=deal.fee_amount or 0,
            conversion_rate=conversion_rate,
            realize_only=realize_only,
        )
        # - deal contains cumulative amount
        # return self.update_position(time, deal.amount, deal.price, deal.aggressive, conversion_rate)

    def update_market_price(
        self, timestamp: dt_64, price: float, conversion_rate: float, *, stamp_update_time: bool = True
    ) -> float:
        # stamp_update_time=False for mark-only ticks (quotes): they refresh the mark/PnL but must
        # NOT move last_update_time, which tracks the venue SIZE/state update (deal / snapshot) so
        # the reconciler's monotonic position guard works (a quote tick is not a venue size change).
        if stamp_update_time:
            # - always store a dt_64 venue timestamp (the deal path passes int-ns via
            #   time_as_nsec, ticks pass int-ns tick.time) so it stays monotonic-comparable
            #   with snapshot stamps and serializes as ISO, not a raw nanosecond integer
            self.last_update_time = (
                timestamp if isinstance(timestamp, np.datetime64) else np.datetime64(int(timestamp), "ns")
            )  # type: ignore
        self.last_update_price = price
        self.last_update_conversion_rate = conversion_rate

        if not np.isnan(price):
            u_pnl = self.unrealized_pnl()
            # r_pnl already includes cumulative funding
            self.pnl = u_pnl + self.r_pnl
            if self.instrument.is_futures():
                # for derivatives market value of the position is the current unrealized PnL
                self.market_value = u_pnl
            else:
                # for spot: market value is the current value of the position
                # TODO: implement market value calculation for margin
                self.market_value = self.quantity * self.last_update_price * self._qty_multiplier

            # calculate mkt value in funded currency
            self.market_value_funds = self.market_value / conversion_rate

            # - update margin requirements
            self._update_initial_margin()
            self._update_maint_margin()

        return self.pnl

    def unrealized_pnl(self) -> float:
        if not np.isnan(self.last_update_price):
            return (
                self.quantity
                * self._qty_multiplier
                * (self.last_update_price - self.position_avg_price)
                / self.last_update_conversion_rate
            )  # type: ignore
        return 0.0

    def apply_funding_payment(self, time: dt_64, amount: float) -> float:
        """Book a settled funding cash delta: ``amount`` is the signed settle-currency
        cash received (+) or paid (−); a settlement is account truth even on a
        since-reduced/closed position."""
        self.cumulative_funding += amount
        self.r_pnl += amount
        self.pnl += amount
        self.last_funding_time = time
        return amount

    def get_realized_price_pnl(self) -> float:
        """
        Get the realized price PnL for this position excluding funding.
        """
        return self.r_pnl - self.cumulative_funding

    def get_total_price_pnl(self) -> float:
        """
        Get the price PnL for this position excluding funding.
        """
        return self.pnl - self.cumulative_funding

    def episode_pnl(self) -> float:
        """Total P&L of the current episode (realized-since-open + unrealized + funding)."""
        return self.pnl - self.r_pnl_at_open

    def episode_funding(self) -> float:
        """Funding accrued within the current episode."""
        return self.cumulative_funding - self.cumulative_funding_at_open

    def episode_commissions(self) -> float:
        """Commissions paid within the current episode."""
        return self.commissions - self.commissions_at_open

    def episode_price_pnl(self) -> float:
        """Episode P&L excluding funding (mirrors get_total_price_pnl)."""
        return self.episode_pnl() - self.episode_funding()

    def episode_net_pnl(self) -> float:
        """Episode P&L with entry/exit costs included (the all-in figure)."""
        return self.episode_pnl() - self.episode_commissions()

    def is_open(self) -> bool:
        return abs(self.quantity) >= self.instrument.lot_size

    def get_amount_released_funds_after_closing(self, to_remain: float = 0.0) -> float:
        """
        Estimate how much funds would be released if part of position closed
        """
        d = np.sign(self.quantity)
        funds_release = self.market_value_funds
        if to_remain != 0 and self.quantity != 0 and np.sign(to_remain) == d:
            qty_to_release = max(self.quantity - to_remain, 0) if d > 0 else min(self.quantity - to_remain, 0)
            funds_release = (
                qty_to_release * self._qty_multiplier * self.last_update_price / self.last_update_conversion_rate
            )
        return abs(funds_release)

    @staticmethod
    def _t2s(t) -> str:
        return (
            np.datetime64(t, "ns").astype("datetime64[ms]").item().strftime("%Y-%m-%d %H:%M:%S")
            if not np.isnan(t)
            else "???"
        )

    def __str__(self):
        return " ".join(
            [
                f"{self._t2s(self.last_update_time)}",
                f"[{self.instrument}]",
                f"qty={self.quantity:.{self.instrument.size_precision}f}",
                f"entryPrice={self.position_avg_price:.{self.instrument.price_precision}f}",
                f"price={self.last_update_price:.{self.instrument.price_precision}f}",
                f"PNL: (unrealized={self.unrealized_pnl():.2f}",
                f"realized={self.r_pnl:.2f}",
                f"pnl={self.pnl:.2f})",
                f"value={self.market_value_funds:.2f}",
            ]
        )

    def __repr__(self):
        return self.__str__()

    def set_external_maint_margin(self, value: float) -> None:
        """
        Set maintenance margin from external source (exchange API).

        When set externally, the margin value won't be recalculated on price updates.
        This is used for live trading where exchanges provide accurate tiered margin values.

        Args:
            value: Maintenance margin value from exchange
        """
        self.maint_margin = value
        self._maint_margin_external = True

    def set_external_initial_margin(self, value: float) -> None:
        """
        Set initial margin from external source (exchange API).

        When set externally, the margin value won't be recalculated on price updates.
        Live exchanges report the actual margin reserved for the position
        (which depends on the per-instrument leverage tier and margin mode);
        we trust that value over any framework computation.

        Args:
            value: Initial margin value from exchange
        """
        self.initial_margin = value
        self._initial_margin_external = True

    def _update_maint_margin(self) -> None:
        # Skip recalculation if margin is managed externally (live trading with exchange-provided values)
        if self._maint_margin_external:
            return

        # Only apply maintenance margin for leveraged instruments (futures/swaps)
        # Spot positions don't have margin requirements since you own the actual asset
        if self.instrument.is_futures():
            maint_margin = self.instrument.maint_margin or DEFAULT_MAINTENANCE_MARGIN
            self.maint_margin = maint_margin * abs(self.quantity) * self._qty_multiplier * self.last_update_price
        else:
            self.maint_margin = 0.0

    def _update_initial_margin(self) -> None:
        # Skip recalculation if margin is managed externally (live trading with exchange-provided values)
        if self._initial_margin_external:
            return

        # Only apply initial margin for leveraged instruments (futures/swaps).
        # Use the per-asset initial_margin fraction from instrument metadata when
        # populated; otherwise leave at 0.0 (the framework can't infer a sensible
        # default without the per-instrument leverage setting, which lives on
        # the account processor).
        if self.instrument.is_futures() and self.instrument.initial_margin > 0:
            self.initial_margin = (
                self.instrument.initial_margin * abs(self.quantity) * self._qty_multiplier * self.last_update_price
            )
        else:
            self.initial_margin = 0.0


class CtrlChannel:
    """Unbounded control channel: producers push events on, the strategy thread drains.

    Sends are non-blocking and never drop — the queue is normally near-empty (events are
    consumed as fast as they arrive). If the consumer stalls, the queue grows rather than
    blocking the producer (the connector loop).
    """

    control: Event
    _queue: Queue  # we need something like disruptor here (Queue is temporary)
    name: str

    def __init__(self, name: str, sentinel=(None, None, None, None)):
        self.name = name
        self.control = Event()
        self._sent = sentinel
        self._queue = Queue()
        self.start()

    def register(self, callback):
        pass

    def stop(self):
        if self.control.is_set():
            self.control.clear()
            self._queue.put_nowait(self._sent)  # wake the consumer with the sentinel

    def start(self):
        self.control.set()

    def send(self, data):
        if not self.control.is_set():
            return
        self._queue.put_nowait(data)  # unbounded: never blocks, never drops

    def receive(self, timeout: int | None = None) -> Any:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            raise QueueTimeout(f"Timeout waiting for data on {self.name} channel")


class DataType(StrEnum):
    """
    Data type constants. Used for specifying the type of data and can be used for subscription to.
    Special value `DataType.ALL` can be used to subscribe to all available data types
    that are currently in use by the broker for other instruments.
    """

    ALL = "__all__"
    NONE = "__none__"
    QUOTE = "quote"
    TRADE = "trade"
    OHLC = "ohlc"
    ORDERBOOK = "orderbook"
    LIQUIDATION = "liquidation"
    AGGREGATED_LIQUIDATIONS = "aggregated_liquidations"
    FUNDING_RATE = "funding_rate"
    FUNDING_PAYMENT = "funding_payment"
    OPEN_INTEREST = "open_interest"
    OHLC_QUOTES = "ohlc_quotes"  # when we want to emulate quotes from OHLC data
    OHLC_TRADES = "ohlc_trades"  # when we want to emulate trades from OHLC data
    RECORD = "record"  # arbitrary timestamped data (actually liquidation and funding rates fall into this type)
    FUNDAMENTAL = "fundamental"  # fundamental data (with parameters)

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataType):
            return self.value == other.value
        return self.value == DataType.from_str(other)[0].value

    def __hash__(self) -> int:
        return hash(self.value)

    def __getitem__(self, *args, **kwargs) -> str:
        match self:
            case DataType.OHLC | DataType.OHLC_QUOTES:
                tf = args[0] if args else kwargs.get("timeframe")
                if not tf:
                    raise ValueError("Timeframe is not provided for OHLC subscription")
                return f"{self.value}({tf})"

            case DataType.AGGREGATED_LIQUIDATIONS:
                tf = args[0] if args else kwargs.get("timeframe")
                if not tf:
                    raise ValueError("Timeframe is not provided for AGGREGATED_LIQUIDATIONS")
                return f"{self.value}({tf})"

            case DataType.QUOTE:
                tf = args[0] if args else kwargs.get("timeframe")
                if tf:
                    return f"{self.value}({tf})"
                return self.value

            case DataType.ORDERBOOK:
                # Check if args is a tuple containing another tuple (the nested case)
                if len(args) == 1 and isinstance(args[0], tuple):
                    # Unpack the nested tuple
                    inner_args = args[0]
                    if len(inner_args) == 2:
                        tick_size_pct, depth = inner_args
                    else:
                        raise ValueError(f"Invalid arguments for ORDERBOOK subscription: {inner_args}")
                elif len(args) == 2:
                    tick_size_pct, depth = args
                elif len(args) > 0:
                    raise ValueError(f"Invalid arguments for ORDERBOOK subscription: {args}")
                else:
                    tick_size_pct = kwargs.get("tick_size_pct", 0.01)
                    depth = kwargs.get("depth", 200)
                return f"{self.value}({tick_size_pct}, {depth})"

            case DataType.FUNDING_RATE:
                if len(args) == 0:
                    return f"{self.value}"
                elif len(args) == 1:
                    inner_args = args[0]
                    if len(inner_args) == 1:
                        return f"{self.value}({inner_args[0]})"
                    elif len(inner_args) == 2:
                        return f"{self.value}({inner_args[0]}, {inner_args[1]})"
                    else:
                        raise ValueError(f"Invalid arguments for FUNDING_RATE subscription: {inner_args}")
                else:
                    raise ValueError(f"Invalid arguments for FUNDING_RATE subscription: {args}")

            case DataType.FUNDAMENTAL:
                if len(args) == 0:
                    return f"{self.value}"
                else:
                    inner_args = args[0]
                    return (
                        f"{self.value}({', '.join(map(str, *args))})"
                        if isinstance(inner_args, tuple)
                        else f"{self.value}({inner_args})"
                    )

            case _:
                return self.value

    @staticmethod
    @cache
    def from_str(value: "str | DataType") -> tuple["DataType", dict[str, Any]]:
        """
        Parse subscription type from string.
        Returns: (subtype, params)

        Example:
        >>> Subtype.from_str("ohlc(1Min)")
        (Subtype.OHLC, {"timeframe": "1Min"})

        >>> Subtype.from_str("orderbook(0.01, 100)")
        (Subtype.ORDERBOOK, {"tick_size_pct": 0.01, "depth": 100})

        >>> Subtype.from_str("quote")
        (Subtype.QUOTE, {})

        >>> Subtype.from_str("funding_rate(all)")
        (Subtype.FUNDING_RATE, {"__all__": True})
        """
        if isinstance(value, DataType):
            return value, {}
        try:
            _value = value.lower()
            _has_params = DataType._str_has_params(value)
            if not _has_params and value.upper() not in DataType.__members__:
                return DataType.NONE, {}
            elif not _has_params:
                return DataType(_value), {}
            else:
                type_name, params_str = value.split("(", 1)
                params = [p.strip() for p in params_str.rstrip(")").split(",")]
                match type_name.lower():
                    case DataType.OHLC.value:
                        return DataType.OHLC, {"timeframe": time_delta_to_str(to_timedelta(params[0]).asm8.item())}

                    case DataType.AGGREGATED_LIQUIDATIONS.value:
                        return DataType.AGGREGATED_LIQUIDATIONS, {
                            "timeframe": time_delta_to_str(to_timedelta(params[0]).asm8.item())
                        }

                    case DataType.OHLC_QUOTES.value:
                        return DataType.OHLC_QUOTES, {
                            "timeframe": time_delta_to_str(to_timedelta(params[0]).asm8.item())
                        }

                    case DataType.OHLC_TRADES.value:
                        return DataType.OHLC_TRADES, {
                            "timeframe": time_delta_to_str(to_timedelta(params[0]).asm8.item())
                        }

                    case DataType.QUOTE.value:
                        return DataType.QUOTE, {"timeframe": time_delta_to_str(to_timedelta(params[0]).asm8.item())}

                    case DataType.ORDERBOOK.value:
                        if len(params) == 1 and not params[0].replace(".", "").isdigit():
                            return DataType.ORDERBOOK, {
                                "timeframe": time_delta_to_str(to_timedelta(params[0]).asm8.item())
                            }
                        return DataType.ORDERBOOK, {"tick_size_pct": float(params[0]), "depth": int(params[1])}

                    case DataType.FUNDING_RATE.value:
                        if len(params) == 1 and params[0] == "all":
                            return DataType.FUNDING_RATE, {"__all__": True}
                        elif len(params) == 2 and params[0] == "all":
                            return DataType.FUNDING_RATE, {"__all__": True, "poll_interval_minutes": int(params[1])}
                        else:
                            raise ValueError(f"Invalid arguments for FUNDING_RATE subscription: {params}")

                    case DataType.FUNDAMENTAL.value:
                        if len(params) > 0:
                            return DataType.FUNDAMENTAL, {"fields": params}
                        return DataType.FUNDAMENTAL, {}

                    case _:
                        return DataType.NONE, {}
        except IndexError:
            raise ValueError(f"Invalid subscription type: {value}")

    @staticmethod
    def _str_has_params(value: str) -> bool:
        return "(" in value


class LiveTimeProvider(ITimeProvider):
    def __init__(self):
        self._start_ntp_thread()

    def time(self) -> dt_64:
        return time_now()

    def _start_ntp_thread(self):
        start_ntp_thread()


@dataclass
class RestoredState:
    """
    Container for state information needed to restart a strategy.

    This includes the current time, signals by instrument, and positions.
    """

    time: np.datetime64
    balances: list[Balance]
    instrument_to_signal_positions: dict[Instrument, list[Signal]]
    instrument_to_target_positions: dict[Instrument, list[TargetPosition]]
    positions: dict[Instrument, Position]


class InstrumentsLookup:
    def get_lookup(self) -> dict[str, Instrument]: ...

    def find(
        self,
        exchange: str,
        base: str,
        quote: str | None = None,
        settle: str | None = None,
        market_type: MarketType | None = None,
    ) -> Instrument | None:
        for i in self.get_lookup().values():
            if (
                i.exchange == exchange
                and (
                    (
                        quote is not None
                        and ((i.base == base and i.quote == quote) or (i.base == quote and i.quote == base))
                    )
                    or (quote is None and i.base == base)
                )
                and (market_type is None or i.market_type == market_type)
            ):
                if settle is not None and i.settle is not None:
                    if i.settle == settle:
                        return i
                else:
                    return i
        return None

    def find_symbol(self, exchange: str, symbol: str, market_type: MarketType | None = None) -> Instrument | None:
        for i in self.get_lookup().values():
            if (
                (i.exchange == exchange)
                and (i.symbol == symbol)
                and (market_type is None or i.market_type == market_type)
            ):
                return i

        return None

    def find_instruments(
        self,
        exchange: str,
        base: str | None = None,
        quote: str | None = None,
        market_type: MarketType | None = None,
        as_of: str | pd.Timestamp | None = None,
    ) -> list[Instrument]:
        """
        Find instruments by exchange, quote, market type and as of date.
        If as_of is not None, then only instruments that are not delisted after as_of date will be returned.
        - exchange: str - exchange name
        - base: str | None - base currency
        - quote: str | None - quote currency
        - market_type: MarketType | None - market type
        - as_of is a string in format YYYY-MM-DD or pd.Timestamp or None
        """
        _limit_time = pd.Timestamp(as_of) if as_of else None
        return [
            i
            for i in self.get_lookup().values()
            if i.exchange == exchange
            and (
                base is None or (i.base == base or i.base == f"1000{base}")
            )  # this is a hack to support 1000DOGEUSDT and others
            and (quote is None or i.quote == quote)
            and (market_type is None or i.market_type == market_type)
            and (
                _limit_time is None
                or (i.onboard_date is None or pd.Timestamp(i.onboard_date).tz_localize(None) <= _limit_time)
            )
            and (
                _limit_time is None
                or (i.delist_date is None or pd.Timestamp(i.delist_date).tz_localize(None) >= _limit_time)
            )
        ]

    def find_aux_instrument_for(
        self, instrument: Instrument, base_currency: str, market_type: MarketType | None = None
    ) -> Instrument | None:
        """
        Tries to find aux instrument (for conversions to funded currency)
        for example:
            ETHBTC -> BTCUSDT for base_currency USDT
            EURGBP -> GBPUSD for base_currency USD
            ...
        """
        if market_type is None:
            market_type = instrument.market_type
        base_currency = base_currency.upper()
        if instrument.quote != base_currency:
            return self.find(instrument.exchange, instrument.quote, base_currency, market_type=market_type)
        return None

    def __getitem__(self, spath: str) -> list[Instrument]:
        """
        Helper method for finding instruments by pattern.
        It's convenient to use in research mode.
        """
        res = []
        c = re.compile(spath)
        for k, v in self.get_lookup().items():
            if re.match(c, k):
                res.append(v)
        return res


class FeesLookup:
    def find_fees(self, exchange: str, spec: str | None) -> TransactionCostsCalculator: ...


class AccountsLookup:
    def get_credentials(self, exchange: str): ...
    def get_settings(self, exchange: str): ...
