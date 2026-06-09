from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.account_manager.config import AccountManagerConfig
from qubx.core.account_manager.reducer import ApplyResult
from qubx.core.account_manager.state import AccountState, VenueAccountFigures
from qubx.core.account_manager.state_machine import can_transition, validate_transition
from qubx.core.basics import (
    ZERO_COSTS,
    Balance,
    Deal,
    Instrument,
    ITimeProvider,
    Order,
    OrderChange,
    OrderOrigin,
    OrderStatus,
    OrderTransition,
    Position,
    TransactionCostsCalculator,
)
from qubx.core.connector import IConnector
from qubx.core.events import (
    AccountMessage,
    AccountSnapshotEvent,
    BalanceUpdateEvent,
    FundingPaymentEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
    PositionUpdateEvent,
)
from qubx.core.exceptions import InvalidOrderTransition
from qubx.core.interfaces import IProcessingManager, IStrategy
from qubx.utils.time import timedelta_to_crontab

# StrategyContext and ProcessingManager both import AccountManager (the context builds it,
# the PM holds/schedules for it), so importing them here at runtime is a circular import —
# they're type-only forward refs. IStrategy has no such cycle and is imported normally above.
if TYPE_CHECKING:
    from qubx.core.context import StrategyContext
    from qubx.core.mixins.processing import ProcessingManager

# Client-id prefix that marks an order as framework-originated. MUST match the prefix
# ClientIdStore._create_id produces in qubx.core.mixins.trading ("qubx_<symbol>_<n>");
# a snapshot order carrying it is a RECOVERED framework order, anything else is EXTERNAL.
_FRAMEWORK_CID_PREFIX = "qubx_"


class AccountManager:
    _pm: IProcessingManager | None
    _connectors: dict[str, IConnector]
    _strategy: IStrategy
    _time: ITimeProvider
    _cfg: AccountManagerConfig
    account_id: str
    _tcc: TransactionCostsCalculator | None
    _states: dict[str, AccountState]
    _ctx: "StrategyContext"
    _handlers: dict[type[AccountMessage], Callable[..., ApplyResult]]
    # Derived timedeltas, precomputed once in _init_state (config is fixed for the AM's life).
    _snapshot_grace: np.timedelta64
    _snapshot_interval: np.timedelta64
    _inflight_threshold: np.timedelta64
    _liveness_threshold: np.timedelta64
    _terminal_retention: np.timedelta64
    _liveness_unready_since: dict[str, np.datetime64]
    # Per-exchange dedup of applied funding buckets, bounded LRU-style (insertion order ≈
    # funding-event time order): old buckets evict once the cap is hit so the set can't grow
    # unbounded over long-running sessions. A re-delivered funding event only needs RECENT
    # buckets to dedup against.
    _FUNDING_BUCKET_CAP: int = 4096
    _applied_funding_buckets: "dict[str, OrderedDict[tuple, None]]"

    def __init__(
        self,
        *,
        pm: "ProcessingManager | None" = None,
        connectors: dict[str, IConnector],
        base_currencies: dict[str, str],
        strategy: IStrategy,
        time: ITimeProvider,
        cfg: AccountManagerConfig | None = None,
        account_id: str = "AccountManager",
        tcc: TransactionCostsCalculator | None = None,
    ):
        self._pm = pm
        self._init_state(
            connectors=connectors,
            base_currencies=base_currencies,
            strategy=strategy,
            time=time,
            cfg=cfg,
            account_id=account_id,
            tcc=tcc,
        )
        # The live runner builds the AM before the ProcessingManager (which lives inside
        # StrategyContext and needs the AM), so pm is None there — ticks register later in
        # set_context once ctx._processing_manager exists. Direct-construction callers
        # (tests) still pass pm and get ticks immediately.
        if self._pm is not None:
            self._register_ticks()

    def set_context(self, ctx: "StrategyContext") -> None:
        """Wire the IStrategyContext after construction.

        AM-fired callbacks (reconcile, inflight-exhaustion) pass this ctx so their
        signature matches PM-fired callbacks — no None placeholder.

        Live: the AM is built before the ProcessingManager (the AM↔pm cycle — pm lives
        inside StrategyContext, which takes the AM), so pm is None at construction and
        the periodic ticks register HERE, once ctx._processing_manager exists.
        """
        self._ctx = ctx
        if self._pm is None:
            self._pm = ctx._processing_manager
            self._register_ticks()

    def add_order(self, order: Order) -> None:
        # exchange is derived from the order's instrument — no separate parameter needed.
        self._states[order.instrument.exchange].add_order(order)

    def transition_order(self, exchange: str, cid: str, new_status: OrderStatus) -> None:
        state = self._states[exchange]
        order = state.get_active_order(cid)
        if order is None:
            raise KeyError(f"order {cid} not found in {exchange}")
        validate_transition(cid, order.status, new_status)
        state.transition_order(cid, new_status, self._time.time())

    def remove_order(self, exchange: str, cid: str) -> None:
        # Drop an order from state entirely (e.g. a submit that raised before reaching the
        # venue) — distinct from a terminal transition, which keeps it in history.
        self._states[exchange].remove_order(cid)

    def get_state(self, exchange: str) -> AccountState:
        return self._states[exchange]

    def get_orders(
        self,
        instrument: Instrument | None = None,
        exchange: str | None = None,
        origin: OrderOrigin | None = None,
    ) -> dict[str, Order]:
        if exchange is not None:
            orders = self._states[exchange].get_orders()
        else:
            orders = {cid: o for s in self._states.values() for cid, o in s.get_orders().items()}
        # Public "open orders" view: terminal orders are retained in active_orders
        # during the grace window for late-event resolution, but callers reading
        # get_orders() want only live orders.
        orders = {cid: o for cid, o in orders.items() if not o.status.is_terminal}
        if instrument is not None:
            orders = {cid: o for cid, o in orders.items() if o.instrument == instrument}
        if origin is not None:
            orders = {cid: o for cid, o in orders.items() if o.origin == origin}
        return orders

    def get_order(self, client_order_id: str) -> Order | None:
        for state in self._states.values():
            if (o := state.get_order(client_order_id)) is not None:
                return o
        return None

    def get_position(self, instrument):
        state = self._states.get(instrument.exchange)
        if state is None:
            return None
        pos = state.get_position(instrument)
        if pos is None:
            # Consumers (gatherers, trackers, sizers, ctx.positions[instrument]) expect
            # a Position object for any known instrument, not None. Materialize an empty
            # one so a never-traded instrument reads as flat rather than KeyError-ing.
            pos = Position(instrument=instrument)
            state.set_position(instrument, pos)
        return pos

    def get_positions(self, exchange: str | None = None) -> dict[Instrument, Position]:
        if exchange is not None:
            return self._states[exchange].get_positions()
        return {ins: pos for s in self._states.values() for ins, pos in s.get_positions().items()}

    @property
    def positions(self) -> dict[Instrument, Position]:
        return self.get_positions()

    def get_balance(self, currency: str, exchange: str | None = None):
        if exchange is not None:
            return self._states[exchange].get_balance(currency)
        for state in self._states.values():
            if (b := state.get_balance(currency)) is not None:
                return b
        return None

    def get_balances(self, exchange: str | None = None) -> list[Balance]:
        if exchange is not None:
            return self._states[exchange].get_balances()
        return [b for s in self._states.values() for b in s.get_balances()]

    def get_fees_calculator(self, exchange: str | None = None) -> TransactionCostsCalculator:
        # `exchange` is accepted for IStrategyContext symmetry but unused: the AM models a
        # single TCC that applies to every exchange (no per-exchange fee modeling).
        # No TCC configured => zero-fee calculator (rather than raising), so callers that
        # only need fee arithmetic still work in fee-agnostic setups.
        return self._tcc if self._tcc is not None else ZERO_COSTS

    def find_order_by_id(self, order_id: str) -> Order | None:
        for state in self._states.values():
            if (o := state.get_order_by_venue_id(order_id)) is not None:
                return o
        return None

    def find_order_by_client_id(self, client_id: str) -> Order | None:
        return self.get_order(client_id)

    def get_order_history(self, client_order_id: str) -> list[OrderTransition]:
        """The status-transition audit trail for an order — searches active orders then the
        terminal-history ring buffer; empty if the order is unknown or already evicted."""
        order = self.get_order(client_order_id)
        return list(order.transitions) if order is not None else []

    def get_metrics(self) -> dict[str, dict[str, int]]:
        """Per-exchange audit counters: exchange -> {order-status: transitions into it}.
        A pull-based hook for emitters/dashboards; never reset within a session."""
        return {exchange: state.get_transition_counts() for exchange, state in self._states.items()}

    def position_report(self, exchange: str | None = None) -> dict:
        report = {}
        for pos in self.get_positions(exchange).values():
            report[pos.instrument.symbol] = {
                "Qty": pos.quantity,
                "Price": pos.position_avg_price_funds,
                "PnL": pos.pnl,
                "MktValue": pos.market_value_funds,
                "Leverage": self.get_leverage(pos.instrument),
            }
        return report

    def apply(self, event: AccountMessage) -> ApplyResult:
        state = self._get_state_for_event(event)
        if state is None:
            return ApplyResult()
        handler = self._handlers.get(type(event))
        if handler is None:
            logger.warning(f"unhandled AccountMessage: {type(event)}")
            return ApplyResult()
        return handler(state, event)

    def _handle_position_balance_noop(self, state: AccountState, event: AccountMessage) -> ApplyResult:
        # No connector emits PositionUpdate/BalanceUpdate yet; positions/balances are derived
        # from fills and corrected by snapshot reconcile. PM still fires on_account_update
        # off the event payload.
        # TODO(account-mgmt): apply venue WS position/balance to AccountState here (via
        # set_position/update_balance) once the live connectors emit them, with the same
        # freshness/ratchet guard as snapshot reconcile.
        return ApplyResult()

    def on_market_quote(self, instrument, quote) -> None:
        state = self._states.get(instrument.exchange)
        if state is None:
            return
        pos = state.get_position(instrument)
        if pos is None:
            return
        # Position.update_market_price expects (timestamp, price, conversion_rate);
        # mark-to-market uses the quote mid.
        pos.update_market_price(self._time.time(), quote.mid_price(), state.conversion_rate(instrument))

    # ---- metrics (aggregated) ------------------------------------------ #
    # Per-exchange math lives on AccountState; the manager only aggregates
    # across exchanges.

    def _sum(self, metric: Callable[[AccountState], float], exchange: str | None) -> float:
        if exchange is not None:
            return metric(self._states[exchange])
        return sum(metric(s) for s in self._states.values())

    def _states_for(self, exchange: str | None) -> list[AccountState]:
        return [self._states[exchange]] if exchange is not None else list(self._states.values())

    def get_total_capital(self, exchange: str | None = None) -> float:
        return self._sum(AccountState.total_capital, exchange)

    def get_available_margin(self, exchange: str | None = None) -> float:
        return self._sum(AccountState.available_margin, exchange)

    def get_total_initial_margin(self, exchange: str | None = None) -> float:
        return self._sum(AccountState.total_initial_margin, exchange)

    def get_total_maint_margin(self, exchange: str | None = None) -> float:
        return self._sum(AccountState.total_maint_margin, exchange)

    def get_margin_ratio(self, exchange: str | None = None) -> float:
        states = self._states_for(exchange)
        if len(states) == 1:
            # single state: state.margin_ratio() so the venue-reported ratio is preferred
            return states[0].margin_ratio()
        # cross-exchange: no venue reports a combined ratio — derive from the sums
        # (total_capital still prefers venue equity per state)
        maint = sum(s.total_maint_margin() for s in states)
        if maint == 0:
            return 100.0
        return min(100.0, sum(s.total_capital() for s in states) / maint)

    def get_base_currency(self, exchange: str | None = None) -> str:
        state = self._states[exchange] if exchange is not None else next(iter(self._states.values()))
        return state.base_currency

    def get_leverage(self, instrument: Instrument) -> float:
        state = self._states.get(instrument.exchange)
        return state.leverage(instrument) if state is not None else 0.0

    def get_net_leverage(self, exchange: str | None = None) -> float:
        return self._aggregate_leverage(exchange, AccountState.net_leverage)

    def get_gross_leverage(self, exchange: str | None = None) -> float:
        return self._aggregate_leverage(exchange, AccountState.gross_leverage)

    def _aggregate_leverage(self, exchange: str | None, per_state: Callable[[AccountState], float]) -> float:
        # capital-weighted: leverage is notional/capital, so recover Σnotional = Σ(lev * capital)
        states = self._states_for(exchange)
        total_capital = sum(s.total_capital() for s in states)
        if total_capital <= 0:
            return 0.0
        return sum(per_state(s) * s.total_capital() for s in states) / total_capital

    def get_leverages(self, exchange: str | None = None) -> dict[Instrument, float]:
        return {ins: self.get_leverage(ins) for ins in self.get_positions(exchange)}

    # Per-instrument exchange-side settings: the venue's configured leverage tier,
    # hard caps and margin mode. Simulation does not model margin, so these return
    # neutral values; the real settings live on the venue.
    # TODO(account-mgmt): back these with venue-sourced leverage/margin/mode held in
    # AccountState once the margin-aware live connectors are wired.
    def get_instrument_leverage(self, instrument: Instrument) -> float | None:
        return None

    def get_max_instrument_leverage(self, instrument: Instrument) -> float | None:
        return None

    def get_max_instrument_notional(self, instrument: Instrument) -> float:
        return float("inf")

    def get_margin_mode(self, instrument: Instrument) -> Literal["cross", "isolated"] | None:
        return None

    def get_adl_level(self, instrument: Instrument) -> int | None:
        pos = self.get_position(instrument)
        return pos.adl_level if pos is not None else None

    def get_reserved(self, instrument: Instrument) -> float:
        return 0.0

    def _init_state(
        self,
        *,
        connectors: dict[str, IConnector],
        base_currencies: dict[str, str],
        strategy: IStrategy,
        time: ITimeProvider,
        cfg: AccountManagerConfig | None,
        account_id: str,
        tcc: TransactionCostsCalculator | None,
    ) -> None:
        # Shared field init for both the live and simulation managers, so the
        # subclass can't silently drift from the parent's field set.
        self._connectors = connectors
        self._strategy = strategy
        self._time = time
        self._cfg = cfg or AccountManagerConfig()
        self.account_id = account_id
        self._tcc = tcc
        # Derived timedeltas (config is fixed for the AM's lifetime) — computed once
        # here rather than rebuilt on every tick/snapshot.
        self._snapshot_grace = np.timedelta64(self._cfg.snapshot_check_threshold_ms, "ms")
        self._snapshot_interval = np.timedelta64(self._cfg.snapshot_check_interval_ms, "ms")
        self._inflight_threshold = np.timedelta64(self._cfg.inflight_check_threshold_ms, "ms")
        self._liveness_threshold = np.timedelta64(self._cfg.liveness_check_threshold_ms, "ms")
        self._terminal_retention = np.timedelta64(self._cfg.terminal_order_retention_ms, "ms")
        # base_currency is explicit, per exchange — resolved from config at the runner
        # boundary, never inferred from balances inside the AM.
        self._states = {
            ex: AccountState(
                exchange=ex,
                base_currency=base_currencies[ex],
                terminal_history_size=self._cfg.terminal_order_history_size,
            )
            for ex in connectors
        }
        self._liveness_unready_since = {}
        # Bounded per-exchange funding-bucket dedup (see _FUNDING_BUCKET_CAP); old
        # buckets evict in _handle_funding_payment so this can't grow unbounded.
        self._applied_funding_buckets = {}
        # Deferred init: always wired via set_context before any tick/callback fires.
        self._ctx = None  # type: ignore[assignment]
        # Event-type → handler dispatch table (consumed by apply()). All handlers in one
        # place; dispatch is an O(1) lookup on the exact event type.
        self._handlers = {
            OrderPartiallyFilledEvent: self._handle_partial_fill,
            OrderFilledEvent: self._handle_fill,
            OrderAcceptedEvent: self._handle_accepted,
            OrderCanceledEvent: self._handle_canceled,
            OrderExpiredEvent: self._handle_expired,
            OrderUpdatedEvent: self._handle_updated,
            OrderRejectedEvent: self._handle_rejected,
            OrderCancelRejectedEvent: self._handle_cancel_rejected,
            OrderUpdateRejectedEvent: self._handle_update_rejected,
            AccountSnapshotEvent: self._handle_snapshot,
            FundingPaymentEvent: self._handle_funding_payment,
            PositionUpdateEvent: self._handle_position_balance_noop,
            BalanceUpdateEvent: self._handle_position_balance_noop,
        }

    def _register_ticks(self) -> None:
        cfg = self._cfg
        if cfg.inflight_check_interval_ms > 0:
            self._pm.schedule(
                timedelta_to_crontab(pd.Timedelta(cfg.inflight_check_interval_ms, "ms")), self._on_inflight_tick
            )
        if cfg.snapshot_check_interval_ms > 0:
            self._pm.schedule(
                timedelta_to_crontab(pd.Timedelta(cfg.snapshot_check_interval_ms, "ms")), self._on_snapshot_tick
            )
        if cfg.liveness_check_interval_ms > 0:
            self._pm.schedule(
                timedelta_to_crontab(pd.Timedelta(cfg.liveness_check_interval_ms, "ms")), self._on_liveness_tick
            )

    def _transition(self, state: AccountState, cid: str, new_status: OrderStatus) -> Order:
        """Single validating chokepoint for every AM-driven status change.

        Raises InvalidOrderTransition on an illegal move (the PM dispatch logs + skips
        it; tick/snapshot callers guard or expect only legal moves). ``transition_order``
        is the low-level setter — the legality check lives here, in AccountManager.
        """
        order = state.get_active_order(cid)
        if order is None:
            raise KeyError(f"order {cid} not found in {state.exchange}")
        validate_transition(cid, order.status, new_status)
        return state.transition_order(cid, new_status, self._time.time())

    def _get_state_for_event(self, event: AccountMessage) -> AccountState | None:
        if isinstance(event, AccountSnapshotEvent):
            return self._states.get(event.snapshot.exchange)
        if event.instrument is not None:
            return self._states.get(event.instrument.exchange)
        # No instrument — route an order event by its identifiers so a reject emitted
        # without one (e.g. SimulatedConnector on update-of-missing-order) still reaches
        # the right state.
        if isinstance(event, OrderEvent):
            for state in self._states.values():
                if state.get_order(event.client_order_id) is not None:
                    return state
            if event.venue_order_id is not None:
                for state in self._states.values():
                    if state.get_order_by_venue_id(event.venue_order_id) is not None:
                        return state
        if len(self._states) == 1:
            return next(iter(self._states.values()))
        logger.debug(f"cannot route {type(event).__name__} with no instrument/identifiers — dropped")
        return None

    def _resolve(self, state: AccountState, event: OrderEvent) -> Order | None:
        # Known by cid (active or terminal-history) or by the venue id it was assigned.
        if (order := state.get_order(event.client_order_id)) is not None:
            return order
        if event.venue_order_id is not None:
            return state.get_order_by_venue_id(event.venue_order_id)
        return None

    def _resolve_or_materialize(self, state: AccountState, event: OrderEvent) -> Order | None:
        if (order := self._resolve(state, event)) is not None:
            return order
        if event.instrument is None:  # can't track a position without an instrument
            return None
        return self._materialize_external(state, event, event.instrument)

    def _materialize_external(self, state: AccountState, event: OrderEvent, instrument: Instrument) -> Order:
        # Unknown to us => external order (manual UI / another bot / pre-existing). A venue
        # lifecycle event always carries the id the venue assigned; fall back to the cid
        # for a stable identity only in the (malformed) case where it doesn't.
        venue_id = event.venue_order_id
        order = Order(
            client_order_id=f"ext:{venue_id or event.client_order_id}",
            venue_order_id=venue_id,
            origin=OrderOrigin.EXTERNAL,
            type="LIMIT",
            instrument=instrument,
            time=self._time.time(),
            quantity=0.0,
            price=0.0,
            side="BUY",
            status=OrderStatus.ACCEPTED,  # it exists at the venue
            time_in_force="gtc",
        )
        state.add_order(order)
        return order

    # A late event for an already-terminal order — and any accepted/canceled/expired event
    # for an order we don't know — is a benign no-op: the handlers below return an empty
    # ApplyResult (the suppress signal) so the ProcessingManager fires nothing. External
    # orders materialize only on money-carrying events (fill/partial-fill/updated).
    def _handle_accepted(self, state: AccountState, event: OrderAcceptedEvent) -> ApplyResult:
        order = self._resolve(state, event)
        if order is None:
            return ApplyResult()
        if order.status.is_terminal:
            # Late accept on an already-terminal order (design "OrderFilled before
            # OrderAccepted"): benign side-effect, no transition, no phantom. Set
            # the venue id ONLY if the order is still in active_orders — an evicted
            # order's venue-id index was already dropped, so set_venue_id would
            # KeyError on active_orders[cid].
            if state.has_active_order(order.client_order_id):
                if event.venue_order_id is not None:
                    state.set_venue_id(order.client_order_id, event.venue_order_id)
                order.accepted_at = event.accepted_at
            return ApplyResult()
        if event.venue_order_id is not None:
            state.set_venue_id(order.client_order_id, event.venue_order_id)
        order.accepted_at = event.accepted_at
        if order.status == OrderStatus.PENDING_CANCEL:
            # A late accept racing an outstanding cancel must NOT wipe PENDING_CANCEL:
            # the sweep keeps polling the cancel and a later cancel-rejected still reverts.
            return ApplyResult()
        if not can_transition(order.status, OrderStatus.ACCEPTED):
            return ApplyResult()
        order = self._transition(state, order.client_order_id, OrderStatus.ACCEPTED)
        return ApplyResult(order=order, order_change=OrderChange.ACCEPTED)

    def _handle_partial_fill(self, state: AccountState, event: OrderPartiallyFilledEvent) -> ApplyResult:
        order = self._resolve_or_materialize(state, event)
        if order is None or order.status.is_terminal:
            return ApplyResult()
        if event.venue_order_id is not None:
            state.set_venue_id(order.client_order_id, event.venue_order_id)
        new_deal = state.apply_fill(order.client_order_id, event.fill, self._time.time())
        deal = event.fill if new_deal else None
        position = (
            self._book_deal(state, order.instrument, deal)
            if deal is not None and order.instrument is not None
            else None
        )
        # a pending cancel/update is resolved by the venue separately — don't disturb its status
        pending = order.status in (OrderStatus.PENDING_CANCEL, OrderStatus.PENDING_UPDATE)
        if pending:
            # filled_quantity mirrors real, irreversible fills (and the position),
            # so it is NEVER reduced. A fill that races a pending modify can push it
            # past the new (smaller) target — surface that as a warning only; the
            # venue resolves the race (OrderUpdated, or OrderUpdateRejected because
            # it can't shrink an order below what's already filled).
            if order.status == OrderStatus.PENDING_UPDATE and order.filled_quantity > order.quantity:
                logger.warning(
                    f"[{order.client_order_id}] fill during pending-update pushed "
                    f"filled_quantity ({order.filled_quantity}) past target "
                    f"({order.quantity}); leaving filled intact, awaiting venue verdict"
                )
            return ApplyResult(deal=deal, position=position)
        if can_transition(order.status, OrderStatus.PARTIALLY_FILLED):
            order = self._transition(state, order.client_order_id, OrderStatus.PARTIALLY_FILLED)
            return ApplyResult(order=order, order_change=OrderChange.PARTIALLY_FILLED, deal=deal, position=position)
        return ApplyResult(deal=deal, position=position)  # no status change -> execution only

    def _handle_fill(self, state: AccountState, event: OrderFilledEvent) -> ApplyResult:
        order = self._resolve_or_materialize(state, event)
        if order is None or order.status.is_terminal:
            return ApplyResult()
        if event.venue_order_id is not None:
            state.set_venue_id(order.client_order_id, event.venue_order_id)
        new_deal = state.apply_fill(order.client_order_id, event.fill, self._time.time())
        deal = event.fill if new_deal else None
        position = (
            self._book_deal(state, order.instrument, deal)
            if deal is not None and order.instrument is not None
            else None
        )
        order = self._transition(state, order.client_order_id, OrderStatus.FILLED)
        return ApplyResult(order=order, order_change=OrderChange.FILLED, deal=deal, position=position)

    def _handle_canceled(self, state: AccountState, event: OrderCanceledEvent) -> ApplyResult:
        order = self._resolve(state, event)
        if order is None or order.status.is_terminal:
            return ApplyResult()
        order = self._transition(state, order.client_order_id, OrderStatus.CANCELED)
        return ApplyResult(order=order, order_change=OrderChange.CANCELED)

    def _handle_expired(self, state: AccountState, event: OrderExpiredEvent) -> ApplyResult:
        order = self._resolve(state, event)
        if order is None or order.status.is_terminal:
            return ApplyResult()
        order = self._transition(state, order.client_order_id, OrderStatus.EXPIRED)
        return ApplyResult(order=order, order_change=OrderChange.EXPIRED)

    def _active_order_for(self, state: AccountState, event: OrderEvent) -> Order | None:
        # Resolve an ACTIVE order by client id, then by venue id — so a reject addressed by
        # venue id alone (a venue-id-only cancel/update) still routes. Active-only (no
        # materialize): a reject has no order to create.
        order = state.get_active_order(event.client_order_id)
        if order is None and event.venue_order_id is not None:
            order = state.get_order_by_venue_id(event.venue_order_id)
        return order

    def _handle_rejected(self, state: AccountState, event: OrderRejectedEvent) -> ApplyResult:
        order = self._active_order_for(state, event)
        if order is None or order.status.is_terminal:
            return ApplyResult()
        order.rejected_reason = event.reason
        order = self._transition(state, order.client_order_id, OrderStatus.REJECTED)
        return ApplyResult(order=order, order_change=OrderChange.REJECTED)

    def _handle_updated(self, state: AccountState, event: OrderUpdatedEvent) -> ApplyResult:
        order = self._resolve_or_materialize(state, event)
        if order is None or order.status.is_terminal:
            return ApplyResult()
        if event.venue_order_id is not None and order.venue_order_id != event.venue_order_id:
            # set_venue_id re-keys internally: it drops the order's previous venue id.
            state.set_venue_id(order.client_order_id, event.venue_order_id)
        if event.new_price is not None:
            order.price = event.new_price
        if event.new_quantity is not None:
            order.quantity = event.new_quantity
        order.last_updated_at = self._time.time()
        if order.status == OrderStatus.PENDING_UPDATE:
            target = state.get_pre_pending(order.client_order_id) or OrderStatus.ACCEPTED
            order = self._transition(state, order.client_order_id, target)
        return ApplyResult(order=order, order_change=OrderChange.UPDATED)

    def _handle_cancel_rejected(self, state: AccountState, event: OrderCancelRejectedEvent) -> ApplyResult:
        order = self._active_order_for(state, event)
        if order is None or order.status != OrderStatus.PENDING_CANCEL:
            return ApplyResult()
        return self._revert_from_pending(state, order, OrderChange.CANCEL_REJECTED)

    def _handle_update_rejected(self, state: AccountState, event: OrderUpdateRejectedEvent) -> ApplyResult:
        order = self._active_order_for(state, event)
        if order is None or order.status != OrderStatus.PENDING_UPDATE:
            return ApplyResult()
        return self._revert_from_pending(state, order, OrderChange.UPDATE_REJECTED)

    def _revert_from_pending(self, state: AccountState, order: Order, change: OrderChange) -> ApplyResult:
        # Revert to the status captured on entry to PENDING_* — never inferred from
        # filled_quantity/venue_id (brittle when venues roll back partial fills). ACCEPTED
        # is the safe default for the rare order with no captured status. The transition
        # itself clears the capture (the target is non-pending).
        target = state.get_pre_pending(order.client_order_id) or OrderStatus.ACCEPTED
        order = self._transition(state, order.client_order_id, target)
        return ApplyResult(order=order, order_change=change)

    def _handle_snapshot(self, state: AccountState, event: AccountSnapshotEvent) -> ApplyResult:
        snapshot = event.snapshot
        if state.is_snapshot_stale(snapshot.as_of):
            return ApplyResult()
        state.mark_snapshot_applied(snapshot.as_of)

        # Reconcile mutates state silently; the strategy is notified once via on_account_update
        # (PM routes the AccountSnapshotEvent there) rather than per applied change. We keep a
        # debug-log tally for observability instead of a per-event callback.
        n_terminal = n_materialized = n_updated = n_positions = n_balances = 0

        if snapshot.open_orders is not None:
            snap_by_vid = {o.venue_order_id: o for o in snapshot.open_orders if o.venue_order_id}
            for cid, cached in state.get_orders().items():
                if cached.status.is_terminal:
                    continue
                vid = cached.venue_order_id
                if vid is not None and vid in snap_by_vid:
                    # Still open at the venue — property drift is reconciled in the
                    # open-orders loop below (_update_from_snapshot), not here.
                    continue
                if (snapshot.as_of - cached.time) < self._snapshot_grace:
                    continue
                terminal = OrderStatus.REJECTED if cached.status == OrderStatus.SUBMITTED else OrderStatus.CANCELED
                cached.rejected_reason = "reconcile: missing from snapshot"
                try:
                    self._transition(state, cid, terminal)
                    n_terminal += 1
                except InvalidOrderTransition:
                    logger.warning(f"reconcile: cannot terminate {cid} from {cached.status}")
            for snap_order in snapshot.open_orders:
                existing = state.get_order_by_venue_id(snap_order.venue_order_id) if snap_order.venue_order_id else None
                if existing is None:
                    self._materialize_from_snapshot(state, snap_order, snapshot.as_of)
                    n_materialized += 1
                elif existing.last_updated_at is None or snapshot.as_of > existing.last_updated_at:
                    self._update_from_snapshot(state, existing, snap_order, snapshot.as_of)
                    n_updated += 1

        # Positions and balances: the snapshot is the venue's authoritative full
        # truth for size/amount, and stale snapshots are already rejected wholesale
        # by the is_snapshot_stale ratchet above — so an accepted snapshot always
        # overwrites. No per-record freshness here (unlike orders, where it guards a
        # fresh fill): Position/Balance carry no reliable last-update timestamp yet.
        # TODO(account-mgmt): once WS PositionUpdate/BalanceUpdate events are wired,
        # add per-record freshness backed by a real timestamp so a snapshot older than
        # a recent WS update can't clobber it.
        if snapshot.positions is not None:
            for snap_pos in snapshot.positions:
                if state.apply_position_snapshot(snap_pos):
                    n_positions += 1

        if snapshot.balances is not None:
            for snap_bal in snapshot.balances:
                if state.apply_balance_snapshot(snap_bal):
                    n_balances += 1

        # Venue-reported account figures: prefer-venue-else-derive happens per metric in
        # AccountState. A snapshot with no figures (sim, or a failed balance leg) keeps the
        # previous capture rather than clearing — absence means "not observed", not "gone".
        if snapshot.equity is not None or snapshot.available_margin is not None or snapshot.margin_ratio is not None:
            state.set_venue_figures(
                VenueAccountFigures(
                    as_of=snapshot.as_of,
                    equity=snapshot.equity,
                    available_margin=snapshot.available_margin,
                    margin_ratio=snapshot.margin_ratio,
                )
            )

        if n_terminal or n_materialized or n_updated or n_positions or n_balances:
            logger.debug(
                f"[{snapshot.exchange}] reconcile applied: {n_terminal} terminated, "
                f"{n_materialized} materialized, {n_updated} updated, "
                f"{n_positions} positions, {n_balances} balances"
            )
        return ApplyResult()

    def _materialize_from_snapshot(self, state: AccountState, snap_order: Order, as_of: np.datetime64) -> None:
        # cid prefix classifies origin: our prefix → a recovered framework order;
        # anything else → external. Keep an already-synthesized ext: cid as-is,
        # otherwise synthesize one from the venue id.
        if snap_order.client_order_id.startswith(_FRAMEWORK_CID_PREFIX):
            origin = OrderOrigin.RECOVERED
            cid = snap_order.client_order_id
        elif snap_order.client_order_id.startswith("ext:"):
            origin = OrderOrigin.EXTERNAL
            cid = snap_order.client_order_id
        else:
            origin = OrderOrigin.EXTERNAL
            cid = f"ext:{snap_order.venue_order_id}"
        state.add_order(
            Order(
                client_order_id=cid,
                venue_order_id=snap_order.venue_order_id,
                origin=origin,
                type=snap_order.type,
                instrument=snap_order.instrument,
                time=snap_order.time,
                quantity=snap_order.quantity,
                price=snap_order.price,
                side=snap_order.side,
                status=snap_order.status,
                time_in_force=snap_order.time_in_force,
                filled_quantity=snap_order.filled_quantity,
                avg_fill_price=snap_order.avg_fill_price,
                last_updated_at=as_of,
            )
        )

    def _update_from_snapshot(
        self, state: AccountState, existing: Order, snap_order: Order, as_of: np.datetime64
    ) -> None:
        existing.status = snap_order.status
        existing.filled_quantity = snap_order.filled_quantity
        existing.avg_fill_price = snap_order.avg_fill_price
        existing.price = snap_order.price
        existing.quantity = snap_order.quantity
        existing.last_updated_at = as_of

    def _sweep_terminal_evictions(self) -> None:
        now = self._time.time()
        for state in self._states.values():
            state.prune_terminal_orders(now, self._terminal_retention)

    def _on_inflight_tick(self, ctx) -> None:
        now = self._time.time()
        for exchange, state in self._states.items():
            for order in state.get_inflight_orders():
                cid = order.client_order_id
                try:
                    last = order.last_updated_at or order.time
                    if (now - last) < self._inflight_threshold:
                        continue
                    if state.get_retry(cid) >= self._cfg.inflight_check_retries:
                        self._resolve_exhausted_inflight(state, exchange, order)
                    else:
                        self._connectors[exchange].request_order_status(
                            client_order_id=cid,
                            venue_order_id=order.venue_order_id,
                        )
                        state.bump_retry(cid)
                        order.last_updated_at = now
                except Exception:
                    # One bad order / raising strategy callback / connector error must not
                    # abort the rest of the sweep or skip terminal eviction (design §1260).
                    logger.exception(f"[{exchange}] inflight sweep failed for {cid}")
        self._sweep_terminal_evictions()

    def _resolve_exhausted_inflight(self, state: AccountState, exchange, order):
        reason = f"reconcile: no venue ack after {state.get_retry(order.client_order_id)} retries"
        if order.status == OrderStatus.SUBMITTED:
            order.rejected_reason = reason
            self._transition(state, order.client_order_id, OrderStatus.REJECTED)
            return
        target = state.get_pre_pending(order.client_order_id) or OrderStatus.ACCEPTED
        was = order.status
        self._transition(state, order.client_order_id, target)
        # Surface the failed cancel/update to the strategy as a synthesized reject event
        # through the unified on_order_update callback (the order is back to a live state).
        if was == OrderStatus.PENDING_CANCEL:
            event = OrderCancelRejectedEvent(
                instrument=order.instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.venue_order_id,
                reason=reason,
            )
        else:
            event = OrderUpdateRejectedEvent(
                instrument=order.instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.venue_order_id,
                reason=reason,
            )
        self._strategy.on_order_update(self._ctx, order, event)

    def _on_snapshot_tick(self, ctx) -> None:
        now = self._time.time()
        for exchange, state in self._states.items():
            last = state.get_last_snapshot_as_of()
            if last is None or (now - last) > self._snapshot_interval:
                self._connectors[exchange].request_snapshot()

    def _on_liveness_tick(self, ctx) -> None:
        now = self._time.time()
        for exchange, connector in self._connectors.items():
            if connector.is_ws_ready():
                self._liveness_unready_since.pop(exchange, None)
                continue
            since = self._liveness_unready_since.setdefault(exchange, now)
            if (now - since) >= self._liveness_threshold:
                logger.warning(f"[{exchange}] WS unready past threshold; reconnecting")
                try:
                    connector.reconnect()
                except Exception:
                    logger.exception(f"reconnect failed for {exchange}")
                self._liveness_unready_since.pop(exchange, None)

    def _book_deal(self, state: AccountState, instrument: Instrument, deal: Deal) -> Position:
        """Apply a deal's effect to the position and balances. Caller dedups first."""
        pos = state.get_position(instrument)
        if pos is None:
            pos = Position(instrument=instrument)
            state.set_position(instrument, pos)
        # update_position_by_deal returns (realized_pnl, fee) for this fill, both in
        # the portfolio funded currency (conversion_rate=1.0 — single-base-currency).
        realized_pnl, fee = pos.update_position_by_deal(deal, conversion_rate=1.0)
        self._apply_deal_to_balances(state, instrument, deal, realized_pnl, fee)
        return pos

    def _apply_deal_to_balances(
        self, state: AccountState, instrument: Instrument, deal: Deal, realized_pnl: float, fee: float
    ) -> None:
        """Propagate a fill's cash impact to balances.

        Futures/swap: credits realized PnL and debits the fee to the
        settle-currency balance. Spot: debits the quote currency by the trade
        cost (notional + fee) and credits the base asset by the filled amount.
        """
        if instrument.is_futures():
            # TODO(account-mgmt): fee is folded into settle here (correct when
            # settle == portfolio base currency); revisit for instruments whose
            # settle currency differs from the portfolio base currency.
            self._adjust_balance(state, instrument.settle, realized_pnl - fee)
        else:
            self._adjust_balance(state, instrument.quote, -(deal.amount * deal.price + fee))
            self._adjust_balance(state, instrument.base, deal.amount)

    def _adjust_balance(self, state: AccountState, currency: str, delta: float) -> None:
        bal = state.get_balance(currency)
        if bal is None:
            bal = Balance(exchange=state.exchange, currency=currency)
        # keep free consistent with total (free == total - locked), as the funding handler does
        bal.total += delta
        bal.free += delta
        state.update_balance(currency, bal)

    def _handle_funding_payment(self, state: AccountState, event: FundingPaymentEvent) -> ApplyResult:
        payment = event.payment
        instrument = event.instrument
        if instrument is None:
            return ApplyResult()
        interval_ns = payment.funding_interval_hours * 3_600_000_000_000
        bucket = (instrument, int(payment.time) // interval_ns)
        seen = self._applied_funding_buckets.setdefault(state.exchange, OrderedDict())
        if bucket in seen:
            return ApplyResult()
        pos = state.get_position(instrument)
        if pos is None:
            return ApplyResult()
        # Funding cash is computed on the mark price; FundingPayment carries no
        # amount. If the position has no mark yet (NaN), we cannot value the
        # payment — skip WITHOUT consuming the bucket so a re-delivered event can
        # apply once a mark exists (rather than poisoning balance/PnL with NaN).
        mark = pos.last_update_price
        if np.isnan(mark):
            logger.warning(f"[{state.exchange}] funding for {instrument} skipped: no mark price yet")
            return ApplyResult()
        seen[bucket] = None
        if len(seen) > self._FUNDING_BUCKET_CAP:
            seen.popitem(last=False)  # evict the oldest bucket
        amount = pos.apply_funding_payment(payment, mark)  # updates cumulative_funding/pnl
        bal = state.get_balance(instrument.settle)
        if bal is not None:
            bal.total += amount
            bal.free += amount  # keep free consistent with total (free == total - locked)
            state.update_balance(instrument.settle, bal)
        return ApplyResult(position=pos)


class SimulatedAccountManager(AccountManager):
    """Backtest variant — no asyncio, no WS, no periodic ticks."""

    def __init__(
        self,
        *,
        connectors,
        base_currencies: dict[str, str],
        strategy,
        time,
        cfg: AccountManagerConfig | None = None,
        account_id: str = "SimulatedAccountManager",
        tcc: TransactionCostsCalculator | None = None,
    ):
        self._pm = None
        self._init_state(
            connectors=connectors,
            base_currencies=base_currencies,
            strategy=strategy,
            time=time,
            cfg=cfg,
            account_id=account_id,
            tcc=tcc,
        )
        # Backtest has no asyncio/WS/periodic scheduling — no ticks registered.

    def set_context(self, ctx: "StrategyContext") -> None:
        # Backtest is synchronous: wire the ctx but NEVER register periodic ticks
        # (no PM scheduler drives them in simulation). Overrides the base, which would
        # otherwise pull pm from ctx and schedule.
        self._ctx = ctx
