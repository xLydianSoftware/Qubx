from collections import deque
from typing import Literal

import numpy as np

from qubx import logger
from qubx.core.account_manager_config import AccountManagerConfig, _ms_to_cron
from qubx.core.account_state import AccountState
from qubx.core.basics import (
    Balance,
    Instrument,
    Order,
    OrderOrigin,
    OrderStatus,
    Position,
    TransactionCostsCalculator,
)
from qubx.core.events import (
    AccountMessage,
    AccountSnapshotEvent,
    FundingPaymentEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
    ReconcileDiff,
)
from qubx.core.exceptions import InvalidOrderTransition

_LEGAL_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.INITIALIZED: {OrderStatus.SUBMITTED, OrderStatus.REJECTED},
    OrderStatus.SUBMITTED: {
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.ACCEPTED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.PENDING_UPDATE,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PENDING_CANCEL,
        OrderStatus.PENDING_UPDATE,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.PENDING_CANCEL: {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.PENDING_UPDATE: {
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    },
}


class AccountManager:
    def __init__(
        self,
        *,
        pm,
        connectors,
        strategy,
        time,
        cfg: AccountManagerConfig | None = None,
        account_id: str = "AccountManager",
        tcc: TransactionCostsCalculator | None = None,
    ):
        self._pm = pm
        self._connectors = connectors
        self._strategy = strategy
        self._time = time
        self._cfg = cfg or AccountManagerConfig()
        self.account_id = account_id
        self._tcc = tcc
        self._states = {ex: AccountState(exchange=ex) for ex in connectors}
        self._liveness_unready_since: dict[str, np.datetime64] = {}
        # TODO(account-mgmt): this set grows unbounded over long sessions; bound it
        # (evict old buckets) in a later PR.
        self._applied_funding_buckets: dict[str, set] = {}
        self._ctx = None  # set via set_context once StrategyContext is built
        if self._cfg.inflight_check_interval_ms > 0:
            pm.schedule(_ms_to_cron(self._cfg.inflight_check_interval_ms), self._on_inflight_tick)
        if self._cfg.snapshot_check_interval_ms > 0:
            pm.schedule(_ms_to_cron(self._cfg.snapshot_check_interval_ms), self._on_snapshot_tick)
        if self._cfg.liveness_check_interval_ms > 0:
            pm.schedule(_ms_to_cron(self._cfg.liveness_check_interval_ms), self._on_liveness_tick)

    def set_context(self, ctx) -> None:
        """Wire the IStrategyContext after construction.

        AM-fired callbacks (reconcile, inflight-exhaustion) pass this ctx so their
        signature matches PM-fired callbacks — no None placeholder.
        """
        self._ctx = ctx

    def add_order(self, exchange: str, order: Order) -> None:
        self._states[exchange]._add_order(order)

    def transition_order(self, exchange: str, cid: str, new_status: OrderStatus) -> None:
        state = self._states[exchange]
        order = state.active_orders.get(cid)
        if order is None:
            raise KeyError(f"order {cid} not found in {exchange}")
        if new_status not in _LEGAL_TRANSITIONS.get(order.status, set()):
            raise InvalidOrderTransition(cid, order.status, new_status)
        if new_status in (OrderStatus.PENDING_CANCEL, OrderStatus.PENDING_UPDATE):
            order.pre_pending_status = order.status
        state._transition_order(cid, new_status, self._time.time())

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
        # get_orders() want only live orders (this is the IAccountViewer contract the
        # old broker/account get_orders honoured by reading the venue's open orders).
        orders = {cid: o for cid, o in orders.items() if not o.status.is_terminal()}
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
            state._set_position(instrument, pos)
        return pos

    def get_positions(self, exchange: str | None = None) -> dict[Instrument, Position]:
        if exchange is not None:
            return self._states[exchange].get_positions()
        return {ins: pos for s in self._states.values() for ins, pos in s.positions.items()}

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
            return list(self._states[exchange].balances.values())
        return [b for s in self._states.values() for b in s.balances.values()]

    def get_fees_calculator(self, exchange: str | None = None) -> TransactionCostsCalculator:
        if self._tcc is None:
            raise NotImplementedError("get_fees_calculator: no TransactionCostsCalculator provided to AccountManager")
        return self._tcc

    def find_order_by_id(self, order_id: str) -> Order | None:
        for state in self._states.values():
            if (o := state.get_order_by_venue_id(order_id)) is not None:
                return o
        return None

    def find_order_by_client_id(self, client_id: str) -> Order | None:
        return self.get_order(client_id)

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

    def apply(self, event: AccountMessage):
        state = self._get_state_for_event(event)
        if state is None:
            return None
        match event:
            case OrderPartiallyFilledEvent():
                return self._handle_partial_fill(state, event)
            case OrderFilledEvent():
                return self._handle_fill(state, event)
            case OrderAcceptedEvent():
                return self._handle_accepted(state, event)
            case OrderCanceledEvent():
                return self._handle_canceled(state, event)
            case OrderExpiredEvent():
                return self._handle_expired(state, event)
            case OrderUpdatedEvent():
                return self._handle_updated(state, event)
            case OrderRejectedEvent():
                return self._handle_rejected(state, event)
            case OrderCancelRejectedEvent():
                return self._handle_cancel_rejected(state, event)
            case OrderUpdateRejectedEvent():
                return self._handle_update_rejected(state, event)
            case AccountSnapshotEvent():
                return self._handle_snapshot(state, event)
            case FundingPaymentEvent():
                return self._handle_funding_payment(state, event)
            case _:
                logger.warning(f"unhandled AccountMessage: {type(event)}")
                return None

    def _get_state_for_event(self, event):
        if isinstance(event, AccountSnapshotEvent):
            return self._states.get(event.snapshot.exchange)
        if event.instrument is not None:
            return self._states.get(event.instrument.exchange)
        return None

    def _resolve_or_materialize(self, state, event):
        cid = getattr(event, "client_order_id", None)
        venue_id = getattr(event, "venue_order_id", None)
        if cid is not None and cid in state.active_orders:
            return state.active_orders[cid]
        if venue_id is not None and venue_id in state._venue_id_index:
            return state.active_orders[state._venue_id_index[venue_id]]
        if cid is not None:
            for hist in state._terminal_history:
                if hist.client_order_id == cid:
                    return hist
        return self._materialize_external(state, event)

    def _materialize_external(self, state, event):
        venue_id = getattr(event, "venue_order_id", None)
        if venue_id is None:
            # No venue id → no stable identity; all such orders would collide on
            # ext:unknown. Real venue events always carry one, so warn loudly.
            venue_id = "unknown"
            logger.warning(f"materializing EXTERNAL order with no venue_order_id: {event}")
        cid = f"ext:{venue_id}"
        order = Order(
            client_order_id=cid,
            venue_order_id=venue_id,
            origin=OrderOrigin.EXTERNAL,
            type="LIMIT",
            instrument=event.instrument,
            time=self._time.time(),
            quantity=0.0,
            price=0.0,
            side="BUY",
            status=OrderStatus.ACCEPTED,
            time_in_force="gtc",
        )
        state._add_order(order)
        return order

    def _handle_accepted(self, state, event: OrderAcceptedEvent):
        order = self._resolve_or_materialize(state, event)
        if order.status.is_terminal():
            # Late accept on an already-terminal order (design "OrderFilled before
            # OrderAccepted"): benign side-effect, no transition, no phantom. Set
            # the venue id ONLY if the order is still in active_orders — an evicted
            # order's venue-id index was already dropped, so _set_venue_id would
            # KeyError on active_orders[cid].
            if order.client_order_id in state.active_orders:
                state._set_venue_id(order.client_order_id, event.venue_order_id)
                order.accepted_at = event.accepted_at
            return order
        state._set_venue_id(order.client_order_id, event.venue_order_id)
        order.accepted_at = event.accepted_at
        if order.status == OrderStatus.PENDING_CANCEL:
            return order
        if order.status == OrderStatus.PENDING_UPDATE:
            return state._transition_order(order.client_order_id, OrderStatus.ACCEPTED, self._time.time())
        if OrderStatus.ACCEPTED in _LEGAL_TRANSITIONS.get(order.status, set()):
            return state._transition_order(order.client_order_id, OrderStatus.ACCEPTED, self._time.time())
        return order

    # Terminal-order guard (shared by the lifecycle handlers below).
    # A late event for an order that is already terminal — possibly still in
    # active_orders during the grace window, possibly already evicted to
    # _terminal_history — must be a benign no-op: NO status change (a terminal
    # state has no legal outgoing edge) and NO active_orders[cid] mutation (which
    # would KeyError for an evicted order). This generalizes the OrderAccepted
    # grace rule (design "OrderFilled before OrderAccepted" / terminal retention)
    # to every lifecycle event, and is what keeps a late OrderCanceled from
    # silently flipping a FILLED order to CANCELED.
    def _is_late_terminal(self, order) -> bool:
        return order.status.is_terminal()

    def _handle_partial_fill(self, state, event: OrderPartiallyFilledEvent):
        order = self._resolve_or_materialize(state, event)
        if self._is_late_terminal(order):
            logger.debug(f"late partial-fill on terminal {order.client_order_id}; ignoring")
            return order
        if event.venue_order_id and order.venue_order_id is None:
            state._set_venue_id(order.client_order_id, event.venue_order_id)
        is_new_trade = event.fill.trade_id not in order.seen_trade_ids
        state._apply_fill(order.client_order_id, event.fill, self._time.time())
        if is_new_trade and order.instrument is not None:
            self._apply_deal_to_position(state, order.instrument, event.fill)
        if order.status.is_pending():
            # filled_quantity mirrors real, irreversible fills (and the position),
            # so it is NEVER reduced. A fill that races a pending modify can push it
            # past the new (smaller) target — surface that as a warning only; the
            # venue resolves the race (OrderUpdated, or OrderUpdateRejected because
            # it can't shrink an order below what's already filled).
            if (order.status == OrderStatus.PENDING_UPDATE
                    and order.filled_quantity > order.quantity):
                logger.warning(
                    f"[{order.client_order_id}] fill during pending-update pushed "
                    f"filled_quantity ({order.filled_quantity}) past target "
                    f"({order.quantity}); leaving filled intact, awaiting venue verdict"
                )
            return state.get_order(order.client_order_id)
        if OrderStatus.PARTIALLY_FILLED in _LEGAL_TRANSITIONS.get(order.status, set()):
            return state._transition_order(order.client_order_id, OrderStatus.PARTIALLY_FILLED, self._time.time())
        return order

    def _handle_fill(self, state, event: OrderFilledEvent):
        order = self._resolve_or_materialize(state, event)
        if self._is_late_terminal(order):
            logger.debug(f"late fill on terminal {order.client_order_id}; ignoring")
            return order
        if event.venue_order_id and order.venue_order_id is None:
            state._set_venue_id(order.client_order_id, event.venue_order_id)
        is_new_trade = event.fill.trade_id not in order.seen_trade_ids
        state._apply_fill(order.client_order_id, event.fill, self._time.time())
        if is_new_trade and order.instrument is not None:
            self._apply_deal_to_position(state, order.instrument, event.fill)
        return state._transition_order(order.client_order_id, OrderStatus.FILLED, self._time.time())

    def _handle_canceled(self, state, event):
        order = self._resolve_or_materialize(state, event)
        if self._is_late_terminal(order):
            logger.debug(f"late cancel on terminal {order.client_order_id}; ignoring")
            return order
        return state._transition_order(order.client_order_id, OrderStatus.CANCELED, self._time.time())

    def _handle_expired(self, state, event):
        order = self._resolve_or_materialize(state, event)
        if self._is_late_terminal(order):
            logger.debug(f"late expire on terminal {order.client_order_id}; ignoring")
            return order
        return state._transition_order(order.client_order_id, OrderStatus.EXPIRED, self._time.time())

    def _handle_rejected(self, state, event: OrderRejectedEvent):
        order = state.active_orders.get(event.client_order_id)
        if order is None:
            logger.warning(f"reject for unknown order {event.client_order_id}")
            return None
        if order.status.is_terminal():
            logger.debug(f"late reject on terminal {order.client_order_id}; ignoring")
            return order
        order.rejected_reason = event.reason
        return state._transition_order(order.client_order_id, OrderStatus.REJECTED, self._time.time())

    def _handle_updated(self, state, event: OrderUpdatedEvent):
        order = self._resolve_or_materialize(state, event)
        if self._is_late_terminal(order):
            logger.debug(f"late update on terminal {order.client_order_id}; ignoring")
            return order
        if order.venue_order_id != event.venue_order_id:
            if order.venue_order_id is not None:
                state._venue_id_index.pop(order.venue_order_id, None)
            state._set_venue_id(order.client_order_id, event.venue_order_id)
        if event.new_price is not None:
            order.price = event.new_price
        if event.new_quantity is not None:
            order.quantity = event.new_quantity
        order.last_updated_at = self._time.time()
        if order.status == OrderStatus.PENDING_UPDATE:
            return state._transition_order(order.client_order_id, OrderStatus.ACCEPTED, self._time.time())
        return order

    def _handle_cancel_rejected(self, state, event):
        order = state.active_orders.get(event.client_order_id)
        if order is None or order.status != OrderStatus.PENDING_CANCEL:
            logger.warning(f"cancel-rejected for unexpected state: {order}")
            return None
        return self._revert_from_pending(state, order)

    def _handle_update_rejected(self, state, event):
        order = state.active_orders.get(event.client_order_id)
        if order is None or order.status != OrderStatus.PENDING_UPDATE:
            logger.warning(f"update-rejected for unexpected state: {order}")
            return None
        return self._revert_from_pending(state, order)

    def _revert_from_pending(self, state, order):
        target = order.pre_pending_status or OrderStatus.ACCEPTED
        order.pre_pending_status = None
        return state._transition_order(order.client_order_id, target, self._time.time())

    def _handle_snapshot(self, state, event: AccountSnapshotEvent):
        snapshot = event.snapshot
        if state._last_snapshot_as_of is not None and snapshot.as_of <= state._last_snapshot_as_of:
            return None
        state._last_snapshot_as_of = snapshot.as_of

        grace = np.timedelta64(self._cfg.snapshot_check_threshold_ms, "ms")
        diff = ReconcileDiff(exchange=snapshot.exchange, as_of=snapshot.as_of)

        if snapshot.open_orders is not None:
            snap_by_vid = {o.venue_order_id: o for o in snapshot.open_orders if o.venue_order_id}
            for cid, cached in list(state.active_orders.items()):
                if cached.status.is_terminal():
                    continue
                vid = cached.venue_order_id
                if vid is not None and vid in snap_by_vid:
                    continue
                if (snapshot.as_of - cached.time) < grace:
                    continue
                terminal = OrderStatus.REJECTED if cached.status == OrderStatus.SUBMITTED else OrderStatus.CANCELED
                cached.rejected_reason = "reconcile: missing from snapshot"
                cached.pre_pending_status = None
                try:
                    state._transition_order(cid, terminal, self._time.time())
                    diff.orders_newly_terminal.append(cached)
                except InvalidOrderTransition:
                    logger.warning(f"reconcile: cannot terminate {cid} from {cached.status}")
            for snap_order in snapshot.open_orders:
                existing = (
                    state.get_order_by_venue_id(snap_order.venue_order_id) if snap_order.venue_order_id else None
                )
                if existing is None:
                    self._materialize_from_snapshot(state, snap_order, snapshot.as_of)
                    diff.orders_materialized.append(snap_order)
                elif existing.last_updated_at is None or snapshot.as_of > existing.last_updated_at:
                    self._update_from_snapshot(state, existing, snap_order, snapshot.as_of)
                    diff.orders_updated.append(existing)

        # Positions and balances: the snapshot is the venue's authoritative full
        # truth for size/amount, and stale snapshots are already rejected wholesale
        # by the _last_snapshot_as_of ratchet above — so an accepted snapshot always
        # overwrites. No per-record freshness here (unlike orders, where it guards a
        # fresh fill): Position/Balance carry no reliable last-update timestamp yet.
        # TODO(account-mgmt): once WS PositionUpdate/BalanceUpdate events are wired
        # (PR 6/7), add per-record freshness backed by a real timestamp so a snapshot
        # older than a recent WS update can't clobber it.
        if snapshot.positions is not None:
            for snap_pos in snapshot.positions:
                state._set_position(snap_pos.instrument, snap_pos)
                diff.positions_updated.append(snap_pos)

        if snapshot.balances is not None:
            for snap_bal in snapshot.balances:
                state._update_balance(snap_bal.currency, snap_bal)
                diff.balances_updated.append(snap_bal)

        try:
            self._strategy.on_reconcile_complete(self._ctx, snapshot.exchange, diff)
        except Exception:
            logger.exception("on_reconcile_complete raised")
        return None

    def _materialize_from_snapshot(self, state, snap_order, as_of):
        # cid prefix classifies origin: our prefix → a recovered framework order;
        # anything else → external. Keep an already-synthesized ext: cid as-is,
        # otherwise synthesize one from the venue id.
        if snap_order.client_order_id.startswith("qubx-"):
            origin = OrderOrigin.RECOVERED
            cid = snap_order.client_order_id
        elif snap_order.client_order_id.startswith("ext:"):
            origin = OrderOrigin.EXTERNAL
            cid = snap_order.client_order_id
        else:
            origin = OrderOrigin.EXTERNAL
            cid = f"ext:{snap_order.venue_order_id}"
        state._add_order(
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

    def _update_from_snapshot(self, state, existing, snap_order, as_of):
        existing.status = snap_order.status
        existing.filled_quantity = snap_order.filled_quantity
        existing.avg_fill_price = snap_order.avg_fill_price
        existing.price = snap_order.price
        existing.quantity = snap_order.quantity
        existing.last_updated_at = as_of

    def _sweep_terminal_evictions(self) -> None:
        grace = np.timedelta64(self._cfg.terminal_order_retention_ms, "ms")
        now = self._time.time()
        for state in self._states.values():
            if state._terminal_history.maxlen != self._cfg.terminal_order_history_size:
                state._terminal_history = deque(
                    state._terminal_history,
                    maxlen=self._cfg.terminal_order_history_size,
                )
            for cid in list(state._pending_evict_index):
                if (now - state._pending_evict_index[cid]) >= grace:
                    state._evict_to_history(cid)

    def _on_inflight_tick(self, ctx) -> None:
        now = self._time.time()
        threshold = np.timedelta64(self._cfg.inflight_check_threshold_ms, "ms")
        for exchange, state in self._states.items():
            for cid in list(state._inflight_index):
                order = state.active_orders.get(cid)
                if order is None or order.status not in (
                    OrderStatus.SUBMITTED,
                    OrderStatus.PENDING_CANCEL,
                    OrderStatus.PENDING_UPDATE,
                ):
                    state._inflight_index.discard(cid)
                    continue
                last = order.last_updated_at or order.time
                if (now - last) < threshold:
                    continue
                if order.retry_count >= self._cfg.inflight_check_retries:
                    self._resolve_exhausted_inflight(state, exchange, order)
                else:
                    self._connectors[exchange].request_order_status(
                        client_order_id=cid,
                        venue_order_id=order.venue_order_id,
                    )
                    order.retry_count += 1
                    order.last_updated_at = now
        self._sweep_terminal_evictions()

    def _resolve_exhausted_inflight(self, state, exchange, order):
        reason = f"reconcile: no venue ack after {order.retry_count} retries"
        if order.status == OrderStatus.SUBMITTED:
            order.rejected_reason = reason
            state._transition_order(order.client_order_id, OrderStatus.REJECTED, self._time.time())
            return
        target = order.pre_pending_status or OrderStatus.ACCEPTED
        order.pre_pending_status = None
        was = order.status
        state._transition_order(order.client_order_id, target, self._time.time())
        if was == OrderStatus.PENDING_CANCEL:
            self._strategy.on_order_cancel_rejected(self._ctx, order, reason)
        else:
            self._strategy.on_order_update_rejected(self._ctx, order, reason)

    def _on_snapshot_tick(self, ctx) -> None:
        now = self._time.time()
        interval = np.timedelta64(self._cfg.snapshot_check_interval_ms, "ms")
        for exchange, state in self._states.items():
            last = state._last_snapshot_as_of
            if last is None or (now - last) > interval:
                self._connectors[exchange].request_snapshot()

    def _on_liveness_tick(self, ctx) -> None:
        now = self._time.time()
        threshold = np.timedelta64(self._cfg.liveness_check_threshold_ms, "ms")
        for exchange, connector in self._connectors.items():
            if connector.is_ws_ready():
                self._liveness_unready_since.pop(exchange, None)
                continue
            since = self._liveness_unready_since.setdefault(exchange, now)
            if (now - since) >= threshold:
                logger.warning(f"[{exchange}] WS unready past threshold; reconnecting")
                try:
                    connector.force_ws_reconnect_sync()
                except Exception:
                    logger.exception(f"force_ws_reconnect_sync failed for {exchange}")
                self._liveness_unready_since.pop(exchange, None)

    def _apply_deal_to_position(self, state, instrument, deal):
        pos = state.get_position(instrument)
        if pos is None:
            pos = Position(instrument=instrument)
            state._set_position(instrument, pos)
        pos.update_position_by_deal(deal, conversion_rate=1.0)
        return pos

    def on_market_quote(self, instrument, quote) -> None:
        state = self._states.get(instrument.exchange)
        if state is None:
            return
        pos = state.get_position(instrument)
        if pos is None:
            return
        # Position.update_market_price expects (timestamp, price, conversion_rate);
        # mark-to-market uses the quote mid.
        pos.update_market_price(self._time.time(), quote.mid_price(), 1.0)

    def _handle_funding_payment(self, state, event: FundingPaymentEvent):
        payment = event.payment
        instrument = event.instrument
        if instrument is None:
            return None
        interval_ns = payment.funding_interval_hours * 3_600_000_000_000
        bucket = (instrument, int(payment.time) // interval_ns)
        seen = self._applied_funding_buckets.setdefault(state.exchange, set())
        if bucket in seen:
            return None
        pos = state.get_position(instrument)
        if pos is None:
            return None
        # Funding cash is computed on the mark price; FundingPayment carries no
        # amount. If the position has no mark yet (NaN), we cannot value the
        # payment — skip WITHOUT consuming the bucket so a re-delivered event can
        # apply once a mark exists (rather than poisoning balance/PnL with NaN).
        mark = pos.last_update_price
        if np.isnan(mark):
            logger.warning(f"[{state.exchange}] funding for {instrument} skipped: no mark price yet")
            return None
        seen.add(bucket)
        amount = pos.apply_funding_payment(payment, mark)  # updates cumulative_funding/pnl
        bal = state.get_balance(instrument.settle)
        if bal is not None:
            bal.total += amount
            bal.free += amount  # keep free consistent with total (free == total - locked)
            state._update_balance(instrument.settle, bal)
        return payment

    def get_total_capital(self, exchange: str | None = None) -> float:
        if exchange is not None:
            return self._total_capital_for(self._states[exchange])
        return sum(self._total_capital_for(s) for s in self._states.values())

    def get_capital(self, exchange: str | None = None) -> float:
        if exchange is not None:
            return self._free_capital_for(self._states[exchange])
        return sum(self._free_capital_for(s) for s in self._states.values())

    def _total_capital_for(self, state) -> float:
        base = self._base_currency_for(state)
        bal = state.get_balance(base)
        if bal is None:
            return 0.0
        total = bal.total
        for pos in state.positions.values():
            if pos.instrument.is_futures():
                total += pos.unrealized_pnl()
        return total

    def _free_capital_for(self, state) -> float:
        base = self._base_currency_for(state)
        bal = state.get_balance(base)
        return bal.free if bal else 0.0

    def get_base_currency(self, exchange: str | None = None) -> str:
        state = self._states[exchange] if exchange is not None else next(iter(self._states.values()), None)
        return self._base_currency_for(state) if state is not None else "USDT"

    def _base_currency_for(self, state) -> str:
        if not state.balances:
            return "USDT"
        return max(state.balances.values(), key=lambda b: b.total).currency

    def _notional(self, pos) -> float:
        # notional_value is NaN for an unmarked position; treat as 0 so a single
        # unmarked position can't poison aggregate leverage (consumed by emitters,
        # loggers, sizers). A real position gets a mark on its first quote.
        return float(np.nan_to_num(pos.notional_value))

    def get_leverage(self, instrument) -> float:
        state = self._states.get(instrument.exchange)
        if state is None:
            return 0.0
        pos = state.get_position(instrument)
        if pos is None:
            return 0.0
        total = self._total_capital_for(state)
        return abs(self._notional(pos)) / total if total > 0 else 0.0

    def get_net_leverage(self, exchange: str | None = None) -> float:
        states = [self._states[exchange]] if exchange else list(self._states.values())
        net = sum(self._notional(p) for s in states for p in s.positions.values())
        total = sum(self._total_capital_for(s) for s in states)
        return abs(net) / total if total > 0 else 0.0

    def get_gross_leverage(self, exchange: str | None = None) -> float:
        states = [self._states[exchange]] if exchange else list(self._states.values())
        gross = sum(abs(self._notional(p)) for s in states for p in s.positions.values())
        total = sum(self._total_capital_for(s) for s in states)
        return gross / total if total > 0 else 0.0

    def get_leverages(self, exchange: str | None = None) -> dict[Instrument, float]:
        return {ins: self.get_leverage(ins) for ins in self.get_positions(exchange)}

    # Per-instrument exchange-side settings: the venue's configured leverage tier,
    # hard caps and margin mode. These are not tracked by AccountManager (they live
    # on the venue), so they default to the same neutral values the base
    # IAccountProcessor uses for connectors without the concept.
    def get_instrument_leverage(self, instrument: Instrument) -> float | None:
        return None

    def get_max_instrument_leverage(self, instrument: Instrument) -> float | None:
        return None

    def get_max_instrument_notional(self, instrument: Instrument) -> float:
        return float("inf")

    def get_margin_mode(self, instrument: Instrument) -> Literal["cross", "isolated"] | None:
        return None

    def get_total_initial_margin(self, exchange: str | None = None) -> float:
        return sum(p.initial_margin for p in self.get_positions(exchange).values())

    def get_total_maint_margin(self, exchange: str | None = None) -> float:
        return sum(p.maint_margin for p in self.get_positions(exchange).values())

    def get_available_margin(self, exchange: str | None = None) -> float:
        return self.get_total_capital(exchange) - self.get_total_initial_margin(exchange)

    def get_margin_ratio(self, exchange: str | None = None) -> float:
        maint = self.get_total_maint_margin(exchange)
        if maint == 0:
            return 100.0
        return min(100.0, self.get_total_capital(exchange) / maint)

    def get_adl_level(self, instrument: Instrument) -> int | None:
        pos = self.get_position(instrument)
        return pos.adl_level if pos is not None else None

    def get_reserved(self, instrument: Instrument) -> float:
        return 0.0


class SimulationAccountManager(AccountManager):
    """Backtest variant — no asyncio, no WS, no periodic ticks."""

    def __init__(
        self,
        *,
        connectors,
        strategy,
        time,
        cfg: AccountManagerConfig | None = None,
        account_id: str = "SimulationAccountManager",
        tcc: TransactionCostsCalculator | None = None,
    ):
        self._pm = None
        self._connectors = connectors
        self._strategy = strategy
        self._time = time
        self._cfg = cfg or AccountManagerConfig()
        self.account_id = account_id
        self._tcc = tcc
        self._states = {ex: AccountState(exchange=ex) for ex in connectors}
        self._liveness_unready_since = {}
        self._applied_funding_buckets = {}
        self._ctx = None
