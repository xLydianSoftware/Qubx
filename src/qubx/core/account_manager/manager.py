"""Central account-state owner: per-exchange AccountStates + the event apply path.

Routes a typed AccountMessage to the right exchange's state, applies it via the reducer
(on the AM clock), and exposes the cross-exchange read facade + aggregated metrics.
Periodic ticks (reconcile, sweep, liveness) call the reconcile.py decision helpers and
fire the connector requests — connector calls stay manager-only, as does PM wiring.
"""

from typing import Callable, Literal

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.account_manager import reconcile, reducer
from qubx.core.account_manager.config import AccountManagerConfig
from qubx.core.account_manager.reducer import ApplyResult
from qubx.core.account_manager.state import AccountState
from qubx.core.basics import (
    ZERO_COSTS,
    Balance,
    Instrument,
    ITimeProvider,
    Order,
    OrderOrigin,
    OrderStatus,
    OrderTransition,
    Position,
    TransactionCostsCalculator,
)
from qubx.core.connector import IConnector
from qubx.core.events import AccountMessage, AccountSnapshotEvent, BalanceUpdateEvent, OrderEvent
from qubx.core.interfaces import IProcessingManager
from qubx.utils.time import timedelta_to_crontab


class AccountManager:
    _pm: IProcessingManager | None
    _connectors: dict[str, IConnector]
    _time: ITimeProvider
    _cfg: AccountManagerConfig
    account_id: str
    _tcc: dict[str, TransactionCostsCalculator]
    _states: dict[str, AccountState]
    _snapshot_grace: np.timedelta64
    _snapshot_interval: np.timedelta64
    _inflight_threshold: np.timedelta64
    _liveness_threshold: np.timedelta64
    _terminal_retention: np.timedelta64
    _liveness_unready_since: dict[str, np.datetime64]
    _last_eviction_sweep: np.datetime64
    _last_drift_snapshot: dict[str, np.datetime64]

    # Min interval between drift-triggered snapshot requests, per exchange: a burst of
    # drifting position pushes collapses into one REST correction.
    _DRIFT_SNAPSHOT_MIN_INTERVAL = np.timedelta64(2_000, "ms")

    def __init__(
        self,
        *,
        pm: IProcessingManager | None = None,
        connectors: dict[str, IConnector],
        base_currencies: dict[str, str],
        time: ITimeProvider,
        cfg: AccountManagerConfig | None = None,
        account_id: str = "AccountManager",
        tcc: dict[str, TransactionCostsCalculator] | None = None,
    ):
        self._pm = pm
        self._init_state(
            connectors=connectors,
            base_currencies=base_currencies,
            time=time,
            cfg=cfg,
            account_id=account_id,
            tcc=tcc,
        )
        # Live passes pm=None and registers the ticks later — see set_processing_manager.
        if self._pm is not None:
            self._register_ticks()

    def set_processing_manager(self, pm: IProcessingManager) -> None:
        """Late wiring hook, called by StrategyContext once its ProcessingManager exists.

        Live: the AM is built before the ProcessingManager (the AM↔pm cycle — the PM lives
        inside StrategyContext, which takes the AM), so pm is None at construction and the
        periodic ticks register HERE. Idempotent: a second call is a no-op.
        """
        if self._pm is None:
            self._pm = pm
            self._register_ticks()

    def _init_state(
        self,
        *,
        connectors: dict[str, IConnector],
        base_currencies: dict[str, str],
        time: ITimeProvider,
        cfg: AccountManagerConfig | None,
        account_id: str,
        tcc: dict[str, TransactionCostsCalculator] | None,
    ) -> None:
        # Shared field init for both the live and simulation managers, so the
        # subclass can't silently drift from the parent's field set.
        self._connectors = connectors
        self._time = time
        self._cfg = cfg or AccountManagerConfig()
        self.account_id = account_id
        # Per-exchange fee schedules, keyed like connectors/base_currencies.
        self._tcc = tcc or {}
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
        self._last_eviction_sweep = time.time()
        self._last_drift_snapshot = {}

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

    # ---- order-entry bookkeeping (TradingManager calls these) ----------- #

    def add_order(self, order: Order) -> None:
        self._states[order.instrument.exchange].add_order(order)

    def transition_order(self, exchange: str, cid: str, new_status: OrderStatus) -> None:
        reconcile.transition(self._states[exchange], cid, new_status, self._time.time())

    def remove_order(self, exchange: str, cid: str) -> None:
        self._states[exchange].remove_order(cid)

    # ---- state seeding & cash adjustments (runner/transfer boundary) ------ #
    # The only sanctioned out-of-AM mutation paths: AccountState mutators stay
    # AM-internal, callers go through these.

    def seed_balance(self, exchange: str, balance: Balance) -> bool:
        """Seed a startup balance (restored state, initial paper/backtest capital).
        Returns False when this AM doesn't manage the exchange (record skipped)."""
        state = self._states.get(exchange)
        if state is None:
            return False
        state.update_balance(balance.currency, balance)
        return True

    def seed_position(self, position: Position) -> bool:
        """Seed a restored position (exchange derived from the instrument).
        Returns False when this AM doesn't manage the exchange (record skipped)."""
        state = self._states.get(position.instrument.exchange)
        if state is None:
            return False
        state.set_position(position.instrument, position)
        return True

    def adjust_balance(self, exchange: str, currency: str, delta: float) -> None:
        """Apply a cash delta (e.g. a simulated transfer leg) to an exchange balance,
        creating the Balance if missing."""
        self._states[exchange].adjust_balance(currency, delta)

    # ---- event path ------------------------------------------------------ #

    def apply(self, event: AccountMessage) -> ApplyResult:
        state = self._state_for_event(event)
        if state is None:
            return ApplyResult()
        now = self._time.time()
        result = reducer.apply(state, event, now, snapshot_grace=self._snapshot_grace)
        if (diff := result.reconcile_diff) is not None:
            if diff.missing:
                self._resolve_missing_orders(state, diff)
            self._log_reconcile_diff(state.exchange, diff)
        if result.position_drift is not None:
            self._on_position_drift(state, result.position_drift, now)
        # Opportunistic terminal eviction: SimulatedAccountManager (paper + backtest)
        # registers no periodic ticks, so without this sweep terminal orders and their
        # side-table entries would grow unbounded there (and in live when
        # inflight_check_interval_ms=0). One timestamp comparison per apply; the sweep
        # itself runs at most once per retention window. Live additionally sweeps from
        # the inflight tick so eviction stays prompt during event silence.
        if now - self._last_eviction_sweep >= self._terminal_retention:
            self._sweep_terminal_evictions()
        return result

    def _log_reconcile_diff(self, exchange: str, diff: reconcile.ReconcileDiff) -> None:
        # The operator surface for snapshot reconcile: one line per applied snapshot that
        # changed anything. WARNING when orders were terminalized/materialized/are still
        # missing (drift an operator must act on), INFO for benign corrections. A diff
        # carrying only venue_figures (present on virtually every live snapshot) stays silent.
        if not (
            diff.missing or diff.terminated or diff.materialized or diff.updated or diff.positions or diff.balances
        ):
            return
        msg = (
            f"[{exchange}] reconcile applied: "
            f"terminated={[o.client_order_id for o in diff.terminated]}, "
            f"materialized={[o.client_order_id for o in diff.materialized]}, "
            f"missing={[o.client_order_id for o in diff.missing]}, "
            f"updated={len(diff.updated)} orders, {len(diff.positions)} positions, {len(diff.balances)} balances"
        )
        if diff.terminated or diff.materialized or diff.missing:
            logger.warning(msg)
        else:
            logger.info(msg)

    def _resolve_missing_orders(self, state: AccountState, diff: reconcile.ReconcileDiff) -> None:
        # Fetch-before-terminalize: an order missing from the snapshot past grace may have
        # FILLED during a WS gap — request its true status (the connector replays the venue's
        # answer through the normal event path: _handle_ws_order emits the real FILLED/
        # CANCELED, OrderNotFound synthesizes the reject). The shared per-order retry counter
        # is the fetch budget; once exhausted across snapshot cycles, give up and terminalize
        # exactly as before. Resolved orders leave the budget via transition_order's reset.
        connector = self._connectors[state.exchange]
        now = self._time.time()
        unresolved: list[Order] = []
        for order in diff.missing:
            cid = order.client_order_id
            try:
                if reconcile.retries_exhausted(state, cid, self._cfg.inflight_check_retries):
                    logger.warning(
                        f"[{state.exchange}] reconcile give-up: terminalizing {cid} "
                        f"(status={order.status}, vid={order.venue_order_id}) after "
                        f"{state.get_retry(cid)} status-fetch attempts"
                    )
                    if reconcile.terminalize_missing(state, order, now):
                        diff.terminated.append(order)
                else:
                    connector.request_order_status(
                        client_order_id=cid, venue_order_id=order.venue_order_id, instrument=order.instrument
                    )
                    state.bump_retry(cid)
                    unresolved.append(order)
            except Exception:
                # One bad order / connector error must not abort the rest of the resolution.
                logger.exception(f"[{state.exchange}] missing-order status fetch failed for {cid}")
                unresolved.append(order)
        diff.missing[:] = unresolved

    def _on_position_drift(self, state: AccountState, pushed: Position, now: np.datetime64) -> None:
        # A venue position push disagreed with local size. The reducer never applies it
        # (sizes are owned by the deal ledger); the correction is the race-safe snapshot
        # reconcile. Rate-limited per exchange so a burst of drifting pushes collapses
        # into one REST request — drifts suppressed inside the window are still corrected
        # by that pending snapshot.
        instrument = pushed.instrument
        local = state.get_position(instrument)
        local_qty = local.quantity if local is not None else 0.0
        last = self._last_drift_snapshot.get(state.exchange)
        if last is not None and (now - last) < self._DRIFT_SNAPSHOT_MIN_INTERVAL:
            logger.debug(
                f"[{state.exchange}] position drift on {instrument.symbol} "
                f"(local={local_qty}, pushed={pushed.quantity}) within rate window; snapshot already requested"
            )
            return
        logger.warning(
            f"[{state.exchange}] position drift on {instrument.symbol}: "
            f"local={local_qty}, pushed={pushed.quantity}; requesting snapshot"
        )
        # Stamp before the request so a connector error doesn't re-fire every push;
        # the periodic snapshot tick is the backstop.
        self._last_drift_snapshot[state.exchange] = now
        try:
            self._connectors[state.exchange].request_snapshot()
        except Exception:
            logger.exception(f"[{state.exchange}] drift snapshot request failed")

    def _state_for_event(self, event: AccountMessage) -> AccountState | None:
        if isinstance(event, AccountSnapshotEvent):
            return self._states.get(event.snapshot.exchange)
        if isinstance(event, BalanceUpdateEvent):
            # Balance pushes carry no instrument — route strictly by the balance's
            # exchange (the single-state fallback below would misroute a push stamped
            # for an unmanaged exchange).
            return self._states.get(event.balance.exchange)
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

    # ---- market data ------------------------------------------------------ #

    def on_market_quote(self, instrument, quote) -> None:
        state = self._states.get(instrument.exchange)
        if state is None:
            return
        pos = state.get_position(instrument)
        if pos is None:  # only mark positions we hold; never create one per quote
            return
        pos.update_market_price(self._time.time(), quote.mid_price(), state.conversion_rate(instrument))

    # ---- reads (cross-exchange) -------------------------------------------- #

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

    def get_order(self, client_order_id: str, exchange: str | None = None) -> Order | None:
        if exchange is not None:
            return self._states[exchange].get_order(client_order_id)
        if len(self._states) == 1:
            return next(iter(self._states.values())).get_order(client_order_id)
        # multi-exchange, caller didn't say which: framework cids are globally unique
        for state in self._states.values():
            if (order := state.get_order(client_order_id)) is not None:
                return order
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
        # No exchange => the first configured one (single-exchange callers omit it).
        # No TCC configured for the exchange => zero-fee calculator (rather than raising),
        # so callers that only need fee arithmetic still work in fee-agnostic setups.
        if exchange is None:
            exchange = next(iter(self._states))
        return self._tcc.get(exchange, ZERO_COSTS)

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
        Operator/debug surface — not wired to emitters; never reset within a session."""
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

    # ---- metrics (aggregated) -------------------------------------------- #
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

    # ---- periodic ticks (thin: reconcile.py decides, the manager acts) ----- #

    def _sweep_terminal_evictions(self) -> None:
        now = self._time.time()
        self._last_eviction_sweep = now
        for state in self._states.values():
            state.prune_terminal_orders(now, self._terminal_retention)

    def _on_inflight_tick(self, ctx) -> None:
        now = self._time.time()
        for exchange, state in self._states.items():
            for order in reconcile.select_overdue_inflight(state, now, self._inflight_threshold):
                cid = order.client_order_id
                try:
                    if reconcile.retries_exhausted(state, cid, self._cfg.inflight_check_retries):
                        # Give-up after the retry budget: synthesize the reject the venue
                        # never sent and route it through pm.process_event — the same path
                        # venue events take — so the normal handlers do the transition/
                        # revert and the PM fires the strategy callback error-isolated
                        # (with metrics). The AM never calls the strategy directly. The
                        # tick runs on the strategy thread (PM-scheduled), so apply stays
                        # single-mutator.
                        # Give-ups (and WS reconnects) are WARNING-log-only counters: the
                        # AM holds no metric emitter — that seam lives on the PM, which
                        # does emit reconcile terminated/materialized off the ApplyResult
                        # diff it already sees.
                        retries = state.get_retry(cid)
                        logger.warning(
                            f"[{exchange}] inflight give-up: no venue answer for {cid} "
                            f"(status={order.status}, vid={order.venue_order_id}) after "
                            f"{retries} status-fetch attempts; synthesizing reject"
                        )
                        assert self._pm is not None  # ticks only register with a pm
                        self._pm.process_event(reconcile.giveup_event(order, retries))
                    else:
                        self._connectors[exchange].request_order_status(
                            client_order_id=cid,
                            venue_order_id=order.venue_order_id,
                            instrument=order.instrument,
                        )
                        state.bump_retry(cid)
                        order.last_updated_at = now
                except Exception:
                    # One bad order / connector error must not abort the rest of the sweep
                    # or skip terminal eviction (design.md "Ticks & sweeps"). Strategy-callback errors are
                    # already isolated inside the PM dispatch the give-up path routes through.
                    logger.exception(f"[{exchange}] inflight sweep failed for {cid}")
        self._sweep_terminal_evictions()

    def _on_snapshot_tick(self, ctx) -> None:
        now = self._time.time()
        for exchange, state in self._states.items():
            if reconcile.snapshot_due(state, now, self._snapshot_interval):
                try:
                    self._connectors[exchange].request_snapshot()
                except Exception:
                    # One bad connector must not abort the snapshot backstop for the rest.
                    logger.exception(f"[{exchange}] periodic snapshot request failed")

    def _on_liveness_tick(self, ctx) -> None:
        now = self._time.time()
        for exchange, connector in self._connectors.items():
            try:
                ws_ready = connector.is_ws_ready()
            except Exception:
                # Same isolation rule: a raising liveness check on one connector must not
                # abort the checks for the rest. The unready timer is left as-is.
                logger.exception(f"[{exchange}] liveness check failed")
                continue
            if ws_ready:
                self._liveness_unready_since.pop(exchange, None)
                continue
            since = self._liveness_unready_since.setdefault(exchange, now)
            if reconcile.liveness_overdue(since, now, self._liveness_threshold):
                logger.warning(f"[{exchange}] WS unready past threshold; reconnecting")
                try:
                    reconnected = connector.reconnect()
                except Exception:
                    logger.exception(f"reconnect failed for {exchange}")
                    continue
                if reconnected:
                    # Clear only on success; on failure keep the timestamp so the next
                    # tick retries instead of restarting the full threshold.
                    self._liveness_unready_since.pop(exchange, None)
                else:
                    logger.warning(f"[{exchange}] reconnect failed; will retry on next liveness tick")


class SimulatedAccountManager(AccountManager):
    """Backtest variant — no asyncio, no WS, no periodic ticks."""

    def __init__(
        self,
        *,
        connectors,
        base_currencies: dict[str, str],
        time,
        cfg: AccountManagerConfig | None = None,
        account_id: str = "SimulatedAccountManager",
        tcc: dict[str, TransactionCostsCalculator] | None = None,
    ):
        super().__init__(
            connectors=connectors,
            base_currencies=base_currencies,
            time=time,
            cfg=cfg,
            account_id=account_id,
            tcc=tcc,
        )

    def set_processing_manager(self, pm: IProcessingManager) -> None:
        # Backtest is synchronous: NEVER register periodic ticks (no PM scheduler drives
        # them in simulation). Overrides the base, which would otherwise store pm and schedule.
        pass
