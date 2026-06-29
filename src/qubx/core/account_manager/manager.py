"""Central account-state owner: per-exchange AccountStates + the event apply path.

Routes a typed AccountMessage to the right exchange's state, applies it via the reducer
(on the AM clock), and exposes the cross-exchange read facade + aggregated metrics.
Periodic ticks (reconcile, sweep, liveness) call the reconcile.py decision helpers and
fire the connector requests — connector calls stay manager-only, as does PM wiring.
"""

from typing import Callable, Literal

import numpy as np
import pandas as pd

from qubx import area_logger
from qubx.core.account_manager import reconcile, reducer
from qubx.core.account_manager.config import AccountManagerConfig
from qubx.core.account_manager.diffs import Differ
from qubx.core.account_manager.reconciler import (
    Reconciler,
    RequestHistDeals,
    RequestSnapshot,
    RequestStatus,
    RouteEvent,
)
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
from qubx.core.interfaces import IAccountViewer, IProcessingManager
from qubx.utils.time import timedelta_to_crontab

# Module logger bound to the "account_manager" area (see reducer.py for the contract).
logger = area_logger("account_manager")


class AccountManager(IAccountViewer):
    account_id: str

    _pm: IProcessingManager | None
    _connectors: dict[str, IConnector]
    _time: ITimeProvider
    _cfg: AccountManagerConfig
    _tcc: dict[str, TransactionCostsCalculator]
    _states: dict[str, AccountState]
    _snapshot_grace: np.timedelta64
    _snapshot_interval: np.timedelta64
    _liveness_threshold: np.timedelta64
    _terminal_retention: np.timedelta64
    _liveness_unready_since: dict[str, np.datetime64]
    _last_eviction_sweep: np.datetime64
    _reconcilers: dict[str, Reconciler]

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
        self._snapshot_grace = np.timedelta64(self._cfg.snapshot_grace_ms, "ms")
        self._snapshot_interval = np.timedelta64(self._cfg.snapshot_interval_ms, "ms")
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
        # Stage-2 Reconciler per exchange. Wired into on_event now (no-op until on_snapshot
        # spawns tasks); on_snapshot/on_tick wiring + the old-path sweep land in a later step.
        self._reconcilers = {
            ex: Reconciler(
                Differ(grace=self._snapshot_grace),
                snapshot_interval=self._snapshot_interval,
                missing_wait=np.timedelta64(self._cfg.missing_order_wait_ms, "ms"),
                missing_max_retries=self._cfg.missing_order_retries,
                position_confirm_wait=np.timedelta64(self._cfg.position_confirm_wait_ms, "ms"),
                order_confirm_wait=np.timedelta64(self._cfg.order_confirm_wait_ms, "ms"),
                order_confirm_max_retries=self._cfg.order_confirm_retries,
            )
            for ex in connectors
        }

    def _register_ticks(self) -> None:
        cfg = self._cfg
        assert self._pm is not None

        # One reconcile heartbeat drives the Reconciler: its on_tick gates the snapshot request
        # (snapshot_interval) AND nudges the order/position tasks. Replaces the old inflight +
        # snapshot ticks. Fires at the (fast) inflight interval so task timers stay responsive.
        if cfg.reconcile_tick_interval_ms > 0:
            self._pm.schedule(
                timedelta_to_crontab(pd.Timedelta(cfg.reconcile_tick_interval_ms, "ms")), self._on_reconcile_tick
            )
        if cfg.liveness_check_interval_ms > 0:
            self._pm.schedule(
                timedelta_to_crontab(pd.Timedelta(cfg.liveness_check_interval_ms, "ms")), self._on_liveness_tick
            )

    def add_order(self, order: Order) -> None:
        exchange = order.instrument.exchange
        state = self._states[exchange]
        state.add_order(order)

        # - spawn AwaitOrderConfirm for the freshly-sent order (no immediate I/O)
        self._reconcilers[exchange].on_order_sent(state, order, self._time.time())

    def transition_order(self, exchange: str, cid: str, new_status: OrderStatus) -> None:
        reconcile.transition(self._states[exchange], cid, new_status, self._time.time())

    def remove_order(self, exchange: str, cid: str) -> None:
        self._states[exchange].remove_order(cid)

    # ---- state seeding & cash adjustments (runner/transfer boundary) ------ #
    # The only sanctioned out-of-AM mutation paths: AccountState mutators stay
    # AM-internal, callers go through these.

    def seed_balance(self, exchange: str, balance: Balance) -> bool:
        """
        Seed a startup balance (restored state, initial paper/backtest capital).
        Returns False when this AM doesn't manage the exchange (record skipped).
        """
        if (state := self._states.get(exchange)) is None:
            return False
        state.update_balance(balance.currency, balance)
        return True

    def seed_position(self, position: Position) -> bool:
        """
        Seed a restored position (exchange derived from the instrument).
        Returns False when this AM doesn't manage the exchange (record skipped).
        """
        if (state := self._states.get(position.instrument.exchange)) is None:
            return False
        state.set_position(position.instrument, position)
        return True

    def adjust_balance(self, exchange: str, currency: str, delta: float) -> None:
        """
        Apply a cash delta (e.g. a simulated transfer leg) to an exchange balance,
        creating the Balance if missing.
        """
        self._states[exchange].adjust_balance(currency, delta)

    def settle_position(self, instrument: Instrument) -> None:
        """
        Flatten a delisted/gone position in place (no trade): the venue already
        cash-settled it, so the universe manager reconciles the in-memory position to
        flat. Routed per-exchange via the instrument; no-op if the exchange is unmanaged
        or the position isn't held.
        """
        if (state := self._states.get(instrument.exchange)) is not None:
            state.settle_position(instrument)

    def apply(self, event: AccountMessage) -> ApplyResult:
        if (state := self._state_for_event(event)) is None:
            return ApplyResult()
        now = self._time.time()
        result = self._apply_to_state(state, event, now)
        self._maybe_sweep_evictions(now)
        return result

    def _apply_to_state(self, state: AccountState, event: AccountMessage, now: np.datetime64) -> ApplyResult:
        rec = self._reconcilers[state.exchange]
        # Snapshots are owned by the Reconciler (diff + apply + tasks + venue figures); it
        # collects the reconciled positions so the PM still fires on_position_change.
        if isinstance(event, AccountSnapshotEvent):
            changed: list[Position] = []
            self._execute(state, rec.on_snapshot(state, event.snapshot, now, changed_positions=changed))
            return ApplyResult(positions=changed)

        # Everything else goes through the reducer; order/deal events additionally drive the
        # Reconciler's tasks (coverage, resolve-by-event).
        result = reducer.apply(state, event, now)
        if isinstance(event, OrderEvent):
            self._execute(state, rec.on_event(state, event, now))

        return result

    def _maybe_sweep_evictions(self, now: np.datetime64) -> None:
        # Opportunistic terminal eviction: SimulatedAccountManager (paper + backtest) registers
        # no periodic ticks, so without this sweep terminal orders and their side-table entries
        # would grow unbounded there (and in live when reconcile_tick_interval_ms=0). One
        # timestamp comparison per apply; the sweep runs at most once per retention window. Live
        # additionally sweeps from the reconcile tick so eviction stays prompt during event silence.
        if now - self._last_eviction_sweep >= self._terminal_retention:
            self._sweep_terminal_evictions()

    def _execute(self, state: AccountState, actions: list) -> None:
        # - perform the I/O the Reconciler asked for (connector calls / event routing). Connector
        #   calls stay manager-only; one bad action must not abort the rest.
        connector = self._connectors.get(state.exchange)
        for action in actions:
            try:
                match action:
                    case RequestStatus(cid=cid):
                        order = state.get_order(cid)
                        if order is not None and connector is not None:
                            connector.request_order_status(order)

                    case RequestSnapshot():
                        if connector is not None:
                            connector.request_snapshot()

                    case RouteEvent(event=routed):
                        # - HACK, do NOT copy: process_event is SYNCHRONOUS, and _execute runs
                        #   inside apply() — so this re-enters apply() recursively (same thread,
                        #   bounded, so it works). The clean form is to enqueue on the channel and
                        #   let the next drain apply it
                        if self._pm is not None:
                            self._pm.process_event(routed)

                    case RequestHistDeals(instrument=instrument, since=since):
                        if connector is not None:
                            logger.debug(
                                f"[{state.exchange}] reconcile: RequestHistDeals "
                                f"<y>{instrument.symbol}</y> since {since} -> connector.request_hist_deals"
                            )
                            connector.request_hist_deals(instrument, since)

                    case _:
                        logger.warning(f"[{state.exchange}] unknown reconcile action: {action!r}")
            except Exception:
                logger.exception(f"[{state.exchange}] reconcile action failed: {action!r}")

    def _state_for_event(self, event: AccountMessage) -> AccountState | None:
        # - route by the most specific locator the event carries
        match event:
            case AccountSnapshotEvent():
                return self._states.get(event.snapshot.exchange)

            case BalanceUpdateEvent():
                # balance pushes carry no instrument — route strictly by the balance's exchange
                # (the sole-state fallback would misroute a push for an unmanaged exchange)
                return self._states.get(event.balance.exchange)

            case _ if event.instrument is not None:
                return self._states.get(event.instrument.exchange)

        # no instrument: an order event (e.g. a reject emitted without one) routes by its own
        # ids; otherwise the sole managed state, else give up.
        state = self._state_holding_order(event) if isinstance(event, OrderEvent) else None
        if state is None and len(self._states) == 1:
            state = next(iter(self._states.values()))
        if state is None:
            logger.warning(f"cannot route {type(event).__name__} with no instrument/identifiers — dropped")
        return state

    def _state_holding_order(self, event: OrderEvent) -> AccountState | None:
        # - the state already tracking this order — by cid (primary id) first, then venue id
        states = self._states.values()
        by_cid = next((s for s in states if s.get_order(event.client_order_id) is not None), None)
        if by_cid is not None or event.venue_order_id is None:
            return by_cid
        return next((s for s in states if s.get_order_by_venue_id(event.venue_order_id) is not None), None)

    def on_market_quote(self, instrument, quote) -> None:
        if (state := self._states.get(instrument.exchange)) is None:
            return

        # only mark positions we hold; never create one per quote
        if (pos := state.get_position(instrument)) is None:
            return

        pos.update_market_price(
            self._time.time(), quote.mid_price(), state.conversion_rate(instrument), stamp_update_time=False
        )

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
        if (state := self._states.get(instrument.exchange)) is None:
            return None

        if (pos := state.get_position(instrument)) is None:
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

    def get_balance(self, currency: str, exchange: str | None = None) -> Balance:
        # IAccountViewer contract: never None — a currency the account never held reads
        # as a detached zero Balance (mirrors get_position's materialize-flat rule,
        # without storing it). Currency codes are uppercase throughout (venue payloads,
        # base_currency normalization) — normalize the lookup so 'usdt' cannot silently
        # read as an empty wallet.
        currency = currency.upper()
        if exchange is not None:
            return self._states[exchange].get_balance(currency) or Balance(exchange, currency)

        for state in self._states.values():
            if (b := state.get_balance(currency)) is not None:
                return b

        return Balance(next(iter(self._states)), currency)

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

    def get_withdrawable_balance(self, exchange: str | None = None) -> float:
        return self._sum(AccountState.withdrawable_balance, exchange)

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

    def _on_reconcile_tick(self, ctx) -> None:
        # The single reconcile heartbeat: drive each exchange's Reconciler (snapshot request +
        # task timers) and execute the I/O it asks for, then run the terminal-eviction sweep.
        now = self._time.time()
        for exchange, state in self._states.items():
            try:
                self._execute(state, self._reconcilers[exchange].on_tick(state, now))
            except Exception:
                logger.exception(f"[{exchange}] reconcile tick failed")
        self._sweep_terminal_evictions()

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
