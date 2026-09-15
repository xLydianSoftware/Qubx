import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

from qubx import logger
from qubx.core.basics import DataType, Instrument, dt_64
from qubx.core.exceptions import NotSupported
from qubx.core.interfaces import IDataProvider, IHealthMonitor, StrategyState
from qubx.core.status import ContextStatus, DegradeReason
from qubx.health.base import STALE_THRESHOLDS
from qubx.health.status import ExchangeDataStatus
from qubx.utils.time import convert_tf_str_td64

# Data types with a continuous feed and a staleness threshold worth policing.
_WATCHDOG_DATA_TYPES = frozenset({DataType.QUOTE, DataType.ORDERBOOK, DataType.TRADE})

# Consecutive DARK ticks before EXCHANGE_MAINTENANCE is published. Every healthy
# reconnect observed during the 2026-09-13 outage completed in 1-4s, so one tick would
# flap the order path on routine blips.
_DARK_TICKS_BEFORE_MAINTENANCE = 2

# Repair backoff caps at threshold/10 - 1min for orderbook/quote, 3min for trade. Tying
# it to the type's own threshold keeps retries aggressive where feeds tick continuously
# and patient where they legitimately do not.
_BACKOFF_CAP_DIVISOR = 10

# Bound on stop()'s join. StrategyContext.stop() calls it AFTER the data providers are
# closed, so an in-flight tick can be sitting in the settle sleep or blocking against a
# just-closed provider; the thread is a daemon, so abandoning it is safer than hanging
# teardown. Comfortably over the settle sleep.
_STOP_JOIN_TIMEOUT_SECONDS = 5.0

# (instrument, base data type): the unit of staleness, repair and verification.
_Key = tuple[Instrument, str]
# subscription key -> instruments, for one exchange
_Subs = dict[str, frozenset[Instrument]]


class ExchangeClassification(StrEnum):
    OK = "ok"
    DARK = "dark"
    PARTIAL = "partial"


@dataclass(slots=True)
class _RepairRecord:
    repaired_at: dt_64
    interval_ticks: int = 1
    ticks_until_retry: int = 1


@dataclass(slots=True)
class _ExchangeState:
    venue: str  # the provider's configured name; the health monitor keys is_connected by it
    provider: IDataProvider
    dark_ticks: int = 0
    maintenance_held: bool = False
    repairs: dict[_Key, _RepairRecord] = field(default_factory=dict)
    unsupported: set[str] = field(default_factory=set)  # subscription keys the venue rejected


def _base_type(sub: str) -> str:
    return str(DataType.from_str(sub)[0])


class SubscriptionWatchdog:
    """Detects instruments whose feed has stopped and re-establishes their transport.

    Owns no subscription state: intent lives in SubscriptionManager and is read here as
    a snapshot, so a repair this class fails to complete is retried rather than lost.
    Runs entirely on its own thread; the only shared state it touches is that snapshot
    (copied under the manager's lock) and ContextStatus (thread-safe by design).
    """

    def __init__(
        self,
        data_providers: list[IDataProvider],
        health_monitor: IHealthMonitor,
        status: ContextStatus,
        snapshot_fn: Callable[[], dict[str, _Subs]],
        strategy_state: StrategyState,
        interval_seconds: float = 30.0,
        settle_seconds: float = 3.0,
    ) -> None:
        # - local import: qubx.core.mixins.__init__ imports subscription.py, which imports
        #   this module, so a module-level import here would close a cycle
        from qubx.core.mixins.utils import canonical_exchange

        self._health_monitor = health_monitor
        self._status = status
        self._snapshot_fn = snapshot_fn
        self._strategy_state = strategy_state
        self._interval_seconds = interval_seconds
        self._settle_seconds = settle_seconds
        # - keyed by the canonical exchange instruments carry, which is also the scope the
        #   order path checks (is_degraded_for(instrument.exchange)); the provider itself
        #   may be registered under a venue alias (BINANCE.PM trades BINANCE.UM)
        self._exchanges: dict[str, _ExchangeState] = {
            canonical_exchange(p.exchange()): _ExchangeState(venue=p.exchange(), provider=p)
            for p in data_providers
            if not p.is_simulation
        }
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ----- policy -----

    @staticmethod
    def classify(status: ExchangeDataStatus) -> ExchangeClassification:
        # - limb (A) needs something to be dark ABOUT: with no eligible instrument (an
        #   exchange carrying only unpoliced types, or one whose whole universe is still
        #   in grace) a flapping or unanswerable is_connected() would otherwise publish
        #   EXCHANGE_MAINTENANCE and refuse every order - reduce-only included - on a
        #   venue this watchdog is not policing at all. The incident had 21 eligible.
        if status.connected is False and status.subscribed >= 1:
            return ExchangeClassification.DARK
        if status.stale == 0:
            return ExchangeClassification.OK
        if status.stale == status.subscribed and status.subscribed >= 2:
            return ExchangeClassification.DARK
        return ExchangeClassification.PARTIAL

    # ----- lifecycle -----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SubscriptionWatchdog")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is None or thread is threading.current_thread():
            # - stop() from inside a tick: joining self raises RuntimeError. The set event
            #   already guarantees the loop exits at its next wait().
            return
        thread.join(timeout=_STOP_JOIN_TIMEOUT_SECONDS)
        if thread.is_alive():
            logger.warning(
                f"[SubscriptionWatchdog] :: thread still running after {_STOP_JOIN_TIMEOUT_SECONDS}s; "
                "abandoning it (daemon) rather than blocking shutdown"
            )

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.tick()

    # ----- tick -----

    def tick(self) -> None:
        try:
            if not self._strategy_state.is_on_warmup_finished_called:
                return
            universe = self._snapshot_fn()
        except Exception as e:
            logger.error(f"[SubscriptionWatchdog] :: tick failed: {e}")
            return

        # - per-exchange try/except: one venue's failure must not skip the rest
        for exchange, state in self._exchanges.items():
            try:
                self._tick_exchange(exchange, state, universe.get(exchange, {}))
            except Exception as e:
                logger.error(f"[{exchange}] :: watchdog tick failed: {e}")

    def _tick_exchange(self, exchange: str, state: _ExchangeState, subs: _Subs) -> None:
        policed = {sub: instrs for sub, instrs in subs.items() if _base_type(sub) in _WATCHDOG_DATA_TYPES}
        status = self._health_monitor.get_exchange_data_status(state.venue, policed)
        classification = self.classify(status)

        if classification is ExchangeClassification.DARK:
            state.dark_ticks += 1
            if state.dark_ticks >= _DARK_TICKS_BEFORE_MAINTENANCE:
                self._hold_maintenance(exchange, state, status)
            return

        # - gated on maintenance having actually been held (not merely "was dark for
        #   one tick"): a single-tick blip tore nothing down and needs no re-assert.
        #   A flapping exchange that never reaches the threshold must not repeatedly
        #   re-subscribe the whole universe - that resubscribe pressure is what cost
        #   the 2026-09-13 recovery.
        recovering = state.dark_ticks >= _DARK_TICKS_BEFORE_MAINTENANCE
        state.dark_ticks = 0
        self._clear_maintenance(exchange, state)
        if recovering:
            # - transport is back: re-assert everything on this exchange once, policed
            #   or not, rather than repairing instrument by instrument
            self._reassert(state, subs)
            return

        if classification is ExchangeClassification.OK:
            state.repairs.clear()
            return

        self._repair_stale(exchange, state, policed, status.stale_keys)

    # ----- repair -----

    def _repair_stale(self, exchange: str, state: _ExchangeState, subs: _Subs, stale: frozenset[_Key]) -> None:
        # - stale_keys already excludes instruments in their grace window, so a freshly
        #   subscribed instrument that has simply not delivered yet is never torn down.
        #   A record no longer stale is moot: it delivered, left the universe, or is fresh.
        for key in [k for k in state.repairs if k not in stale]:
            del state.repairs[key]

        due: list[_Key] = []
        for key in stale:
            record = state.repairs.get(key)
            if record is None:
                due.append(key)
                continue
            last_event = self._health_monitor.get_last_event_time(*key)
            if last_event is not None and last_event > record.repaired_at:
                # - verified by advancement: a message arrived after the repair, so the
                #   subscription is live whatever the staleness threshold says
                del state.repairs[key]
                continue
            record.ticks_until_retry -= 1
            if record.ticks_until_retry <= 0:
                due.append(key)

        if not due:
            return

        self._repair(state, subs, set(due))

        # - stamped AFTER the repair, which sleeps at least the settle delay: a message
        #   that arrived before the unsubscribe must not read as proof the repair worked,
        #   or a permanently wedged instrument never escalates to ERROR
        now = self._health_monitor.time_provider.time()
        for key in due:
            instrument, base_type = key
            record = state.repairs.get(key)
            if record is None:
                state.repairs[key] = _RepairRecord(now)
                logger.info(f"[{exchange}] :: repairing {base_type} for {instrument.symbol}")
                continue
            cap = self._backoff_cap_ticks(base_type)
            record.repaired_at = now
            record.interval_ticks = min(record.interval_ticks * 2, cap)
            record.ticks_until_retry = record.interval_ticks
            level = logger.error if record.interval_ticks >= cap else logger.warning
            level(
                f"[{exchange}] :: {base_type} for {instrument.symbol} still not delivering after "
                f"repair; retrying every {record.interval_ticks} ticks"
            )

    def _repair(self, state: _ExchangeState, subs: _Subs, refresh: set[_Key]) -> None:
        """Re-establish transport for `refresh` and nothing else.

        Keyed by (instrument, base type), so a wedged trade feed does not tear down that
        instrument's orderbook. Every unsubscribe is paired with a subscribe of the FULL
        desired set for that subscription. The settle delay between them guards against
        the venue processing the two out of order and bounds churn; it is not the safety
        net - the caller's retry is.
        """
        by_type: dict[str, set[Instrument]] = defaultdict(set)
        for instrument, base_type in refresh:
            by_type[base_type].add(instrument)

        targets = [
            (sub, desired, hit)
            for sub, desired in subs.items()
            if sub not in state.unsupported and (hit := by_type.get(_base_type(sub), set()) & desired)
        ]
        for sub, _, hit in targets:
            try:
                state.provider.unsubscribe(sub, set(hit))
            except Exception as e:
                # - still gets its paired subscribe: re-subscribing is idempotent on every
                #   connector, so trying is never worse than leaving it torn down
                logger.error(f"[{state.venue}] :: unsubscribe of {sub} failed: {e}")
        if targets:
            time.sleep(self._settle_seconds)
        for sub, desired, _ in targets:
            self._subscribe(state, sub, desired)

    def _reassert(self, state: _ExchangeState, subs: _Subs) -> None:
        for sub, desired in subs.items():
            if sub not in state.unsupported:
                self._subscribe(state, sub, desired)

    def _subscribe(self, state: _ExchangeState, sub: str, desired: frozenset[Instrument]) -> None:
        try:
            state.provider.subscribe(sub, set(desired), reset=True)
        except NotSupported as e:
            state.unsupported.add(sub)
            logger.warning(f"[{state.venue}] :: {sub} not supported: {e}")
        except Exception as e:
            logger.error(f"[{state.venue}] :: subscribe of {sub} failed: {e}")

    def _backoff_cap_ticks(self, base_type: str) -> int:
        threshold = STALE_THRESHOLDS.get(base_type)
        if threshold is None:
            return 1
        cap_seconds = convert_tf_str_td64(threshold).astype("timedelta64[s]").astype(int) / _BACKOFF_CAP_DIVISOR
        return max(1, int(cap_seconds // self._interval_seconds))

    # ----- status -----

    def _hold_maintenance(self, exchange: str, state: _ExchangeState, status: ExchangeDataStatus) -> None:
        if state.maintenance_held:
            return
        state.maintenance_held = True
        reason = "connection down" if status.connected is False else "no instrument delivering"
        logger.error(f"[{exchange}] :: exchange is dark ({reason}) - holding EXCHANGE_MAINTENANCE")
        self._status.add(
            DegradeReason.EXCHANGE_MAINTENANCE,
            self._health_monitor.time_provider.time(),
            scope=exchange,
            message=reason,
        )

    def _clear_maintenance(self, exchange: str, state: _ExchangeState) -> None:
        if not state.maintenance_held:
            return
        state.maintenance_held = False
        logger.info(f"[{exchange}] :: data resumed - clearing EXCHANGE_MAINTENANCE")
        self._status.clear(DegradeReason.EXCHANGE_MAINTENANCE, scope=exchange)
