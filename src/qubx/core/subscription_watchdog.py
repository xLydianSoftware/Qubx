import threading
from collections import defaultdict
from enum import StrEnum
from typing import Callable

from qubx import logger
from qubx.core.basics import DataType, Instrument, dt_64
from qubx.core.interfaces import IDataProvider, IHealthMonitor, StrategyState
from qubx.core.status import ContextStatus, DegradeReason
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


class ExchangeClassification(StrEnum):
    OK = "ok"
    DARK = "dark"
    PARTIAL = "partial"


class _RepairRecord:
    __slots__ = ("repaired_at", "interval_ticks", "ticks_until_retry")

    def __init__(self, repaired_at: dt_64, interval_ticks: int) -> None:
        self.repaired_at = repaired_at
        self.interval_ticks = interval_ticks
        self.ticks_until_retry = interval_ticks


class SubscriptionWatchdog:
    """Detects instruments whose feed has stopped and drives repair through reconcile().

    Owns no subscription state: intent lives in SubscriptionManager, so a repair this
    class fails to complete is retried rather than lost.
    """

    def __init__(
        self,
        data_providers: list[IDataProvider],
        health_monitor: IHealthMonitor,
        status: ContextStatus,
        reconcile_fn: Callable[..., None],
        subscriptions_fn: Callable[[], dict[str, set[Instrument]]],
        strategy_state: StrategyState,
        interval_seconds: float = 30.0,
    ) -> None:
        self._data_providers = data_providers
        self._health_monitor = health_monitor
        self._status = status
        self._reconcile_fn = reconcile_fn
        self._subscriptions_fn = subscriptions_fn
        self._strategy_state = strategy_state
        self._interval_seconds = interval_seconds
        self._repair_state: dict[tuple[Instrument, str], _RepairRecord] = {}
        self._dark_ticks: dict[str, int] = defaultdict(int)
        self._maintenance_held: set[str] = set()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ----- policy -----

    @staticmethod
    def classify(status: ExchangeDataStatus) -> ExchangeClassification:
        if status.connected is False:
            return ExchangeClassification.DARK
        if status.stale == 0:
            return ExchangeClassification.OK
        if status.stale == status.subscribed and status.subscribed >= 2:
            return ExchangeClassification.DARK
        return ExchangeClassification.PARTIAL

    # ----- loop -----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SubscriptionWatchdog")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.tick()

    def tick(self) -> None:
        try:
            if not self._strategy_state.is_on_warmup_finished_called:
                return
            universe = self._watchdog_subscriptions()
        except Exception as e:
            logger.error(f"[SubscriptionWatchdog] :: tick failed: {e}")
            return

        # - per-provider try/except: one exchange's failure must not skip the rest
        for provider in self._data_providers:
            try:
                if provider.is_simulation:
                    continue
                self._tick_exchange(provider.exchange(), universe)
            except Exception as e:
                logger.error(f"[SubscriptionWatchdog] :: tick failed: {e}")

    def _watchdog_subscriptions(self) -> dict[str, set[Instrument]]:
        return {
            sub: instrs
            for sub, instrs in self._subscriptions_fn().items()
            if DataType.from_str(sub)[0] in _WATCHDOG_DATA_TYPES and instrs
        }

    def _tick_exchange(self, exchange: str, universe: dict[str, set[Instrument]]) -> None:
        status = self._health_monitor.get_exchange_data_status(exchange, universe)
        classification = self.classify(status)

        if classification is ExchangeClassification.DARK:
            self._dark_ticks[exchange] += 1
            if self._dark_ticks[exchange] >= _DARK_TICKS_BEFORE_MAINTENANCE:
                self._hold_maintenance(exchange, status)
            return

        # - gated on maintenance having actually been held (not merely "was dark for
        #   one tick"): a single-tick blip tore nothing down and needs no re-assert.
        #   A flapping exchange that never reaches the threshold must not repeatedly
        #   re-subscribe the whole universe - that resubscribe pressure is what cost
        #   the 2026-09-13 recovery.
        recovering_from_maintenance = self._dark_ticks[exchange] >= _DARK_TICKS_BEFORE_MAINTENANCE
        self._dark_ticks[exchange] = 0
        self._clear_maintenance(exchange)
        if recovering_from_maintenance:
            # - transport is back: re-assert the whole universe once rather than
            #   repairing instrument by instrument
            self._reconcile_fn()
            return

        if classification is ExchangeClassification.OK:
            self._forget_repairs(exchange)
            return

        self._repair_stale(exchange, universe)

    # ----- repair -----

    def _repair_stale(self, exchange: str, universe: dict[str, set[Instrument]]) -> None:
        now = self._health_monitor.time_provider.time()
        self._prune_dropped_from_universe(exchange, universe)

        # - keyed by (instrument, base_type), not bare instrument: an instrument
        #   subscribed to multiple types must not have a due repair on one type stamp
        #   a bogus record on a type that was never stale
        due: set[tuple[Instrument, str]] = set()

        for sub, instruments in universe.items():
            base_type = str(DataType.from_str(sub)[0])
            for instrument in instruments:
                if instrument.exchange != exchange:
                    continue
                key = (instrument, base_type)
                if not self._health_monitor.is_stale(instrument, base_type):
                    self._repair_state.pop(key, None)
                    continue
                record = self._repair_state.get(key)
                if record is None:
                    due.add(key)
                    continue
                last_event = self._health_monitor.get_last_event_time(instrument, base_type)
                if last_event is not None and last_event > record.repaired_at:
                    # - verified by advancement: a message arrived after the repair, so
                    #   the subscription is live whatever the staleness threshold says
                    self._repair_state.pop(key, None)
                    continue
                record.ticks_until_retry -= 1
                if record.ticks_until_retry <= 0:
                    due.add(key)

        if not due:
            return

        self._reconcile_fn(refresh={instrument for instrument, _ in due})

        for sub, instruments in universe.items():
            base_type = str(DataType.from_str(sub)[0])
            cap = self._backoff_cap_ticks(base_type)
            for instrument in instruments:
                key = (instrument, base_type)
                if key not in due:
                    continue
                record = self._repair_state.get(key)
                if record is None:
                    self._repair_state[key] = _RepairRecord(now, 1)
                    logger.info(f"[{exchange}] :: repairing {sub} for {instrument.symbol}")
                else:
                    record.repaired_at = now
                    record.interval_ticks = min(record.interval_ticks * 2, cap)
                    record.ticks_until_retry = record.interval_ticks
                    level = logger.error if record.interval_ticks >= cap else logger.warning
                    level(
                        f"[{exchange}] :: {sub} for {instrument.symbol} still not delivering "
                        f"after repair; retrying every {record.interval_ticks} ticks"
                    )

    def _prune_dropped_from_universe(self, exchange: str, universe: dict[str, set[Instrument]]) -> None:
        current_keys = {
            (instrument, str(DataType.from_str(sub)[0]))
            for sub, instruments in universe.items()
            for instrument in instruments
            if instrument.exchange == exchange
        }
        for key in [k for k in self._repair_state if k[0].exchange == exchange and k not in current_keys]:
            self._repair_state.pop(key, None)

    def _backoff_cap_ticks(self, base_type: str) -> int:
        from qubx.health.base import STALE_THRESHOLDS

        threshold = STALE_THRESHOLDS.get(base_type)
        if threshold is None:
            return 1
        cap_seconds = convert_tf_str_td64(threshold).astype("timedelta64[s]").astype(int)
        cap_seconds = cap_seconds / _BACKOFF_CAP_DIVISOR
        return max(1, int(cap_seconds // self._interval_seconds))

    def _forget_repairs(self, exchange: str) -> None:
        for key in [k for k in self._repair_state if k[0].exchange == exchange]:
            self._repair_state.pop(key, None)

    # ----- status -----

    def _hold_maintenance(self, exchange: str, status: ExchangeDataStatus) -> None:
        if exchange in self._maintenance_held:
            return
        self._maintenance_held.add(exchange)
        reason = "connection down" if status.connected is False else "no instrument delivering"
        logger.error(f"[{exchange}] :: exchange is dark ({reason}) - holding EXCHANGE_MAINTENANCE")
        self._status.add(
            DegradeReason.EXCHANGE_MAINTENANCE,
            self._health_monitor.time_provider.time(),
            scope=exchange,
            message=reason,
        )

    def _clear_maintenance(self, exchange: str) -> None:
        if exchange not in self._maintenance_held:
            return
        self._maintenance_held.discard(exchange)
        logger.info(f"[{exchange}] :: data resumed - clearing EXCHANGE_MAINTENANCE")
        self._status.clear(DegradeReason.EXCHANGE_MAINTENANCE, scope=exchange)
