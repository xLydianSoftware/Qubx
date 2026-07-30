import pprint
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from qubx import logger
from qubx.core.basics import DataType, Instrument
from qubx.core.exceptions import NotSupported
from qubx.core.fit_executor import SingleThreadWorker
from qubx.core.interfaces import IDataProvider, IHealthMonitor, ISubscriptionManager, ITimeProvider, StrategyState
from qubx.utils.misc import synchronized

from .utils import EXCHANGE_MAPPINGS

# Channel d_type under which a deferred subscription swap rides the CtrlChannel as a
# (None, SUBSCRIPTION_SWAP_EVENT, apply_callable, False) tuple; posted by the warmup
# worker when its fetch completes, dispatched to ProcessingManager._handle_subscription_swap
# and applied on the ProcessorThread.
SUBSCRIPTION_SWAP_EVENT = "subscription_swap"


@dataclass(frozen=True, slots=True)
class _CommitPlan:
    """Snapshot of the pending subscription operations taken by one commit().

    The swap plan is fixed at commit time so subscribe()/unsubscribe() calls arriving
    while a deferred (background-warmup) commit is in flight accumulate toward the NEXT
    commit instead of mutating this one.
    """

    stream_subscriptions: dict[str, set[Instrument]] = field(default_factory=dict)
    stream_unsubscriptions: dict[str, set[Instrument]] = field(default_factory=dict)
    global_subscriptions: set[str] = field(default_factory=set)
    global_unsubscriptions: set[str] = field(default_factory=set)


class SubscriptionManager(ISubscriptionManager):
    _time_provider: ITimeProvider
    _data_providers: list[IDataProvider]
    _exchange_to_data_provider: dict[str, IDataProvider]
    _health_monitor: IHealthMonitor
    _base_sub: str
    _sub_to_warmup: dict[str, str]
    _auto_subscribe: bool

    _pending_global_subscriptions: set[str]
    _pending_global_unsubscriptions: set[str]

    _pending_stream_subscriptions: dict[str, set[Instrument]]
    _pending_stream_unsubscriptions: dict[str, set[Instrument]]
    _pending_warmups: dict[tuple[str, Instrument], str]

    def __init__(
        self,
        time_provider: ITimeProvider,
        data_providers: list[IDataProvider],
        health_monitor: IHealthMonitor,
        strategy_state: StrategyState,
        auto_subscribe: bool = True,
        default_base_subscription: str = DataType.NONE,
        monitor_interval_seconds: float = 30.0,
    ) -> None:
        self._time_provider = time_provider
        self._data_providers = data_providers
        self._exchange_to_data_provider = {data_provider.exchange(): data_provider for data_provider in data_providers}
        self._health_monitor = health_monitor
        self._strategy_state = strategy_state
        self._base_sub = default_base_subscription
        self._sub_to_warmup = {}
        self._pending_warmups = {}
        self._pending_global_subscriptions = set()
        self._pending_global_unsubscriptions = set()
        self._pending_stream_subscriptions = defaultdict(set)
        self._pending_stream_unsubscriptions = defaultdict(set)
        self._auto_subscribe = auto_subscribe
        self._monitor_interval_seconds = monitor_interval_seconds
        self._is_simulation = all(data_provider.is_simulation for data_provider in data_providers)
        # Live only: warmup fetches run on this worker so they never block the
        # ProcessorThread; the swap of a deferred commit applies when its fetch is done.
        # The thread itself starts lazily on the first deferred commit.
        self._warmup_worker: SingleThreadWorker | None = (
            None if self._is_simulation else SingleThreadWorker("WarmupThread")
        )
        # In-flight deferred commits: incremented at submit, decremented on the
        # ProcessorThread once the commit's swap has been applied (a stopped channel at
        # shutdown may strand it — harmless, the process is exiting). Read via
        # is_warming_up from any thread, including the StrategyFitThread.
        self._warmup_inflight = 0
        self._warmup_inflight_lock = threading.Lock()
        self._init_connection_status_callbacks()
        self._init_subscription_monitoring()

    @synchronized
    def subscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None) -> None:
        self._subscribe(subscription_type, instruments)

    @synchronized
    def unsubscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None) -> None:
        self._unsubscribe(subscription_type, instruments)

    @property
    def is_warming_up(self) -> bool:
        return self._warmup_inflight > 0

    @synchronized
    def commit(self) -> None:
        """Apply the pending subscription operations. Always called on the ProcessorThread.

        The plan (swap + warmup set) is snapshotted here; subsequent subscribe/unsubscribe
        calls accumulate toward the next commit. Simulation (and any commit that needs no
        warmup) runs synchronously — warmup fetch then swap, today's behavior. A live
        commit that needs warmup DEFERS: the fetch runs on the WarmupThread so it never
        blocks the ProcessorThread, and the swap applies only after the fetch completes
        (see _warmup_and_post_swap for why that ordering is mandatory).
        """
        if not self._has_operations_to_commit():
            return

        plan = _CommitPlan(
            stream_subscriptions={sub: set(instrs) for sub, instrs in self._pending_stream_subscriptions.items()},
            stream_unsubscriptions={sub: set(instrs) for sub, instrs in self._pending_stream_unsubscriptions.items()},
            global_subscriptions=set(self._pending_global_subscriptions),
            global_unsubscriptions=set(self._pending_global_unsubscriptions),
        )
        self._pending_stream_subscriptions.clear()
        self._pending_stream_unsubscriptions.clear()
        self._pending_global_subscriptions.clear()
        self._pending_global_unsubscriptions.clear()

        _warmups = self._collect_warmups(plan)

        if self._warmup_worker is None or not _warmups:
            # - simulation, or nothing to warm: synchronous warmup + swap (today's path)
            self._run_warmup(_warmups)
            self._apply_swap(plan)
            return

        with self._warmup_inflight_lock:
            self._warmup_inflight += 1
        self._warmup_worker.submit(partial(self._warmup_and_post_swap, _warmups, plan))

    def _run_warmup(self, warmups: dict[IDataProvider, dict[tuple[str, Instrument], str]]) -> None:
        for _data_provider, _configs in warmups.items():
            _data_provider.warmup(_configs)

    def _warmup_and_post_swap(
        self, warmups: dict[IDataProvider, dict[tuple[str, Instrument], str]], plan: _CommitPlan
    ) -> None:
        """Deferred-commit body, runs on the WarmupThread.

        warmup() blocks this thread until the fetched history has been SENT on the
        channel (as is_historical events, destined for the market cache), then the swap
        application is posted on the same channel. FIFO ordering therefore guarantees the
        ProcessorThread lands all history in the cache BEFORE the swap subscribes the
        added instruments — mandatory, because the OHLC series reject past-data updates:
        once a live bar lands, older warmup bars would be silently dropped.

        On warmup failure the swap still applies (ERROR-logged, same degradation as a
        failed synchronous warmup today): the strategy trades the added instruments with
        whatever history is present rather than never getting them.
        """
        try:
            self._run_warmup(warmups)
        except Exception as e:
            logger.error(f"[SubscriptionManager] :: deferred warmup failed: {e}; applying the subscription swap anyway")
            logger.opt(colors=False).error(traceback.format_exc())
        finally:
            self._data_providers[0].channel.send(
                (None, SUBSCRIPTION_SWAP_EVENT, partial(self._apply_deferred_swap, plan), False)
            )

    def _apply_deferred_swap(self, plan: _CommitPlan) -> None:
        """ProcessorThread-side application of a deferred commit's swap (posted by the
        WarmupThread); error-isolated so a failing swap cannot kill the processing loop,
        and always clears the in-flight marker."""
        try:
            self._apply_swap(plan)
        except Exception as e:
            logger.error(f"[SubscriptionManager] :: deferred subscription swap failed: {e}")
            logger.opt(colors=False).error(traceback.format_exc())
        finally:
            with self._warmup_inflight_lock:
                self._warmup_inflight -= 1

    def _apply_swap(self, plan: _CommitPlan) -> None:
        # - apply: the subscribe/unsubscribe swap (fast)
        for _sub in self._get_updated_subs(plan):
            _current_sub_instruments = set(self.get_subscribed_instruments(_sub))
            _removed_instruments = plan.stream_unsubscriptions.get(_sub, set())
            _added_instruments = plan.stream_subscriptions.get(_sub, set())

            if _sub in plan.global_unsubscriptions:
                _removed_instruments.update(_current_sub_instruments)

            if _sub in plan.global_subscriptions:
                _added_instruments.update(self.get_subscribed_instruments())

            # - subscribe collection
            _updated_instruments = _current_sub_instruments.union(_added_instruments).difference(_removed_instruments)
            _exchange_to_updated_instruments = defaultdict(set)
            _exchange_to_current_sub_instruments = defaultdict(set)
            for instr in _updated_instruments:
                _exchange_to_updated_instruments[instr.exchange].add(instr)
            for instr in _current_sub_instruments:
                _exchange_to_current_sub_instruments[instr.exchange].add(instr)

            _exchanges_to_update = set(_exchange_to_updated_instruments.keys()).union(
                _exchange_to_current_sub_instruments.keys()
            )
            for _exchange in _exchanges_to_update:
                _data_provider = self._get_data_provider(_exchange)
                _exchange_updated_instruments = _exchange_to_updated_instruments[_exchange]
                _exchange_current_instruments = _exchange_to_current_sub_instruments[_exchange]
                if _exchange_updated_instruments != _exchange_current_instruments:
                    try:
                        _data_provider.subscribe(_sub, _exchange_updated_instruments, reset=True)
                    except NotSupported as e:
                        logger.warning(f"Subscription not supported for {_exchange}: {e}")

            # Notify health monitor of new subscriptions
            for instr in _added_instruments:
                self._health_monitor.subscribe(instr, _sub)

            # - unsubscribe instruments
            _exchange_to_removed_instruments = defaultdict(set)
            for instr in _removed_instruments:
                _exchange_to_removed_instruments[instr.exchange].add(instr)

            for _exchange, _exchange_removed_instruments in _exchange_to_removed_instruments.items():
                _data_provider = self._get_data_provider(_exchange)
                _data_provider.unsubscribe(_sub, _exchange_removed_instruments)

            # Notify health monitor to cleanup unsubscribed data
            for instr in _removed_instruments:
                self._health_monitor.unsubscribe(instr, _sub)

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        _data_provider = self._get_data_provider(instrument.exchange)
        return _data_provider.has_subscription(instrument, subscription_type)

    def get_subscriptions(self, instrument: Instrument | None = None) -> list[str]:
        _data_provider = (
            self._get_data_provider(instrument.exchange) if instrument is not None else self._data_providers[0]
        )
        return list(
            set(_data_provider.get_subscriptions(instrument))
            | {self.get_base_subscription()}
            | self._pending_global_subscriptions
        )

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
        _current_instruments = []
        for _data_provider in self._data_providers:
            _current_instruments.extend(_data_provider.get_subscribed_instruments(subscription_type))
        return _current_instruments

    def get_base_subscription(self) -> str:
        return self._base_sub

    def set_base_subscription(self, subscription_type: str) -> None:
        self._base_sub = subscription_type

    def get_warmup(self, subscription_type: str) -> str | None:
        return self._sub_to_warmup.get(subscription_type)

    def set_warmup(self, configs: dict[Any, str]) -> None:
        for subscription_type, period in configs.items():
            self._sub_to_warmup[subscription_type] = period

    @property
    def auto_subscribe(self) -> bool:
        return self._auto_subscribe

    @auto_subscribe.setter
    def auto_subscribe(self, value: bool) -> None:
        self._auto_subscribe = value

    def _subscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None) -> None:
        # - figure out which instruments to subscribe to (all or specific)
        if instruments is None:
            self._pending_global_subscriptions.add(subscription_type)
            return

        if isinstance(instruments, Instrument):
            instruments = [instruments]

        # - get instruments that are not already subscribed to
        _current_instruments = self.get_subscribed_instruments(subscription_type)
        instruments = list(set(instruments).difference(_current_instruments))

        # - subscribe to all existing subscriptions if subscription_type is ALL
        if subscription_type == DataType.ALL:
            subscriptions = self.get_subscriptions()
            for sub in subscriptions:
                self._subscribe(sub, instruments)
            return

        self._pending_stream_subscriptions[subscription_type].update(instruments)
        self._update_pending_warmups(subscription_type, instruments)

    def _unsubscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None) -> None:
        if instruments is None:
            self._pending_global_unsubscriptions.add(subscription_type)
            return

        if isinstance(instruments, Instrument):
            instruments = [instruments]

        # - subscribe to all existing subscriptions if subscription_type is ALL
        if subscription_type == DataType.ALL:
            subscriptions = self.get_subscriptions()
            for sub in subscriptions:
                self._unsubscribe(sub, instruments)
            return

        self._pending_stream_unsubscriptions[subscription_type].update(instruments)

    def _get_updated_subs(self, plan: _CommitPlan) -> list[str]:
        return list(
            set(plan.stream_unsubscriptions.keys())
            | set(plan.stream_subscriptions.keys())
            | plan.global_subscriptions
            | plan.global_unsubscriptions
        )

    def _has_operations_to_commit(self) -> bool:
        return any(
            (
                self._pending_stream_unsubscriptions,
                self._pending_stream_subscriptions,
                self._pending_global_subscriptions,
                self._pending_global_unsubscriptions,
            )
        )

    def _update_pending_warmups(self, subscription_type: str, instruments: list[Instrument]) -> None:
        # TODO: refactor pending warmups in a way that would allow to subscribe and then call set_warmup in the same iteration
        # - ohlc is handled separately
        if DataType.from_str(subscription_type) != DataType.OHLC:
            _warmup_period = self._sub_to_warmup.get(subscription_type)
            if _warmup_period is not None:
                for instrument in instruments:
                    self._pending_warmups[(subscription_type, instrument)] = _warmup_period

        # - if base subscription, then we need to fetch historical OHLC data for warmup
        if subscription_type == self._base_sub:
            self._pending_warmups.update(
                {
                    (sub, instrument): period
                    for sub, period in self._sub_to_warmup.items()
                    for instrument in instruments
                    if DataType.OHLC == sub
                }
            )

    def _collect_warmups(self, plan: _CommitPlan) -> dict[IDataProvider, dict[tuple[str, Instrument], str]]:
        """Resolve this commit's full warmup set (pending pairs from subscribe() plus the
        global-subscription expansion), partitioned per data provider. Consumes and clears
        _pending_warmups; providers with nothing to warm are omitted."""
        # - handle warmup for global subscriptions
        for _data_provider in self._data_providers:
            _subscribed_instruments = set(_data_provider.get_subscribed_instruments())
            _new_instruments = (
                set.union(*plan.stream_subscriptions.values()) if plan.stream_subscriptions else set()
            )

            _subs_to_warm = set(plan.global_subscriptions)
            if _new_instruments:
                _subs_to_warm.update(self._sub_to_warmup.keys())

            for sub in _subs_to_warm:
                _warmup_period = self._sub_to_warmup.get(sub)
                if _warmup_period is None:
                    continue
                _sub_instruments = _data_provider.get_subscribed_instruments(sub)
                _add_instruments = _subscribed_instruments.union(_new_instruments).difference(_sub_instruments)
                for instr in _add_instruments:
                    self._pending_warmups[(sub, instr)] = _warmup_period

        # TODO: think about appropriate handling of timeouts
        _warmups: dict[IDataProvider, dict[tuple[str, Instrument], str]] = {}
        for _data_provider in self._data_providers:
            _warmup_configs = self._get_pending_warmups_for_exchange(_data_provider.exchange())
            if _warmup_configs:
                _warmups[_data_provider] = _warmup_configs
        self._pending_warmups.clear()
        return _warmups

    def _get_pending_warmups_for_exchange(self, exchange: str) -> dict[tuple[str, Instrument], str]:
        return {
            (sub, instr): period for (sub, instr), period in self._pending_warmups.items() if instr.exchange == exchange
        }

    def _get_data_provider(self, exchange: str) -> IDataProvider:
        if exchange in self._exchange_to_data_provider:
            return self._exchange_to_data_provider[exchange]
        if exchange in EXCHANGE_MAPPINGS and EXCHANGE_MAPPINGS[exchange] in self._exchange_to_data_provider:
            return self._exchange_to_data_provider[EXCHANGE_MAPPINGS[exchange]]
        raise ValueError(f"Data provider for exchange {exchange} not found")

    def _init_connection_status_callbacks(self) -> None:
        for data_provider in self._data_providers:
            self._health_monitor.set_is_connected(
                exchange=data_provider.exchange(),
                is_connected=data_provider.is_connected,
            )

    def _init_subscription_monitoring(self) -> None:
        if self._is_simulation:
            return
        # - start monitoring thread only if there is at least one live data provider
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        while True:
            try:
                time.sleep(self._monitor_interval_seconds)
                if self._strategy_state.is_on_warmup_finished_called:
                    self._monitor_subscription_status()
            except Exception as e:
                logger.error(f"Error in subscription monitoring: {e}")

    def _monitor_subscription_status(self) -> None:
        exch_sub_to_stale_instr = defaultdict(lambda: defaultdict(set))
        for data_type in ["quote", "orderbook", "trade"]:
            for data_provider in self._data_providers:
                if data_provider.is_simulation or not data_provider.is_connected():
                    continue
                instruments = data_provider.get_subscribed_instruments(data_type)
                for instrument in instruments:
                    if self._health_monitor.is_stale(instrument, data_type):
                        exch_sub_to_stale_instr[data_provider.exchange()][data_type].add(instrument)

        if not exch_sub_to_stale_instr:
            return

        for exchange, sub_to_stale_instr in exch_sub_to_stale_instr.items():
            logger.warning(
                f"[<yellow>{exchange}</yellow>] :: Stale data detected for {pprint.pformat(dict(sub_to_stale_instr))} instruments"
            )
            logger.info("[1/4] Unsubscribing stale instruments..")
            data_provider = self._get_data_provider(exchange)
            # - unsubscribe stale instruments
            for data_type, stale_instruments in sub_to_stale_instr.items():
                data_provider.unsubscribe(data_type, stale_instruments)
            logger.info("[2/4] Waiting for 3 seconds before resubscribing..")
            # - wait for 3 seconds before resubscribing
            time.sleep(3)
            logger.info("[3/4] Resubscribing stale instruments..")
            # - resubscribe stale instruments
            for data_type, stale_instruments in sub_to_stale_instr.items():
                data_provider.subscribe(data_type, stale_instruments)
            logger.info("[4/4] Resubscription complete")
