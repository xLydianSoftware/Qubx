"""Tests for the threaded fit executor (B.2): on_fit off the ProcessorThread.

The calling test thread plays the ProcessorThread role: it submits the fit via
``_handle_fit`` and drains the CtrlChannel (processing events and finally the
FitCommit) exactly like ``StrategyContext.__process_incoming_data_loop`` does.
"""

import threading
import time
from unittest.mock import ANY, MagicMock

import numpy as np
import pytest

from qubx.core.basics import CtrlChannel, DataType, Instrument, Signal
from qubx.core.exceptions import QueueTimeout
from qubx.core.fit_executor import FitCycleState
from qubx.core.interfaces import StrategyState
from qubx.core.mixins.market import CachedMarketDataHolder
from qubx.core.mixins.processing import ProcessingManager
from qubx.core.mixins.subscription import SubscriptionManager
from qubx.core.mixins.universe import UniverseManager
from qubx.core.series import Bar, Quote
from qubx.utils.runner.configs import LiveConfig

T0 = np.datetime64("2025-01-01T00:00:00", "ns")


def _mock_instrument(symbol: str = "BTCUSDT") -> MagicMock:
    instrument = MagicMock(spec=Instrument)
    instrument.symbol = symbol
    instrument.min_size = 0.001
    instrument.exchange = "BINANCE"
    instrument.delist_date = None
    return instrument


def _mock_signal(instrument: MagicMock) -> MagicMock:
    signal = MagicMock(spec=Signal)
    signal.instrument = instrument
    signal.is_service = False
    signal.group = None
    signal.reference_price = None
    return signal


def make_thread_pm(fit_executor: str = "thread", is_simulation: bool = False, **kwargs):
    """ProcessingManager with mocked collaborators, wired for the threaded-fit tests."""
    channel = CtrlChannel("test-databus")

    context = MagicMock()
    context.is_simulation = is_simulation
    context.is_paper_trading = False
    context.instruments = []
    context._strategy_state = StrategyState(
        is_on_init_called=True,
        is_on_start_called=True,
        is_on_warmup_finished_called=True,
        is_on_fit_called=False,
        is_warmup_in_progress=False,
    )
    context._data_providers = [MagicMock(channel=channel)]
    context.emitter = None

    strategy = MagicMock()
    strategy.__class__.__name__ = "TestStrategy"

    cache = MagicMock()
    cache.default_timeframe = "1h"
    market_data = MagicMock()
    market_data.get_market_data_cache.return_value = cache

    subscription_manager = MagicMock()
    subscription_manager.get_base_subscription.return_value = "quote"

    time_provider = MagicMock()
    time_provider.time.return_value = T0

    position_tracker = MagicMock()
    position_tracker.process_signals.return_value = []
    position_tracker.update.return_value = []

    universe_manager = MagicMock()
    universe_manager.is_trading_allowed.return_value = True

    health_monitor = MagicMock()
    health_monitor.return_value.__enter__ = MagicMock(return_value=None)
    health_monitor.return_value.__exit__ = MagicMock(return_value=False)

    pm = ProcessingManager(
        context=context,
        strategy=strategy,
        logging=MagicMock(),
        market_data=market_data,
        subscription_manager=subscription_manager,
        time_provider=time_provider,
        account_manager=MagicMock(),
        connectors={},
        position_tracker=position_tracker,
        position_gathering=MagicMock(),
        universe_manager=universe_manager,
        scheduler=MagicMock(),
        is_simulation=is_simulation,
        health_monitor=health_monitor,
        delisting_detector=MagicMock(),
        fit_executor=fit_executor,
        **kwargs,
    )
    pm._is_ready = lambda: True  # type: ignore[method-assign]
    pm._init_stage_position_tracker = MagicMock()
    pm._init_stage_position_tracker.process_signals.return_value = []
    pm._init_stage_position_tracker.update.return_value = []
    return pm, context, channel


def drain_until_committed(pm: ProcessingManager, channel: CtrlChannel, timeout: float = 5.0) -> None:
    """Play the ProcessorThread: drain the channel until the FitCommit clears the gate."""
    deadline = time.monotonic() + timeout
    while pm._fit_is_running:
        if time.monotonic() > deadline:
            raise TimeoutError("FitCommit was not applied within the timeout")
        try:
            msg = channel.receive(timeout=1)
        except QueueTimeout:
            continue
        instrument, d_type, data, hist = msg
        pm.process_data(instrument, d_type, data, hist)


class TestThreadedFitExecution:
    def test_fit_runs_on_strategy_fit_thread_and_commit_clears_flags(self):
        pm, context, channel = make_thread_pm()
        seen: dict[str, str] = {}

        def on_fit(ctx):
            seen["thread"] = threading.current_thread().name

        pm._strategy.on_fit.side_effect = on_fit

        pm._handle_fit(None, "fit", (None, T0))
        # - submission returns immediately; the gate stays up until the commit is applied
        assert pm._fit_is_running is True
        assert context._strategy_state.is_on_fit_called is False

        drain_until_committed(pm, channel)

        assert seen["thread"] == "StrategyFitThread"
        assert context._strategy_state.is_on_fit_called is True
        assert pm._fit_is_running is False

    def test_fit_duration_gauge_recorded(self):
        pm, _, channel = make_thread_pm()
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)
        pm._health_monitor.record_gauge.assert_any_call("fit_duration_s", ANY)

    def test_events_processed_during_slow_fit(self):
        """The incident scenario: a slow fit must not stop the ProcessorThread from
        applying market/account events; the strategy itself stays gated (same as today)."""
        pm, _, channel = make_thread_pm()
        started, release = threading.Event(), threading.Event()

        def slow_fit(ctx):
            started.set()
            assert release.wait(5.0), "fit was never released"

        pm._strategy.on_fit.side_effect = slow_fit
        pm._handle_fit(None, "fit", (None, T0))
        assert started.wait(5.0)

        try:
            # - ProcessorThread (this thread) applies a quote WHILE the fit computes
            instrument = _mock_instrument()
            quote = MagicMock(spec=Quote)
            quote.time = T0
            pm.process_data(instrument, "quote", quote, False)

            assert pm._fit_is_running is True  # fit is still computing
            pm._cache.update.assert_called_once()  # market cache updated mid-fit
            pm._account_manager.on_market_quote.assert_called_once()  # mark-to-market mid-fit
            pm._strategy.on_market_data.assert_not_called()  # strategy stays gated (as today)
        finally:
            release.set()

        drain_until_committed(pm, channel)
        assert pm._context._strategy_state.is_on_fit_called is True

    def test_overlapping_fit_trigger_dropped(self):
        pm, _, channel = make_thread_pm()
        started, release = threading.Event(), threading.Event()

        def slow_fit(ctx):
            started.set()
            assert release.wait(5.0)

        pm._strategy.on_fit.side_effect = slow_fit
        pm._handle_fit(None, "fit", (None, T0))
        assert started.wait(5.0)
        try:
            pm._handle_fit(None, "fit", (None, T0))  # overlapping trigger — dropped
        finally:
            release.set()
        drain_until_committed(pm, channel)

        assert pm._strategy.on_fit.call_count == 1

    def test_exception_in_fit_still_posts_commit_and_clears_flags(self):
        pm, context, channel = make_thread_pm()
        applied: list[str] = []

        def failing_fit(ctx):
            pm._fit_state.record(lambda: applied.append("op"))
            raise RuntimeError("fit blew up")

        pm._strategy.on_fit.side_effect = failing_fit
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)

        assert context._strategy_state.is_on_fit_called is True
        assert pm._fit_is_running is False
        # - ops recorded before the raise are applied, mirroring inline semantics
        #   (mutations before an inline raise stay applied)
        assert applied == ["op"]

    def test_soft_deadline_warning_fires_for_slow_fit(self):
        pm, _, channel = make_thread_pm(fit_soft_deadline_s=0.05)
        pm._warn_fit_soft_deadline = MagicMock()  # type: ignore[method-assign]
        pm._strategy.on_fit.side_effect = lambda ctx: time.sleep(0.3)
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)
        pm._warn_fit_soft_deadline.assert_called_once()

    def test_soft_deadline_warning_not_fired_for_fast_fit(self):
        pm, _, channel = make_thread_pm(fit_soft_deadline_s=5.0)
        pm._warn_fit_soft_deadline = MagicMock()  # type: ignore[method-assign]
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)
        time.sleep(0.05)  # give a mis-armed timer a chance to fire
        pm._warn_fit_soft_deadline.assert_not_called()


class TestDeferredMutationsAndSignals:
    def test_emitted_signals_drain_in_order_at_commit(self):
        pm, _, channel = make_thread_pm()
        instrument = _mock_instrument()
        sig1, sig2 = _mock_signal(instrument), _mock_signal(instrument)

        def emitting_fit(ctx):
            pm.emit_signal(sig1)
            pm.emit_signal([sig2])
            # - buffered on the fit-cycle state, NOT in the live pipeline list
            assert pm._emitted_signals == []

        pm._strategy.on_fit.side_effect = emitting_fit
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)

        # - the commit handed both signals to the normal pipeline in emission order
        pm._logging.save_signals.assert_called_once()
        assert list(pm._logging.save_signals.call_args[0][0]) == [sig1, sig2]

    def test_schedule_from_fit_thread_is_deferred_and_keeps_event_id(self):
        pm, _, channel = make_thread_pm()
        scheduler = pm._scheduler
        returned: dict[str, str] = {}

        def scheduling_fit(ctx):
            returned["event_id"] = pm.schedule("0 0 * * *", lambda c: None)
            # - scheduler untouched while the fit runs (mutation deferred to commit)
            scheduler.schedule_event.assert_not_called()

        scheduler.schedule_event.reset_mock()  # drop registrations from __init__
        pm._strategy.on_fit.side_effect = scheduling_fit
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)

        event_id = returned["event_id"]
        assert event_id.startswith("custom_schedule_")
        assert event_id in pm._custom_scheduled_methods
        scheduler.schedule_event.assert_called_once_with("0 0 * * *", event_id)

    def test_set_fit_schedule_from_fit_thread_is_deferred(self):
        pm, _, channel = make_thread_pm()
        scheduler = pm._scheduler

        def scheduling_fit(ctx):
            pm.set_fit_schedule("0 0 * * *")
            scheduler.unschedule_event.assert_not_called()

        scheduler.unschedule_event.reset_mock()
        pm._strategy.on_fit.side_effect = scheduling_fit
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)

        scheduler.unschedule_event.assert_called_once_with("fit")
        scheduler.schedule_event.assert_any_call("0 0 * * *", "fit")

    def test_deferred_set_universe_applies_atomically_at_commit(self):
        """End-to-end through a REAL UniverseManager: set_universe from the fit thread is
        recorded (universe untouched mid-fit), prepared (warmup prefetch on the fit
        thread), and applied at the commit — with on_universe_change fired on the
        ProcessorThread."""
        pm, context, channel = make_thread_pm()
        instrument = _mock_instrument()

        mkt = MagicMock()
        mkt.is_instrument_listed.return_value = True
        delisting = MagicMock()
        delisting.filter_delistings.side_effect = lambda instruments: list(instruments)
        account = MagicMock()
        account.positions = {}
        subscription_manager = MagicMock()

        um = UniverseManager(
            context=context,
            strategy=pm._strategy,
            market_data_manager=mkt,
            logging=MagicMock(),
            subscription_manager=subscription_manager,
            trading_manager=MagicMock(),
            time_provider=MagicMock(),
            account=account,
            position_gathering=MagicMock(),
            delisting_detector=delisting,
            fit_state=pm._fit_state,
        )

        prepare_thread: list[str] = []
        subscription_manager.prepare.side_effect = lambda instruments: prepare_thread.append(
            threading.current_thread().name
        )
        callback_thread: list[int] = []
        pm._strategy.on_universe_change.side_effect = lambda ctx, added, removed: callback_thread.append(
            threading.get_ident()
        )

        started, release = threading.Event(), threading.Event()

        def universe_fit(ctx):
            um.set_universe([instrument])
            started.set()
            assert release.wait(5.0)

        pm._strategy.on_fit.side_effect = universe_fit
        pm._handle_fit(None, "fit", (None, T0))
        assert started.wait(5.0)

        try:
            # - mid-fit: intent recorded + prefetched, but NOTHING applied yet
            assert um.instruments == []
            pm._strategy.on_universe_change.assert_not_called()
            subscription_manager.prepare.assert_called_once_with([instrument])
            assert prepare_thread == ["StrategyFitThread"]
        finally:
            release.set()

        drain_until_committed(pm, channel)

        # - the commit (on this thread, the ProcessorThread) applied the swap and fired
        #   the callback here
        assert um.instruments == [instrument]
        pm._strategy.on_universe_change.assert_called_once()
        assert callback_thread == [threading.get_ident()]

    def test_remove_instruments_from_fit_thread_is_deferred(self):
        pm, context, _ = make_thread_pm()
        instrument = _mock_instrument()

        um = UniverseManager(
            context=context,
            strategy=pm._strategy,
            market_data_manager=MagicMock(),
            logging=MagicMock(),
            subscription_manager=MagicMock(),
            trading_manager=MagicMock(),
            time_provider=MagicMock(),
            account=MagicMock(),
            position_gathering=MagicMock(),
            delisting_detector=MagicMock(),
            fit_state=pm._fit_state,
        )

        pm._fit_state.begin(threading.get_ident())
        um.remove_instruments([instrument])
        pm._strategy.on_universe_change.assert_not_called()  # nothing applied mid-fit
        ops, _ = pm._fit_state.end()
        assert len(ops) == 1


class TestSubscriptionManagerFitPath:
    def _make_sm(self, fit_state: FitCycleState) -> tuple[SubscriptionManager, MagicMock]:
        dp = MagicMock()
        dp.exchange.return_value = "BINANCE"
        dp.get_subscribed_instruments.return_value = []
        dp.is_simulation = True  # keeps the monitoring thread off
        sm = SubscriptionManager(
            time_provider=MagicMock(),
            data_providers=[dp],
            health_monitor=MagicMock(),
            strategy_state=StrategyState(),
            fit_state=fit_state,
        )
        return sm, dp

    def test_subscribe_from_fit_thread_is_recorded_not_queued(self):
        fit_state = FitCycleState()
        sm, _ = self._make_sm(fit_state)
        instrument = _mock_instrument()

        fit_state.begin(threading.get_ident())
        sm.subscribe(DataType.TRADE, [instrument])
        assert not sm._pending_stream_subscriptions  # nothing queued from the fit thread
        ops, _ = fit_state.end()
        assert len(ops) == 1

        ops[0]()  # ProcessorThread replay takes the normal path
        assert instrument in sm._pending_stream_subscriptions[DataType.TRADE]

    def test_prepare_prefetches_and_commit_skips_prewarmed(self):
        fit_state = FitCycleState()
        sm, dp = self._make_sm(fit_state)
        instrument = _mock_instrument()
        sub = str(DataType.OHLC["1h"])
        sm.set_warmup({sub: "30d"})

        # - fit thread: prepare() fetches the warmup for the addition
        fit_state.begin(threading.get_ident())
        sm.prepare([instrument])
        fit_state.end()
        dp.warmup.assert_called_once_with({(sub, instrument): "30d"})

        # - ProcessorThread: the replayed subscribe + commit must NOT refetch
        sm.subscribe(sub, [instrument])
        sm.commit()
        assert dp.warmup.call_count == 2
        assert dp.warmup.call_args[0][0] == {}  # pre-warmed pair was skipped
        assert not sm._prewarmed  # consumed

    def test_discard_prewarmed_clears_ledger(self):
        fit_state = FitCycleState()
        sm, _ = self._make_sm(fit_state)
        instrument = _mock_instrument()
        sm.set_warmup({str(DataType.OHLC["1h"]): "30d"})
        fit_state.begin(threading.get_ident())
        sm.prepare([instrument])
        fit_state.end()
        assert sm._prewarmed
        sm.discard_prewarmed()
        assert not sm._prewarmed


class TestCacheConcurrentReads:
    def test_concurrent_append_and_fit_read_is_consistent(self):
        """Race test: ProcessorThread appends while the fit thread snapshots — no
        exception, and every snapshot is internally consistent."""
        holder = CachedMarketDataHolder("1Min")
        fit_state = FitCycleState()
        holder.enable_concurrent_fit_reads(fit_state)
        instrument = _mock_instrument()
        holder.init_ohlcv(instrument)

        t0 = T0.astype("datetime64[ns]").astype(int)
        one_min = 60_000_000_000
        errors: list[Exception] = []
        stop_writer = threading.Event()

        def writer():
            try:
                for i in range(20_000):
                    if stop_writer.is_set():
                        return
                    holder.update_by_bar(
                        instrument,
                        Bar(t0 + i * one_min, 1.0 + i, 2.0 + i, 0.5 + i, 1.5 + i, volume=1.0, bought_volume=0.5),
                    )
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        def reader():
            try:
                fit_state.begin(threading.get_ident())
                for _ in range(300):
                    snapshot = holder.get_ohlcv(instrument)
                    n = len(snapshot)
                    # - internal consistency of the snapshot across all sub-series
                    assert len(snapshot.close) == n
                    assert len(snapshot.open) == n
                    assert len(snapshot.times) == n
                    if n:
                        _ = snapshot.pd()
            except Exception as e:
                errors.append(e)
            finally:
                fit_state.end()

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start(), r.start()
        r.join(30.0)
        stop_writer.set()
        w.join(30.0)

        assert not errors, f"concurrent access raised: {errors}"

    def test_fit_thread_read_returns_snapshot_not_live_series(self):
        holder = CachedMarketDataHolder("1Min")
        fit_state = FitCycleState()
        holder.enable_concurrent_fit_reads(fit_state)
        instrument = _mock_instrument()
        holder.init_ohlcv(instrument)
        t0 = T0.astype("datetime64[ns]").astype(int)
        holder.update_by_bar(instrument, Bar(t0, 1.0, 2.0, 0.5, 1.5, volume=1.0, bought_volume=0.5))

        live = holder._get_ohlcv_series(instrument)
        # - on-thread (no armed fit): the live series, same as today
        assert holder.get_ohlcv(instrument) is live

        fit_state.begin(threading.get_ident())
        try:
            snapshot = holder.get_ohlcv(instrument)
            assert snapshot is not live
            assert len(snapshot) == len(live)
        finally:
            fit_state.end()

    def test_lock_free_path_when_not_enabled(self):
        holder = CachedMarketDataHolder("1Min")
        instrument = _mock_instrument()
        holder.init_ohlcv(instrument)
        live = holder._get_ohlcv_series(instrument)
        assert holder.get_ohlcv(instrument) is live  # identical semantics to today


class TestSimulationPurity:
    def test_simulation_never_creates_executor_even_when_configured(self):
        pm, _, _ = make_thread_pm(fit_executor="thread", is_simulation=True)
        assert pm._fit_executor is None
        assert pm._fit_executor_mode == "inline"

    def test_live_inline_default_creates_no_executor(self):
        pm, _, _ = make_thread_pm(fit_executor="inline")
        assert pm._fit_executor is None
        assert pm._fit_executor_mode == "inline"

    def test_simulation_fit_runs_inline_on_caller_thread(self):
        pm, context, _ = make_thread_pm(fit_executor="thread", is_simulation=True)
        seen: dict[str, int] = {}
        pm._strategy.on_fit.side_effect = lambda ctx: seen.setdefault("ident", threading.get_ident())

        pm._handle_fit(None, "fit", (None, T0))

        assert seen["ident"] == threading.get_ident()  # ran synchronously, right here
        assert context._strategy_state.is_on_fit_called is True
        assert pm._fit_is_running is False

    def test_live_thread_mode_creates_executor(self):
        pm, _, _ = make_thread_pm(fit_executor="thread")
        assert pm._fit_executor is not None
        assert pm._fit_executor_mode == "thread"

    def test_config_default_is_inline(self):
        assert LiveConfig.model_fields["fit_executor"].default == "inline"
        assert LiveConfig.model_fields["fit_soft_deadline_s"].default == pytest.approx(120.0)
