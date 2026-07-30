"""Tests for the threaded fit executor: on_fit off the ProcessorThread via FitContext.

The calling test thread plays the ProcessorThread role: it submits the fit via
``_handle_fit`` and drains the CtrlChannel (processing events and finally the
FitCommit) exactly like ``StrategyContext.__process_incoming_data_loop`` does.
The strategy's ``on_fit`` receives a :class:`FitContext` proxy in thread mode.
"""

import threading
import time
from unittest.mock import ANY, MagicMock

import numpy as np
import pytest

from qubx.core.basics import CtrlChannel, DataType, Instrument, ITimeProvider, Signal
from qubx.core.context import StrategyContext
from qubx.core.exceptions import QueueTimeout
from qubx.core.fit_context import FitContext, UnclassifiedFitContextAccess
from qubx.core.fit_executor import FitCycleState
from qubx.core.interfaces import IStrategyContext, StrategyState
from qubx.core.mixins.market import CachedMarketDataHolder
from qubx.core.mixins.processing import ProcessingManager
from qubx.core.mixins.subscription import SUBSCRIPTION_SWAP_EVENT, SubscriptionManager
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
    context._market_data_provider = MagicMock()  # FitContext snapshot reads

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
    # - FitContext reaches the schedule/delay/fit-schedule seams through the context
    context._processing_manager = pm
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
    def test_fit_runs_on_strategy_fit_thread_with_fit_context(self):
        pm, context, channel = make_thread_pm()
        seen: dict[str, object] = {}

        def on_fit(ctx):
            seen["thread"] = threading.current_thread().name
            seen["ctx"] = ctx

        pm._strategy.on_fit.side_effect = on_fit

        pm._handle_fit(None, "fit", (None, T0))
        # - submission returns immediately; the gate stays up until the commit is applied
        assert pm._fit_is_running is True
        assert context._strategy_state.is_on_fit_called is False

        drain_until_committed(pm, channel)

        assert seen["thread"] == "StrategyFitThread"
        assert isinstance(seen["ctx"], FitContext)  # NOT the real context
        assert context._strategy_state.is_on_fit_called is True
        assert pm._fit_is_running is False

    def test_fit_duration_gauge_recorded(self):
        pm, _, channel = make_thread_pm()
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)
        pm._health_monitor.record_gauge.assert_any_call("fit_duration_s", ANY)

    def test_infra_applied_and_strategy_gated_during_first_slow_fit(self):
        """The incident scenario: a slow fit must not stop the ProcessorThread from
        applying market/account events. During the FIRST fit the strategy is not yet
        fitted, so its callbacks stay gated even in thread mode."""
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
            pm._strategy.on_market_data.assert_not_called()  # never-fitted strategy stays gated
        finally:
            release.set()

        drain_until_committed(pm, channel)
        assert pm._context._strategy_state.is_on_fit_called is True

    def test_events_delivered_to_strategy_during_threaded_refit(self):
        """Thread mode delivers instead of skipping: during a slow RE-fit (strategy
        already fitted once) a market-data event reaches on_market_data, concurrently
        with the running fit, and ctx.is_fitting is True inside the handler."""
        pm, context, channel = make_thread_pm()
        context._strategy_state.is_on_fit_called = True  # a previous fit committed
        started, release = threading.Event(), threading.Event()
        seen_fitting_in_handler: list[bool] = []

        def slow_fit(ctx):
            started.set()
            assert release.wait(5.0), "fit was never released"

        pm._strategy.on_fit.side_effect = slow_fit
        pm._strategy.on_market_data.side_effect = lambda ctx, event: seen_fitting_in_handler.append(pm.is_fitting)

        pm._handle_fit(None, "fit", (None, T0))
        assert started.wait(5.0)
        try:
            instrument = _mock_instrument()
            quote = MagicMock(spec=Quote)
            quote.time = T0
            pm.process_data(instrument, "quote", quote, False)

            pm._cache.update.assert_called_once()  # infra applied mid-fit as well
            pm._strategy.on_market_data.assert_called_once()  # DELIVERED mid-fit
            assert seen_fitting_in_handler == [True]  # handler observed the running fit
        finally:
            release.set()

        drain_until_committed(pm, channel)
        assert pm.is_fitting is False

    def test_inline_mode_gating_unchanged(self):
        """Inline mode (the default, and simulation) keeps today's behavior: while an
        inline fit is running no strategy callback fires."""
        pm, context, _ = make_thread_pm(fit_executor="inline")
        context._strategy_state.is_on_fit_called = True
        pm._fit_is_running = True  # an inline fit is on the stack (re-entrant pump)

        instrument = _mock_instrument()
        quote = MagicMock(spec=Quote)
        quote.time = T0
        pm.process_data(instrument, "quote", quote, False)

        pm._cache.update.assert_called_once()  # infra still applies
        pm._strategy.on_market_data.assert_not_called()  # strategy stays gated

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
            ctx.emit_signal(sig1)
            ctx.emit_signal([sig2])
            # - buffered on the fit-cycle state, NOT in the live pipeline list
            assert pm._emitted_signals == []

        pm._strategy.on_fit.side_effect = emitting_fit
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)

        # - the commit handed both signals to the normal pipeline in emission order
        pm._logging.save_signals.assert_called_once()
        assert list(pm._logging.save_signals.call_args[0][0]) == [sig1, sig2]

    def test_schedule_via_fit_context_is_deferred_and_keeps_event_id(self):
        pm, _, channel = make_thread_pm()
        scheduler = pm._scheduler
        returned: dict[str, str] = {}

        def scheduling_fit(ctx):
            returned["event_id"] = ctx.schedule("0 0 * * *", lambda c: None)
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

    def test_set_fit_schedule_via_fit_context_is_deferred(self):
        pm, _, channel = make_thread_pm()
        scheduler = pm._scheduler

        def scheduling_fit(ctx):
            ctx.set_fit_schedule("0 0 * * *")
            scheduler.unschedule_event.assert_not_called()

        scheduler.unschedule_event.reset_mock()
        pm._strategy.on_fit.side_effect = scheduling_fit
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)

        scheduler.unschedule_event.assert_called_once_with("fit")
        scheduler.schedule_event.assert_any_call("0 0 * * *", "fit")

    def test_deferred_set_universe_applies_atomically_at_commit(self):
        """End-to-end through a REAL UniverseManager: set_universe on the FitContext is
        recorded (universe untouched mid-fit) and applied at the commit — with
        on_universe_change fired on the ProcessorThread."""
        pm, context, channel = make_thread_pm()
        instrument = _mock_instrument()

        mkt = MagicMock()
        mkt.is_instrument_listed.return_value = True
        delisting = MagicMock()
        delisting.filter_delistings.side_effect = lambda instruments: list(instruments)
        account = MagicMock()
        account.positions = {}

        um = UniverseManager(
            context=context,
            strategy=pm._strategy,
            market_data_manager=mkt,
            logging=MagicMock(),
            subscription_manager=MagicMock(),
            trading_manager=MagicMock(),
            time_provider=MagicMock(),
            account=account,
            position_gathering=MagicMock(),
            delisting_detector=delisting,
        )
        # - the proxy records partial(context.set_universe, ...): route it to the real manager
        context.set_universe = um.set_universe

        callback_thread: list[int] = []
        pm._strategy.on_universe_change.side_effect = lambda ctx, added, removed: callback_thread.append(
            threading.get_ident()
        )

        started, release = threading.Event(), threading.Event()

        def universe_fit(ctx):
            ctx.set_universe([instrument])
            started.set()
            assert release.wait(5.0)

        pm._strategy.on_fit.side_effect = universe_fit
        pm._handle_fit(None, "fit", (None, T0))
        assert started.wait(5.0)

        try:
            # - mid-fit: intent recorded, but NOTHING applied yet
            assert um.instruments == []
            pm._strategy.on_universe_change.assert_not_called()
        finally:
            release.set()

        drain_until_committed(pm, channel)

        # - the commit (on this thread, the ProcessorThread) applied the swap and fired
        #   the callback here
        assert um.instruments == [instrument]
        pm._strategy.on_universe_change.assert_called_once()
        assert callback_thread == [threading.get_ident()]

    def test_realistic_fit_body_smoke(self):
        """A frab-shaped on_fit through the proxy: reads (time/ohlc/get_data/account
        views) + set_universe + emit_signal + schedule — completes and commits atomically."""
        pm, context, channel = make_thread_pm()
        instrument = _mock_instrument()
        signal = _mock_signal(instrument)
        context.get_positions.return_value = {}
        context.get_active_targets.return_value = {}
        mm = context._market_data_provider
        scheduled: dict[str, str] = {}

        def realistic_fit(ctx):
            assert isinstance(ctx, FitContext)
            _ = ctx.time()
            _ = ctx.ohlc(instrument, "1h", 100)
            _ = ctx.quote(instrument)
            _ = ctx.get_cached_market_data(instrument, "trade")
            positions = ctx.get_positions()
            positions["mutating-the-copy-is-safe"] = object()
            _ = ctx.get_active_targets()
            ctx.set_universe([instrument])
            ctx.emit_signal(signal)
            scheduled["event_id"] = ctx.schedule("0 0 * * *", lambda c: None)

        pm._strategy.on_fit.side_effect = realistic_fit
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)

        # - reads went through the snapshot-mode paths
        mm._ohlc.assert_called_once_with(instrument, "1h", 100, snapshot=True)
        mm._quote.assert_called_once_with(instrument, snapshot=True)
        # - the copy mutation never reached the real view
        assert "mutating-the-copy-is-safe" not in context.get_positions.return_value
        # - mutations applied atomically at the commit, on this thread
        context.set_universe.assert_called_once_with([instrument], False, "close")
        assert scheduled["event_id"] in pm._custom_scheduled_methods
        assert list(pm._logging.save_signals.call_args[0][0]) == [signal]
        assert pm._context._strategy_state.is_on_fit_called is True


class TestFitContextClassification:
    def _make_fit_context(self) -> tuple[FitContext, MagicMock]:
        context = MagicMock()
        return FitContext(context, FitCycleState()), context

    def test_unknown_attribute_raises_unclassified(self):
        fit_ctx, _ = self._make_fit_context()
        with pytest.raises(NotImplementedError, match="not classified on FitContext"):
            _ = fit_ctx.definitely_not_a_context_member
        # - duck-typing probes degrade to "absent" (UnclassifiedFitContextAccess is
        #   also an AttributeError) instead of crashing the prober
        assert not hasattr(fit_ctx, "definitely_not_a_context_member")
        assert getattr(fit_ctx, "definitely_not_a_context_member", None) is None

    def test_istrategycontext_surface_fully_classified(self):
        """Fail-closed guarantee: every public IStrategyContext member must be explicitly
        classified on FitContext. A future addition to the interface that nobody
        classified fails THIS test instead of silently passing through."""
        fit_ctx, _ = self._make_fit_context()
        names: set[str] = set()
        for klass in IStrategyContext.__mro__:
            names.update(n for n in vars(klass) if not n.startswith("_"))
            names.update(n for n in getattr(klass, "__annotations__", {}) if not n.startswith("_"))
        unclassified = []
        for name in sorted(names):
            try:
                getattr(fit_ctx, name)
            except UnclassifiedFitContextAccess:
                unclassified.append(name)
            except Exception:
                pass  # DENIED-on-access members raise plain NotImplementedError — classified
        assert unclassified == [], f"unclassified IStrategyContext members on FitContext: {unclassified}"

    def test_denied_members_raise_with_clear_message(self):
        fit_ctx, _ = self._make_fit_context()
        with pytest.raises(NotImplementedError, match="not allowed inside a threaded on_fit"):
            fit_ctx.trade(_mock_instrument(), 1.0)
        with pytest.raises(NotImplementedError, match="not allowed inside a threaded on_fit"):
            fit_ctx.commit()
        with pytest.raises(NotImplementedError, match="not allowed inside a threaded on_fit"):
            _ = fit_ctx.initializer

    def test_copy_reads_return_copies_real_state_untouched(self):
        fit_ctx, context = self._make_fit_context()
        instrument = _mock_instrument()
        real_positions = {instrument: MagicMock()}
        real_orders = {"o1": MagicMock()}
        real_balances = [MagicMock()]
        real_targets = {instrument: MagicMock()}
        context.get_positions.return_value = real_positions
        context.get_orders.return_value = real_orders
        context.get_balances.return_value = real_balances
        context.get_active_targets.return_value = real_targets
        context.account.get_positions.return_value = dict(real_positions)

        positions = fit_ctx.get_positions()
        assert positions == real_positions and positions is not real_positions
        positions.clear()
        assert real_positions  # the real view is untouched

        orders = fit_ctx.get_orders()
        assert orders == real_orders and orders is not real_orders
        orders.clear()
        assert real_orders

        balances = fit_ctx.get_balances()
        assert balances == real_balances and balances is not real_balances
        balances.clear()
        assert real_balances

        targets = fit_ctx.get_active_targets()
        targets.clear()
        assert real_targets

        # - account view copies too
        acct_positions = fit_ctx.account.get_positions()
        acct_positions.clear()
        assert context.account.get_positions.return_value

    def test_recorded_mutations_do_not_touch_real_context(self):
        fit_ctx, context = self._make_fit_context()
        instrument = _mock_instrument()
        context.instruments = []

        fit_ctx.remove_instruments([instrument])
        fit_ctx.subscribe(DataType.TRADE, [instrument])
        fit_ctx.unsubscribe(DataType.TRADE, [instrument])
        context.remove_instruments.assert_not_called()
        context.subscribe.assert_not_called()
        context.unsubscribe.assert_not_called()

        ops, _ = fit_ctx._fit_state.end()
        assert len(ops) == 3
        for op in ops:
            op()  # ProcessorThread replay takes the real path
        context.remove_instruments.assert_called_once_with([instrument], "close")
        context.subscribe.assert_called_once_with(DataType.TRADE, [instrument])
        context.unsubscribe.assert_called_once_with(DataType.TRADE, [instrument])

    def test_time_passthrough_supports_instrument_signal(self):
        fit_ctx, context = self._make_fit_context()
        context.time.return_value = T0
        assert isinstance(fit_ctx, ITimeProvider)
        assert fit_ctx.time() == T0

    def test_ctx_flags_pass_through(self):
        # - is_fitting / is_warming_up are plain passthrough reads: a threaded on_fit
        #   observes the same flags as any other strategy callback
        fit_ctx, context = self._make_fit_context()
        context.is_fitting = True
        context.is_warming_up = False
        assert fit_ctx.is_fitting is True
        assert fit_ctx.is_warming_up is False


class TestStashedContextTripwires:
    def _make_real_ctx_half_object(self) -> StrategyContext:
        ctx = StrategyContext.__new__(StrategyContext)
        ctx._fit_state = FitCycleState()
        ctx._universe_manager = MagicMock()
        ctx._subscription_manager = MagicMock()
        ctx._processing_manager = MagicMock()
        return ctx

    def test_stashed_ctx_mutators_raise_from_fit_thread(self):
        ctx = self._make_real_ctx_half_object()
        ctx._fit_state.begin(threading.get_ident())  # we ARE the fit thread now
        try:
            for call in (
                lambda: ctx.set_universe([]),
                lambda: ctx.add_instruments([]),
                lambda: ctx.remove_instruments([]),
                lambda: ctx.subscribe("trade"),
                lambda: ctx.unsubscribe("trade"),
                lambda: ctx.emit_signal(MagicMock()),
                lambda: ctx.schedule("0 0 * * *", lambda c: None),
            ):
                with pytest.raises(RuntimeError, match="outside FitContext"):
                    call()
        finally:
            ctx._fit_state.end()
        ctx._universe_manager.set_universe.assert_not_called()
        ctx._processing_manager.emit_signal.assert_not_called()

    def test_same_calls_pass_on_processor_thread_while_fit_runs(self):
        ctx = self._make_real_ctx_half_object()
        # - a fit is in flight on ANOTHER thread (fake ident): this thread is the
        #   ProcessorThread and must pass straight through (the FitCommit replay path)
        ctx._fit_state.begin(threading.get_ident() + 1)
        try:
            ctx.set_universe([])
            ctx.emit_signal(MagicMock())
            ctx.subscribe("trade")
            ctx.schedule("0 0 * * *", lambda c: None)
        finally:
            ctx._fit_state.end()
        ctx._universe_manager.set_universe.assert_called_once()
        ctx._processing_manager.emit_signal.assert_called_once()
        ctx._subscription_manager.subscribe.assert_called_once()
        ctx._processing_manager.schedule.assert_called_once()

    def test_no_overhead_path_when_no_fit_in_flight(self):
        ctx = self._make_real_ctx_half_object()
        ctx.set_universe([])  # disarmed fit state: plain delegation
        ctx._universe_manager.set_universe.assert_called_once()


class TestCtxFlags:
    """ctx.is_fitting / ctx.is_warming_up across both fit-executor modes."""

    def test_is_fitting_true_during_inline_fit(self):
        pm, context, _ = make_thread_pm(fit_executor="inline")
        seen: list[bool] = []
        pm._strategy.on_fit.side_effect = lambda ctx: seen.append(pm.is_fitting)

        assert pm.is_fitting is False
        pm._handle_fit(None, "fit", (None, T0))  # inline: runs synchronously right here

        assert seen == [True]
        assert pm.is_fitting is False
        assert context._strategy_state.is_on_fit_called is True

    def test_is_fitting_spans_submit_to_commit_in_thread_mode(self):
        pm, _, channel = make_thread_pm()
        seen: list[bool] = []
        pm._strategy.on_fit.side_effect = lambda ctx: seen.append(pm.is_fitting)

        assert pm.is_fitting is False
        pm._handle_fit(None, "fit", (None, T0))
        assert pm.is_fitting is True  # submitted, FitCommit not applied yet
        drain_until_committed(pm, channel)

        assert seen == [True]  # observed True on the fit thread too
        assert pm.is_fitting is False

    def test_strategy_context_delegates_flags(self):
        ctx = StrategyContext.__new__(StrategyContext)
        pm = MagicMock()
        pm.is_fitting = True
        sm = MagicMock()
        sm.is_warming_up = False
        ctx._processing_manager = pm
        ctx._subscription_manager = sm
        assert ctx.is_fitting is True
        assert ctx.is_warming_up is False
        pm.is_fitting = False
        sm.is_warming_up = True
        assert ctx.is_fitting is False
        assert ctx.is_warming_up is True

    def test_interface_defaults_are_false(self):
        # - honest defaults for consumers holding the bare interfaces (and simulation
        #   contexts before any fit)
        assert IStrategyContext.is_fitting.fget(object()) is False  # type: ignore[attr-defined]
        assert IStrategyContext.is_warming_up.fget(object()) is False  # type: ignore[attr-defined]


class TestDeferredSubscriptionCommit:
    """SubscriptionManager background warmup: a live commit that needs warmup fetches on
    the WarmupThread and applies its swap only AFTER the fetch completed (posted on the
    channel, applied by the ProcessorThread — history must land before live bars start)."""

    def _make_sm(self, is_simulation: bool) -> tuple[SubscriptionManager, MagicMock, CtrlChannel]:
        channel = CtrlChannel("test-databus")
        dp = MagicMock()
        dp.exchange.return_value = "BINANCE"
        dp.get_subscribed_instruments.return_value = []
        dp.is_simulation = is_simulation
        dp.channel = channel
        sm = SubscriptionManager(
            time_provider=MagicMock(),
            data_providers=[dp],
            health_monitor=MagicMock(),
            strategy_state=StrategyState(),
            monitor_interval_seconds=3600.0,  # keep the live monitoring thread quiet
        )
        return sm, dp, channel

    def _receive_swap(self, channel: CtrlChannel, timeout: float = 5.0):
        instrument, d_type, apply_swap, hist = channel.receive(timeout=timeout)
        assert d_type == SUBSCRIPTION_SWAP_EVENT
        return apply_swap

    def test_simulation_commit_stays_synchronous(self):
        sm, dp, channel = self._make_sm(is_simulation=True)
        instrument = _mock_instrument()
        sub = str(DataType.OHLC["1h"])
        sm.set_warmup({sub: "30d"})

        warmup_thread: list[str] = []
        dp.warmup.side_effect = lambda configs: warmup_thread.append(threading.current_thread().name)

        sm.subscribe(sub, [instrument])
        sm.commit()

        # - warmup fetched on the calling thread, swap applied before commit() returned
        dp.warmup.assert_called_once_with({(sub, instrument): "30d"})
        assert warmup_thread == [threading.current_thread().name]
        dp.subscribe.assert_called_once_with(sub, {instrument}, reset=True)
        assert sm.is_warming_up is False
        assert channel._queue.empty()  # nothing was deferred

    def test_live_commit_without_warmup_applies_swap_immediately(self):
        sm, dp, channel = self._make_sm(is_simulation=False)
        instrument = _mock_instrument()

        sm.subscribe(DataType.TRADE, [instrument])
        sm.commit()

        dp.warmup.assert_not_called()
        dp.subscribe.assert_called_once_with(DataType.TRADE, {instrument}, reset=True)
        assert sm.is_warming_up is False
        assert channel._queue.empty()

    def test_live_commit_defers_swap_until_warmup_completes(self):
        sm, dp, channel = self._make_sm(is_simulation=False)
        instrument = _mock_instrument()
        sub = str(DataType.OHLC["1h"])
        sm.set_warmup({sub: "30d"})

        release = threading.Event()
        warmup_thread: list[str] = []

        def blocking_warmup(configs):
            warmup_thread.append(threading.current_thread().name)
            assert release.wait(5.0), "warmup was never released"

        dp.warmup.side_effect = blocking_warmup

        sm.subscribe(sub, [instrument])
        sm.commit()

        # - commit() returned immediately: fetch runs on the WarmupThread, no swap yet
        assert sm.is_warming_up is True
        dp.subscribe.assert_not_called()
        release.set()

        # - the swap application rides the channel BEHIND the warmed history and is
        #   applied by the ProcessorThread (this thread here)
        apply_swap = self._receive_swap(channel)
        dp.warmup.assert_called_once_with({(sub, instrument): "30d"})
        assert warmup_thread == ["WarmupThread"]
        dp.subscribe.assert_not_called()  # not until the ProcessorThread applies it
        assert sm.is_warming_up is True

        apply_swap()
        dp.subscribe.assert_called_once_with(sub, {instrument}, reset=True)
        assert sm.is_warming_up is False

    def test_live_warmup_failure_logs_and_applies_swap_anyway(self):
        sm, dp, channel = self._make_sm(is_simulation=False)
        instrument = _mock_instrument()
        sub = str(DataType.OHLC["1h"])
        sm.set_warmup({sub: "30d"})
        dp.warmup.side_effect = RuntimeError("venue exploded")

        sm.subscribe(sub, [instrument])
        sm.commit()

        apply_swap = self._receive_swap(channel)
        apply_swap()

        # - same degradation as a failed synchronous warmup: the swap still applies
        dp.subscribe.assert_called_once_with(sub, {instrument}, reset=True)
        assert sm.is_warming_up is False

    def test_overlapping_deferred_commits_keep_warming_up_until_all_applied(self):
        sm, dp, channel = self._make_sm(is_simulation=False)
        first, second = _mock_instrument("BTCUSDT"), _mock_instrument("ETHUSDT")
        sub = str(DataType.OHLC["1h"])
        sm.set_warmup({sub: "30d"})

        sm.subscribe(sub, [first])
        sm.commit()
        sm.subscribe(sub, [second])
        sm.commit()
        assert sm.is_warming_up is True

        apply_first = self._receive_swap(channel)
        apply_second = self._receive_swap(channel)
        apply_first()
        assert sm.is_warming_up is True  # second deferred commit still in flight
        apply_second()
        assert sm.is_warming_up is False
        # - each commit swapped its own snapshot plan
        assert dp.subscribe.call_count == 2

    def test_processing_manager_applies_posted_swap(self):
        """The ProcessorThread side: a subscription_swap tuple dispatches to the handler,
        which invokes the posted apply callable."""
        pm, context, _ = make_thread_pm()
        context._strategy_state.is_on_fit_called = True  # don't auto-trigger a fit
        applied: list[str] = []
        pm.process_data(None, SUBSCRIPTION_SWAP_EVENT, lambda: applied.append("swap"), False)
        assert applied == ["swap"]

    def test_threaded_fit_universe_change_warms_up_in_background_end_to_end(self):
        """FitCommit replay -> UniverseManager.set_universe -> REAL SubscriptionManager:
        the replay records the subscription and defers the warmup to the WarmupThread;
        the added instrument's subscription goes live only after the fetch completed and
        the posted swap was applied on the ProcessorThread."""
        pm, context, channel = make_thread_pm()
        instrument = _mock_instrument()
        sub = str(DataType.OHLC["1h"])

        dp = MagicMock()
        dp.exchange.return_value = "BINANCE"
        dp.get_subscribed_instruments.return_value = []
        dp.is_simulation = False
        dp.channel = channel
        sm = SubscriptionManager(
            time_provider=MagicMock(),
            data_providers=[dp],
            health_monitor=MagicMock(),
            strategy_state=StrategyState(),
            auto_subscribe=False,
            default_base_subscription=sub,
            monitor_interval_seconds=3600.0,
        )
        sm.set_warmup({sub: "30d"})

        mkt = MagicMock()
        mkt.is_instrument_listed.return_value = True
        delisting = MagicMock()
        delisting.filter_delistings.side_effect = lambda instruments: list(instruments)
        account = MagicMock()
        account.positions = {}
        um = UniverseManager(
            context=context,
            strategy=pm._strategy,
            market_data_manager=mkt,
            logging=MagicMock(),
            subscription_manager=sm,
            trading_manager=MagicMock(),
            time_provider=MagicMock(),
            account=account,
            position_gathering=MagicMock(),
            delisting_detector=delisting,
        )
        context.set_universe = um.set_universe

        pm._strategy.on_fit.side_effect = lambda ctx: ctx.set_universe([instrument])
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)

        # - FitCommit applied: universe swapped and callback fired, but the DATA swap is
        #   still deferred behind the background warmup
        assert um.instruments == [instrument]
        dp.subscribe.assert_not_called()

        # - drain the posted swap exactly like the ProcessorThread would
        instrument_, d_type, apply_swap, hist = channel.receive(timeout=5)
        assert d_type == SUBSCRIPTION_SWAP_EVENT
        pm.process_data(instrument_, d_type, apply_swap, hist)

        dp.warmup.assert_called_once_with({(sub, instrument): "30d"})
        dp.subscribe.assert_called_once_with(sub, {instrument}, reset=True)
        assert sm.is_warming_up is False


class TestCacheConcurrentReads:
    def test_concurrent_append_and_snapshot_read_is_consistent(self):
        """Race test: ProcessorThread appends while the fit thread snapshots — no
        exception, and every snapshot is internally consistent."""
        holder = CachedMarketDataHolder("1Min")
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
                for _ in range(300):
                    snapshot = holder.get_ohlcv_snapshot(instrument)
                    n = len(snapshot)
                    # - internal consistency of the snapshot across all sub-series
                    assert len(snapshot.close) == n
                    assert len(snapshot.open) == n
                    assert len(snapshot.times) == n
                    if n:
                        _ = snapshot.pd()
            except Exception as e:
                errors.append(e)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start(), r.start()
        r.join(30.0)
        stop_writer.set()
        w.join(30.0)

        assert not errors, f"concurrent access raised: {errors}"

    def test_snapshot_read_returns_clone_not_live_series(self):
        holder = CachedMarketDataHolder("1Min")
        instrument = _mock_instrument()
        holder.init_ohlcv(instrument)
        t0 = T0.astype("datetime64[ns]").astype(int)
        holder.update_by_bar(instrument, Bar(t0, 1.0, 2.0, 0.5, 1.5, volume=1.0, bought_volume=0.5))

        live = holder.get_ohlcv(instrument)
        snapshot = holder.get_ohlcv_snapshot(instrument)
        assert snapshot is not live
        assert len(snapshot) == len(live)

    def test_live_read_path_unchanged(self):
        holder = CachedMarketDataHolder("1Min")
        instrument = _mock_instrument()
        holder.init_ohlcv(instrument)
        # - identical semantics to today: the live series, same object on every read
        assert holder.get_ohlcv(instrument) is holder.get_ohlcv(instrument)


class TestSimulationPurity:
    def test_simulation_never_creates_executor_even_when_configured(self):
        pm, _, _ = make_thread_pm(fit_executor="thread", is_simulation=True)
        assert pm._fit_executor is None
        assert pm._fit_executor_mode == "inline"

    def test_live_inline_default_creates_no_executor(self):
        pm, _, _ = make_thread_pm(fit_executor="inline")
        assert pm._fit_executor is None
        assert pm._fit_executor_mode == "inline"

    def test_simulation_fit_runs_inline_on_caller_thread_with_real_context(self):
        pm, context, _ = make_thread_pm(fit_executor="thread", is_simulation=True)
        seen: dict[str, object] = {}

        def on_fit(ctx):
            seen["ident"] = threading.get_ident()
            seen["ctx"] = ctx

        pm._strategy.on_fit.side_effect = on_fit

        pm._handle_fit(None, "fit", (None, T0))

        assert seen["ident"] == threading.get_ident()  # ran synchronously, right here
        assert seen["ctx"] is context  # the REAL context, never a FitContext
        assert context._strategy_state.is_on_fit_called is True
        assert pm._fit_is_running is False

    def test_live_thread_mode_creates_executor(self):
        pm, _, _ = make_thread_pm(fit_executor="thread")
        assert pm._fit_executor is not None
        assert pm._fit_executor_mode == "thread"

    def test_config_default_is_inline(self):
        assert LiveConfig.model_fields["fit_executor"].default == "inline"
        assert LiveConfig.model_fields["fit_soft_deadline_s"].default == pytest.approx(120.0)
