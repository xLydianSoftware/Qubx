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

from qubx.core.basics import CtrlChannel, DataType, FundingPayment, Instrument, ITimeProvider, Signal
from qubx.core.context import StrategyContext
from qubx.core.exceptions import QueueTimeout
from qubx.core.fit_context import FitContext, UnclassifiedFitContextAccess
from qubx.core.fit_executor import FitCycleState, FitExecutorMode
from qubx.core.interfaces import IStrategyContext, StrategyState
from qubx.core.mixins.market import CachedMarketDataHolder, MarketManager
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
    """Play the ProcessorThread: drain the channel until the FitCommit clears the flag."""
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
        # - submission returns immediately; the flag stays set until the commit is applied
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
        fitted, so its callbacks stay suppressed even in thread mode."""
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
            pm._strategy.on_market_data.assert_not_called()  # never-fitted strategy stays suppressed
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
        pm._strategy.on_market_data.assert_not_called()  # strategy stays suppressed

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
            _ = ctx.get_cached_market_data(instrument, "trade")  # detached snapshot clone
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
        mm._get_cached_market_data.assert_called_once_with(instrument, "trade", snapshot=True)
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

        fit_ctx._fit_state.begin(threading.get_ident())  # recording requires an armed cycle
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
        ctx._trading_manager = MagicMock()
        ctx._transfer_manager = MagicMock()
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
                # full mutator surface — trading, venue config, plumbing, scheduling
                lambda: ctx.trade(MagicMock(), 1.0),
                lambda: ctx.submit_orders([]),
                lambda: ctx.set_target_position(MagicMock(), 1.0),
                lambda: ctx.set_target_leverage(MagicMock(), 1.0),
                lambda: ctx.close_position(MagicMock()),
                lambda: ctx.close_positions(),
                lambda: ctx.cancel_order(order_id="x"),
                lambda: ctx.cancel_orders(),
                lambda: ctx.update_order(1.0, 1.0, order_id="x"),
                lambda: ctx.settle_position(MagicMock()),
                lambda: ctx.commit(),
                lambda: ctx.set_warmup({}),
                lambda: ctx.set_base_subscription("trade"),
                lambda: ctx.set_fit_schedule("0 0 * * *"),
                lambda: ctx.set_event_schedule("0 0 * * *"),
                lambda: ctx.unschedule("x"),
                lambda: ctx.delay("1m", lambda c: None),
                lambda: ctx.set_margin_mode(MagicMock(), "cross"),
                lambda: ctx.set_instrument_leverage(MagicMock(), 5.0),
                lambda: ctx.transfer_funds("A", "B", "USDT", 1.0),
                lambda: ctx.set_warmup_positions({}),
                lambda: ctx.set_warmup_orders({}),
                lambda: ctx.set_warmup_active_targets({}),
            ):
                with pytest.raises(RuntimeError, match="outside FitContext"):
                    call()
        finally:
            ctx._fit_state.end()
        ctx._universe_manager.set_universe.assert_not_called()
        ctx._processing_manager.emit_signal.assert_not_called()
        ctx._trading_manager.trade.assert_not_called()
        ctx._transfer_manager.transfer_funds.assert_not_called()

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


class TestReviewFixes:
    """Regression tests for the adversarial-review findings."""

    def _make_fit_ctx(self, positions: dict | None = None, exchanges: list[str] | None = None):
        context = MagicMock()
        context.get_positions.return_value = positions if positions is not None else {}
        context.exchanges = exchanges if exchanges is not None else ["BINANCE"]
        return FitContext(context, FitCycleState()), context

    def test_get_position_never_materializes_into_account_state(self):
        instrument = _mock_instrument()
        fit_ctx, context = self._make_fit_ctx(exchanges=[instrument.exchange])

        pos = fit_ctx.get_position(instrument)
        # - detached empty Position: same consumer semantics as the real materialization
        assert pos is not None and pos.instrument is instrument
        # - the real (inserting) get_position was never touched, on ctx nor on account
        context.get_position.assert_not_called()
        context.account.get_position.assert_not_called()
        # - a second read hands out a fresh detached object — nothing was stored
        assert fit_ctx.get_position(instrument) is not pos

    def test_get_position_returns_live_position_when_present(self):
        instrument = _mock_instrument()
        live_pos = MagicMock()
        fit_ctx, context = self._make_fit_ctx(positions={instrument: live_pos})
        assert fit_ctx.get_position(instrument) is live_pos
        context.get_position.assert_not_called()

    def test_get_position_unknown_exchange_returns_none(self):
        instrument = _mock_instrument()
        fit_ctx, _ = self._make_fit_ctx(exchanges=["SOME_OTHER_EXCHANGE"])
        assert fit_ctx.get_position(instrument) is None

    def test_position_derivatives_read_from_peek(self):
        instrument = _mock_instrument()
        fit_ctx, context = self._make_fit_ctx(exchanges=[instrument.exchange])
        assert fit_ctx.get_max_instrument_leverage(instrument) is None
        assert fit_ctx.get_max_instrument_notional(instrument) == float("inf")
        assert fit_ctx.get_margin_mode(instrument) is None
        assert fit_ctx.get_adl_level(instrument) is None
        context.get_position.assert_not_called()
        # - the account view routes through the same peek
        assert fit_ctx.account.get_max_instrument_notional(instrument) == float("inf")
        context.account.get_position.assert_not_called()

    def test_is_trading_allowed_answers_without_executing_removal(self):
        instrument = _mock_instrument()
        fit_ctx, context = self._make_fit_ctx()
        context._universe_manager._removal_queue = {instrument: ("wait_for_change", False)}
        assert fit_ctx.is_trading_allowed(instrument) is False
        # - the real call would EXECUTE the queued removal (cancels/trades) — never invoked
        context.is_trading_allowed.assert_not_called()

        context._universe_manager._removal_queue = {}
        assert fit_ctx.is_trading_allowed(instrument) is True
        context._universe_manager._removal_queue = {instrument: ("close", False)}
        assert fit_ctx.is_trading_allowed(instrument) is True

    def test_record_outside_fit_cycle_raises(self):
        state = FitCycleState()
        state.begin(threading.get_ident())
        state.end()
        # - fit over: a leaked FitContext must fail loudly, not record into the void
        with pytest.raises(RuntimeError, match="outside its fit cycle"):
            state.record(lambda: None)
        with pytest.raises(RuntimeError, match="outside its fit cycle"):
            state.buffer_signals([MagicMock()])

    def test_record_from_foreign_thread_while_armed_raises(self):
        state = FitCycleState()
        state.begin(threading.get_ident() + 1)  # some OTHER thread's fit is in flight
        try:
            with pytest.raises(RuntimeError, match="outside its fit cycle"):
                state.record(lambda: None)
        finally:
            state.end()

    def test_commit_posted_even_when_fitcontext_construction_raises(self, monkeypatch):
        pm, context, channel = make_thread_pm()
        monkeypatch.setattr(
            "qubx.core.mixins.processing.FitContext",
            MagicMock(side_effect=RuntimeError("boom at proxy construction")),
        )
        pm._handle_fit(None, "fit", (None, T0))
        # - the pre-fit framework raise still posts the FitCommit: flag clears, bot keeps fitting
        drain_until_committed(pm, channel)
        assert pm._fit_is_running is False
        pm._strategy.on_fit.assert_not_called()


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

    def test_no_warmup_commit_queues_behind_inflight_deferred_commit(self):
        """A pure-removal commit must NOT jump ahead of an in-flight deferred commit:
        applied synchronously, the older deferred plan would land afterwards and
        re-subscribe (current ∪ added − removed at apply time) what the removal removed."""
        sm, dp, channel = self._make_sm(is_simulation=False)
        instrument = _mock_instrument()
        sub = str(DataType.OHLC["1h"])
        sm.set_warmup({sub: "30d"})

        release = threading.Event()
        dp.warmup.side_effect = lambda configs: release.wait(5.0)

        sm.subscribe(sub, [instrument])
        sm.commit()  # deferred: warmup in flight on the WarmupThread
        assert sm.is_warming_up is True

        sm.unsubscribe(sub, [instrument])
        sm.commit()  # pure removal, no warmup — must still queue behind the first
        dp.subscribe.assert_not_called()  # nothing applied synchronously

        release.set()
        # - both swaps ride the channel in submission order
        apply_first = self._receive_swap(channel)
        apply_first()
        dp.subscribe.assert_called_once_with(sub, {instrument}, reset=True)

        dp.get_subscribed_instruments.return_value = [instrument]
        apply_second = self._receive_swap(channel)
        apply_second()
        # - the LAST intent (removal) wins: final swap unsubscribes the instrument
        dp.subscribe.assert_called_with(sub, set(), reset=True)
        assert sm.is_warming_up is False

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

    def test_config_default_is_thread(self):
        assert LiveConfig.model_fields["fit_executor"].default == FitExecutorMode.THREAD
        assert LiveConfig.model_fields["fit_soft_deadline_s"].default == pytest.approx(120.0)


class TestFitContextCachedMarketData:
    """The fit reads cached non-OHLC data (frab reads funding payments this way) — it must
    get a detached snapshot rather than the live series the ProcessorThread appends to."""

    def _fit_context_over(self, cache: CachedMarketDataHolder) -> FitContext:
        # - real MarketManager (bypassing its heavy __init__) so the read goes through the
        #   same _get_cached_market_data seam the fit uses, not a mock of it
        market_manager = MarketManager.__new__(MarketManager)
        market_manager._cache = cache
        context = MagicMock()
        context._market_data_provider = market_manager
        return FitContext(context, FitCycleState())

    def test_the_fit_reads_a_detached_copy_carrying_the_stored_records(self):
        instrument = _mock_instrument()
        cache = CachedMarketDataHolder(default_timeframe="1h")
        t0 = np.datetime64("2023-01-01T10:00:00", "ns").astype(np.int64)
        cache.update(instrument, DataType.FUNDING_PAYMENT, FundingPayment(t0, 0.0001, 8))
        fit_ctx = self._fit_context_over(cache)

        series = fit_ctx.get_cached_market_data(instrument, DataType.FUNDING_PAYMENT)
        cache.update(instrument, DataType.FUNDING_PAYMENT, FundingPayment(t0 + 8 * 3_600_000_000_000, 0.0002, 8))

        # - the stored object survives the clone: GenericSeries overrides _add_new_item to
        #   keep the record, the base TimeSeries would coerce it to a double
        assert series[0].funding_rate == pytest.approx(0.0001)
        # - and the copy is detached from the series the ProcessorThread keeps appending to
        assert len(series) == 1


class TestFitCloneDestruction:
    """Clones handed to the fit are emptied when it returns: a clone (or an indicator
    attached to one) that outlives its fit serves data frozen at fit time, silently. After
    destruction it raises instead."""

    def _cache_with_bars(self, instrument, n: int = 10) -> CachedMarketDataHolder:
        cache = CachedMarketDataHolder(default_timeframe="1h")
        t0 = np.datetime64("2023-01-01T00:00:00", "ns").astype(np.int64)
        hour = 3_600_000_000_000
        cache.update_by_bars(
            instrument,
            "1h",
            [
                Bar(time=t0 + i * hour, open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.0 + i, volume=1.0)
                for i in range(n)
            ],
        )
        return cache

    def _fit_ctx_over(self, cache: CachedMarketDataHolder) -> FitContext:
        market_manager = MarketManager.__new__(MarketManager)
        market_manager._cache = cache
        provider = MagicMock(is_simulation=False)
        # - _ohlc compares now against the newest bar to decide on a history fetch
        provider.time_provider.time.return_value = np.datetime64("2023-01-01T09:00:00", "ns")
        market_manager._get_data_provider = MagicMock(return_value=provider)
        context = MagicMock()
        context._market_data_provider = market_manager
        return FitContext(context, FitCycleState())

    def test_an_indicator_attached_during_the_fit_raises_after_it_returns(self):
        from qubx.ta.indicators import sma

        instrument = _mock_instrument()
        cache = self._cache_with_bars(instrument)
        fit_ctx = self._fit_ctx_over(cache)

        ohlc = fit_ctx.ohlc(instrument, "1h", None)
        stashed = sma(ohlc.close, 3)
        assert stashed[0] == pytest.approx(108.0)  # - valid while the fit runs

        fit_ctx.destroy_clones()

        with pytest.raises(IndexError):
            _ = stashed[0]
        assert len(ohlc) == 0

    def test_a_stashed_generic_series_is_emptied_too(self):
        instrument = _mock_instrument()
        cache = CachedMarketDataHolder(default_timeframe="1h")
        t0 = np.datetime64("2023-01-01T10:00:00", "ns").astype(np.int64)
        cache.update(instrument, DataType.FUNDING_PAYMENT, FundingPayment(t0, 0.0001, 8))
        fit_ctx = self._fit_ctx_over(cache)

        stashed = fit_ctx.get_cached_market_data(instrument, DataType.FUNDING_PAYMENT)
        assert len(stashed) == 1

        fit_ctx.destroy_clones()

        assert len(stashed) == 0

    def test_the_live_series_and_its_indicators_are_untouched(self):
        from qubx.ta.indicators import sma

        instrument = _mock_instrument()
        cache = self._cache_with_bars(instrument)
        live = cache.get_ohlcv(instrument, "1h")
        live_sma = sma(live.close, 3)
        fit_ctx = self._fit_ctx_over(cache)

        fit_ctx.ohlc(instrument, "1h", None)
        fit_ctx.destroy_clones()

        # - the ProcessorThread's own series must survive its fit untouched
        assert len(live) == 10
        assert live_sma[0] == pytest.approx(108.0)
        assert len(cache.get_ohlcv(instrument, "1h")) == 10

    def test_the_fit_lifecycle_destroys_what_the_body_kept(self):
        """End-to-end through _handle_fit: whatever the body stashed is dead once the
        FitCommit lands, without the strategy doing anything."""
        pm, context, channel = make_thread_pm()
        instrument = _mock_instrument()
        cache = self._cache_with_bars(instrument)
        market_manager = MarketManager.__new__(MarketManager)
        market_manager._cache = cache
        provider = MagicMock(is_simulation=False)
        provider.time_provider.time.return_value = np.datetime64("2023-01-01T09:00:00", "ns")
        market_manager._get_data_provider = MagicMock(return_value=provider)
        context._market_data_provider = market_manager
        kept: dict[str, object] = {}

        pm._strategy.on_fit.side_effect = lambda ctx: kept.update(ohlc=ctx.ohlc(instrument, "1h", None))
        pm._handle_fit(None, "fit", (None, T0))
        drain_until_committed(pm, channel)

        assert len(kept["ohlc"]) == 0  # type: ignore[arg-type]
        assert len(cache.get_ohlcv(instrument, "1h")) == 10  # - the live series is untouched
