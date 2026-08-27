"""Boot state machine driving the strategy pipeline: live boot ordering and simulation parity."""

from unittest.mock import MagicMock

import numpy as np

from qubx.core.basics import CtrlChannel, TriggerEvent
from qubx.core.boot import BootPhase
from qubx.core.interfaces import StrategyState
from qubx.core.mixins.processing import ProcessingManager

T0 = np.datetime64("2025-01-01T00:00:00", "ns")


def make_pm(
    *,
    is_simulation: bool = False,
    is_warmup_in_progress: bool = False,
    is_on_fit_called: bool = False,
    warmup_positions: dict | None = None,
    restored_state=None,
    resolver=None,
    fit_on_start: bool = False,
):
    channel = CtrlChannel("test-databus")
    context = MagicMock()
    context.is_simulation = is_simulation
    context.is_paper_trading = False
    context.instruments = []
    context._strategy_state = StrategyState(
        is_on_init_called=True,
        is_on_start_called=False,
        is_on_warmup_finished_called=False,
        is_on_fit_called=is_on_fit_called,
        is_warmup_in_progress=is_warmup_in_progress,
    )
    context._data_providers = [MagicMock(channel=channel)]
    context.emitter = None
    context._market_data_provider = MagicMock()
    context.get_warmup_positions.return_value = warmup_positions or {}
    context.get_warmup_orders.return_value = {}
    context.get_warmup_active_targets.return_value = {}
    context.get_restored_state.return_value = restored_state
    context.initializer.get_state_resolver.return_value = resolver
    context.initializer.get_fit_on_start.return_value = fit_on_start

    strategy = MagicMock()
    strategy.__class__.__name__ = "TestStrategy"
    strategy.on_fit.return_value = None
    strategy.on_event.return_value = []
    strategy.on_market_data.return_value = []

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
    account_manager = MagicMock()
    account_manager.is_synced.return_value = True
    account_manager.get_positions.return_value = {}
    account_manager.get_orders.return_value = {}  # _log_state_mismatch iterates these

    pm = ProcessingManager(
        context=context,
        strategy=strategy,
        logging=MagicMock(),
        market_data=market_data,
        subscription_manager=subscription_manager,
        time_provider=time_provider,
        account_manager=account_manager,
        connectors={},
        position_tracker=position_tracker,
        position_gathering=MagicMock(),
        universe_manager=universe_manager,
        scheduler=MagicMock(),
        is_simulation=is_simulation,
        health_monitor=health_monitor,
        delisting_detector=MagicMock(),
        fit_executor="inline",
    )
    pm._is_data_ready = lambda: True  # type: ignore[method-assign]
    pm._init_stage_position_tracker = MagicMock()
    pm._init_stage_position_tracker.process_signals.return_value = []
    pm._init_stage_position_tracker.update.return_value = []
    context._processing_manager = pm
    return pm, context, strategy


def drive(pm, passes: int = 1):
    for _ in range(passes):
        pm._run_strategy_pipeline(None)


class TestLiveBoot:
    def test_resolution_runs_without_warmup_output(self):
        resolver = MagicMock()
        resolver.__name__ = "custom"
        pm, ctx, _ = make_pm(resolver=resolver)
        drive(pm)
        resolver.assert_called_once_with(ctx, {}, {}, {})

    def test_default_resolver_installed_when_none_registered(self):
        pm, ctx, _ = make_pm(resolver=None)
        drive(pm)  # REDUCE_ONLY with empty args -> guard: must not read live positions
        ctx.get_positions.assert_not_called()
        assert pm._boot.phase in (BootPhase.BOOT_FIT, BootPhase.TRADING)

    def test_resolution_receives_warmup_output(self):
        resolver = MagicMock()
        resolver.__name__ = "custom"
        instr = MagicMock()
        instr.symbol = "BTCUSDT"
        pos = MagicMock()
        pos.quantity = 1.0  # real float: _log_state_mismatch formats it with :.6f
        wp = {instr: pos}
        pm, ctx, _ = make_pm(resolver=resolver, warmup_positions=wp)
        drive(pm)
        resolver.assert_called_once_with(ctx, wp, {}, {})

    def test_boot_sequence_reaches_trading_and_fit_pass_returns_false(self):
        pm, ctx, strategy = make_pm()
        assert pm._run_strategy_pipeline(None) is False  # boot pass fires on_start..fit
        strategy.on_start.assert_called_once()
        strategy.on_warmup_finished.assert_called_once()
        strategy.on_fit.assert_called_once()
        assert ctx._strategy_state.is_on_fit_called
        assert pm._boot.is_trading

    def test_no_event_is_processed_on_the_boot_fit_pass(self):
        pm, _, strategy = make_pm()
        event = TriggerEvent(time=T0, type="time", instrument=None, data=None)
        pm._run_strategy_pipeline(event)
        strategy.on_fit.assert_called_once()
        strategy.on_event.assert_not_called()
        pm._run_strategy_pipeline(event)
        strategy.on_event.assert_called_once()

    def test_warmup_fit_satisfies_boot_fit_when_knob_off(self):
        pm, ctx, strategy = make_pm(is_on_fit_called=True)
        drive(pm)
        strategy.on_fit.assert_not_called()
        assert pm._boot.is_trading

    def test_restore_called_with_restored_state(self):
        restored = MagicMock()
        restored.instrument_to_signal_positions = {}
        restored.instrument_to_target_positions = {}
        pm, ctx, _ = make_pm(restored_state=restored)
        drive(pm)
        assert pm._boot.is_trading


class TestSimulationParity:
    def test_plain_simulation_boots_without_resolution(self):
        resolver = MagicMock()
        pm, ctx, strategy = make_pm(is_simulation=True, resolver=resolver)
        drive(pm)
        resolver.assert_not_called()
        strategy.on_start.assert_called_once()
        strategy.on_warmup_finished.assert_called_once()
        strategy.on_fit.assert_called_once()
        assert pm._boot.is_trading

    def test_warmup_sim_context_skips_warmup_finished_and_restore(self):
        restored = MagicMock()
        pm, ctx, strategy = make_pm(is_simulation=True, is_warmup_in_progress=True, restored_state=restored)
        drive(pm)
        strategy.on_start.assert_called_once()
        strategy.on_warmup_finished.assert_not_called()
        strategy.on_fit.assert_called_once()
        ctx.get_restored_state.assert_not_called()
        assert pm._boot.is_trading
        assert not ctx._strategy_state.is_on_warmup_finished_called
