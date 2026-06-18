"""Tests that ProcessingManager refreshes the instrument-service cache before on_fit."""

from unittest.mock import MagicMock, patch

import pytest

from qubx.core.mixins.processing import ProcessingManager


@pytest.fixture
def processing_manager():
    """ProcessingManager with fully mocked dependencies (mirrors the harness in
    processing_pending_signals_test.py)."""
    context = MagicMock()
    context.is_simulation = True
    context.instruments = []
    context._strategy_state = MagicMock()
    context._strategy_state.is_on_fit_called = False

    strategy = MagicMock()
    strategy.__class__.__name__ = "TestStrategy"

    logging = MagicMock()

    cache = MagicMock()
    cache.default_timeframe = "1h"

    market_data = MagicMock()
    market_data.get_market_data_cache.return_value = cache

    subscription_manager = MagicMock()
    subscription_manager.get_base_subscription.return_value = "quote"

    time_provider = MagicMock()

    account = MagicMock()
    position_tracker = MagicMock()
    position_tracker.process_signals.return_value = []
    position_gathering = MagicMock()
    universe_manager = MagicMock()
    universe_manager.is_trading_allowed.return_value = True
    scheduler = MagicMock()

    health_monitor = MagicMock()
    health_monitor.return_value.__enter__ = MagicMock(return_value=None)
    health_monitor.return_value.__exit__ = MagicMock(return_value=False)

    delisting_detector = MagicMock()

    pm = ProcessingManager(
        context=context,
        strategy=strategy,
        logging=logging,
        market_data=market_data,
        subscription_manager=subscription_manager,
        time_provider=time_provider,
        account=account,
        position_tracker=position_tracker,
        position_gathering=position_gathering,
        universe_manager=universe_manager,
        scheduler=scheduler,
        is_simulation=True,
        health_monitor=health_monitor,
        delisting_detector=delisting_detector,
    )
    pm._test_context = context
    pm._test_strategy = strategy
    return pm


def test_invoke_on_fit_refreshes_instrument_service_before_on_fit(processing_manager):
    context = processing_manager._test_context
    strategy = processing_manager._test_strategy

    # Attach both calls to a shared parent mock so their relative order is recorded.
    order = MagicMock()
    order.attach_mock(context._instrument_service_manager.enforce_at_fit, "enforce_at_fit")
    order.attach_mock(strategy.on_fit, "on_fit")

    # name-mangled: ProcessingManager.__invoke_on_fit
    processing_manager._ProcessingManager__invoke_on_fit()

    context._instrument_service_manager.enforce_at_fit.assert_called_once_with()
    strategy.on_fit.assert_called_once_with(context)

    called = [c[0] for c in order.mock_calls]
    assert called.index("enforce_at_fit") < called.index("on_fit")


def test_invoke_on_fit_marks_fit_called_even_if_refresh_raises(processing_manager):
    context = processing_manager._test_context
    strategy = processing_manager._test_strategy
    context._instrument_service_manager.enforce_at_fit.side_effect = RuntimeError("boom")

    # Should not propagate: the on_fit try/except swallows it (refresh shares on_fit's
    # error handling — acceptable per design §5).
    processing_manager._ProcessingManager__invoke_on_fit()

    strategy.on_fit.assert_not_called()  # refresh raised before on_fit ran
    assert context._strategy_state.is_on_fit_called is True


def test_trigger_fit_schedules_handle_fit_on_manager_not_context(processing_manager):
    """trigger_fit must schedule a callback that invokes _handle_fit on the
    ProcessingManager (self), not on the context the scheduler passes in — the
    context facade has no _handle_fit. Regression for the no-op trigger_fit bug."""
    pm = processing_manager

    pm.trigger_fit()

    # delay() stored exactly one one-off callback.
    assert len(pm._custom_scheduled_methods) == 1
    scheduled = next(iter(pm._custom_scheduled_methods.values()))

    # The scheduler dispatches the callback as method(self._context); that context
    # exposes no _handle_fit. Using a spec'd mock makes any c._handle_fit access raise.
    from unittest.mock import MagicMock

    ctx_without_handle_fit = MagicMock(spec=["time"])
    ctx_without_handle_fit.time.return_value = 0

    with patch.object(pm, "_handle_fit") as mock_handle_fit:
        scheduled(ctx_without_handle_fit)  # must not raise AttributeError
        mock_handle_fit.assert_called_once()
