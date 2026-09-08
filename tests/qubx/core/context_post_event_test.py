import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from qubx.core.context import StrategyContext
from qubx.core.fit_context import FitContext
from qubx.core.fit_executor import FitCycleState
from qubx.core.interfaces import IStrategyContext


def _ctx_shell(providers, custom_scheduled_methods=None):
    # Exercise the unbound methods on a shell object: post_event needs _data_providers and
    # _processing_manager._custom_scheduled_methods (the registered-handler membership check),
    # register_handler needs _processing_manager and the fit-thread tripwire.
    shell = SimpleNamespace(
        _data_providers=providers,
        _processing_manager=MagicMock(),
        _fit_state=SimpleNamespace(is_fit_thread=lambda: False),
    )
    shell._processing_manager._custom_scheduled_methods = (
        custom_scheduled_methods if custom_scheduled_methods is not None else {}
    )
    shell._assert_not_fit_thread = lambda name: StrategyContext._assert_not_fit_thread(shell, name)
    return shell


def test_post_event_sends_tuple_on_the_data_channel():
    channel = MagicMock()
    shell = _ctx_shell(
        [SimpleNamespace(channel=channel)],
        custom_scheduled_methods={"agg.sources": lambda ctx: None},
    )

    StrategyContext.post_event(shell, "agg.sources")

    channel.send.assert_called_once_with((None, "agg.sources", None, False))


def test_post_event_without_data_provider_raises():
    shell = _ctx_shell([])
    with pytest.raises(RuntimeError):
        StrategyContext.post_event(shell, "agg.sources")


def test_post_event_for_unregistered_name_raises_value_error():
    # A typo'd name must never fall through to the tuple dispatch: unknown event types
    # decay into a bogus MarketEvent(instrument=None) delivered to on_market_data
    # (processing.py's _process_custom_event), which repeatedly crashes user strategies.
    channel = MagicMock()
    shell = _ctx_shell([SimpleNamespace(channel=channel)])  # _custom_scheduled_methods == {}

    with pytest.raises(ValueError):
        StrategyContext.post_event(shell, "agg.sorces")  # typo

    channel.send.assert_not_called()


def test_register_handler_delegates_to_processing_manager():
    shell = _ctx_shell([SimpleNamespace(channel=MagicMock())])
    fn = lambda ctx: None  # noqa: E731
    StrategyContext.register_handler(shell, "agg.sources", fn)
    shell._processing_manager.register_handler.assert_called_once_with("agg.sources", fn)


def test_register_handler_from_fit_thread_raises():
    shell = _ctx_shell([SimpleNamespace(channel=MagicMock())])
    shell._fit_state = SimpleNamespace(is_fit_thread=lambda: True)

    with pytest.raises(RuntimeError):
        StrategyContext.register_handler(shell, "agg.sources", lambda ctx: None)

    shell._processing_manager.register_handler.assert_not_called()


def test_interface_declares_both_methods():
    assert callable(getattr(IStrategyContext, "post_event"))
    assert callable(getattr(IStrategyContext, "register_handler"))


def test_fit_context_register_handler_is_deferred_via_fit_state_record():
    context = MagicMock()
    fit_state = FitCycleState()
    fit_ctx = FitContext(context, fit_state)
    fit_state.begin(threading.get_ident())

    fn = lambda ctx: None  # noqa: E731
    fit_ctx.register_handler("agg.sources", fn)

    ops, _ = fit_state.end()
    assert len(ops) == 1
    context._processing_manager.register_handler.assert_not_called()  # deferred, not applied yet

    ops[0]()  # simulate the ProcessorThread replaying it at the FitCommit
    context._processing_manager.register_handler.assert_called_once_with("agg.sources", fn)


def test_fit_context_post_event_passes_through_to_real_context():
    context = MagicMock()
    fit_ctx = FitContext(context, FitCycleState())

    fit_ctx.post_event("agg.sources")

    context.post_event.assert_called_once_with("agg.sources")
