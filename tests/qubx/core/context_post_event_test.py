import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from qubx.core.context import StrategyContext
from qubx.core.fit_context import FitContext
from qubx.core.fit_executor import FitCycleState
from qubx.core.interfaces import IStrategyContext
from tests.qubx.core.conftest import make_pm, real_handler_map


def _ctx_shell(channel, event_handlers=None):
    # Exercise the unbound methods on a shell object: post_event needs _channel and
    # _processing_manager.has_handler (the registered-handler membership check),
    # register_handler needs _processing_manager and the fit-thread tripwire.
    registry = event_handlers if event_handlers is not None else {}
    shell = SimpleNamespace(
        _channel=channel,
        _processing_manager=MagicMock(),
        _fit_state=SimpleNamespace(is_fit_thread=lambda: False),
    )
    shell._processing_manager._event_handlers = registry
    # a bare MagicMock would answer every name truthily and silently hide the guard
    shell._processing_manager.has_handler.side_effect = lambda name: name in registry
    shell._assert_not_fit_thread = lambda name: StrategyContext._assert_not_fit_thread(shell, name)
    return shell


def _fit_ctx_with_real_validation() -> tuple[FitContext, FitCycleState]:
    """FitContext over a context whose _processing_manager is a REAL ProcessingManager
    half-object, so _validate_handler_name actually runs (a MagicMock would swallow it)."""
    pm = make_pm()
    pm._handlers = real_handler_map()
    pm._custom_scheduled_methods = {}
    context = MagicMock()
    context._processing_manager = pm
    fit_state = FitCycleState()
    return FitContext(context, fit_state), fit_state


def test_post_event_sends_tuple_on_the_data_channel():
    channel = MagicMock()
    shell = _ctx_shell(channel, event_handlers={"agg.sources": lambda ctx, payload: None})

    StrategyContext.post_event(shell, "agg.sources")

    channel.send.assert_called_once_with((None, "agg.sources", None, False))


def test_post_event_sends_the_payload_in_the_tuple():
    # The payload rides the data tuple's data slot, which is what _process_custom_event
    # hands to the registered handler as its second argument.
    channel = MagicMock()
    shell = _ctx_shell(channel, event_handlers={"agg.sources": lambda ctx, payload: None})
    payload = {"sources": ["a"]}

    StrategyContext.post_event(shell, "agg.sources", payload)

    channel.send.assert_called_once_with((None, "agg.sources", payload, False))
    assert channel.send.call_args[0][0][2] is payload  # handed over, not copied


def test_post_event_allowed_from_fit_thread():
    # post_event is the one scheduling-family call with NO fit-thread tripwire: live it is a
    # thread-safe Queue.put_nowait, which is the whole point of the hook (wake the strategy
    # thread from anywhere). Pin it so a future _assert_not_fit_thread sweep can't add one.
    channel = MagicMock()
    shell = _ctx_shell(channel, event_handlers={"agg.sources": lambda ctx, payload: None})
    shell._fit_state = SimpleNamespace(is_fit_thread=lambda: True)

    StrategyContext.post_event(shell, "agg.sources")

    channel.send.assert_called_once_with((None, "agg.sources", None, False))


def test_post_event_for_unregistered_name_raises_value_error():
    # A typo'd name must never fall through to the tuple dispatch: unknown event types
    # decay into a bogus MarketEvent(instrument=None) delivered to on_market_data
    # (processing.py's _process_custom_event), which repeatedly crashes user strategies.
    channel = MagicMock()
    shell = _ctx_shell(channel)  # _event_handlers == {}

    with pytest.raises(ValueError):
        StrategyContext.post_event(shell, "agg.sorces")  # typo

    channel.send.assert_not_called()


def test_post_event_for_a_scheduled_id_raises_value_error():
    # schedule()/delay() ids live in a separate registry and their callbacks take (ctx)
    # only, so has_handler does not cover them: posting to one must be rejected here rather
    # than reach the channel and call the scheduled method with a payload it cannot take.
    channel = MagicMock()
    shell = _ctx_shell(channel, event_handlers={"agg.sources": lambda ctx, p: None})

    with pytest.raises(ValueError):
        StrategyContext.post_event(shell, "custom_schedule_x")

    channel.send.assert_not_called()


def test_register_handler_delegates_to_processing_manager():
    shell = _ctx_shell(MagicMock())
    fn = lambda ctx, payload: None  # noqa: E731
    StrategyContext.register_handler(shell, "agg.sources", fn)
    shell._processing_manager.register_handler.assert_called_once_with("agg.sources", fn)


def test_register_handler_from_fit_thread_raises():
    shell = _ctx_shell(MagicMock())
    shell._fit_state = SimpleNamespace(is_fit_thread=lambda: True)

    with pytest.raises(RuntimeError):
        StrategyContext.register_handler(shell, "agg.sources", lambda ctx, payload: None)

    shell._processing_manager.register_handler.assert_not_called()


def test_interface_declares_both_methods():
    assert callable(getattr(IStrategyContext, "post_event"))
    assert callable(getattr(IStrategyContext, "register_handler"))


def test_fit_context_register_handler_is_deferred_via_fit_state_record():
    context = MagicMock()
    fit_state = FitCycleState()
    fit_ctx = FitContext(context, fit_state)
    fit_state.begin(threading.get_ident())

    fn = lambda ctx, payload: None  # noqa: E731
    fit_ctx.register_handler("agg.sources", fn)

    ops, _ = fit_state.end()
    assert len(ops) == 1
    context._processing_manager.register_handler.assert_not_called()  # deferred, not applied yet

    ops[0]()  # simulate the ProcessorThread replaying it at the FitCommit
    context._processing_manager.register_handler.assert_called_once_with("agg.sources", fn)


@pytest.mark.parametrize("bad_name", ["trade", "funding_rate", "fit", ""])
def test_fit_context_register_handler_validates_eagerly(bad_name):
    # _handle_fit_commit only LOGS a deferred op's exception, so deferring the name checks
    # would make a bad name fail silently on the fit thread and leave every later post_event
    # raising forever. Validation must happen in the on_fit call itself, before any commit.
    fit_ctx, fit_state = _fit_ctx_with_real_validation()
    fit_state.begin(threading.get_ident())

    with pytest.raises(ValueError):
        fit_ctx.register_handler(bad_name, lambda ctx, payload: None)

    ops, _ = fit_state.end()
    assert ops == ()  # nothing recorded -> nothing to fail silently at the commit


def test_fit_context_register_handler_rejects_the_scheduled_arity_eagerly():
    # Same reason the name checks are eager: _handle_fit_commit only LOGS a deferred op's
    # exception, so a wrong-arity handler deferred to the commit would register silently and
    # then no-op on every post. It must raise into on_fit instead.
    fit_ctx, fit_state = _fit_ctx_with_real_validation()
    fit_state.begin(threading.get_ident())

    with pytest.raises(ValueError, match=r"must accept \(ctx, payload\)"):
        fit_ctx.register_handler("agg.sources", lambda ctx: None)

    ops, _ = fit_state.end()
    assert ops == ()  # nothing recorded -> nothing to fail silently at the commit


def test_fit_context_post_event_passes_through_to_real_context():
    context = MagicMock()
    fit_ctx = FitContext(context, FitCycleState())

    fit_ctx.post_event("agg.sources")

    context.post_event.assert_called_once_with("agg.sources", None)


def test_fit_context_post_event_forwards_the_payload():
    context = MagicMock()
    fit_ctx = FitContext(context, FitCycleState())
    payload = object()

    fit_ctx.post_event("agg.sources", payload)

    context.post_event.assert_called_once_with("agg.sources", payload)
