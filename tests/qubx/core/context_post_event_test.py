from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from qubx.core.context import StrategyContext
from qubx.core.interfaces import IStrategyContext


def _ctx_shell(providers):
    # Exercise the unbound methods on a shell object: post_event needs only _data_providers,
    # register_handler needs _processing_manager and the fit-thread tripwire.
    shell = SimpleNamespace(
        _data_providers=providers,
        _processing_manager=MagicMock(),
        _fit_state=SimpleNamespace(is_fit_thread=lambda: False),
    )
    shell._assert_not_fit_thread = lambda name: StrategyContext._assert_not_fit_thread(shell, name)
    return shell


def test_post_event_sends_tuple_on_the_data_channel():
    channel = MagicMock()
    shell = _ctx_shell([SimpleNamespace(channel=channel)])

    StrategyContext.post_event(shell, "agg.sources")

    channel.send.assert_called_once_with((None, "agg.sources", None, False))


def test_post_event_without_data_provider_raises():
    shell = _ctx_shell([])
    with pytest.raises(RuntimeError):
        StrategyContext.post_event(shell, "agg.sources")


def test_register_handler_delegates_to_processing_manager():
    shell = _ctx_shell([SimpleNamespace(channel=MagicMock())])
    fn = lambda ctx: None  # noqa: E731
    StrategyContext.register_handler(shell, "agg.sources", fn)
    shell._processing_manager.register_handler.assert_called_once_with("agg.sources", fn)


def test_interface_declares_both_methods():
    assert callable(getattr(IStrategyContext, "post_event"))
    assert callable(getattr(IStrategyContext, "register_handler"))
