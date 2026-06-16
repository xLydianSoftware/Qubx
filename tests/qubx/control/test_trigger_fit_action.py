from pytest_mock import MockerFixture

from qubx.control.builtin import BUILTIN_ACTIONS


def test_trigger_fit_action_registered_and_dangerous():
    assert "trigger_fit" in BUILTIN_ACTIONS
    action_def, handler = BUILTIN_ACTIONS["trigger_fit"]
    assert action_def.dangerous is True
    assert action_def.read_only is False


def test_trigger_fit_handler_calls_ctx_trigger_fit(mocker: MockerFixture):
    _, handler = BUILTIN_ACTIONS["trigger_fit"]
    ctx = mocker.Mock()
    result = handler(ctx)
    ctx.trigger_fit.assert_called_once_with()
    assert result.status == "ok"
