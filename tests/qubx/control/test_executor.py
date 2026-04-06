import concurrent.futures
from queue import Queue
from unittest.mock import MagicMock

import pytest

from qubx.control.executor import ActionExecutor, CommandEvent


def _make_mock_ctx():
    ctx = MagicMock()
    # Use a plain object as strategy so isinstance(strategy, IControllable) is False
    ctx.strategy = type("FakeStrategy", (), {})()

    instr = MagicMock()
    instr.__str__ = lambda self: "BTCUSDT"
    ctx.instruments = [instr]

    pos = MagicMock()
    pos.quantity = 0.5
    pos.position_avg_price = 67500.0
    pos.last_update_price = 68000.0
    pos.pnl = 250.0
    pos.r_pnl = 100.0
    pos.market_value_funds = 34000.0
    ctx.get_positions.return_value = {instr: pos}
    ctx.get_balances.return_value = []
    ctx.get_orders.return_value = {}
    ctx.get_total_capital.return_value = 10000.0
    ctx.get_net_leverage.return_value = 0.15
    ctx.get_gross_leverage.return_value = 0.15
    ctx.is_warmup_in_progress = False
    ctx.is_simulation = False
    ctx.health = MagicMock()
    ctx.health.is_connected.return_value = True
    ctx.health.get_queue_size.return_value = 0
    ctx.health.is_stale.return_value = False
    return ctx


class TestActionExecutor:
    def test_list_actions_includes_builtins(self):
        ctx = _make_mock_ctx()
        executor = ActionExecutor(ctx, Queue())
        actions = executor.list_actions()
        names = {a.name for a in actions}
        assert "get_positions" in names
        assert "get_universe" in names
        assert "trade" in names

    def test_resolve_known_action(self):
        ctx = _make_mock_ctx()
        executor = ActionExecutor(ctx, Queue())
        action_def, handler = executor.resolve("get_positions")
        assert action_def is not None
        assert action_def.name == "get_positions"

    def test_resolve_unknown_action(self):
        ctx = _make_mock_ctx()
        executor = ActionExecutor(ctx, Queue())
        action_def, handler = executor.resolve("nonexistent")
        assert action_def is None

    @pytest.mark.asyncio
    async def test_execute_read_only_directly(self):
        ctx = _make_mock_ctx()
        executor = ActionExecutor(ctx, Queue())
        result = await executor.execute("get_positions", {})
        assert result.status == "ok"
        assert "positions" in result.data

    @pytest.mark.asyncio
    async def test_execute_unknown_returns_error(self):
        ctx = _make_mock_ctx()
        executor = ActionExecutor(ctx, Queue())
        result = await executor.execute("nonexistent", {})
        assert result.status == "error"
        assert result.error is not None and "Unknown action" in result.error


class TestCommandEvent:
    def test_command_event_creation(self):
        future = concurrent.futures.Future()
        cmd = CommandEvent("trade", {"symbol": "BTC"}, future)
        assert cmd.name == "trade"
        assert cmd.params == {"symbol": "BTC"}
        assert cmd.future is future

    def test_execute_command_on_strategy_thread(self):
        ctx = _make_mock_ctx()
        queue = Queue()
        executor = ActionExecutor(ctx, queue)
        future = concurrent.futures.Future()
        cmd = CommandEvent("get_positions", {}, future)
        result = executor.execute_command(cmd)
        assert result.status == "ok"

    def test_execute_command_unknown(self):
        ctx = _make_mock_ctx()
        queue = Queue()
        executor = ActionExecutor(ctx, queue)
        future = concurrent.futures.Future()
        cmd = CommandEvent("nonexistent", {}, future)
        result = executor.execute_command(cmd)
        assert result.status == "error"
