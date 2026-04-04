from unittest.mock import MagicMock

from qubx.control.builtin import BUILTIN_ACTIONS


def _make_mock_ctx():
    """Create a mock IStrategyContext with common attributes."""
    ctx = MagicMock()

    # Mock instruments
    instr1 = MagicMock()
    instr1.__str__ = lambda self: "BTCUSDT"
    instr1.min_size = 0.001
    instr2 = MagicMock()
    instr2.__str__ = lambda self: "ETHUSDT"
    instr2.min_size = 0.01
    ctx.instruments = [instr1, instr2]

    # Mock positions
    pos1 = MagicMock()
    pos1.quantity = 0.5
    pos1.position_avg_price = 67500.0
    pos1.last_update_price = 68000.0
    pos1.pnl = 250.0
    pos1.r_pnl = 100.0
    pos1.market_value_funds = 34000.0
    pos2 = MagicMock()
    pos2.quantity = 0.0
    pos2.position_avg_price = 0.0
    pos2.last_update_price = 3800.0
    pos2.pnl = 0.0
    pos2.r_pnl = 0.0
    pos2.market_value_funds = 0.0
    ctx.get_positions.return_value = {instr1: pos1, instr2: pos2}

    # Mock balances
    bal = MagicMock()
    bal.exchange = "BINANCE.UM"
    bal.currency = "USDT"
    bal.total = 10000.0
    bal.free = 9500.0
    bal.locked = 500.0
    ctx.get_balances.return_value = [bal]

    # Mock orders
    ctx.get_orders.return_value = {}

    # Mock state
    ctx.get_total_capital.return_value = 10000.0
    ctx.get_net_leverage.return_value = 0.15
    ctx.get_gross_leverage.return_value = 0.15
    ctx.is_warmup_in_progress = False
    ctx.is_simulation = False

    # Mock health
    ctx.health = MagicMock()
    ctx.health.is_connected.return_value = True
    ctx.health.get_queue_size.return_value = 0
    ctx.health.is_stale.return_value = False

    return ctx


class TestGetUniverse:
    def test_returns_instrument_list(self):
        ctx = _make_mock_ctx()
        _, handler = BUILTIN_ACTIONS["get_universe"]
        result = handler(ctx)
        assert result.status == "ok"
        assert result.data["count"] == 2
        assert "BTCUSDT" in result.data["instruments"]
        assert "ETHUSDT" in result.data["instruments"]


class TestGetPositions:
    def test_returns_positions(self):
        ctx = _make_mock_ctx()
        _, handler = BUILTIN_ACTIONS["get_positions"]
        result = handler(ctx)
        assert result.status == "ok"
        assert "BTCUSDT" in result.data["positions"]
        pos = result.data["positions"]["BTCUSDT"]
        assert pos["quantity"] == 0.5
        assert pos["avg_price"] == 67500.0
        assert pos["market_price"] == 68000.0


class TestGetBalances:
    def test_returns_balances(self):
        ctx = _make_mock_ctx()
        _, handler = BUILTIN_ACTIONS["get_balances"]
        result = handler(ctx)
        assert result.status == "ok"
        assert result.data["balances"]["BINANCE.UM:USDT"]["total"] == 10000.0
        assert result.data["balances"]["BINANCE.UM:USDT"]["free"] == 9500.0


class TestGetState:
    def test_returns_full_state(self):
        ctx = _make_mock_ctx()
        _, handler = BUILTIN_ACTIONS["get_state"]
        result = handler(ctx)
        assert result.status == "ok"
        assert result.data["total_capital"] == 10000.0
        assert result.data["net_leverage"] == 0.15
        assert len(result.data["instruments"]) == 2


class TestGetHealth:
    def test_returns_health_info(self):
        ctx = _make_mock_ctx()
        ctx.exchanges = ["BINANCE.UM"]
        _, handler = BUILTIN_ACTIONS["get_health"]
        result = handler(ctx)
        assert result.status == "ok"
        assert "connected" in result.data
        assert "queue_size" in result.data
        assert "data_latencies_ms" in result.data


class TestGetOrders:
    def test_returns_empty_orders(self):
        ctx = _make_mock_ctx()
        _, handler = BUILTIN_ACTIONS["get_orders"]
        result = handler(ctx)
        assert result.status == "ok"
        assert result.data["orders"] == []


class TestBuiltinRegistry:
    def test_all_actions_have_definitions(self):
        for name, (action_def, handler) in BUILTIN_ACTIONS.items():
            assert action_def.name == name
            assert handler is not None
            assert action_def.description

    def test_read_only_actions_are_marked(self):
        read_only = {
            "get_universe", "get_positions", "get_balances", "get_orders", "get_quote",
            "get_ohlc", "get_state", "get_health", "get_total_capital", "get_leverages",
            "get_subscriptions",
        }
        for name in read_only:
            action_def, _ = BUILTIN_ACTIONS[name]
            assert action_def.read_only is True, f"{name} should be read_only"

    def test_dangerous_actions_are_marked(self):
        dangerous = {
            "trade", "set_target_position", "set_target_leverage", "close_position",
            "close_positions", "emit_signal", "remove_instruments", "set_universe",
        }
        for name in dangerous:
            action_def, _ = BUILTIN_ACTIONS[name]
            assert action_def.dangerous is True, f"{name} should be dangerous"
