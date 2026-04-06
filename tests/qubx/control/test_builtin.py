from unittest.mock import MagicMock

from qubx.control.builtin import BUILTIN_ACTIONS


def _make_mock_ctx():
    """Create a mock IStrategyContext with common attributes."""
    ctx = MagicMock()

    # Mock instruments
    instr1 = MagicMock()
    instr1.__str__ = lambda self: "BTCUSDT"
    instr1.symbol = "BTCUSDT"
    instr1.exchange = "BINANCE.UM"
    instr1.market_type = "SWAP"
    instr1.base = "BTC"
    instr1.quote = "USDT"
    instr1.tick_size = 0.1
    instr1.lot_size = 0.001
    instr1.min_size = 0.001
    instr1.min_notional = 5.0
    instr1.contract_size = 1.0
    instr1.min_size = 0.001
    instr2 = MagicMock()
    instr2.__str__ = lambda self: "ETHUSDT"
    instr2.symbol = "ETHUSDT"
    instr2.exchange = "BINANCE.UM"
    instr2.market_type = "SWAP"
    instr2.base = "ETH"
    instr2.quote = "USDT"
    ctx.instruments = [instr1, instr2]

    # Mock account (multi-exchange aware)
    account = MagicMock()
    pos1 = MagicMock()
    pos1.quantity = 0.5
    pos1.position_avg_price = 67500.0
    pos1.last_update_price = 68000.0
    pos1.unrealized_pnl.return_value = 250.0
    pos1.r_pnl = 100.0
    pos1.market_value_funds = 34000.0
    pos1.is_open.return_value = True
    pos2 = MagicMock()
    pos2.quantity = 0.0
    pos2.position_avg_price = 0.0
    pos2.last_update_price = 3800.0
    pos2.unrealized_pnl.return_value = 0.0
    pos2.r_pnl = 0.0
    pos2.market_value_funds = 0.0
    pos2.is_open.return_value = False

    ctx.get_positions.return_value = {instr1: pos1, instr2: pos2}
    account.get_positions.return_value = {instr1: pos1, instr2: pos2}
    account.get_orders.return_value = {}
    account.get_leverage.return_value = 0.34
    account.get_total_capital.return_value = 10000.0
    account.get_capital.return_value = 9500.0
    account.get_net_leverage.return_value = 0.15
    account.get_gross_leverage.return_value = 0.15
    account.get_base_currency.return_value = "USDT"

    # Mock balances
    bal = MagicMock()
    bal.exchange = "BINANCE.UM"
    bal.currency = "USDT"
    bal.total = 10000.0
    bal.free = 9500.0
    bal.locked = 500.0
    ctx.get_balances.return_value = [bal]
    account.get_balances.return_value = [bal]

    ctx.account = account

    # Mock orders
    ctx.get_orders.return_value = {}

    # Mock state
    ctx.get_total_capital.return_value = 10000.0
    ctx.get_net_leverage.return_value = 0.15
    ctx.get_gross_leverage.return_value = 0.15
    ctx.is_warmup_in_progress = False
    ctx.is_simulation = False
    ctx.exchanges = ["BINANCE.UM"]
    ctx.time.return_value = "2026-04-05T10:00:00"

    # Mock strategy (no @state decorators)
    ctx.strategy = type("FakeStrategy", (), {})()

    # Mock health
    ctx.health = MagicMock()
    ctx.health.is_connected.return_value = True
    ctx.health.get_queue_size.return_value = 0
    ctx.health.get_data_latencies.return_value = {}

    # Mock query_instrument
    ctx.query_instrument.side_effect = lambda s, exchange=None: instr1 if "BTC" in s else (instr2 if "ETH" in s else None)

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
        assert pos["unrealized_pnl"] == 250.0


class TestGetBalances:
    def test_returns_balances(self):
        ctx = _make_mock_ctx()
        _, handler = BUILTIN_ACTIONS["get_balances"]
        result = handler(ctx)
        assert result.status == "ok"
        assert result.data["balances"]["BINANCE.UM:USDT"]["total"] == 10000.0
        assert result.data["balances"]["BINANCE.UM:USDT"]["free"] == 9500.0


class TestGetState:
    def test_returns_multi_exchange_state(self):
        ctx = _make_mock_ctx()
        _, handler = BUILTIN_ACTIONS["get_state"]
        result = handler(ctx)
        assert result.status == "ok"
        data = result.data

        # Top-level fields
        assert data["total_capital"] == 10000.0
        assert "timestamp" in data
        assert "instruments" in data

        # Per-exchange structure
        assert "BINANCE.UM" in data["exchanges"]
        exch = data["exchanges"]["BINANCE.UM"]
        assert exch["base_currency"] == "USDT"
        assert "capital" in exch
        assert exch["capital"]["total"] == 10000.0
        assert exch["capital"]["available"] == 9500.0
        assert "net_leverage" in exch
        assert "gross_leverage" in exch
        assert "open_positions" in exch
        assert "positions" in exch
        assert "orders" in exch
        assert "balances" in exch

    def test_includes_custom_state(self):
        ctx = _make_mock_ctx()

        # Add a @state-decorated method to the strategy
        from qubx.control import state

        class StrategyWithState:
            @state(description="Current regime")
            def regime(self, ctx):
                return "trending"

        ctx.strategy = StrategyWithState()
        _, handler = BUILTIN_ACTIONS["get_state"]
        result = handler(ctx)
        assert result.data["custom"]["regime"] == "trending"


class TestGetHealth:
    def test_returns_health_info(self):
        ctx = _make_mock_ctx()
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


class TestGetAvailableInstruments:
    def test_returns_instrument_list(self):
        ctx = _make_mock_ctx()
        _, handler = BUILTIN_ACTIONS["get_available_instruments"]
        # This calls lookup.find_instruments which we can't easily mock,
        # but we can at least verify it doesn't crash
        result = handler(ctx, exchange="BINANCE.UM", quote="USDT", market_type="SWAP")
        assert result.status == "ok"
        assert "count" in result.data
        assert "instruments" in result.data
        assert result.data["exchange"] == "BINANCE.UM"
        assert result.data["market_type"] == "SWAP"


class TestGetInstrumentDetails:
    def test_returns_details(self):
        ctx = _make_mock_ctx()
        _, handler = BUILTIN_ACTIONS["get_instrument_details"]
        result = handler(ctx, symbols=["BTCUSDT"])
        assert result.status == "ok"
        assert len(result.data["instruments"]) == 1
        details = list(result.data["instruments"].values())[0]
        assert "tick_size" in details
        assert "lot_size" in details
        assert "min_notional" in details

    def test_returns_not_found(self):
        ctx = _make_mock_ctx()
        _, handler = BUILTIN_ACTIONS["get_instrument_details"]
        result = handler(ctx, symbols=["XYZUSDT"])
        assert result.status == "ok"
        assert "not_found" in result.data
        assert "XYZUSDT" in result.data["not_found"]


class TestBuiltinRegistry:
    def test_all_actions_have_definitions(self):
        for name, (action_def, handler) in BUILTIN_ACTIONS.items():
            assert action_def.name == name
            assert handler is not None
            assert action_def.description

    def test_read_only_actions_are_marked(self):
        read_only = {
            "get_universe", "get_positions", "get_balances", "get_orders", "get_quote",
            "get_ohlc", "get_state", "get_health", "get_state_schema", "get_total_capital",
            "get_leverages", "get_subscriptions", "get_available_instruments",
            "get_instrument_details", "get_top_instruments",
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

    def test_expected_action_count(self):
        assert len(BUILTIN_ACTIONS) == 25
