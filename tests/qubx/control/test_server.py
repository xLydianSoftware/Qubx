from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from qubx.control.server import ControlServer


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


@pytest.fixture
def server_with_ctx():
    """Create a ControlServer with a mock context attached, return the TestClient."""
    server = ControlServer(port=0, ready_check=lambda: True)
    ctx = _make_mock_ctx()
    server.attach_context(ctx)
    client = TestClient(server._app)
    return client, ctx


@pytest.fixture
def server_no_ctx():
    """Create a ControlServer without a context attached."""
    server = ControlServer(port=0, ready_check=lambda: False)
    return TestClient(server._app)


class TestHealthEndpoints:
    def test_health_returns_ok(self, server_with_ctx):
        client, _ = server_with_ctx
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_returns_ok_when_ready(self, server_with_ctx):
        client, _ = server_with_ctx
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_ready_returns_503_when_not_ready(self, server_no_ctx):
        client = server_no_ctx
        resp = client.get("/ready")
        assert resp.status_code == 503


class TestActionsEndpoint:
    def test_list_actions(self, server_with_ctx):
        client, _ = server_with_ctx
        resp = client.get("/actions")
        assert resp.status_code == 200
        data = resp.json()
        assert "actions" in data
        assert "bot_id" in data
        names = {a["name"] for a in data["actions"]}
        assert "get_positions" in names
        assert "get_universe" in names
        assert "trade" in names

    def test_actions_have_required_fields(self, server_with_ctx):
        client, _ = server_with_ctx
        resp = client.get("/actions")
        for action in resp.json()["actions"]:
            assert "name" in action
            assert "description" in action
            assert "category" in action
            assert "read_only" in action
            assert "dangerous" in action
            assert "params" in action

    def test_actions_503_without_context(self, server_no_ctx):
        client = server_no_ctx
        resp = client.get("/actions")
        assert resp.status_code == 503


class TestExecEndpoint:
    def test_exec_read_only_action(self, server_with_ctx):
        client, _ = server_with_ctx
        resp = client.post("/actions/get_positions", json={"params": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "positions" in data["data"]

    def test_exec_get_universe(self, server_with_ctx):
        client, _ = server_with_ctx
        resp = client.post("/actions/get_universe", json={"params": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["count"] == 1

    def test_exec_get_state(self, server_with_ctx):
        client, _ = server_with_ctx
        resp = client.post("/actions/get_state", json={"params": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["total_capital"] == 10000.0

    def test_exec_unknown_action(self, server_with_ctx):
        client, _ = server_with_ctx
        resp = client.post("/actions/nonexistent", json={"params": {}})
        assert resp.status_code == 400

    def test_exec_503_without_context(self, server_no_ctx):
        client = server_no_ctx
        resp = client.post("/actions/get_positions", json={"params": {}})
        assert resp.status_code == 503

    def test_exec_with_empty_params(self, server_with_ctx):
        client, _ = server_with_ctx
        resp = client.post("/actions/get_health", json={})
        assert resp.status_code == 200

    def test_action_schema_for_llm_conversion(self, server_with_ctx):
        """Verify action schema contains enough info for LLM tool conversion."""
        client, _ = server_with_ctx
        resp = client.get("/actions")
        actions = resp.json()["actions"]

        trade_action = next(a for a in actions if a["name"] == "trade")
        assert len(trade_action["params"]) >= 2
        param_names = {p["name"] for p in trade_action["params"]}
        assert "symbol" in param_names
        assert "amount" in param_names

        for param in trade_action["params"]:
            assert "name" in param
            assert "type" in param
            assert "description" in param
            assert "required" in param
