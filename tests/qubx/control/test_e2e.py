"""End-to-end test for the control server with a real StrategyContext.

Uses a minimal strategy and mocked exchange dependencies to test
the full HTTP → executor → context pipeline.
"""

import socket
import time
from threading import Thread
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

from qubx.control import IControllable, action
from qubx.control.server import ControlServer
from qubx.control.types import ActionResult
from qubx.core.basics import CtrlChannel, DataType, Instrument, ITimeProvider
from qubx.core.context import StrategyContext
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import IStrategy, IStrategyContext, IStrategyInitializer, StrategyState
from qubx.core.lookups import lookup

# --- Test strategy with custom actions ---


class TestStrategy(IStrategy, IControllable):
    threshold: float = 0.7
    paused: bool = False

    def on_init(self, initializer: IStrategyInitializer) -> None:
        initializer.set_base_subscription(DataType.QUOTE)

    @action(description="Get strategy parameters", category="diagnostics", read_only=True)
    def get_params(self, ctx: IStrategyContext):
        return {"threshold": self.threshold, "paused": self.paused}

    @action(description="Update threshold", category="config")
    def set_threshold(self, ctx: IStrategyContext, value: float):
        if not 0.0 <= value <= 1.0:
            return ActionResult(status="error", error="Must be between 0 and 1")
        old = self.threshold
        self.threshold = value
        return ActionResult(status="ok", data={"old": old, "new": value})

    @action(description="Pause trading", category="config")
    def pause(self, ctx: IStrategyContext):
        self.paused = True

    @action(description="Resume trading", category="config")
    def resume(self, ctx: IStrategyContext):
        self.paused = False


# --- Helpers ---


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_mock_data_provider(instruments: list[Instrument], is_simulation: bool = False):
    dp = MagicMock()
    dp.is_simulation = is_simulation
    dp.channel = CtrlChannel("test")
    dp.instruments = instruments
    return dp


def _make_mock_broker():
    broker = MagicMock()
    broker.exchange.return_value = "BINANCE.UM"
    return broker


def _make_mock_account(instruments: list[Instrument]):
    account = MagicMock()
    pos = MagicMock()
    pos.quantity = 0.0
    pos.position_avg_price = 0.0
    pos.last_update_price = 0.0
    pos.unrealized_pnl.return_value = 0.0
    pos.r_pnl = 0.0
    pos.market_value_funds = 0.0
    pos.is_open.return_value = False
    account.get_positions.return_value = {i: pos for i in instruments}
    bal = MagicMock()
    bal.currency = "USDT"
    bal.total = 10000.0
    bal.free = 10000.0
    bal.locked = 0.0
    account.get_balances.return_value = [bal]
    account.get_orders.return_value = {}
    account.get_total_capital.return_value = 10000.0
    account.get_capital.return_value = 10000.0
    account.get_net_leverage.return_value = 0.0
    account.get_gross_leverage.return_value = 0.0
    account.get_base_currency.return_value = "USDT"
    account.get_leverage.return_value = 0.0
    return account


def _make_time_provider():
    tp = MagicMock(spec=ITimeProvider)
    tp.time.return_value = np.datetime64("2026-01-01T00:00:00", "ns")
    return tp


def _make_logging():
    return MagicMock()


@pytest.fixture
def e2e_server():
    """Build a real StrategyContext + ControlServer, yield the running server URL, then tear down."""
    instruments = [lookup.find_symbol("BINANCE.UM", "BTCUSDT"), lookup.find_symbol("BINANCE.UM", "ETHUSDT")]

    dp = _make_mock_data_provider(instruments, is_simulation=False)
    broker = _make_mock_broker()
    account = _make_mock_account(instruments)
    tp = _make_time_provider()
    logging = _make_logging()
    scheduler = MagicMock()
    aux_storage = MagicMock()

    ctx = StrategyContext(
        strategy=TestStrategy,
        brokers=[broker],
        data_providers=[dp],
        account=account,
        scheduler=scheduler,
        time_provider=tp,
        instruments=instruments,
        logging=logging,
        aux_data_storage=aux_storage,
        strategy_name="test-e2e",
        strategy_state=StrategyState(is_on_warmup_finished_called=True),
        initializer=BasicStrategyInitializer(simulation=False),
    )

    port = _find_free_port()

    def ready_check() -> bool:
        return ctx._strategy_state.is_on_warmup_finished_called

    server = ControlServer(port, ready_check=ready_check)
    server.attach_context(ctx)

    # Start the data processing loop in a thread (so commands get drained)
    ctx._thread_data_loop = Thread(
        target=ctx._StrategyContext__process_incoming_data_loop,
        args=(dp.channel,),
        daemon=True,
        name="TestProcessorThread",
    )
    ctx._thread_data_loop.start()

    server.start()
    # Wait for server to be ready
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            r = requests.get(f"{base_url}/health", timeout=0.5)
            if r.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(0.1)

    yield base_url, ctx

    # Teardown
    dp.channel.stop()
    server.stop()


@pytest.mark.e2e
class TestControlServerE2E:
    def test_health(self, e2e_server):
        url, _ = e2e_server
        resp = requests.get(f"{url}/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready(self, e2e_server):
        url, _ = e2e_server
        resp = requests.get(f"{url}/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_list_actions_includes_builtin_and_custom(self, e2e_server):
        url, _ = e2e_server
        resp = requests.get(f"{url}/actions")
        assert resp.status_code == 200
        data = resp.json()
        names = {a["name"] for a in data["actions"]}

        # Built-in
        assert "get_positions" in names
        assert "get_universe" in names
        assert "trade" in names
        assert "get_state" in names

        # Custom from TestStrategy
        assert "get_params" in names
        assert "set_threshold" in names
        assert "pause" in names
        assert "resume" in names

    def test_exec_get_universe(self, e2e_server):
        url, _ = e2e_server
        resp = requests.post(f"{url}/actions/get_universe", json={"params": {}})
        assert resp.status_code == 200
        data = resp.json()
        # Universe is empty because ctx.start() wasn't called (no set_universe),
        # but the endpoint works correctly
        assert "instruments" in data["data"]
        assert "count" in data["data"]

    def test_exec_get_state(self, e2e_server):
        url, _ = e2e_server
        resp = requests.post(f"{url}/actions/get_state", json={"params": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["total_capital"] == 10000.0

    def test_exec_custom_read_only_action(self, e2e_server):
        url, ctx = e2e_server
        resp = requests.post(f"{url}/actions/get_params", json={"params": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["threshold"] == 0.7
        assert data["data"]["paused"] is False

    def test_exec_custom_write_action_via_command_queue(self, e2e_server):
        """Write actions go through the command queue → data loop thread."""
        url, ctx = e2e_server
        resp = requests.post(f"{url}/actions/set_threshold", json={"params": {"value": 0.9}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["old"] == 0.7
        assert data["data"]["new"] == 0.9

        # Verify state actually changed
        assert ctx.strategy.threshold == 0.9

    def test_exec_custom_write_action_error(self, e2e_server):
        url, _ = e2e_server
        resp = requests.post(f"{url}/actions/set_threshold", json={"params": {"value": 5.0}})
        assert resp.status_code == 400
        assert "between 0 and 1" in resp.json()["detail"]

    def test_exec_pause_resume(self, e2e_server):
        url, ctx = e2e_server

        resp = requests.post(f"{url}/actions/pause", json={"params": {}})
        assert resp.status_code == 200
        assert ctx.strategy.paused is True

        resp = requests.post(f"{url}/actions/resume", json={"params": {}})
        assert resp.status_code == 200
        assert ctx.strategy.paused is False

    def test_exec_unknown_action(self, e2e_server):
        url, _ = e2e_server
        resp = requests.post(f"{url}/actions/nonexistent", json={"params": {}})
        assert resp.status_code == 400

    def test_action_schema_valid_for_llm_tools(self, e2e_server):
        """The /actions response should be convertible to LLM tool definitions."""
        url, _ = e2e_server
        resp = requests.get(f"{url}/actions")
        actions = resp.json()["actions"]

        for a in actions:
            assert isinstance(a["name"], str)
            assert isinstance(a["description"], str)
            assert isinstance(a["params"], list)
            for p in a["params"]:
                assert "name" in p
                assert "type" in p
                assert p["type"] in ("string", "number", "integer", "boolean", "array", "object", "enum")
