import os
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from qubx.core.basics import (
    Instrument,
    MarketType,
    Order,
    OrderOrigin,
    OrderRequest,
    OrderStatus,
)
from qubx.core.context import StrategyContext
from qubx.core.exceptions import OrderGateRejected
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import IStrategyContext
from qubx.core.mixins.trading import TradingManager
from qubx.core.order_gate import IOrderGate, OrderGateDecision, ProteusOrderGate, SoftwareOrderGate
from qubx.health.dummy import DummyHealthMonitor


def _make_instrument(symbol: str = "BTCUSDT", exchange: str = "BINANCE.UM") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange=exchange,
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.1,
        lot_size=0.001,
        min_size=0.001,
    )


def _make_request(
    instrument: Instrument,
    qty: float = 1.0,
    price: float | None = 95000.0,
    reduce_only: bool = False,
) -> OrderRequest:
    return OrderRequest(
        client_id="qubx_BTCUSDT_1",
        instrument=instrument,
        quantity=qty,
        price=price,
        order_type="LIMIT" if price is not None else "MARKET",
        side="BUY",
        time_in_force="gtc",
        options={"reduceOnly": True} if reduce_only else {},
    )


def _make_order(instrument: Instrument, cid: str = "qubx_BTCUSDT_1", qty: float = 1.0) -> Order:
    return Order(
        client_order_id=cid,
        venue_order_id="venue-1",
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=instrument,
        submitted_at=pd.Timestamp("2026-01-01").asm8,
        quantity=qty,
        price=95000.0,
        side="BUY",
        status=OrderStatus.ACCEPTED,
        time_in_force="gtc",
    )


class _FakeAccountManager:
    def __init__(self) -> None:
        self.orders: dict[str, Order] = {}

    def get_position(self, instrument: Instrument):
        return None

    def add_order(self, order: Order) -> None:
        self.orders[order.client_order_id] = order

    def remove_order(self, exchange: str, cid: str) -> None:
        self.orders.pop(cid, None)

    def find_order_by_id(self, order_id: str) -> Order | None:
        return None

    def find_order_by_client_id(self, cid: str) -> Order | None:
        return self.orders.get(cid)

    def get_orders(self, instrument: Instrument | None = None) -> dict[str, Order]:
        return dict(self.orders)

    def transition_order(self, exchange: str, cid: str, status: OrderStatus) -> None:
        if cid in self.orders:
            self.orders[cid].status = status


class _FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _MockContext(IStrategyContext):
    def __init__(self) -> None:
        self._timestamp = pd.Timestamp("2026-01-01").asm8

    def time(self):
        return self._timestamp

    def is_blacklisted(self, instrument):
        return False


def _make_trading_manager(gate: IOrderGate | None = None):
    instr = _make_instrument()
    ctx = _MockContext()
    connector = Mock()
    connector.exchange_name = instr.exchange
    connector.make_client_id.side_effect = lambda cid: cid
    account_mgr = _FakeAccountManager()
    tm = TradingManager(
        context=ctx,
        connectors={instr.exchange: connector},
        account_manager=account_mgr,  # type: ignore[arg-type]
        health_monitor=DummyHealthMonitor(),
        strategy_name="test_gate",
        order_gate=gate,
    )
    return tm, connector, account_mgr, instr


def test_software_order_gate_rate_limit_cancel_bypass_and_refill():
    clock = _FakeClock(100.0)
    gate = SoftwareOrderGate(max_orders=2, window_seconds=1.0, clock=clock)
    assert isinstance(gate, IOrderGate)
    instr = _make_instrument()

    assert gate.check_order(_make_request(instr, qty=1.0)).allowed
    assert gate.check_order(_make_request(instr, qty=1.0)).allowed

    throttled = gate.check_order(_make_request(instr, qty=1.0))
    assert not throttled.allowed
    assert throttled.code == "RATE_LIMIT"

    # Cancels and reduce-only closes remain allowed while rate-limited
    assert gate.check_cancel(_make_order(instr)).allowed
    assert gate.check_order(_make_request(instr, qty=1.0, reduce_only=True)).allowed

    # Window refills cleanly
    clock.advance(1.01)
    assert gate.check_order(_make_request(instr, qty=1.0)).allowed


def test_software_order_gate_fat_finger_and_kill_switch():
    gate = SoftwareOrderGate(max_order_qty=10.0, max_order_notional=100_000.0, block_cancels_on_kill=False)
    instr = _make_instrument()

    bad_qty = gate.check_order(_make_request(instr, qty=25.0, price=100.0))
    assert not bad_qty.allowed
    assert bad_qty.code == "FAT_FINGER"

    bad_notional = gate.check_order(_make_request(instr, qty=2.0, price=95_000.0))
    assert not bad_notional.allowed
    assert bad_notional.code == "FAT_FINGER"

    gate.trip()
    assert not gate.is_armed
    killed = gate.check_order(_make_request(instr, qty=0.5, price=95_000.0))
    assert not killed.allowed
    assert killed.code == "KILL_SWITCH"
    # Default allows cancels even when kill-switch is tripped
    assert gate.check_cancel(_make_order(instr)).allowed

    gate.arm()
    assert gate.check_order(_make_request(instr, qty=0.5, price=95_000.0)).allowed


def test_trading_manager_enforces_gate_without_phantom_orders():
    clock = _FakeClock(0.0)
    gate = SoftwareOrderGate(max_orders=1, window_seconds=10.0, max_order_qty=5.0, clock=clock)
    tm, connector, account_mgr, instr = _make_trading_manager(gate)

    # First order passes and submits normally
    order = tm.trade(instr, 1.0, price=95_000.0)
    assert order is not None
    assert connector.submit_order.call_count == 1
    assert account_mgr.find_order_by_client_id(order.client_order_id) is not None

    # Second order hits rate limit -> raises OrderGateRejected, never hits connector or account manager
    with pytest.raises(OrderGateRejected) as exc_info:
        tm.trade(instr, 1.0, price=95_000.0)
    assert exc_info.value.code == "RATE_LIMIT"
    assert connector.submit_order.call_count == 1
    assert len(account_mgr.get_orders(instr)) == 1

    # Cancel still succeeds while throttled
    assert tm.cancel_order(client_order_id=order.client_order_id) is True
    assert connector.cancel_order.call_count == 1


def test_trading_manager_kill_switch_blocks_cancel_when_configured():
    gate = SoftwareOrderGate(block_cancels_on_kill=True)
    tm, connector, _, instr = _make_trading_manager(gate)

    order = tm.trade(instr, 1.0, price=95_000.0)
    assert order is not None

    gate.set_kill_switch(False)
    with pytest.raises(OrderGateRejected) as exc_info:
        tm.cancel_order(client_order_id=order.client_order_id)
    assert exc_info.value.code == "KILL_SWITCH"
    assert connector.cancel_order.call_count == 0


def test_initializer_and_context_delegation():
    init = BasicStrategyInitializer()
    gate = SoftwareOrderGate(max_orders=5)
    assert init.get_order_gate() is None
    init.set_order_gate(gate)
    assert init.get_order_gate() is gate

    ctx = StrategyContext.__new__(StrategyContext)
    tm = Mock()
    tm.get_order_gate.return_value = gate
    ctx._trading_manager = tm
    ctx._fit_state = Mock()
    ctx._fit_state.is_fit_thread.return_value = False

    ctx.set_order_gate(gate)
    tm.set_order_gate.assert_called_once_with(gate)
    assert ctx.get_order_gate() is gate


def test_proteus_order_gate_protocol_with_mock_transport():
    calls: list[str] = []

    def fake_transport(line: str) -> dict:
        calls.append(line)
        parts = line.split()
        if parts[0] == "LOAD":
            return {"ok": True, "running": True}
        if parts[0] == "SET_KILL":
            return {"ok": True, "kill_n": int(parts[1])}
        if parts[0] == "SEND_FRAME":
            frame_hex = parts[1]
            frame = bytes.fromhex(frame_hex)
            if frame[0] == 0x4F and frame[3] >= 128:
                return {"ok": True, "tx_hex": "", "audits_hex": "e4", "lat_clks": -1}
            if frame[0] == 0x58:
                return {"ok": True, "tx_hex": frame_hex, "audits_hex": "a1", "lat_clks": 5}
            return {"ok": True, "tx_hex": frame_hex, "audits_hex": "a0", "lat_clks": 9}
        return {"ok": True}

    gate = ProteusOrderGate(image_hex="0102", transport=fake_transport)
    instr = _make_instrument()

    ok = gate.check_order(_make_request(instr, qty=10.0, price=130.0))
    assert ok == OrderGateDecision(True, "PASS_NEW_ORDER", "", pytest.approx(150.0))

    ff = gate.check_order(_make_request(instr, qty=200.0, price=130.0))
    assert not ff.allowed
    assert ff.code == "FAT_FINGER"

    cxl = gate.check_cancel(_make_order(instr, qty=10.0))
    assert cxl.allowed
    assert cxl.code == "PASS_CANCEL"
    assert cxl.latency_ns == pytest.approx(83.3333, rel=1e-3)

    gate.set_kill_switch(False)
    assert "SET_KILL 0" in calls


def test_proteus_order_gate_live_verilator_if_available():
    env_root = os.environ.get("PROTEUS_ROOT")
    proteus_root = Path(env_root) if env_root else Path(__file__).resolve().parents[4] / "proteus"
    if not (proteus_root / "examples/qubx_risk_gate/build/sim_risk_gate").exists():
        alt = Path(__file__).resolve().parents[4] / "protoemu"
        if (alt / "examples/qubx_risk_gate/build/sim_risk_gate").exists():
            proteus_root = alt
    sim_bin = proteus_root / "examples/qubx_risk_gate/build/sim_risk_gate"
    fsm_path = proteus_root / "examples/qubx_risk_gate/risk_gate.fsm"
    if not (sim_bin.exists() and fsm_path.exists()):
        pytest.skip("Proteus Verilator risk-gate simulator not built locally")

    import sys

    sys.path.insert(0, str(proteus_root))
    from bakeoff.cand_table.compiler import compile_file

    img_hex = compile_file(fsm_path).image.hex()
    with ProteusOrderGate(image_hex=img_hex, sim_bin=sim_bin) as gate:
        tm, connector, _, instr = _make_trading_manager(gate)

        # 3 normal orders pass on real 60 MHz Proteus Verilog (0 ns jitter)
        for _ in range(3):
            o = tm.trade(instr, 10.0, price=130.0)
            assert o is not None
        assert connector.submit_order.call_count == 3

        # 4th order is throttled in Proteus silicon!
        with pytest.raises(OrderGateRejected) as exc_info:
            tm.trade(instr, 10.0, price=130.0)
        assert exc_info.value.code == "RATE_LIMIT"
        assert connector.submit_order.call_count == 3

        # Cancel still passes through Proteus silicon while rate-limited!
        assert tm.cancel_order(client_order_id=o.client_order_id) is True
        assert connector.cancel_order.call_count == 1
