from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import AccountManager, AccountManagerConfig, SimulatedAccountManager
from qubx.core.basics import Deal, Instrument, MarketType, Order, OrderOrigin, OrderStatus
from qubx.core.events import OrderAcceptedEvent, OrderFilledEvent


class _T:
    def __init__(self, t="2026-05-28T00:00:00"):
        self.t = np.datetime64(t)

    def time(self):
        return self.t

    def adv(self, ms):
        self.t = self.t + np.timedelta64(ms, "ms")


def _instrument(symbol="BTCUSDT", exchange="binance") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange=exchange,
        base=symbol.replace("USDT", ""),
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def _am(cfg=None):
    return AccountManager(
        connectors={"binance": MagicMock()},
        base_currencies={"binance": "USDT"},
        time=_T(),
        cfg=cfg or AccountManagerConfig(terminal_order_retention_ms=30_000),
        account_id="test",
    )


def _add(state, cid="cid-1", status=OrderStatus.ACCEPTED, instrument=None):
    state.add_order(
        Order(
            client_order_id=cid,
            venue_order_id=None,
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT",
            instrument=instrument,
            submitted_at=np.datetime64("2026-05-28T00:00:00"),
            quantity=1.0,
            price=50_000.0,
            side="BUY",
            status=status,
            time_in_force="gtc",
        )
    )


def _fill(trade_id="t1", amount=1.0, price=50_000.0):
    return Deal(
        trade_id=trade_id,
        order_id="V1",
        time=np.datetime64("2026-05-28T00:00:00"),
        amount=amount,
        price=price,
        aggressive=True,
    )


def test_terminal_in_active_during_grace():
    am = _am()
    state = am.get_state("binance")
    inst = _instrument()
    _add(state, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill()))
    assert state.get_order("cid-1").status is OrderStatus.FILLED
    # sweep before grace elapses -> still in active_orders
    am._time.adv(10_000)
    am._sweep_terminal_evictions()
    assert state.has_active_order("cid-1")


def test_terminal_evicted_after_grace():
    am = _am()
    state = am.get_state("binance")
    inst = _instrument()
    _add(state, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill()))
    am._time.adv(31_000)
    am._sweep_terminal_evictions()
    assert not state.has_active_order("cid-1")
    # still resolvable from terminal history
    assert state.get_order("cid-1").status is OrderStatus.FILLED


def test_terminal_eviction_runs_on_reconcile_tick():
    # live cadence: the reconcile heartbeat itself must run the eviction sweep
    am = _am()
    state = am.get_state("binance")
    inst = _instrument()
    _add(state, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill()))
    am._time.adv(31_000)
    am._on_reconcile_tick(None)
    assert not state.has_active_order("cid-1")
    assert state.get_order("cid-1") is not None  # retained in terminal history


def test_terminal_eviction_runs_on_apply_path_in_simulation():
    # F6 regression: SimulatedAccountManager (paper + backtest) registers no periodic
    # ticks — terminal orders must still be evicted via the opportunistic sweep on the
    # apply path once the sim clock passes the retention window.
    time = _T()
    am = SimulatedAccountManager(
        connectors={"binance": MagicMock()},
        base_currencies={"binance": "USDT"},
        time=time,
        cfg=AccountManagerConfig(terminal_order_retention_ms=30_000),
    )
    state = am.get_state("binance")
    inst = _instrument()
    for i in range(5):
        _add(state, cid=f"cid-{i}", instrument=inst)
        am.apply(
            OrderFilledEvent(
                instrument=inst, client_order_id=f"cid-{i}", venue_order_id=f"V{i}", fill=_fill(trade_id=f"t{i}")
            )
        )
    assert all(state.has_active_order(f"cid-{i}") for i in range(5))

    time.adv(31_000)
    # any applied account event triggers the sweep once retention has elapsed
    _add(state, cid="cid-new", status=OrderStatus.SUBMITTED, instrument=inst)
    am.apply(OrderAcceptedEvent(instrument=inst, client_order_id="cid-new", venue_order_id="VN", accepted_at=time.t))

    for i in range(5):
        assert not state.has_active_order(f"cid-{i}")
        assert state.get_order(f"cid-{i}").status is OrderStatus.FILLED  # resolvable from terminal history
    assert state.has_active_order("cid-new")  # the live order is untouched
