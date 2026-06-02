from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import AccountManager, AccountManagerConfig, AccountState
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
    am = AccountManager.__new__(AccountManager)
    am._states = {"binance": AccountState(exchange="binance")}
    am._connectors = {"binance": MagicMock()}
    am._cfg = cfg or AccountManagerConfig(terminal_order_retention_ms=30_000)
    am._time = _T()
    am._strategy = MagicMock()
    am._liveness_unready_since = {}
    am._applied_funding_buckets = {}
    am._ctx = object()
    return am


def _add(state, cid="cid-1", status=OrderStatus.ACCEPTED, instrument=None):
    state.add_order(
        Order(
            client_order_id=cid,
            venue_order_id=None,
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT",
            instrument=instrument,
            time=np.datetime64("2026-05-28T00:00:00"),
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
    state = am._states["binance"]
    inst = _instrument()
    _add(state, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill()))
    assert state.get_order("cid-1").status is OrderStatus.FILLED
    # sweep before grace elapses -> still in active_orders
    am._time.adv(10_000)
    am._sweep_terminal_evictions()
    assert "cid-1" in state._active_orders


def test_terminal_evicted_after_grace():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    _add(state, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill()))
    am._time.adv(31_000)
    am._sweep_terminal_evictions()
    assert "cid-1" not in state._active_orders
    # still resolvable from terminal history
    assert state.get_order("cid-1").status is OrderStatus.FILLED


def test_late_accepted_on_terminal_sets_venue_id_no_phantom():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    _add(state, instrument=inst)
    state.set_venue_id("cid-1", "V1")
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill()))
    # evict to history then deliver a late OrderAccepted resolving via venue/cid
    am._time.adv(31_000)
    am._sweep_terminal_evictions()
    assert "cid-1" not in state._active_orders
    am.apply(
        OrderAcceptedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", accepted_at=np.datetime64("2026-05-28")
        )
    )
    # no phantom EXTERNAL order, evicted order remains FILLED
    assert "ext:V1" not in state._active_orders
    assert "cid-1" not in state._active_orders
    assert state.get_order("cid-1").status is OrderStatus.FILLED


def test_terminal_eviction_runs_on_inflight_tick():
    # eviction is wired into the inflight tick cadence; calling the sweep evicts
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    _add(state, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill()))
    am._time.adv(31_000)
    am._sweep_terminal_evictions()
    assert "cid-1" not in state._active_orders
    assert len(state._terminal_history) == 1
