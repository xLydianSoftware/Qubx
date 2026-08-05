"""Sim integration test pinning `TradingManager.update_order`'s native-amend semantics
(amount = desired remaining) through the REAL backtester OME: SimulatedConnector + real
AccountManager + real TradingManager, with fills driven by actual OME executions.

The OME fills orders atomically (no partial fills) — there is no depth/size-aware matching
in `qubx.backtester.ome`, so a resting order can never end up partially filled while still
open. That means this test only covers the `filled == 0` reprice (native amend) path; the
`filled > 0` cancel+replace chain is covered at the account-manager integration level and by
a planned real-venue end-to-end test.
"""

import numpy as np
import pytest

from qubx.backtester.connector import SimulatedConnector
from qubx.backtester.utils import SimulatedCtrlChannel
from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.basics import ZERO_COSTS, ITimeProvider
from qubx.core.interfaces import IStrategyContext
from qubx.core.lookups import lookup
from qubx.core.mixins.trading import TradingManager
from qubx.core.series import Quote
from qubx.core.utils import recognize_time
from qubx.health.dummy import DummyHealthMonitor


class _TimeService(ITimeProvider):
    """Shared clock for the OME and the AM/TradingManager stack, advanced by feeding quotes."""

    _time: np.datetime64 = np.datetime64(0, "ns")

    def feed(self, quote: Quote) -> Quote:
        self._time = np.datetime64(quote.time, "ns")
        return quote

    def time(self) -> np.datetime64:
        return self._time


def Q(when: str, bid: float, ask: float) -> Quote:
    return Quote(recognize_time(when), bid, ask, 0, 0)


class _RoutingContext(IStrategyContext):
    """process_event applies straight to the AM — the PM dispatch leg
    (ProcessingManager._dispatch_account) minus strategy callbacks.
    Mirrors tests/qubx/core/mixins/trading_test.py::_AccountRoutingContext."""

    def __init__(self, account_manager: SimulatedAccountManager, time_provider: _TimeService):
        self._am = account_manager
        self._time = time_provider

    def time(self) -> np.datetime64:
        return self._time.time()

    def process_event(self, event) -> None:
        self._am.apply(event)

    def is_blacklisted(self, instrument) -> bool:
        return False


@pytest.fixture
def sim():
    """SimulatedConnector (real OME) + real SimulatedAccountManager + real TradingManager,
    all sharing one clock, wired the same way ProcessingManager wires the live stack minus
    strategy callbacks."""
    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert instr is not None
    time = _TimeService()
    channel = SimulatedCtrlChannel("sim", sentinel=(None, None, None, None))
    conn = SimulatedConnector(channel=channel, exchange_name="BINANCE.UM", time_provider=time, tcc=ZERO_COSTS)

    am = SimulatedAccountManager(
        connectors={"BINANCE.UM": conn}, base_currencies={"BINANCE.UM": "USDT"}, time=time
    )
    ctx = _RoutingContext(am, time)
    channel.register(ctx)  # connector.send() -> channel.send() -> ctx.process_event() -> am.apply()
    tm = TradingManager(
        context=ctx,
        connectors={"BINANCE.UM": conn},
        account_manager=am,
        strategy_name="test_strategy",
        health_monitor=DummyHealthMonitor(),
    )

    # Prime the OME's book so it's ready to accept orders.
    exchange = conn._ome
    list(exchange.process_market_data(instr, time.feed(Q("2020-01-01 10:00", 32000.0, 32001.0))))
    return tm, am, conn, instr, time


def test_reprice_unfilled_limit_keeps_remaining_truthful_through_sim(sim):
    tm, am, conn, instr, time = sim

    # 1. BUY limit well below touch (32000/32001) -> rests unfilled.
    order = tm.trade(instr, amount=0.5, price=31000.0)
    assert order is not None
    live = am.find_order_by_client_id(order.client_order_id)
    assert live is not None
    assert live.filled_quantity == 0.0

    # 2. Reprice AND shrink in one native amend: still below touch, so it keeps resting.
    tm.update_order(price=30800.0, amount=0.3, client_order_id=order.client_order_id)
    order = am.find_order_by_client_id(order.client_order_id)
    assert order is not None
    assert order.price == pytest.approx(30800.0)
    remaining = order.quantity - order.filled_quantity
    assert remaining == pytest.approx(0.3)
    assert remaining >= 0.0

    # 3. Move the market down through the new price -> the REAL OME fills the BUY.
    conn.process_market_data(instr, time.feed(Q("2020-01-01 10:01", 30700.0, 30701.0)))

    order = am.find_order_by_client_id(order.client_order_id)
    assert order is not None
    assert order.filled_quantity == pytest.approx(0.3)
    assert order.quantity - order.filled_quantity == pytest.approx(0.0)

    position = am.get_position(instr)
    assert position is not None
    assert position.quantity == pytest.approx(0.3)
