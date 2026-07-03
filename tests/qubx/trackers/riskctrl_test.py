"""Unit tests for BrokerSideRiskController.on_execution_report boundary behaviour.

Focus: the post-open 'is the position closed?' check must not misread a position whose
size is exactly the instrument min_size as flat (which tore down the just-sent stop+take).
"""

from unittest.mock import Mock

from qubx.core.basics import Instrument, MarketType, OrderStatus
from qubx.trackers.riskctrl import BrokerSideRiskController, RiskCalculator, SgnCtrl, State


def _instrument(min_size: float = 0.001) -> Instrument:
    return Instrument(
        symbol="BTCUSDT",
        market_type=MarketType.SWAP,
        exchange="BINANCE.UM",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.1,
        lot_size=0.001,
        min_size=min_size,
        min_notional=0.0,
    )


def _make_tracker() -> BrokerSideRiskController:
    return BrokerSideRiskController(name="t", risk_calculator=RiskCalculator(), sizer=Mock())


def _ctx(instrument: Instrument, pos_qty: float) -> Mock:
    ctx = Mock()
    ctx.positions = {instrument: Mock(quantity=pos_qty)}
    ctx.cancelled = []
    ctx.cancel_order = Mock(side_effect=lambda client_order_id=None, **kw: ctx.cancelled.append(client_order_id))
    orders: dict[str, Mock] = {}
    counter = {"n": 0}

    def _trade(_instr, _amount, _price, **_kwargs):
        counter["n"] += 1
        cid = f"prot_{counter['n']}"
        order = Mock(client_order_id=cid, status=OrderStatus.SUBMITTED, venue_order_id=f"V{counter['n']}")
        orders[cid] = order
        return order

    ctx.trade = Mock(side_effect=_trade)
    ctx.find_order_by_client_id = Mock(side_effect=lambda cid: orders.get(cid))
    return ctx


def test_min_size_entry_keeps_both_protective_legs() -> None:
    # A position opened at EXACTLY min_size must not read as flat. The post-open close-check
    # used abs(pos) <= min_size, which fired the 'closed externally' branch for a min-size
    # position and cancelled the stop+take it had just sent. Both legs must survive.
    instr = _instrument(min_size=0.001)
    tracker = _make_tracker()
    target = Mock(target_position_size=0.001, take=59528.7, stop=58588.9)
    tracker._waiting[instr] = SgnCtrl(signal=Mock(instrument=instr, signal=1.0), target=target, status=State.NEW)

    ctx = _ctx(instr, pos_qty=0.001)
    entry_deal = Mock(order_id="ENTRY_V", price=59064.4)

    tracker.on_execution_report(ctx, instr, entry_deal)

    assert tracker.is_active(instr)  # - still tracking after the opening fill
    ctrl = tracker._trackings[instr]
    assert ctrl.take_order_id is not None  # - take leg live
    assert ctrl.stop_order_id is not None  # - stop leg live
    assert ctx.cancelled == []  # - nothing torn down


def test_protective_orders_are_reduce_only() -> None:
    # Both protective legs (take + stop) must be reduce-only: a stale/late trigger (e.g. one
    # the tracker failed to cancel) then can never FLIP the position into a reverse one — the
    # venue rejects a reduce-only that would increase. Without it, an orphaned stop firing on a
    # flat position opens a fresh short/long.
    instr = _instrument(min_size=0.001)
    tracker = _make_tracker()
    target = Mock(target_position_size=0.001, take=59528.7, stop=58588.9)
    tracker._waiting[instr] = SgnCtrl(signal=Mock(instrument=instr, signal=1.0), target=target, status=State.NEW)

    ctx = _ctx(instr, pos_qty=0.001)
    tracker.on_execution_report(ctx, instr, Mock(order_id="ENTRY_V", price=59064.4))

    assert ctx.trade.call_count == 2  # - take + stop
    for call in ctx.trade.call_args_list:
        assert call.kwargs.get("reduce_only") is True, f"protective order not reduce_only: {call}"


def test_take_fill_closing_position_still_triggers_risk() -> None:
    # Guard: the fix (< instead of <=) must NOT break genuine close detection. A take fill
    # that flattens the position (pos -> 0) must still flip to RISK_TRIGGERED and cancel the stop.
    instr = _instrument(min_size=0.001)
    tracker = _make_tracker()
    tracker._trackings[instr] = SgnCtrl(
        signal=Mock(instrument=instr, signal=1.0),
        target=Mock(target_position_size=0.001, take=59528.7, stop=58588.9),
        status=State.OPEN,
        take_order_id="take1",
        stop_order_id="stop1",
    )

    ctx = _ctx(instr, pos_qty=0.0)
    ctx.find_order_by_client_id = Mock(
        side_effect=lambda cid: {"take1": Mock(venue_order_id="TAKE_V"), "stop1": Mock(venue_order_id="STOP_V")}.get(cid)
    )
    take_deal = Mock(order_id="TAKE_V", price=59528.7)

    tracker.on_execution_report(ctx, instr, take_deal)

    ctrl = tracker._trackings[instr]
    assert ctrl.status == State.RISK_TRIGGERED  # - close detected
    assert ctx.cancelled == ["stop1"]  # - the opposite (stop) leg cancelled
