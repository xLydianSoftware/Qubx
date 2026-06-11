from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
from pytest import approx

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import (
    DataType,
    Deal,
    Instrument,
    Order,
    OrderOrigin,
    OrderStatus,
    Position,
    Signal,
    TargetPosition,
)
from qubx.core.interfaces import (
    IPositionGathering,
    IStrategy,
    IStrategyContext,
    PositionsTracker,
    TriggerEvent,
)
from qubx.core.lookups import lookup
from qubx.core.metrics import portfolio_metrics
from qubx.core.series import Quote
from qubx.core.utils import recognize_time
from qubx.data import CsvStorage
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.ta.indicators import sma
from qubx.trackers.advanced import TimeExpirationTracker
from qubx.trackers.composite import CompositeTracker, CompositeTrackerPerSide, LongTracker
from qubx.trackers.riskctrl import (
    AtrRiskTracker,
    BrokerSideRiskController,
    RiskCalculator,
    StopTakePositionTracker,
    TrailingStopPositionTracker,
)
from qubx.trackers.sizers import FixedLeverageSizer, FixedRiskSizer, FixedSizer
from tests.qubx.core.utils_test import StubAccount

N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)  # noqa: E731


def Q(time: str, bid: float, ask: float) -> Quote:
    return Quote(recognize_time(time), bid, ask, 0, 0)


class TestingPositionGatherer(IPositionGathering):
    def alter_position_size(self, ctx: IStrategyContext, target: TargetPosition) -> float:
        instrument, new_size, at_price = target.instrument, target.target_position_size, target.price
        position = ctx.positions[instrument]
        current_position = position.quantity
        to_trade = new_size - current_position
        if abs(to_trade) < instrument.min_size:
            logger.warning(
                f"Can't change position size for {instrument}. Current position: {current_position}, requested size: {new_size}"
            )
        else:
            # position.quantity = new_size
            q = ctx.quote(instrument)
            assert q is not None
            position.update_position(ctx.time(), new_size, q.mid_price())
            r = ctx.trade(instrument, to_trade, at_price)
            logger.info(
                f"{instrument.symbol} >>> (TESTS) Adjusting position from {current_position} to {new_size} : {r}"
            )
        return current_position

    def on_execution_report(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal): ...


class DebugStratageyCtx(IStrategyContext):
    def __init__(self, instrs, capital) -> None:
        self._instruments = instrs
        self.capital = capital

        positions = {i: Position(i) for i in instrs}
        self.account = StubAccount(base_currency="USDT", exchange="TEST")
        self.account.update_balance("USDT", capital, 0)
        self.account.attach_positions(*positions.values())
        self._n_orders = 0
        self._n_orders_buy = 0
        self._n_orders_sell = 0
        self._orders_size = 0

    @property
    def instruments(self) -> list[Instrument]:
        return self._instruments

    @property
    def positions(self) -> dict[Instrument, Position]:
        return self.account.positions

    def quote(self, symbol: str) -> Quote | None:
        return Q("2020-01-01", 1000.0, 1000.5)

    def get_available_margin(self) -> float:
        return self.capital

    def get_total_capital(self) -> float:
        return self.capital

    def time(self) -> np.datetime64:
        return np.datetime64("2020-01-01T00:00:00", "ns")

    def get_min_size(self, instrument: Instrument, amount: float | None = None) -> float:
        """Return the minimum trade size for an instrument."""
        return instrument.min_size

    def trade(
        self,
        instrument: Instrument,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        **optional,
    ) -> Order:
        # fmt: off
        self._n_orders += 1
        self._orders_size += amount
        if amount > 0:
            self._n_orders_buy += 1
        if amount < 0:
            self._n_orders_sell += 1
        return Order(
            client_order_id="test1", venue_order_id="test", origin=OrderOrigin.FRAMEWORK, type="MARKET", instrument=instrument,
            submitted_at=np.datetime64(0, "ns"), quantity=amount, price=price if price is not None else 0, side="BUY" if amount > 0 else "SELL", status=OrderStatus.SUBMITTED, time_in_force="gtc")
        # fmt: on


class PricedGathererCtx(DebugStratageyCtx):
    """Stub ctx with fire-and-forget order semantics: trade() returns an Order whose
    venue_order_id is still None (populated later, in place, as the AM does on ack)."""

    def __init__(self, instrs, capital) -> None:
        super().__init__(instrs, capital)
        self.submitted: list[Order] = []
        self.cancelled: list[str] = []
        self._orders_by_cid: dict[str, Order] = {}

    def trade(
        self,
        instrument: Instrument,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        **optional,
    ) -> Order:
        order = Order(
            client_order_id=f"cid-{len(self.submitted) + 1}",
            venue_order_id=None,
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT" if price else "MARKET",
            instrument=instrument,
            submitted_at=self.time(),
            quantity=amount,
            price=price if price is not None else 0,
            side="BUY" if amount > 0 else "SELL",
            status=OrderStatus.SUBMITTED,
            time_in_force=time_in_force,
        )
        self.submitted.append(order)
        self._orders_by_cid[order.client_order_id] = order
        return order

    def cancel_order(
        self, order_id: str | None = None, client_order_id: str | None = None, exchange: str | None = None
    ) -> bool:
        assert client_order_id is not None, "must cancel via the reliable client_order_id path"
        self.cancelled.append(client_order_id)
        return True

    def find_order_by_client_id(self, client_id: str) -> Order | None:
        return self._orders_by_cid.get(client_id)


class BrokerRiskCtx(PricedGathererCtx):
    """PricedGathererCtx plus service-signal recording for risk-controller tests."""

    def __init__(self, instrs, capital) -> None:
        super().__init__(instrs, capital)
        self.emitted: list[Signal] = []

    def emit_signal(self, signal: Signal) -> None:
        self.emitted.append(signal)


class ImmediateTakeFillCtx(BrokerRiskCtx):
    """Synchronous-channel semantics: a plain limit order crosses and returns FILLED."""

    def trade(
        self,
        instrument: Instrument,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        **optional,
    ) -> Order:
        order = super().trade(instrument, amount, price, time_in_force, **optional)
        if price is not None and "stop_type" not in optional:
            order.status = OrderStatus.FILLED
            order.venue_order_id = f"V-{order.client_order_id}"
        return order


class ZeroTracker(PositionsTracker):
    def __init__(self) -> None:
        pass

    def process_signals(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        return [s.target_for_amount(0.0) for s in signals]


class GuineaPig(IStrategy):
    """
    Simple signals player
    """

    tests = {}

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(DataType.OHLC["1Min"])
        ctx.set_event_schedule("1Min")

    def on_fit(self, ctx: IStrategyContext):
        self.tests = {recognize_time(k): v for k, v in self.tests.items()}

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | None:
        r = []
        for k in list(self.tests.keys()):
            if event.time >= k:
                s = self.tests.pop(k)
                match s:
                    case Signal():
                        r.append(s)
                    case (Signal(), Callable()):
                        r.append(s[0])
                        cast(Callable, s[1])(ctx, s[0])
                    case _:
                        logger.warning(f" - - - | {s} | - - - ")
        return r


class TestTrackersAndGatherers:
    def test_simple_tracker_sizer(self):
        ctx = DebugStratageyCtx(instrs := [lookup.find_symbol("BINANCE.UM", "BTCUSDT")], 10000)
        tracker = PositionsTracker(FixedSizer(1000.0, amount_in_quote=False))

        gathering = SimplePositionGatherer()
        i = instrs[0]
        assert i is not None

        gathering.alter_positions(
            ctx, tracker.process_signals(ctx, [i.signal(ctx, 1), i.signal(ctx, 0.5), i.signal(ctx, -0.5)])
        )

        assert ctx._n_orders == 3
        assert ctx._orders_size == 1000.0
        assert ctx._n_orders_buy == 2
        assert ctx._n_orders_sell == 1

    def test_simple_gatherer_priced_targets(self):
        ctx = PricedGathererCtx(instrs := [lookup.find_symbol("BINANCE.UM", "BTCUSDT")], 10000)
        i = instrs[0]
        assert i is not None
        gathering = SimplePositionGatherer()

        # - priced (limit) entry: buy below the bid (quote is 1000.0/1000.5)
        gathering.alter_position_size(ctx, TargetPosition(ctx.time(), i, 0.5, entry_price=999.0))
        assert len(ctx.submitted) == 1
        o1 = ctx.submitted[0]
        assert gathering.entry_order_id == o1.client_order_id
        assert not ctx.cancelled

        # - replacing the target must cancel the previous working entry order (the leak pin)
        gathering.alter_position_size(ctx, TargetPosition(ctx.time(), i, 0.7, entry_price=998.0))
        assert ctx.cancelled == [o1.client_order_id]
        o2 = ctx.submitted[1]
        assert gathering.entry_order_id == o2.client_order_id

        # - venue ack arrives after trade() returned: AM populates venue id in place
        o2.venue_order_id = "VENUE-2"

        # - a fill of an unrelated order must not clear the tracked entry
        gathering.on_execution_report(ctx, i, Deal("t0", "VENUE-OTHER", ctx.time(), 0.1, 998.0, False))
        assert gathering.entry_order_id == o2.client_order_id

        # - the entry fill arrives with the VENUE id and must match the stored cid (the fill-match pin)
        gathering.on_execution_report(ctx, i, Deal("t1", "VENUE-2", ctx.time(), 0.7, 998.0, False))
        assert gathering.entry_order_id is None

        # - next target must not try to cancel the already-filled entry
        gathering.alter_position_size(ctx, TargetPosition(ctx.time(), i, 0.9, entry_price=997.0))
        assert ctx.cancelled == [o1.client_order_id]

    def _open_broker_risk_position(self, ctx: BrokerRiskCtx, i: Instrument) -> BrokerSideRiskController:
        ctrl = BrokerSideRiskController("test", RiskCalculator(), FixedSizer(1.0, amount_in_quote=False))
        ctrl.process_signals(ctx, [i.signal(ctx, 0.5, take=1010.0, stop=990.0)])

        # - entry fill opens the position: the controller sends protective take + stop (fire-and-forget)
        ctx.positions[i].quantity = 0.5
        ctrl.on_execution_report(ctx, i, Deal("t-entry", "V-ENTRY", ctx.time(), 0.5, 1000.5, False))
        return ctrl

    def test_broker_risk_take_fill_matches_venue_deal_and_cancels_surviving_stop(self):
        ctx = BrokerRiskCtx(instrs := [lookup.find_symbol("BINANCE.UM", "BTCUSDT")], 10000)
        i = instrs[0]
        assert i is not None
        ctrl = self._open_broker_risk_position(ctx, i)

        assert [o.price for o in ctx.submitted] == [1010.0, 990.0]
        take, stop = ctx.submitted

        # - venue acks arrive after trade() returned (populated in place, as the AM does)
        take.venue_order_id = "V-TAKE"
        stop.venue_order_id = "V-STOP"

        # - the take fills: the deal carries the VENUE id and must match the stored cid;
        #   only the surviving stop is cancelled (misclassifying as closed-externally cancels both)
        ctx.positions[i].quantity = 0.0
        ctrl.on_execution_report(ctx, i, Deal("t-take", "V-TAKE", ctx.time(), -0.5, 1010.0, False))
        assert ctx.cancelled == [stop.client_order_id]

        ctrl.update(ctx, i, Q("2020-01-01", 1000.0, 1000.5))
        assert not ctrl.is_active(i)
        assert [s.comment for s in ctx.emitted] == ["Take triggered"]

    def test_broker_risk_stop_fill_matches_venue_deal_and_cancels_surviving_take(self):
        ctx = BrokerRiskCtx(instrs := [lookup.find_symbol("BINANCE.UM", "BTCUSDT")], 10000)
        i = instrs[0]
        assert i is not None
        ctrl = self._open_broker_risk_position(ctx, i)

        take, stop = ctx.submitted
        take.venue_order_id = "V-TAKE"
        stop.venue_order_id = "V-STOP"

        ctx.positions[i].quantity = 0.0
        ctrl.on_execution_report(ctx, i, Deal("t-stop", "V-STOP", ctx.time(), -0.5, 990.0, False))
        assert ctx.cancelled == [take.client_order_id]

        ctrl.update(ctx, i, Q("2020-01-01", 1000.0, 1000.5))
        assert not ctrl.is_active(i)
        assert [s.comment for s in ctx.emitted] == ["Stop triggered"]

    def test_broker_risk_immediate_take_fill_sends_no_stale_stop(self):
        ctx = ImmediateTakeFillCtx(instrs := [lookup.find_symbol("BINANCE.UM", "BTCUSDT")], 10000)
        i = instrs[0]
        assert i is not None
        self._open_broker_risk_position(ctx, i)

        # - the take limit crossed immediately (synchronous channel returns FILLED):
        #   no stop order may be sent for the already-closed position
        assert [o.price for o in ctx.submitted] == [1010.0]
        assert not ctx.cancelled

    def test_fixed_risk_sizer(self):
        ctx = DebugStratageyCtx(instrs := [lookup.find_symbol("BINANCE.UM", "BTCUSDT")], 10000)
        i = instrs[0]
        assert i is not None

        sizer = FixedRiskSizer(10.0)
        s = sizer.calculate_target_positions(ctx, [i.signal(ctx, 1, stop=900.0)])
        _entry, _stop, _cap_in_risk = 1000.5, 900, 10000 * 10 / 100
        assert s[0].target_position_size == i.round_size_down((_cap_in_risk / ((_entry - _stop) / _entry)) / _entry)

    def test_atr_tracker(self):
        r = CsvStorage("tests/data/storages/csv")
        assert (I := lookup.find_symbol("BINANCE.UM", "BTCUSDT")) is not None

        class StrategyForTracking(IStrategy):
            timeframe: str = "1Min"
            fast_period = 5
            slow_period = 12
            high_low_risk = False

            def on_init(self, ctx: IStrategyContext) -> None:
                ctx.set_base_subscription(DataType.OHLC[self.timeframe])
                ctx.set_event_schedule(self.timeframe)

            def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | None:
                signals = []
                for i in ctx.instruments:
                    ohlc = ctx.ohlc(i, self.timeframe)
                    fast = sma(ohlc.close, self.fast_period)
                    slow = sma(ohlc.close, self.slow_period)
                    pos = ctx.positions[i].quantity

                    if pos <= 0 and (fast[0] > slow[0]) and (fast[1] < slow[1]):
                        if self.high_low_risk:
                            signals.append(i.signal(ctx, +1, stop=min(ohlc[0].low, ohlc[1].low)))
                        else:
                            signals.append(i.signal(ctx, +1))

                    if pos >= 0 and (fast[0] < slow[0]) and (fast[1] > slow[1]):
                        if self.high_low_risk:
                            signals.append(i.signal(ctx, -1, stop=max(ohlc[0].high, ohlc[1].high)))
                        else:
                            signals.append(i.signal(ctx, -1))

                return signals

            def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
                return PositionsTracker(FixedRiskSizer(1, 10_000, reinvest_profit=True))

        # fmt: off
        rep = simulate(
            strategies={
                "Strategy ST client  (0)": [
                    StrategyForTracking(timeframe="5Min", fast_period=10, slow_period=25, high_low_risk=True),
                    StopTakePositionTracker(
                        None, None, sizer=FixedRiskSizer(1, 10_000), risk_controlling_side="client"
                    ),
                ],
                "Strategy ST broker  (1)": [
                    StrategyForTracking(timeframe="5Min", fast_period=10, slow_period=25, high_low_risk=True),
                    StopTakePositionTracker(
                        None, None, sizer=FixedRiskSizer(1, 10_000), risk_controlling_side="broker"
                    ),
                ],
                "Strategy ATR client (2)": [
                    StrategyForTracking(timeframe="5Min", fast_period=10, slow_period=25, high_low_risk=False),
                    t2 := AtrRiskTracker(
                        5, 5, "5Min", 25, atr_smoother="sma", sizer=FixedRiskSizer(1, 10_000), 
                        risk_controlling_side="client",
                    ),
                ],
                "Strategy ATR broker (3)": [
                    StrategyForTracking(timeframe="5Min", fast_period=10, slow_period=25, high_low_risk=False),
                    t3 := AtrRiskTracker(
                        5, 5, "5Min", 25, atr_smoother="sma", sizer=FixedRiskSizer(1, 10_000), 
                        risk_controlling_side="broker",
                    ),
                ],
            },
            data = r,
            capital=10000, instruments=["BINANCE.UM:BTCUSDT"], commissions="vip0_usdt",
            accurate_stop_orders_execution=True,
            start="2024-01-01", stop="2024-01-02 08:00",
        )
        # fmt: on

        # - check first stop: client executed at the price worse than actual stop level
        # - Update: broker and client executed at correct stop level so returns are the same
        # assert rep[2].signals_log.iloc[1].stop == rep[2].executions_log.iloc[2].price
        assert abs(rep[2].signals_log.iloc[1].price - rep[2].executions_log.iloc[1].price) <= I.tick_size

        # -                 : broker executed at correct stop level
        # assert abs(rep[3].signals_log.iloc[1].stop - rep[3].executions_log.iloc[2].price) <= I.tick_size
        assert abs(rep[3].signals_log.iloc[1].price - rep[3].executions_log.iloc[1].price) <= I.tick_size

        # assert t0.is_active(I) and t1.is_active(I)
        assert not t2.is_active(I) and not t3.is_active(I)

    def test_composite_tracker(self):
        ctx = DebugStratageyCtx(
            I := [
                lookup.find_symbol("BINANCE.UM", "BTCUSDT"),
                lookup.find_symbol("BINANCE.UM", "ETHUSDT"),
                lookup.find_symbol("BINANCE.UM", "SOLUSDT"),
            ],
            30000,
        )
        assert I[0] is not None and I[1] is not None and I[2] is not None

        # 1. Check that we get 0 targets for all symbols
        tracker = CompositeTracker(ZeroTracker(), StopTakePositionTracker())
        targets = tracker.process_signals(ctx, [I[0].signal(ctx, +0.5), I[1].signal(ctx, +0.3), I[2].signal(ctx, +0.2)])
        assert all(t.target_position_size == 0 for t in targets)

        # 2. Check that we get nonzero target positions
        tracker = CompositeTracker(StopTakePositionTracker(sizer=FixedSizer(1.0, amount_in_quote=False)))
        targets = tracker.process_signals(ctx, [I[0].signal(ctx, +0.5), I[1].signal(ctx, +0.3), I[2].signal(ctx, +2.0)])
        assert targets[0].target_position_size == 0.5
        assert targets[1].target_position_size == 0.3
        assert (  # SOL has 1 as min_size_step so anything below 1 would be rounded to 0
            targets[2].target_position_size == 2.0
        )

        # 3. Check that allow_override works
        tracker = CompositeTracker(StopTakePositionTracker())
        targets = tracker.process_signals(
            ctx, [I[0].signal(ctx, 0, options=dict(allow_override=True)), I[0].signal(ctx, +0.5)]
        )
        assert targets[0].target_position_size == 0.5

    def test_long_short_trackers(self):
        ctx = DebugStratageyCtx(
            I := [
                lookup.find_symbol("BINANCE.UM", "BTCUSDT"),
            ],
            30000,
        )
        assert I[0] is not None

        # 1. Check that tracker skips the signal if it is not long
        tracker = LongTracker(StopTakePositionTracker(risk_controlling_side="client"))
        targets = tracker.process_signals(ctx, [I[0].signal(ctx, -0.5)])
        assert not targets

        # 2. Check that tracker sends 0 target if it was active before
        tracker = LongTracker(StopTakePositionTracker(risk_controlling_side="client"))
        _ = tracker.process_signals(ctx, [I[0].signal(ctx, +0.5)])

        # - now tracker works only by execution reports, so we 'emulate' it here
        ctx.positions[I[0]].quantity = +0.5
        tracker.on_execution_report(ctx, I[0], Deal(0, "0", np.datetime64(10000, "ns"), +0.5, 1.0, True))

        targets = tracker.process_signals(ctx, [I[0].signal(ctx, -0.5)])
        assert isinstance(targets, list) and targets[0].target_position_size == 0

    def test_composite_per_side_tracker(self):
        ctx = DebugStratageyCtx(
            I := [
                lookup.find_symbol("BINANCE.UM", "BTCUSDT"),
                lookup.find_symbol("BINANCE.UM", "ETHUSDT"),
            ],
            30000,
        )
        assert I[0] is not None and I[1] is not None

        # 1. Check that long and short signals are processed by corresponding trackers
        tracker = CompositeTrackerPerSide(
            long_trackers=[StopTakePositionTracker(10, 5)], short_trackers=[StopTakePositionTracker(5, 5)]
        )
        targets = tracker.process_signals(ctx, [I[0].signal(ctx, -0.5), I[1].signal(ctx, +0.5)])
        short_target = StopTakePositionTracker(5, 5).process_signals(ctx, [I[0].signal(ctx, -0.5)])
        long_target = StopTakePositionTracker(10, 5).process_signals(ctx, [I[1].signal(ctx, +0.5)])
        assert targets[0].stop_price == short_target[0].stop_price
        assert targets[1].stop_price == long_target[0].stop_price

        # 2. Check that sending an opposite side signal is processed correctly
        targets = tracker.process_signals(ctx, [I[0].signal(ctx, +0.5)])
        assert targets[0].target_position_size == 0.5

    def test_tracker_with_stop_loss_in_advance(self):
        I = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert I is not None

        r = CsvStorage("tests/data/storages/csv")

        # fmt: off
        result = simulate(
            {
                "TEST_StopTakePositionTracker (client)": [
                    GuineaPig(tests={"2024-01-01 20:00:00": I.signal("2024-01-01 20:00:00", -1, stop=43800)}),
                    t1 := StopTakePositionTracker(None, None, sizer=FixedRiskSizer(1), risk_controlling_side="client"),
                ],
                "TEST2_AdvancedStopTakePositionTracker (broker)": [
                    GuineaPig(
                        tests={
                            "2024-01-01 20:00:00": I.signal("2024-01-01 20:00:00", -1, stop=43800, take=43400),
                            "2024-01-01 23:10:00": I.signal("2024-01-01 23:10:00", +1, stop=43400, take=44200),
                            "2024-01-02 00:00:00": I.signal("2024-01-02 00:00:00", +1, stop=43500, take=45500),
                            "2024-01-02 01:10:00": I.signal("2024-01-02 01:10:00", -1, stop=45500, take=44800),
                        }
                    ),
                    t2 := StopTakePositionTracker(None, None, sizer=FixedRiskSizer(1), risk_controlling_side="broker"),
                ],
            },
            r, 10000, instruments=["BINANCE.UM:BTCUSDT"], silent=True, debug="DEBUG", commissions="vip0_usdt",
            start="2024-01-01", stop="2024-01-03",
        )
        # fmt: on
        assert len(result[0].executions_log) == 2
        assert not t1.is_active(I)

        assert len(result[1].executions_log) == 7
        assert len(result[1].signals_log) == 7
        assert result[1].signals_log.iloc[-1]["service"]
        assert not t2.is_active(I)

    def test_stop_loss_broker_side(self):
        T = pd.Timestamp
        r = CsvStorage("tests/data/storages/csv")
        assert (i1 := lookup.find_symbol("BINANCE.UM", "BTCUSDT")) is not None

        # fmt: off
        S = pd.DataFrame({ 
            i1: {
                    T("2024-01-10 15:08:59.716000"): 1,
                    T("2024-01-10 15:10:52.679000"): 1,
                    T("2024-01-10 15:32:44.798000"): 1,
                    T("2024-01-10 15:59:55.303000"): 1,
                    T("2024-01-10 16:09:00.970000"): 1,
                    T("2024-01-10 16:12:34.233000"): 1,
                    T("2024-01-10 19:04:00.905000"): 1,
                    # T("2024-01-10 19:16:00.905000"): 1,
                    T("2024-01-10 19:44:37.785000"): 1,
                    T("2024-01-10 20:06:00.322000"): 1,
                }
            }
        )

        rep = simulate(
            strategies={
                "liq_buy_bounces_c": [S, StopTakePositionTracker(2.5, 0.5, FixedLeverageSizer(0.1), "client")],
                "liq_buy_bounces_b": [S, StopTakePositionTracker(2.5, 0.5, FixedLeverageSizer(0.1), "broker")],
            },
            data=r,
            capital=10000,
            start=S.index[0] - pd.Timedelta("5Min"),
            stop=S.index[-1] + pd.Timedelta("5Min"),
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip9_usdt",
            signal_timeframe="1Min",
            debug="DEBUG"
        )
        assert len(rep[0].executions_log) == len(rep[1].executions_log)

        mtrx0 = portfolio_metrics(
            rep[0].portfolio_log, rep[0].executions_log, rep[0].capital, account_transactions=False, commission_factor=1
        )
        mtrx1 = portfolio_metrics(
            rep[1].portfolio_log, rep[1].executions_log, rep[1].capital, account_transactions=False, commission_factor=1
        )
        assert 27.8332 == N(mtrx0["gain"])  # - broker and client executed at correct stop level so returns are the same
        assert 27.8332 == N(mtrx1["gain"])
        # fmt: on

    def test_composite_trackers_broker_side(self):
        class ComplexCompositeTest(GuineaPig):
            def on_init(self, ctx: IStrategyContext) -> None:
                ctx.set_base_subscription(DataType.OHLC["1h"])
                ctx.set_event_schedule("1h")

            def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
                sizer = FixedLeverageSizer(1.0)
                # fmt: off
                return CompositeTrackerPerSide(
                    long_trackers=[
                        AtrRiskTracker(
                            take_target=5, stop_risk=3, atr_timeframe="1h", atr_period=5,
                            sizer=sizer, risk_controlling_side="broker",
                        ),
                        StopTakePositionTracker(stop_risk=10, sizer=sizer),
                    ],
                    short_trackers=[
                        AtrRiskTracker(
                            take_target=5, stop_risk=3, atr_timeframe="1h", atr_period=5,
                            sizer=sizer, risk_controlling_side="broker",
                        ),
                        StopTakePositionTracker(stop_risk=10, sizer=sizer),
                    ],
                )
                # fmt: on

        assert (I := lookup.find_symbol("BINANCE.UM", "BTCUSDT")) is not None

        strategy = ComplexCompositeTest(
            tests={
                "2023-07-05 00:00:00": I.signal("2023-07-05 00:00:00", -1),
            }
        )

        r = CsvStorage("tests/data/storages/csv")
        rep = simulate(
            strategies={"Composited": strategy},
            data=r,
            capital=10000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-07-01",
            stop="2023-08-01",
            debug="DEBUG",
        )
        # - stop execution at signal's stop price
        # assert abs(rep[0].signals_log.iloc[0].stop - rep[0].executions_log.iloc[1].price) < I.tick_size

        # - we expect second signal as service from the tracker and executed at the price
        assert rep[0].signals_log.iloc[1].service, "Second signal should be service"
        assert abs(rep[0].signals_log.iloc[1].price - rep[0].executions_log.iloc[1].price) < I.tick_size

    def test_time_expiration_tracker(self):
        assert (I := lookup.find_symbol("BINANCE.UM", "BTCUSDT")) is not None
        r = CsvStorage("tests/data/storages/csv")

        class TimeExpiratorTest(GuineaPig):
            def on_init(self, ctx: IStrategyContext) -> None:
                ctx.set_base_subscription(DataType.OHLC["1h"])
                ctx.set_event_schedule("1h")

            def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
                return TimeExpirationTracker("3h", FixedLeverageSizer(1.0))

        rep = simulate(
            strategies={
                "TimeExpiratorTest": TimeExpiratorTest(
                    tests={
                        "2023-07-01 10:00:00": I.signal("2023-07-01 10:00:00", +1),
                        "2023-07-01 14:00:00": I.signal("2023-07-01 14:00:00", -1),
                        "2023-07-01 15:00:00": I.signal("2023-07-01 15:00:00", 0),
                    }
                )
            },
            data=r,
            capital=10000,
            instruments=["BINANCE.UM:BTCUSDT"],
            # silent=True,
            debug="DEBUG",
            commissions="vip0_usdt",
            start="2023-07-01",
            stop="2023-07-02",
        )

        print(rep[0].signals_log)
        assert len(rep[0].executions_log) == 4
        assert rep[0].signals_log.iloc[1].comment == "Time expired: 0 days 03:00:00"

    def test_trailing_stop_position_tracker(self):
        assert (I := lookup.find_symbol("BINANCE.UM", "BTCUSDT")) is not None
        r = CsvStorage("tests/data/storages/csv")

        class TrailingTestClient(GuineaPig):
            def on_init(self, ctx: IStrategyContext) -> None:
                ctx.set_base_subscription(DataType.OHLC["1h"])
                ctx.set_event_schedule("1h")

            def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
                return TrailingStopPositionTracker(1.0, 100, FixedLeverageSizer(1.0), risk_controlling_side="client")

        class TrailingTestBroker(GuineaPig):
            def on_init(self, ctx: IStrategyContext) -> None:
                ctx.set_base_subscription(DataType.OHLC["1h"])
                ctx.set_event_schedule("1h")

            def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
                return TrailingStopPositionTracker(1.0, 100, FixedLeverageSizer(1.0), risk_controlling_side="broker")

        rep = simulate(
            strategies={
                "Trailing.Clent": TrailingTestClient(
                    tests={
                        "2023-07-05 14:00:00": I.signal("2023-07-05 14:00:00", +1),
                        "2023-07-06 10:00:00": I.signal("2023-07-06 10:00:00", -1),
                    }
                ),
                "Trailing.Broker": TrailingTestBroker(
                    tests={
                        "2023-07-05 14:00:00": I.signal("2023-07-05 14:00:00", +1),
                        "2023-07-06 10:00:00": I.signal("2023-07-06 10:00:00", -1),
                    }
                ),
            },
            data=r,
            capital=10000,
            instruments=["BINANCE.UM:BTCUSDT"],
            silent=True,
            debug="DEBUG",
            commissions="vip0_usdt",
            start="2023-07-05",
            stop="2023-07-07",
        )

        # - client side
        assert len(rep[0].executions_log) == 4

        assert "Stop triggered" in rep[0].signals_log.iloc[1].comment
        assert rep[0].signals_log.price[1] >= 31252.00

        assert "Stop triggered" in rep[0].signals_log.iloc[3].comment
        assert rep[0].signals_log.price[3] <= 30163.65

        # - broker side
        assert len(rep[1].executions_log) == 4

        assert "Stop triggered" in rep[1].signals_log.iloc[1].comment
        assert rep[1].signals_log.price[1] >= 31252.00

        assert "Stop triggered" in rep[1].signals_log.iloc[3].comment
        assert rep[1].signals_log.price[3] <= 30163.65
