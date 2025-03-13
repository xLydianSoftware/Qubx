import numpy as np
import pandas as pd

from qubx import logger
from qubx.backtester.ome import OrdersManagementEngine
from qubx.backtester.simulator import simulate
from qubx.core.basics import ZERO_COSTS, DataType, Deal, Instrument, ITimeProvider, Order
from qubx.core.exceptions import SimulationError
from qubx.core.interfaces import IStrategy, IStrategyContext, TriggerEvent
from qubx.core.lookups import lookup
from qubx.core.series import Quote, Trade, TradeArray
from qubx.core.utils import recognize_time
from qubx.data.readers import AsOhlcvSeries, CsvStorageDataReader, RestoreTicksFromOHLC
from qubx.pandaz.utils import shift_series
from qubx.ta.indicators import ema, sma


class _TimeService(ITimeProvider):
    _time: np.datetime64

    def g(self, quote: Quote) -> Quote:
        self._time = quote.time  # type: ignore
        return quote

    def time(self) -> np.datetime64:
        return self._time


def Q(time: str, bid: float, ask: float) -> Quote:
    return Quote(recognize_time(time), bid, ask, 0, 0)


class TestBacktesterStuff:
    def test_basic_ome(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr
        ome = OrdersManagementEngine(instr, t := _TimeService(), tcc=ZERO_COSTS)

        q0 = Q("2020-01-01 10:00", 32000, 32001)
        ome.process_market_data(t.g(q0))

        r0 = ome.place_order("BUY", "MARKET", 0.04, 0, "Test1")
        assert r0.order.status == "CLOSED"
        assert r0.exec is not None
        assert r0.exec.amount == 0.04

        r1 = ome.place_order("SELL", "LIMIT", 0.1, q0.bid, "Test2")
        assert r1.order.status == "CLOSED"
        assert r1.exec is not None
        assert r1.exec.amount == -0.1

        r2 = ome.place_order("BUY", "LIMIT", 0.04, q0.bid - 100, "Test2")
        assert r2.order.status == "OPEN"
        assert r2.exec is None

        r3 = ome.place_order("BUY", "LIMIT", 0.1, q0.bid - 100, "Test3")
        assert r3.order.status == "OPEN"
        assert r3.exec is None

        r4 = ome.place_order("SELL", "LIMIT", 0.04, q0.ask + 100, "Test4")
        assert r4.order.status == "OPEN"
        assert r4.exec is None

        r5 = ome.place_order("SELL", "LIMIT", 0.14, q0.ask + 50, "Test5")
        assert r5.order.status == "OPEN"
        assert r5.order.client_id == "Test5"
        assert r5.exec is None

        r6 = ome.place_order("SELL", "LIMIT", 0.3, q0.ask, "Test6")
        assert r6.order.status == "OPEN"
        assert r6.exec is None

        r7 = ome.place_order("BUY", "LIMIT", 0.12, q0.bid, "Test7")
        assert r7.order.status == "OPEN"
        assert r7.exec is None

        assert len(ome.get_open_orders()) == 6

        ome.cancel_order(r6.order.id)
        assert len(ome.get_open_orders()) == 5

        try:
            ome.cancel_order(r6.order.id)
            assert False
        except:
            assert True

        ome.cancel_order(r7.order.id)
        ome.cancel_order(r5.order.id)
        ome.cancel_order(r4.order.id)
        ome.cancel_order(r3.order.id)
        rc2 = ome.cancel_order(r2.order.id)
        assert rc2.order.status == "CANCELED"
        assert len(ome.get_open_orders()) == 0

    def test_ome_execution(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr
        ome = OrdersManagementEngine(instr, t := _TimeService(), tcc=ZERO_COSTS)

        q0 = Q("2020-01-01 10:00", 32000, 32001)
        ome.process_market_data(t.g(q0))

        r1 = ome.place_order("SELL", "LIMIT", 0.3, 32001, "Test1")
        r2 = ome.place_order("BUY", "LIMIT", 0.3, 32000, "Test2")

        # - nothing changed - no reports
        rs = ome.process_market_data(t.g(Q("2020-01-01 10:01", 32000, 32001)))
        assert not rs

        rs = ome.process_market_data(t.g(Q("2020-01-01 10:01", 32002, 32003)))
        assert rs[0].exec is not None
        assert rs[0].exec.aggressive == False
        assert rs[0].exec.price == 32001

        rs = ome.process_market_data(t.g(Q("2020-01-01 10:01", 31899, 31900)))
        assert rs[0].exec is not None
        assert rs[0].exec.aggressive == False
        assert rs[0].exec.price == 32000

        assert len(ome.get_open_orders()) == 0

    def test_ome_inside_spread_execution(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None

        ome = OrdersManagementEngine(instr, t := _TimeService(), tcc=ZERO_COSTS)
        q0 = t.g(Q("2020-01-01 10:00", 2000.0, 2003.0))
        ome.process_market_data(q0)

        r1 = ome.place_order("BUY", "LIMIT", 0.3, 2001.0, "Test1")
        assert r1.order.status == "OPEN"
        assert r1.exec is None

        r2 = ome.place_order("SELL", "LIMIT", 0.3, 2002.0, "Test2")
        assert r2.order.status == "OPEN"
        assert r2.exec is None

        r3 = ome.process_market_data(t.g(Q("2020-01-01 10:01", 2000.0, 2001.0)))
        assert r3[0].exec is not None
        assert r3[0].exec.price == 2001.0
        assert r3[0].exec.amount == 0.3

        r4 = ome.process_market_data(t.g(Q("2020-01-01 10:03", 2003.0, 2005.0)))
        assert r4[0].exec is not None
        assert r4[0].exec.price == 2002.0
        assert r4[0].exec.amount == -0.3

        # - quote at bid
        r5 = ome.place_order("BUY", "LIMIT", 0.3, 2003.0, "Test 3")
        assert r5.order.status == "OPEN"
        assert r5.exec is None

        # - no exec - price goes up
        r51 = ome.process_market_data(t.g(Q("2020-01-01 10:04", 2004.0, 2005.0)))
        assert r51 == []

        # - executed - price goes down
        r52 = ome.process_market_data(t.g(Q("2020-01-01 10:05", 2002.0, 2000.0)))
        assert r52[0].exec is not None
        assert r52[0].exec.price == 2003.0
        assert r52[0].exec.amount == 0.3

    def test_executions_on_single_trade(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None

        ome = OrdersManagementEngine(instr, t := _TimeService(), tcc=ZERO_COSTS)
        ome.process_market_data(t.g(Q("2020-01-01 10:00", 100.0, 103.0)))

        s4 = ome.place_order("SELL", "LIMIT", 0.05, 106.0, "s3.2")
        s3 = ome.place_order("SELL", "LIMIT", 0.05, 106.0, "s3.1")
        s2 = ome.place_order("SELL", "LIMIT", 0.2, 105.0, "s2")
        s1 = ome.place_order("SELL", "LIMIT", 0.3, 104.0, "s1")
        s0 = ome.place_order("SELL", "LIMIT", 0.4, 103.0, "s0")  # <- ask

        b0 = ome.place_order("BUY", "LIMIT", 0.4, 101.0, "b0")  #  - inside spread -
        b1 = ome.place_order("BUY", "LIMIT", 0.3, 100.0, "b1")  # <- bid
        b2 = ome.place_order("BUY", "LIMIT", 0.2, 99.0, "b2")
        b3 = ome.place_order("BUY", "LIMIT", 0.1, 98.0, "b3")

        x1 = ome.process_market_data(
            Trade(
                recognize_time("2020-01-01 10:01"),
                110.0,
                0.1,
                1,
            )
        )
        for i in x1:
            print(f"  - {i.order.client_id} {i.order.status} {str(i.exec)}")

        x2 = ome.process_market_data(
            Trade(
                recognize_time("2020-01-01 10:02"),
                90.0,
                0.1,
                -1,
            )
        )
        for i in x2:
            print(f"  - {i.order.client_id} {i.order.status} {str(i.exec)}")

        # - quote
        qr = ome.process_market_data(t.g(Q("2020-01-01 10:05", 50.0, 51.0)))
        assert len(qr) == 0  # no execs

        assert len(ome.active_orders) == 0
        assert len(ome.stop_orders) == 0

    def test_executions_on_array_of_trades(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None

        ome = OrdersManagementEngine(instr, t := _TimeService(), tcc=ZERO_COSTS)
        ome.process_market_data(t.g(Q("2020-01-01 10:00", 100.0, 103.0)))

        s4 = ome.place_order("SELL", "LIMIT", 0.05, 106.0, "s3.2")
        s3 = ome.place_order("SELL", "LIMIT", 0.05, 106.0, "s3.1")
        s2 = ome.place_order("SELL", "LIMIT", 0.2, 105.0, "s2")
        ob = ome.place_order("BUY", "STOP_MARKET", 0.2, 104.0, "sm2")
        s1 = ome.place_order("SELL", "LIMIT", 0.3, 104.0, "s1")
        s0 = ome.place_order("SELL", "LIMIT", 0.4, 103.0, "s0")  # <- ask

        b0 = ome.place_order("BUY", "LIMIT", 0.4, 101.0, "b0")  #  - inside spread -
        b1 = ome.place_order("BUY", "LIMIT", 0.3, 100.0, "b1")  # <- bid
        b2 = ome.place_order("BUY", "LIMIT", 0.2, 99.0, "b2")
        b3 = ome.place_order("BUY", "LIMIT", 0.1, 98.0, "b3")

        ta1 = TradeArray()

        # - buys
        ta1.add(recognize_time("2020-01-01 10:01"), 102.0, 0.1, 1)
        ta1.add(recognize_time("2020-01-01 10:02"), 103.0, 0.1, 1)
        ta1.add(recognize_time("2020-01-01 10:03"), 103.5, 0.1, 1)
        ta1.add(recognize_time("2020-01-01 10:03"), 110.0, 0.1, 1)

        # - sells
        ta1.add(recognize_time("2020-01-01 10:01:01"), 101.0, 0.1, -1)
        ta1.add(recognize_time("2020-01-01 10:02:01"), 99.0, 0.1, -1)
        ta1.add(recognize_time("2020-01-01 10:03:01"), 97.5, 0.1, -1)
        ta1.add(recognize_time("2020-01-01 10:03:01"), 96.0, 0.1, -1)

        # - step 1
        x1 = ome.process_market_data(ta1)
        for i in x1:
            print(f"  - {i.order.client_id} {i.order.status} {str(i.exec)}")

        # - quote
        qr = ome.process_market_data(t.g(Q("2020-01-01 10:05", 50.0, 51.0)))
        assert len(qr) == 0  # no execs

        assert len(ome.active_orders) == 0
        assert len(ome.stop_orders) == 0

    def test_ome_loop(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None
        r = CsvStorageDataReader("tests/data/csv")
        stream = r.read("BTCUSDT_ohlcv_M1", transform=RestoreTicksFromOHLC(trades=False, spread=instr.tick_size))
        assert isinstance(stream, list)

        ome = OrdersManagementEngine(instr, t := _TimeService(), tcc=ZERO_COSTS)
        ome.process_market_data(t.g(stream[0]))
        l1 = ome.place_order("BUY", "LIMIT", 0.5, 39500.0, "Test1")
        l2 = ome.place_order("SELL", "LIMIT", 0.5, 52000.0, "Test2")

        execs = []
        for i in range(len(stream)):
            rs = ome.process_market_data(t.g(stream[i]))
            if rs:
                execs.append(rs[0].exec)

        assert l1.order.status == "CLOSED"
        assert l2.order.status == "CLOSED"
        assert execs[0].price == 39500.0
        assert execs[1].price == 52000.0

    def test_simulator(self):
        class CrossOver(IStrategy):
            timeframe: str = "1Min"
            fast_period = 5
            slow_period = 12

            def on_init(self, ctx: IStrategyContext):
                ctx.set_base_subscription(DataType.OHLC[self.timeframe])

            def on_event(self, ctx: IStrategyContext, event: TriggerEvent):
                for i in ctx.instruments:
                    ohlc = ctx.ohlc(i, self.timeframe)
                    fast = ema(ohlc.close, self.fast_period)
                    slow = ema(ohlc.close, self.slow_period)
                    pos = ctx.positions[i].quantity
                    if pos <= 0:
                        if (fast[0] > slow[0]) and (fast[1] < slow[1]):
                            ctx.trade(i, abs(pos) + i.min_size * 10)
                    if pos >= 0:
                        if (fast[0] < slow[0]) and (fast[1] > slow[1]):
                            ctx.trade(i, -pos - i.min_size * 10)
                return None

            def ohlcs(self, timeframe: str) -> dict[str, pd.DataFrame]:
                return {s.symbol: self.ctx.ohlc(s, timeframe).pd() for s in self.ctx.instruments}

        r = CsvStorageDataReader("tests/data/csv")
        ohlc = r.read("BINANCE.UM:BTCUSDT", "2024-01-01", "2024-01-02", AsOhlcvSeries("5Min"))
        fast = ema(ohlc.close, 5)  # type: ignore
        slow = ema(ohlc.close, 15)  # type: ignore
        sigs = (((fast > slow) + (fast.shift(1) < slow.shift(1))) == 2) - (
            ((fast < slow) + (fast.shift(1) > slow.shift(1))) == 2
        )
        sigs = sigs.pd()
        sigs = sigs[sigs != 0]
        i1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert i1 is not None
        # s2 = shift_series(sigs, "4Min59Sec").rename(i1) / 100  # type: ignore
        s2 = shift_series(sigs, "5Min").rename(i1) / 100  # type: ignore

        # fmt: off
        rep1 = simulate(
            {
                # - generated signals as series
                "test0": CrossOver(timeframe="5Min", fast_period=5, slow_period=15),
                "test1": s2,
            },
            {'ohlc(5Min)': r}, 10000, ["BINANCE.UM:BTCUSDT"], "vip0_usdt", "2024-01-01", "2024-01-02", n_jobs=1
        ) 
        # fmt:on

        assert all(
            rep1[0].executions_log[["filled_qty", "price", "side"]]
            == rep1[1].executions_log[["filled_qty", "price", "side"]]
        )

    def test_ome_stop_orders(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None

        r = CsvStorageDataReader("tests/data/csv")
        stream = r.read(
            "BTCUSDT_ohlcv_M1",
            start="2024-01-01",
            stop="2024-01-15",
            transform=RestoreTicksFromOHLC(trades=False, spread=instr.tick_size),
        )
        assert isinstance(stream, list)

        ome = OrdersManagementEngine(instr, t := _TimeService(), tcc=ZERO_COSTS)
        q0 = t.g(stream[0])
        ome.process_market_data(t.g(q0))

        # - trigger immediate exception test
        try:
            ome.place_order("BUY", "STOP_MARKET", 0.5, q0.mid_price() - 100.0, "Failed")
            assert False
        except Exception as e:
            print(f" -> {e}")

        try:
            ome.place_order("SELL", "STOP_MARKET", 0.5, q0.mid_price() + 100.0, "Failed")
            assert False
        except Exception as e:
            print(f" -> {e}")

        ent0 = q0.mid_price() + 1000.0
        ent1 = q0.mid_price() - 150.0

        # - stop orders
        stp1 = ome.place_order("BUY", "STOP_MARKET", 0.5, ent0, "Buy1", fill_at_signal_price=True)
        stp2 = ome.place_order("BUY", "STOP_MARKET", 0.5, ent0, "Buy2", fill_at_signal_price=False)
        stp3 = ome.place_order("SELL", "STOP_MARKET", 0.5, ent1, "Sell1", fill_at_signal_price=True)
        stp4 = ome.place_order("SELL", "STOP_MARKET", 0.5, ent1, "Sell2", fill_at_signal_price=False)

        # - just to put limit orders to test it together
        ent2 = q0.mid_price() + 2000.0
        ent3 = q0.mid_price() - 1900.0
        ent4 = q0.mid_price() - 5000.0
        lmt1 = ome.place_order("SELL", "LIMIT", 0.5, ent2, "LimitSell1")
        lmt2 = ome.place_order("BUY", "LIMIT", 0.5, ent3, "LimitBuy2")
        lmt3 = ome.place_order("BUY", "LIMIT", 0.5, ent4, "LimitBuy3")

        [print(" --> " + str(s)) for s in [stp1, stp2, stp3, stp4]]
        [print(" --> " + str(l)) for l in [lmt1, lmt2, lmt3]]

        execs = []
        for i in range(len(stream)):
            rs = ome.process_market_data(t.g(stream[i]))
            if rs:
                execs.extend([r.exec for r in rs])

        assert stp1.order.status == "CLOSED"
        assert stp2.order.status == "CLOSED"
        assert stp3.order.status == "CLOSED"
        assert stp4.order.status == "CLOSED"
        assert lmt1.order.status == "CLOSED"
        assert lmt2.order.status == "CLOSED"
        assert lmt3.order.status == "OPEN"
        assert execs[0].price == ent0
        assert execs[1].price > ent0
        assert execs[2].price == ent2
        assert execs[3].price == ent1
        assert execs[4].price < ent1
        assert execs[5].price == ent3

        assert len(ome.get_open_orders()) == 1
        [print(" ::::: " + str(s)) for s in ome.get_open_orders()]

    def test_ome_special_execution_price_case(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None

        ome = OrdersManagementEngine(instr, t := _TimeService(), tcc=ZERO_COSTS)
        ome.process_market_data(t.g(Q("2020-01-01 10:00", 100.0, 101.0)))

        # - executed at usual ask price - no previous update !
        ex1 = ome.place_order(
            "BUY", "MARKET", 1, None, "not at desired price", fill_at_signal_price=True, signal_price=111.0
        )
        assert ex1.exec is not None
        assert ex1.exec.price == 101.0
        print(f" -> {ex1}")

        ome.process_market_data(t.g(Q("2020-01-01 10:01", 110.0, 111.0)))

        ex2 = ome.place_order(
            "BUY", "MARKET", 1, None, "at custom desired price", fill_at_signal_price=True, signal_price=105.0
        )
        assert ex2.exec is not None
        assert ex2.exec.price == 105.0
        print(f" -> {ex2}")

        # - not reacheable price (was not crossed)
        ome.process_market_data(t.g(Q("2020-01-01 10:02", 100.0, 100.1)))
        try:
            ome.place_order("SELL", "MARKET", 1, None, "not reacheable", fill_at_signal_price=True, signal_price=1000.0)
            assert False
        except SimulationError:
            pass
