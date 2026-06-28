from dataclasses import dataclass
from typing import List, Union

import pandas as pd
from pytest import approx

from qubx.core.basics import OrderOrigin, Position, TransactionCostsCalculator, classify_origin
from qubx.core.lookups import FileInstrumentsLookupWithCCXT, lookup
from qubx.core.series import Quote, Trade, time_as_nsec
from qubx.utils.time import convert_seconds_to_str

N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)

TIME = lambda x: pd.Timestamp(x, unit="ns").asm8


@dataclass
class Deal:
    time: int
    position: int
    exec_price: float
    aggressive: bool

    def __init__(self, time, pos, price, aggressive=False):
        self.time = time_as_nsec(time)
        self.position = pos
        self.exec_price = price
        self.aggressive = aggressive


def run_deals_updates(p: Position, qs: List[Union[Deal, Trade, Quote]], tcc: TransactionCostsCalculator) -> pd.Series:
    pnls = {}
    for q in qs:
        if isinstance(q, Deal):
            fee_amount = tcc.get_execution_fees(p.instrument, q.exec_price, p.quantity - q.position, q.aggressive)
            pnls[pd.Timestamp(q.time, unit="ns")] = p.update_position(q.time, q.position, q.exec_price, fee_amount)[0]
            print(p, f"\t<-(exec -> {q.position})-")
        else:
            pnls[pd.Timestamp(q.time, unit="ns")] = p.update_market_price_by_tick(q)
            print(p)
    return pd.Series(pnls)


pos_round = lambda s, p, i: (p * round(s / p, i.size_precision), p, round(s / p, i.size_precision))


class TestBasics:
    def test_convertors(self):
        assert "3w" == convert_seconds_to_str(int(pd.Timedelta("3w").total_seconds()))
        assert "1d5h" == convert_seconds_to_str(int(pd.Timedelta("1d5h").total_seconds()))
        assert "1month" == convert_seconds_to_str(int(pd.Timedelta("4w").total_seconds()), convert_months=True)
        assert "1month" == convert_seconds_to_str(int(pd.Timedelta("28d").total_seconds()), convert_months=True)
        assert "4w" == convert_seconds_to_str(int(pd.Timedelta("28d").total_seconds()))

    def test_lookup(self):
        lookup = FileInstrumentsLookupWithCCXT()
        s0 = lookup["BINANCE:.*:ETH.*"]
        s1 = lookup["DUKAS:.*:EURGBP"]
        assert (
            lookup.find_aux_instrument_for(s0[0], "USDT").symbol,
            lookup.find_aux_instrument_for(s0[1], "USDT"),
            lookup.find_aux_instrument_for(s1[0], "USD").symbol,
        ) == ("BTCUSDT", None, "GBPUSD")

    def test_spot_positions(self):
        tcc = TransactionCostsCalculator("SPOT", 0.04, 0.04)
        i, s = lookup["BINANCE:SPOT:BTCUSDT"][0], 1
        D = "2024-01-01 "
        qs = [
            Quote(D + "12:00:00", 45000, 45000.5, 100, 50),
            Deal(D + "12:00:30", s, 45010),
            Trade(D + "12:01:00", 45010, 10, 1),
            Trade(D + "12:02:00", 45015, 10, 1),
            Deal(D + "12:02:30", -s, 45015),
            Quote(D + "12:03:00", 45020, 45021, 0, 0),
            Deal(D + "12:03:30", -2 * s, 45020),
            Quote(D + "12:04:00", 45120, 45121, 0, 0),
            Quote(D + "12:05:00", 45014, 45014, 0, 0),
            Deal(D + "12:06:30", 0, 45010),
            Quote(D + "12:10:00", 45020, 45020, 0, 0),
            Deal(D + "12:11:00", -1, 45020),
            Quote(D + "12:12:00", 45030, 45030, 0, 0),
            Deal(D + "12:13:00", 0, 45100),
        ]

        p = Position(i)
        pnls = run_deals_updates(p, qs, tcc)
        print(pnls)
        assert p.commissions == (1 * 45010 + 2 * 45015 + 1 * 45020 + 2 * 45010 + 45020 + 45100) * 0.04 / 100
        assert p.pnl == -60

    def test_average_price(self):
        p = Position(lookup.find_symbol("BINANCE", "ACAUSDT"))  # type: ignore
        for _p, _s in [
            (0.1763, 35.96),
            (0.1762, 14.04),
            (0.1716, 50.0),
            (0.165, 50.0),
            (0.1534, -40.0),
            (0.1612, 50.0),
            (0.1606, -50.0),
            (0.1611, 51.0),
            (0.1621, 50.0),
        ]:
            p.change_position_by(0, _s, _p)

        p.update_market_price(0, 0.1538, 1)
        # - 2024-10-12: fixed after part position closing fix
        assert p.position_avg_price == 0.1647  # , p.r_pnl, p.pnl - p.r_pnl, p.commissions, p.market_value

        p.change_position_by(0, -211, 0.1620)
        assert p.position_avg_price == 0.0

        for _p, _s in [
            (0.1620, -100.0),
            (0.1630, -100.0),
            (0.1640, -100.0),
            (0.1620, 100.0),
        ]:
            p.change_position_by(0, _s, _p)
        assert p.position_avg_price == 0.1630

    def test_futures_positions(self):
        D = "2024-01-01 "
        fi = lookup["BINANCE.UM:SWAP:BTCUSDT"][0]
        pos = Position(fi)
        tcc = TransactionCostsCalculator("UM", 0.02, 0.05)
        q1 = pos_round(239.9, 47980, fi)[2]
        q2 = q1 + pos_round(143.6, 47860, fi)[2]
        q3 = q2 - pos_round(300, 48050, fi)[2]
        rpnls = run_deals_updates(
            pos,
            [
                Deal(D + "00:00", q1, 47980, False),
                Deal(D + "00:10", q2, 47860, False),
                Trade(D + "00:15", 47984.7, 1),
                Deal(D + "00:20", q3, 48050, False),
                Deal(D + "00:30", 0, 48158.7, True),
            ],
            tcc,
        )
        assert N(rpnls.values) == [0.0, 0.0, 0.3976, 0.69, 0.4474]
        assert N(pos.pnl) == 1.1374
        assert N(pos.commissions) == 0.04815870 + 0.05766 + 0.028716 + 0.04798

        D = "2024-01-01 "
        i = lookup["BINANCE.UM:SWAP:BTCUSDT"][0]
        px0 = Position(i)

        run_deals_updates(
            px0,
            [
                Deal(D + "12:00:00", 1000 / 45000.0, 45000.0),
                Deal(D + "12:01:00", 1000 / 45000.0 + 1000 / 46000.0, 46000.0),
                Deal(D + "12:03:00", 0, 47000.0),
                Trade(D + "12:04:00", 47000.0, 0),
                Trade(D + "12:06:00", 48000.0, 0),
            ],
            tcc,
        )

        px1 = Position(i)
        px2 = Position(i)
        run_deals_updates(
            px1,
            [
                Deal(D + "12:00:00", 1000 / 45000, 45000),
                Deal(D + "12:03:00", 0, 47000),
            ],
            tcc,
        )
        run_deals_updates(
            px2,
            [
                Deal(D + "12:01:00", 1000 / 46000, 46000),
                Deal(D + "12:03:00", 0, 47000),
            ],
            tcc,
        )
        assert px0.pnl == N(px1.pnl + px2.pnl)

    def test_released_funds_estimations(self):
        fi = lookup["BINANCE:SPOT:BNBUSDT"][0]
        pos = Position(fi)
        pos.update_position(TIME(0), 5, 350)
        pos.update_market_price(TIME(1), 355, 1)
        assert 355 * 5 == pos.get_amount_released_funds_after_closing()
        assert 355 * 1 == pos.get_amount_released_funds_after_closing(4)
        assert 0 == pos.get_amount_released_funds_after_closing(10)

        pos2 = Position(fi)
        pos2.update_position(TIME(0), -5, 350)
        pos2.update_market_price(TIME(1), 355, 1)
        assert 355 * 5 == pos2.get_amount_released_funds_after_closing(10)
        assert 355 * 1 == pos2.get_amount_released_funds_after_closing(-4)
        assert 355 * 5 == pos2.get_amount_released_funds_after_closing()


def test_classify_origin_default_prefix():
    assert classify_origin("qubx_BTCUSDT_1") is OrderOrigin.RECOVERED
    assert classify_origin("manual-123") is OrderOrigin.EXTERNAL
    assert classify_origin("ext:VENUE-1") is OrderOrigin.EXTERNAL
    # No-weakening regression: a bare "qubx" lead without the underscore must stay
    # EXTERNAL on default venues — only a connector whose venue mangles the prefix
    # (OKX strips "_") opts into a shorter framework_prefix explicitly.
    assert classify_origin("qubxfoo") is OrderOrigin.EXTERNAL


def test_classify_origin_custom_prefix():
    # The OKX-sanitized form of a framework cid ("qubx_BTCUSDT_1" with "_" stripped).
    assert classify_origin("qubxBTCUSDT1", framework_prefix="qubx") is OrderOrigin.RECOVERED
    assert classify_origin("manual-123", framework_prefix="qubx") is OrderOrigin.EXTERNAL


def test_position_flatten_zeroes_quantity_but_keeps_realized(mocker):
    from qubx.core.basics import Instrument, Position

    instr = mocker.Mock(spec=Instrument)
    instr.lot_size = 0.001
    pos = Position(instr, quantity=10.0, pos_average_price=100.0, r_pnl=42.0)
    pos.market_value = 1000.0
    pos.market_value_funds = 1000.0
    pos.initial_margin = 250.0
    pos.maint_margin = 125.0

    pos.flatten()

    assert pos.quantity == 0.0
    assert pos.market_value == 0.0
    assert pos.market_value_funds == 0.0
    assert pos.initial_margin == 0.0
    assert pos.maint_margin == 0.0
    assert pos.pnl == 42.0  # unrealized is 0 at qty 0 -> pnl == r_pnl
    assert pos.r_pnl == 42.0  # realized preserved
    assert pos.position_avg_price == 100.0  # entry preserved
    assert pos.is_open() is False


def test_position_mark_tick_does_not_clobber_last_update_time():
    # last_update_time must track the venue SIZE/state update (deal / snapshot reconcile),
    # NOT every mark-price tick — otherwise quotes ratchet it to local time and the
    # reconciler's monotonic position guard breaks.
    import numpy as np

    inst = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert inst is not None
    pos = Position(inst)

    t_size = np.datetime64("2026-05-28T00:00:00", "ns")  # a size/state update (e.g. snapshot/deal)
    pos.update_market_price(t_size, 50_000.0, 1.0)
    assert pos.last_update_time == t_size

    t_tick = np.datetime64("2026-05-28T00:05:00", "ns")  # a later mark-only quote tick
    pos.update_market_price_by_tick(Quote(t_tick, 50_010.0, 50_011.0, 0, 0), 1.0)
    assert pos.last_update_price != 50_000.0  # mark IS updated...
    assert pos.last_update_time == t_size  # ...but last_update_time is NOT clobbered


def test_position_deal_stamps_last_update_time_as_datetime64():
    # a deal (venue size change) must stamp last_update_time as a dt_64 venue timestamp,
    # NOT a raw int-ns: it has to be monotonic-comparable with snapshot stamps (which are
    # dt_64) and serialize consistently (control protocol does str() -> ISO, not a raw
    # nanosecond integer like "1782478555172000000").
    import numpy as np

    inst = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert inst is not None
    pos = Position(inst)

    t_deal = np.datetime64("2026-06-26T12:55:55.172", "ns")
    pos.change_position_by(t_deal, 0.003, 59_251.6)

    assert isinstance(pos.last_update_time, np.datetime64)  # dt_64, not int-ns
    assert pos.last_update_time == t_deal


def test_update_position_realize_only_books_pnl_keeps_size():
    # realize_only=True realizes the closing pnl but leaves size/avg/last_update_time untouched
    # (situation II: the deal is already in an authoritative snapshot size).
    import numpy as np

    inst = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert inst is not None
    pos = Position(inst, quantity=0.003, pos_average_price=59_000.0)
    pos.last_update_time = np.datetime64("2026-06-26T00:00:00", "ns")  # type: ignore

    # close 0.002 @ 59_500 (target 0.001), realize-only
    pnl, _ = pos.update_position(TIME("2026-06-26T01:00:00"), 0.001, 59_500.0, realize_only=True)

    assert pnl == approx(1.0)  # 0.002 * (59_500 - 59_000)
    assert pos.r_pnl == approx(1.0)
    assert pos.quantity == 0.003  # size NOT moved
    assert pos.position_avg_price == 59_000.0  # avg NOT touched
    assert pos.last_update_time == np.datetime64("2026-06-26T00:00:00", "ns")  # not restamped


def test_update_position_realize_only_opening_realizes_nothing():
    # a same-side (opening) move under realize_only realizes nothing and never touches size.
    inst = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert inst is not None
    pos = Position(inst, quantity=0.003, pos_average_price=59_000.0)

    pnl, _ = pos.update_position(TIME("2026-06-26T01:00:00"), 0.005, 59_500.0, realize_only=True)

    assert pnl == 0.0
    assert pos.r_pnl == 0.0
    assert pos.quantity == 0.003


def test_update_position_by_deal_realize_only():
    # the deal-driven entry point: realize_only books r_pnl from the deal, size stays put.
    from qubx.core.basics import Deal as CoreDeal

    inst = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert inst is not None
    pos = Position(inst, quantity=0.003, pos_average_price=59_000.0)

    deal = CoreDeal(
        trade_id="t1", order_id="v1", time=TIME("2026-06-26T01:00:00"), amount=-0.002, price=59_500.0, aggressive=True
    )
    pos.update_position_by_deal(deal, realize_only=True)

    assert pos.r_pnl == approx(1.0)
    assert pos.quantity == 0.003  # size untouched


def test_update_position_realize_only_books_fee_to_commissions():
    # realize_only still books the fee into commissions (a tracker; balance stays snapshot-owned)
    inst = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert inst is not None
    pos = Position(inst, quantity=0.003, pos_average_price=59_000.0)

    pos.update_position(TIME("2026-06-26T01:00:00"), 0.001, 59_500.0, fee_amount=0.5, realize_only=True)

    assert pos.r_pnl == approx(1.0)
    assert pos.commissions == approx(0.5)
    assert pos.quantity == 0.003


def test_update_position_with_conversion_rate():
    # the multi-currency seam (dormant at conv=1.0 in prod): r_pnl and avg_funds are scaled into
    # the funded currency by conversion_rate; the returned deal_pnl stays in quote units.
    inst = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert inst is not None
    pos = Position(inst)

    pos.update_position(TIME(0), 0.002, 50_000.0, conversion_rate=2.0)  # open long 0.002
    assert pos.position_avg_price == 50_000.0
    assert pos.position_avg_price_funds == 25_000.0  # avg / conv

    pnl, _ = pos.update_position(TIME(1), 0.0, 51_000.0, conversion_rate=2.0)  # close
    assert pnl == approx(2.0)  # returned realized in QUOTE units: 0.002 * (51_000 - 50_000)
    assert pos.r_pnl == approx(1.0)  # accumulated in FUNDED currency: 2.0 / conv
    assert pos.quantity == 0.0
