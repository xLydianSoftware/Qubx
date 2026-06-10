import numpy as np
import pandas as pd
from pytest import approx

from qubx.connectors.ccxt.utils import (
    ccxt_convert_deal_info,
    ccxt_convert_order_info,
    ccxt_restore_position_from_deals,
)
from qubx.core.basics import Deal, Instrument, OrderOrigin, Position
from qubx.core.lookups import lookup
from tests.qubx.connectors.ccxt.data.ccxt_responses import (
    C1,
    C2,
    C3,
    C4,
    HIST,
    C5new,
    C6ex,
    C7cancel,
)

N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)


class TestStrats:
    def test_ccxt_exec_report_conversion(self):
        instrument = lookup.find_symbol("BINANCE", "ACAUSDT")
        assert instrument is not None
        # - execution reports
        for o in [
            ccxt_convert_order_info(instrument, C1),
            ccxt_convert_order_info(instrument, C2),
            ccxt_convert_order_info(instrument, C3),
            ccxt_convert_order_info(instrument, C4),
            ccxt_convert_order_info(instrument, C5new),
            ccxt_convert_order_info(instrument, C6ex),
            ccxt_convert_order_info(instrument, C7cancel),
        ]:
            print(o)
        print("-" * 50)

        print(ccxt_convert_order_info(instrument, C5new))
        print(ccxt_convert_order_info(instrument, C6ex))
        print(ccxt_convert_order_info(instrument, C7cancel))

        print("#" * 50)

        # - historical records
        for h in HIST:
            i = lookup.find_symbol("BINANCE.UM", h["info"]["symbol"])
            if i is not None:
                o = ccxt_convert_order_info(i, h)
                print(o)

    def test_ccxt_hist_trades_conversion(self):
        raw = {
            "info": {
                "symbol": "RAYUSDT",
                "id": "56324015",
                "orderId": "536752004",
                "orderListId": "-1",
                "price": "2.11290000",
                "qty": "2.40000000",
                "quoteQty": "5.07096000",
                "commission": "0.00000648",
                "commissionAsset": "BNB",
                "time": "1712497717270",
                "isBuyer": True,
                "isMaker": False,
                "isBestMatch": True,
            },
            "timestamp": 1712497717270,
            "datetime": "2024-04-07T13:48:37.270Z",
            "symbol": "RAY/USDT",
            "id": "56324015",
            "order": "536752004",
            "type": None,
            "side": "buy",
            "takerOrMaker": "taker",
            "price": 2.1129,
            "amount": 2.4,
            "cost": 5.07096,
            "fee": {"cost": 6.48e-06, "currency": "BNB"},
            "fees": [{"cost": 6.48e-06, "currency": "BNB"}],
        }
        print(ccxt_convert_deal_info(raw))

    def test_deal_synthesizes_trade_id_when_venue_omits_it(self):
        # Some venues omit a per-fill id; the converter must synthesize a deterministic
        # one from (order_id, timestamp, qty, price) so fill dedup still works.
        raw = {
            "order": "ORD-1",
            "timestamp": 1712497717270,
            "side": "buy",
            "takerOrMaker": "taker",
            "price": 2.1129,
            "amount": 2.4,
        }
        deal = ccxt_convert_deal_info(raw)  # must not raise KeyError on missing "id"
        assert deal.trade_id == "ORD-1:1712497717270:2.4:2.1129"
        # deterministic: same fill -> same id (so seen_trade_ids dedups it)
        assert ccxt_convert_deal_info(dict(raw)).trade_id == deal.trade_id
        # a real venue id is used verbatim when present
        assert ccxt_convert_deal_info({**raw, "id": "T9"}).trade_id == "T9"

    def test_position_restoring_from_deals(self):
        deals = [
            Deal(
                "0",
                1,
                time=pd.Timestamp("2024-04-07 13:04:36.975000"),
                amount=0.5,
                price=180.84,
                aggressive=True,
                fee_amount=0.00011542,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "1",
                1,
                time=pd.Timestamp("2024-04-07 13:09:22.644000"),
                amount=-0.5,
                price=181.12,
                aggressive=True,
                fee_amount=0.00011562,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "2",
                1,
                time=pd.Timestamp("2024-04-07 13:48:37.611000"),
                amount=0.11,
                price=181.67,
                aggressive=True,
                fee_amount=2.544e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "3",
                1,
                time=pd.Timestamp("2024-04-07 13:48:37.611000"),
                amount=0.11,
                price=181.68,
                aggressive=True,
                fee_amount=2.544e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "4",
                1,
                time=pd.Timestamp("2024-04-07 13:48:37.611000"),
                amount=0.11,
                price=181.69,
                aggressive=True,
                fee_amount=2.544e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "5",
                1,
                time=pd.Timestamp("2024-04-07 13:48:37.611000"),
                amount=0.22,
                price=181.69,
                aggressive=True,
                fee_amount=5.09e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "6",
                1,
                time=pd.Timestamp("2024-04-07 14:12:34.624000"),
                amount=-0.55,
                price=181.29,
                aggressive=True,
                fee_amount=0.00012728,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "7",
                1,
                time=pd.Timestamp("2024-04-07 14:16:46.048000"),
                amount=0.7,
                price=181.32,
                aggressive=True,
                fee_amount=0.00016175,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "8",
                1,
                time=pd.Timestamp("2024-04-07 14:17:47.396000"),
                amount=-0.7,
                price=181.36,
                aggressive=True,
                fee_amount=0.00016176,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "9",
                1,
                time=pd.Timestamp("2024-04-07 14:18:25.864000"),
                amount=0.13,
                price=181.36,
                aggressive=True,
                fee_amount=3.005e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "a",
                1,
                time=pd.Timestamp("2024-04-07 14:18:25.864000"),
                amount=0.11,
                price=181.36,
                aggressive=True,
                fee_amount=2.543e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "b",
                1,
                time=pd.Timestamp("2024-04-07 14:18:25.864000"),
                amount=0.76,
                price=181.36,
                aggressive=True,
                fee_amount=0.00076,
                fee_currency="SOL",
            ),  # type: ignore
        ]

        instr1: Instrument = lookup.find_symbol("BINANCE", "SOLUSDT")  # type: ignore
        pos1 = Position(instr1)  # type: ignore
        vol1 = np.sum([d.amount for d in deals]) - instr1.round_size_up(
            deals[-1].fee_amount if deals[-1].fee_amount else 0
        )

        pos1 = ccxt_restore_position_from_deals(pos1, vol1, deals)
        assert N(pos1.quantity, instr1.lot_size) == vol1

        deals = [
            Deal(
                "0",
                2,
                time=pd.Timestamp("2024-04-07 12:40:41.717000"),
                amount=0.154,
                price=587.1,
                aggressive=True,
                fee_amount=0.0001155,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "1",
                2,
                time=pd.Timestamp("2024-04-07 12:41:59.307000"),
                amount=-0.154,
                price=586.6,
                aggressive=True,
                fee_amount=0.00011472,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "2",
                2,
                time=pd.Timestamp("2024-04-07 12:44:45.991000"),
                amount=-0.199,
                price=588.5,
                aggressive=True,
                fee_amount=0.00014922,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "3",
                2,
                time=pd.Timestamp("2024-04-08 12:45:49.738000"),
                amount=0.025,
                price=594.1,
                aggressive=True,
                fee_amount=1.875e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "4",
                2,
                time=pd.Timestamp("2024-04-08 12:48:37.543000"),
                amount=0.011,
                price=594.0,
                aggressive=True,
                fee_amount=8.25e-06,
                fee_currency="BNB",
            ),  # type: ignore
        ]

        instr2 = lookup.find_symbol("BINANCE", "BNBUSDT")
        assert instr2 is not None
        pos2 = Position(instr2)
        vol2 = np.sum([d.amount for d in deals]) - instr2.round_size_up(np.sum([d.fee_amount for d in deals]))  # type: ignore

        pos2 = ccxt_restore_position_from_deals(pos2, vol2, deals)
        assert N(pos2.quantity, instr2.lot_size) == vol2


def _raw_order(**overrides):
    raw = {
        "info": {},
        "amount": 1.0,
        "price": 50_000.0,
        "status": "open",
        "side": "buy",
        "type": "limit",
        "timestamp": 1_716_854_400_000,
        "id": "VENUE-999",
        "cost": 0.0,
    }
    raw.update(overrides)
    return raw


def test_convert_order_info_without_client_order_id_falls_back_to_ext():
    # A venue that omits clientOrderId must not KeyError; the order reads as EXTERNAL with a
    # stable ext:<venue_id> client_order_id.
    instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    order = ccxt_convert_order_info(instrument, _raw_order())  # no clientOrderId
    assert order.client_order_id == "ext:VENUE-999"
    assert order.origin == OrderOrigin.EXTERNAL
    assert order.venue_order_id == "VENUE-999"


def test_convert_order_info_with_framework_client_order_id():
    # A framework cid parsed back from venue data classifies as RECOVERED (classify_origin);
    # FRAMEWORK is reserved for orders the trading mixin creates itself.
    instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    order = ccxt_convert_order_info(instrument, _raw_order(clientOrderId="qubx_BTCUSDT_1"))
    assert order.client_order_id == "qubx_BTCUSDT_1"
    assert order.origin == OrderOrigin.RECOVERED


def test_convert_order_info_market_order_price_is_none():
    # Market orders carry no limit price; the converter must yield None, not a fake 0.0
    # (matches Order.price: float | None).
    instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    order = ccxt_convert_order_info(instrument, _raw_order(price=None, type="market"))
    assert order.price is None


def test_convert_deal_info_tolerates_empty_fee_and_missing_taker():
    # CCXT may send fee={} (or cost=None) and omit takerOrMaker — must not KeyError/TypeError.
    raw = {"order": "O1", "timestamp": 1_716_854_400_000, "side": "buy", "price": 100.0, "amount": 1.0, "fee": {}}
    deal = ccxt_convert_deal_info(raw)
    assert deal.fee_amount is None
    assert deal.fee_currency is None
    assert deal.aggressive is False  # takerOrMaker absent -> treated as maker
