import pandas as pd
from pytest import approx

from qubx.connectors.ccxt.utils import (
    ccxt_convert_deal_info,
    ccxt_convert_order_info,
)
from qubx.core.basics import OrderOrigin, OrderStatus
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

def test_ccxt_order_stamps_venue_last_update_time():
    # the order's last_update_time must carry the VENUE update ts (ccxt lastUpdateTimestamp,
    # else raw info.updateTime), so the reconciler's monotonic guard works in prod.
    from qubx.core.utils import recognize_time

    instrument = lookup.find_symbol("BINANCE", "ACAUSDT")
    o = ccxt_convert_order_info(instrument, C5new)
    assert o.last_update_time == recognize_time(C5new["lastUpdateTimestamp"])

    # fallback to raw info.updateTime when the unified field is absent
    raw = {**C5new, "lastUpdateTimestamp": None, "info": {"updateTime": "1712231523596"}}
    o2 = ccxt_convert_order_info(instrument, raw)
    assert o2.last_update_time == recognize_time(1712231523596)

    # neither present -> None (no venue ts)
    raw3 = {**C5new, "lastUpdateTimestamp": None, "info": {}}
    assert ccxt_convert_order_info(instrument, raw3).last_update_time is None


def test_ccxt_position_stamps_venue_last_update_time():
    from qubx.connectors.ccxt.utils import ccxt_convert_position
    from tests.qubx.connectors.ccxt.test_utils import BINANCE_MARKETS, POSITIONS_BINANCE_UM

    p = POSITIONS_BINANCE_UM[0]
    expected = pd.Timestamp(p["timestamp"], unit="ms").asm8
    assert ccxt_convert_position(p, "BINANCE.UM", BINANCE_MARKETS).last_update_time == expected
    # stamped even when the venue omits markPrice (the relaxed guard)
    assert ccxt_convert_position({**p, "markPrice": None}, "BINANCE.UM", BINANCE_MARKETS).last_update_time == expected


class TestStrats:
    def test_ccxt_exec_report_conversion(self):
        instrument = lookup.find_symbol("BINANCE", "ACAUSDT")
        assert instrument is not None
        # - execution reports: (raw, status, side, unsigned qty, price, type) — quantity is
        #   unsigned for SELL too (the framework's positive-amount rule; direction lives in side)
        expected = [
            (C1, OrderStatus.ACCEPTED, "BUY", 50.0, None, "MARKET"),
            (C2, OrderStatus.FILLED, "BUY", 50.0, 0.1612, "MARKET"),
            (C3, OrderStatus.ACCEPTED, "SELL", 50.0, None, "MARKET"),
            (C4, OrderStatus.FILLED, "SELL", 50.0, 0.1606, "MARKET"),
            (C5new, OrderStatus.ACCEPTED, "BUY", 51.0, 0.1611, "LIMIT"),
            (C6ex, OrderStatus.FILLED, "BUY", 51.0, 0.1611, "LIMIT"),
            (C7cancel, OrderStatus.CANCELED, "BUY", 50.0, 0.1, "LIMIT"),
        ]
        for raw, status, side, quantity, price, order_type in expected:
            o = ccxt_convert_order_info(instrument, raw)
            assert o.venue_order_id == raw["id"]
            assert o.client_order_id == raw["clientOrderId"]
            assert o.status is status
            assert o.side == side
            assert o.quantity == quantity
            assert o.price == price
            assert o.type == order_type
            # fill state rides the unified ccxt fields (R2: snapshots must carry real fills)
            assert o.filled_quantity == (raw.get("filled") or 0.0)
            assert o.avg_fill_price == raw.get("average")

        # - historical records (spot ACAUSDT order history) all convert to recognized statuses
        hist_orders = [ccxt_convert_order_info(instrument, h) for h in HIST]
        assert [o.venue_order_id for o in hist_orders] == [h["id"] for h in HIST]
        assert all(o.status in (OrderStatus.ACCEPTED, OrderStatus.FILLED, OrderStatus.CANCELED) for o in hist_orders)

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
        deal = ccxt_convert_deal_info(raw)
        assert deal.trade_id == "56324015"
        assert deal.order_id == "536752004"
        assert deal.time == pd.Timestamp("2024-04-07 13:48:37.270000")
        assert deal.amount == 2.4
        assert deal.price == 2.1129
        assert deal.aggressive is True  # taker
        assert deal.fee_amount == N(6.48e-06)
        assert deal.fee_currency == "BNB"

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


def test_convert_order_info_maps_fill_state_and_unsigned_quantity():
    # R2: a partially-filled SELL must keep an unsigned quantity (positive-amount rule)
    # and carry the unified filled/average fields into the framework Order.
    instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    order = ccxt_convert_order_info(instrument, _raw_order(side="sell", filled=0.4, average=49_900.0))
    assert order.quantity == 1.0
    assert order.side == "SELL"
    assert order.filled_quantity == 0.4
    assert order.avg_fill_price == 49_900.0
    assert order.status is OrderStatus.PARTIALLY_FILLED


def test_convert_order_info_open_with_fills_maps_partially_filled_venue_agnostic():
    # ccxt's unified status collapses partial fills into "open"; OKX carries the venue
    # state under info.state (not info.status), so the refinement must not be the only
    # path — filled > 0 on an open order is the venue-agnostic signal (R2, OKX leg).
    instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    okx_like = _raw_order(side="buy", filled=0.25, average=50_010.0, info={"state": "partially_filled"})
    order = ccxt_convert_order_info(instrument, okx_like)
    assert order.status is OrderStatus.PARTIALLY_FILLED
    # an open order with no fills stays ACCEPTED
    untouched = ccxt_convert_order_info(instrument, _raw_order(filled=0.0, average=None))
    assert untouched.status is OrderStatus.ACCEPTED
    assert untouched.filled_quantity == 0.0
    assert untouched.avg_fill_price is None


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
