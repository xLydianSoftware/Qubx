import copy
import gzip
import json

import pytest

from qubx.connectors.ccxt.utils import (
    FRAMEWORK_ONLY_OPTIONS,
    ccxt_convert_balance,
    ccxt_convert_liquidation,
    ccxt_convert_order_info,
    ccxt_convert_orderbook,
    ccxt_convert_position,
    ccxt_convert_positions,
    prepare_ccxt_order_payload,
)
from qubx.core.basics import (
    OPTION_AVOID_STOP_ORDER_PRICE_VALIDATION,
    OPTION_FILL_AT_SIGNAL_PRICE,
    OPTION_SIGNAL_PRICE,
    OPTION_SKIP_PRICE_CROSS_CONTROL,
)
from qubx.core.exceptions import InvalidOrderParameters
from qubx.core.lookups import lookup
from qubx.core.series import Quote
from qubx.utils.marketdata.ccxt import ccxt_symbol_to_instrument
from tests.qubx.connectors.ccxt.data.ccxt_responses import (
    BALANCE_BINANCE_MARGIN,
    BINANCE_MARKETS,
    L1,
    M1,
    POSITIONS_BINANCE_UM,
)


class TestCcxtOrderbookRelatedStuff:
    def test_ccxt_orderbook_conversion(self):
        i1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert i1 is not None

        orderbooks_path = "tests/data/BTCUSDT_ccxt_orderbooks.txt.gz"

        with gzip.open(orderbooks_path, "rt") as f:
            orderbooks = [json.loads(line) for line in f]
        print(f"Loaded {len(orderbooks)} orderbooks")

        obs = [ccxt_convert_orderbook(ob, i1) for ob in orderbooks]

        assert len(obs) == len(orderbooks)
        assert all([o is not None for o in obs])

        ob = obs[0]
        assert ob is not None
        assert ob.top_bid < ob.top_ask

        quote = ob.to_quote()
        assert quote.bid == ob.top_bid and quote.ask == ob.top_ask
        assert quote.mid_price() == ob.mid_price()

    def test_ccxt_orderbook_conversion_with_zero_tick_size_pct(self):
        i1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert i1 is not None

        orderbooks_path = "tests/data/BTCUSDT_ccxt_orderbooks.txt.gz"

        with gzip.open(orderbooks_path, "rt") as f:
            orderbooks = [json.loads(line) for line in f][:5]  # Just use first 5 for this test

        # Test with tick_size_pct=0 to use instrument's tick_size
        obs = [ccxt_convert_orderbook(ob, i1, tick_size_pct=0) for ob in orderbooks]

        assert len(obs) == len(orderbooks)
        assert all([o is not None for o in obs])

        # Verify that all orderbooks use the instrument's tick size
        for ob in obs:
            assert ob is not None
            assert ob.tick_size == i1.tick_size

    def test_ccxt_liquidation_conversion(self):
        liquidations = []
        for ccxt_liq in L1:
            liquidations.append(ccxt_convert_liquidation(ccxt_liq))
        assert len(liquidations) == len(L1)

    def test_ccxt_symbol_conversion(self):
        instr = ccxt_symbol_to_instrument("BINANCE.UM", M1)
        assert instr is not None
        assert instr.symbol == "BTCUSDT"

    def test_ccxt_balance_conversion(self):
        balances = ccxt_convert_balance(BALANCE_BINANCE_MARGIN, "BINANCE")
        # Convert list to dict for easier testing
        balance_dict = {b.currency: b for b in balances}
        assert "USDT" in balance_dict and "ETH" in balance_dict
        assert balance_dict["USDT"].total == pytest.approx(642.657)
        assert balance_dict["ETH"].total == pytest.approx(0.10989)
        # Verify exchange is set correctly
        assert all(b.exchange == "BINANCE" for b in balances)

    def test_ccxt_position_conversion(self):
        positions = ccxt_convert_positions(POSITIONS_BINANCE_UM, "BINANCE.UM", BINANCE_MARKETS)
        assert len(positions) > 0

    def test_ccxt_position_unknown_symbol_is_skipped(self):
        """An unknown symbol yields no Position rather than blowing up the caller."""
        assert ccxt_convert_position({"symbol": "XYZ/USDT:USDT"}, "BINANCE.UM", BINANCE_MARKETS) is None

    def test_ccxt_positions_skips_only_the_unknown_row(self):
        """One unconvertible row must not cost the rows that did convert."""
        positions = ccxt_convert_positions(
            [*POSITIONS_BINANCE_UM, {"symbol": "XYZ/USDT:USDT"}], "BINANCE.UM", BINANCE_MARKETS
        )
        assert len(positions) == len(POSITIONS_BINANCE_UM)

    def test_ccxt_position_takes_venue_margins(self):
        """Both margins must come from the venue, which knows the leverage tier.

        Regression: only maintenanceMargin was read, so initial_margin fell back to the internal
        calc -> 0.0 whenever instrument.initial_margin metadata is absent (it is 0.0 for
        BINANCE.UM), and get_total_initial_margin under-reported the whole ccxt side.
        """
        info = POSITIONS_BINANCE_UM[0]
        pos = ccxt_convert_position(info, "BINANCE.UM", BINANCE_MARKETS)
        assert pos is not None
        assert pos.initial_margin == pytest.approx(float(info["initialMargin"]))
        assert pos.maint_margin == pytest.approx(float(info["maintenanceMargin"]))
        # external flags -> price updates must not recalculate these away
        assert pos._initial_margin_external is True
        assert pos._maint_margin_external is True

    def test_ccxt_position_margins_absent_leaves_defaults(self):
        """A venue that omits the margin fields must not blow up or fake a value."""
        info = {k: v for k, v in POSITIONS_BINANCE_UM[0].items() if k not in ("initialMargin", "maintenanceMargin")}
        pos = ccxt_convert_position(info, "BINANCE.UM", BINANCE_MARKETS)
        assert pos is not None
        assert pos._initial_margin_external is False
        assert pos._maint_margin_external is False

    def test_ccxt_position_adl_from_v3_payload(self):
        """fetch_positions defaults to Binance v3 positionRisk, where the ADL field is named `adl`.

        Confirmed against a live account 2026-07-16: raw payload had `adl: 3` and no `adlQuantile`.
        Binance scale is 0..4, higher = closer to the front of the ADL queue.
        """
        info = copy.deepcopy(POSITIONS_BINANCE_UM[0])
        info["info"].pop("leverage", None)  # v3 dropped these two
        info["info"].pop("maxNotionalValue", None)
        info["info"]["adl"] = "3"
        pos = ccxt_convert_position(info, "BINANCE.UM", BINANCE_MARKETS)
        assert pos is not None
        assert pos.adl_level == 3

    def test_ccxt_position_adl_from_v2_payload(self):
        """v2 positionRisk (params.useV2) spells the same value `adlQuantile`."""
        info = copy.deepcopy(POSITIONS_BINANCE_UM[0])
        info["info"]["adlQuantile"] = "2"
        pos = ccxt_convert_position(info, "BINANCE.UM", BINANCE_MARKETS)
        assert pos is not None
        assert pos.adl_level == 2

    def test_ccxt_position_adl_absent_stays_none(self):
        """No ADL field -> None. None means 'venue reported no rank', not 'no ADL risk'."""
        info = copy.deepcopy(POSITIONS_BINANCE_UM[0])
        pos = ccxt_convert_position(info, "BINANCE.UM", BINANCE_MARKETS)
        assert pos is not None
        assert pos.adl_level is None

    def test_ccxt_position_adl_zero_is_kept(self):
        """0 is a real Binance rank (safest bucket) — it must not be dropped as falsy."""
        info = copy.deepcopy(POSITIONS_BINANCE_UM[0])
        info["info"]["adl"] = "0"
        pos = ccxt_convert_position(info, "BINANCE.UM", BINANCE_MARKETS)
        assert pos is not None
        assert pos.adl_level == 0


def _instrument():
    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert instr is not None
    return instr


def _quote(bid: float = 49_990.0, ask: float = 50_010.0) -> Quote:
    return Quote(0, bid, ask, 1.0, 1.0)


def _payload(order_type: str, price: float | None, side: str = "BUY", tif: str = "gtc", **kwargs):
    return prepare_ccxt_order_payload(
        instrument=_instrument(),
        order_side=side,
        order_type=order_type,
        amount=kwargs.pop("amount", 0.01),
        price=price,
        client_id="qubx_BTCUSDT_1",
        time_in_force=tif,
        quote=kwargs.pop("quote", None) or _quote(),
        reduce_only=kwargs.pop("reduce_only", False),
    )


class TestPrepareCcxtOrderPayloadTriggers:
    """A trigger MARKET order must be built like a plain MARKET order plus a trigger level."""

    def test_stop_market_carries_no_tif_and_no_post_only(self):
        # ccxt drops `price` on a market type
        p = _payload("STOP_MARKET", price=48_000.0, side="SELL")
        assert p["type"] == "market"
        assert p["params"]["triggerPrice"] == 48_000.0
        # framework convention: a stop's Order.price IS its trigger (see ccxt_convert_order_info)
        assert p["price"] == 48_000.0
        assert "timeInForce" not in p["params"]
        assert "postOnly" not in p["params"]

    def test_gtx_stop_market_trigger_is_not_repriced(self):
        # GTX repricing is a limit-book rule; a BUY stop rests above the ask by construction
        p = _payload("STOP_MARKET", price=50_050.0, side="BUY", tif="gtx")
        assert p["price"] == 50_050.0
        assert p["params"]["triggerPrice"] == 50_050.0
        assert "postOnly" not in p["params"]

    def test_stop_market_without_price_raises(self):
        with pytest.raises(InvalidOrderParameters):
            _payload("STOP_MARKET", price=None)

    def test_stop_limit_still_carries_tif_and_trigger(self):
        p = _payload("STOP_LIMIT", price=48_000.0, side="SELL")
        assert p["type"] == "limit"
        assert p["params"]["triggerPrice"] == 48_000.0
        assert p["params"]["timeInForce"] == "GTC"

    def test_plain_limit_unchanged(self):
        p = _payload("LIMIT", price=49_000.0)
        assert p["type"] == "limit"
        assert p["params"]["timeInForce"] == "GTC"
        assert "triggerPrice" not in p["params"]

    def test_gtx_limit_is_post_only_and_strict_by_default(self):
        p = _payload("LIMIT", price=50_050.0, side="BUY", tif="gtx")
        assert p["params"]["postOnly"] is True
        assert "timeInForce" not in p["params"]
        assert p["price"] == pytest.approx(50_050.0)


def test_framework_only_options_are_the_ones_no_venue_takes():
    assert {
        OPTION_FILL_AT_SIGNAL_PRICE,
        OPTION_SIGNAL_PRICE,
        OPTION_SKIP_PRICE_CROSS_CONTROL,
        OPTION_AVOID_STOP_ORDER_PRICE_VALIDATION,
        "stop_type",
    } <= FRAMEWORK_ONLY_OPTIONS
    # the connector resolves these into the payload itself rather than forwarding them raw
    assert {"reduceOnly", "reduce_only", "post_only"} <= FRAMEWORK_ONLY_OPTIONS
    assert FRAMEWORK_ONLY_OPTIONS.isdisjoint(
        {"postOnly", "triggerBy", "triggerDirection", "workingType", "lighter_client_order_index"}
    )


def _raw_order(**overrides):
    raw = {
        "info": {},
        "id": "VENUE-1",
        "clientOrderId": "qubx_BTCUSDT_1",
        "amount": 1.0,
        "price": 50_000.0,
        "status": "open",
        "side": "buy",
        "type": "limit",
        "timeInForce": "GTC",
        "timestamp": 1_716_854_400_000,
    }
    raw.update(overrides)
    return raw


class TestCcxtOrderReadBackTypes:
    """Venues type a conditional by how it executes, so a resting stop reads back as MARKET/LIMIT
    while cancel_order and request_order_status route on the order type."""

    def test_conditional_market_row_reads_as_stop_market(self):
        order = ccxt_convert_order_info(_instrument(), _raw_order(type="market", price=None, triggerPrice="57966.5"))
        assert order.type == "STOP_MARKET"
        assert order.price == 57966.5

    def test_conditional_limit_row_reads_as_stop_limit_and_keeps_its_limit_price(self):
        order = ccxt_convert_order_info(_instrument(), _raw_order(type="limit", price=49_000.0, triggerPrice=48_000.0))
        assert order.type == "STOP_LIMIT"
        assert order.price == 49_000.0

    def test_bybit_reduce_only_stop_is_retyped_despite_the_backfilled_stop_loss_price(self):
        # ccxt's bybit parse_order copies triggerPrice into stopLossPrice on a reduce-only conditional
        raw = _raw_order(type="market", price=None, triggerPrice="57966.5", stopLossPrice="57966.5", reduceOnly=True)
        order = ccxt_convert_order_info(_instrument(), raw)
        assert order.type == "STOP_MARKET"
        assert order.reduce_only is True

    def test_binance_pm_algo_row_reads_as_stop_market(self):
        # parse_algo_order emits "market"/"limit" with the trigger under triggerPrice
        raw = _raw_order(type="market", price=None, triggerPrice=0.05064, info={"algoStatus": "NEW"})
        assert ccxt_convert_order_info(_instrument(), raw).type == "STOP_MARKET"

    def test_binance_um_stop_market_row_reads_as_stop_market(self):
        # ccxt's binance parse_order_type collapses stop_market -> market, stop -> limit
        raw = _raw_order(type="market", price=None, triggerPrice=48_000.0, info={"origType": "STOP_MARKET"})
        assert ccxt_convert_order_info(_instrument(), raw).type == "STOP_MARKET"

    def test_binance_um_trailing_stop_row_is_not_retyped(self):
        # a trailing stop carries stopPrice="0" with its activation under activatePrice
        raw = _raw_order(type="market", price=None, info={"origType": "TRAILING_STOP_MARKET", "stopPrice": "0"})
        assert ccxt_convert_order_info(_instrument(), raw).type == "MARKET"

    def test_plain_rows_are_untouched(self):
        assert ccxt_convert_order_info(_instrument(), _raw_order()).type == "LIMIT"
        plain_market = _raw_order(type="market", price=None, info={"stopPrice": "0"})
        order = ccxt_convert_order_info(_instrument(), plain_market)
        assert order.type == "MARKET"
        assert order.price is None

    def test_entry_order_with_attached_tp_sl_is_not_retyped(self):
        # a position-attached TP/SL rides the ENTRY order's takeProfitPrice/stopLossPrice
        raw = _raw_order(takeProfitPrice=60_000.0, stopLossPrice=40_000.0)
        assert ccxt_convert_order_info(_instrument(), raw).type == "LIMIT"

    def test_already_typed_stop_is_left_alone(self):
        # OKX's own parse_order override retypes before the converter sees the row
        raw = _raw_order(type="stop_market", price=None, triggerPrice=57966.5)
        assert ccxt_convert_order_info(_instrument(), raw).type == "STOP_MARKET"


class TestCcxtOrderReadBackFlags:
    def test_post_only_lands_on_the_boolean_the_send_path_already_sets(self):
        """The venue's TIF spelling is passed through as reported; post_only is the one channel."""
        for spelling in ("PO", "GTX", "Alo", "post_only"):
            order = ccxt_convert_order_info(_instrument(), _raw_order(timeInForce=spelling))
            assert order.post_only is True
            assert order.time_in_force == spelling

    def test_a_plain_order_is_not_post_only(self):
        assert ccxt_convert_order_info(_instrument(), _raw_order(timeInForce="GTC")).post_only is False

    def test_other_tifs_pass_through_untouched(self):
        assert ccxt_convert_order_info(_instrument(), _raw_order(timeInForce="GTC")).time_in_force == "GTC"
        assert ccxt_convert_order_info(_instrument(), _raw_order(timeInForce="IOC")).time_in_force == "IOC"
        assert ccxt_convert_order_info(_instrument(), _raw_order(timeInForce=None)).time_in_force is None

    def test_reduce_only_lands_on_the_field_not_only_in_options(self):
        assert ccxt_convert_order_info(_instrument(), _raw_order(reduceOnly=True)).reduce_only is True
        assert ccxt_convert_order_info(_instrument(), _raw_order(reduceOnly=False)).reduce_only is False
        assert ccxt_convert_order_info(_instrument(), _raw_order()).reduce_only is False
