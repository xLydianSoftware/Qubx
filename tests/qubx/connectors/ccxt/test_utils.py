import copy
import gzip
import json

import pytest

from qubx.connectors.ccxt.utils import (
    ccxt_convert_balance,
    ccxt_convert_liquidation,
    ccxt_convert_orderbook,
    ccxt_convert_position,
    ccxt_convert_positions,
)
from qubx.core.lookups import lookup
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
