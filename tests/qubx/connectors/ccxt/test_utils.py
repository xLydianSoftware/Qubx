import gzip
import json

import pytest

from qubx.connectors.ccxt.utils import (
    ccxt_convert_balance,
    ccxt_convert_liquidation,
    ccxt_convert_orderbook,
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
