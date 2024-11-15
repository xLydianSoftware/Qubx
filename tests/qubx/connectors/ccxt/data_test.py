import pandas as pd
import json
import gzip

from qubx import lookup
from qubx.connectors.ccxt.ccxt_utils import ccxt_convert_orderbook


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
        assert ob.top_bid < ob.top_ask

        quote = ob.to_quote()
        assert quote.bid == ob.top_bid and quote.ask == ob.top_ask
        assert quote.mid_price() == ob.mid_price()
