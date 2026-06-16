from dataclasses import dataclass

import pytest

from qubx.core.basics import Instrument, MarketType
from qubx.core.instrument_service import BlacklistEntry


def _instr(symbol: str, market_type: MarketType, exchange: str, base: str, quote: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=market_type,
        exchange=exchange,
        base=base,
        quote=quote,
        settle=quote,
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


BTC_SWAP = _instr("BTCUSDT", MarketType.SWAP, "BINANCE.UM", "BTC", "USDT")
ETH_SPOT = _instr("ETHUSDT", MarketType.SPOT, "BINANCE", "ETH", "USDT")


class TestBlacklistMatches:
    def test_exchange_mismatch_never_matches(self):
        e = BlacklistEntry(exchange="KRAKEN", market_type=None, asset="BTC", symbol=None)
        assert e.matches(BTC_SWAP) is False

    def test_wildcard_market_type_matches_any(self):
        e = BlacklistEntry(exchange="BINANCE.UM", market_type=None, asset="BTC", symbol=None)
        assert e.matches(BTC_SWAP) is True

    def test_market_type_filter_excludes_other_types(self):
        e = BlacklistEntry(exchange="BINANCE.UM", market_type="SWAP", asset="BTC", symbol=None)
        assert e.matches(BTC_SWAP) is True
        spot_btc = _instr("BTCUSDT", MarketType.SPOT, "BINANCE.UM", "BTC", "USDT")
        assert e.matches(spot_btc) is False

    def test_asset_only_case_insensitive(self):
        e = BlacklistEntry(exchange="BINANCE.UM", market_type=None, asset="btc", symbol=None)
        assert e.matches(BTC_SWAP) is True

    def test_symbol_only_case_insensitive(self):
        e = BlacklistEntry(exchange="BINANCE.UM", market_type=None, asset=None, symbol="btcusdt")
        assert e.matches(BTC_SWAP) is True
        assert e.matches(_instr("ETHUSDT", MarketType.SWAP, "BINANCE.UM", "ETH", "USDT")) is False

    def test_both_asset_and_symbol_must_match(self):
        e = BlacklistEntry(exchange="BINANCE.UM", market_type=None, asset="BTC", symbol="BTCUSDT")
        assert e.matches(BTC_SWAP) is True
        e2 = BlacklistEntry(exchange="BINANCE.UM", market_type=None, asset="BTC", symbol="BTCUSD")
        assert e2.matches(BTC_SWAP) is False

    def test_neither_asset_nor_symbol_matches_whole_exchange(self):
        e = BlacklistEntry(exchange="BINANCE.UM", market_type=None, asset=None, symbol=None)
        assert e.matches(BTC_SWAP) is True
        assert e.matches(ETH_SPOT) is False
