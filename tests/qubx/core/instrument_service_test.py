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


from qubx.core.instrument_service import (
    IInstrumentService,
    InstrumentServiceDiff,
    NullInstrumentService,
)


class TestNullInstrumentService:
    def test_is_iinstrument_service(self):
        svc = NullInstrumentService()
        assert isinstance(svc, IInstrumentService)

    def test_empty_entries(self):
        assert NullInstrumentService().get_blacklist_entries() == []

    def test_refresh_returns_empty_diff(self):
        diff = NullInstrumentService().refresh([BTC_SWAP, ETH_SPOT])
        assert isinstance(diff, InstrumentServiceDiff)
        assert diff.blacklisted_added == []
        assert diff.blacklisted_removed == []
        assert diff.entries_changed is False

    def test_is_blacklisted_always_false(self):
        assert NullInstrumentService().is_blacklisted(BTC_SWAP) is False

    def test_matching_instruments_empty(self):
        assert NullInstrumentService().matching_instruments([BTC_SWAP, ETH_SPOT]) == []


from unittest.mock import MagicMock

from qubx.core.instrument_service import HttpInstrumentService


def _resp(payload: dict) -> MagicMock:
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


class TestHttpInstrumentService:
    def test_refresh_parses_entries(self):
        client = MagicMock()
        client.get.return_value = _resp(
            {"entries": [{"exchange": "BINANCE.UM", "market_type": None, "asset": "BTC", "symbol": None}]}
        )
        svc = HttpInstrumentService(base_url="http://svc", exchanges=["BINANCE.UM"], client=client)
        svc.refresh([BTC_SWAP, ETH_SPOT])
        entries = svc.get_blacklist_entries()
        assert len(entries) == 1
        assert entries[0] == BlacklistEntry("BINANCE.UM", None, "BTC", None)

    def test_refresh_calls_correct_url_with_exchange_params(self):
        client = MagicMock()
        client.get.return_value = _resp({"entries": []})
        svc = HttpInstrumentService(base_url="http://svc", exchanges=["A", "B"], client=client)
        svc.refresh([])
        args, kwargs = client.get.call_args
        assert args[0] == "http://svc/internal/instrument-service/blacklist"
        assert kwargs["params"] == [("exchange", "A"), ("exchange", "B")]

    def test_is_blacklisted_and_matching(self):
        client = MagicMock()
        client.get.return_value = _resp(
            {"entries": [{"exchange": "BINANCE.UM", "market_type": "SWAP", "asset": "BTC", "symbol": None}]}
        )
        svc = HttpInstrumentService(base_url="http://svc", exchanges=["BINANCE.UM"], client=client)
        svc.refresh([BTC_SWAP, ETH_SPOT])
        assert svc.is_blacklisted(BTC_SWAP) is True
        assert svc.is_blacklisted(ETH_SPOT) is False
        assert svc.matching_instruments([BTC_SWAP, ETH_SPOT]) == [BTC_SWAP]

    def test_refresh_diff_added_then_removed(self):
        client = MagicMock()
        svc = HttpInstrumentService(base_url="http://svc", exchanges=["BINANCE.UM"], client=client)
        # first refresh: BTC blacklisted
        client.get.return_value = _resp(
            {"entries": [{"exchange": "BINANCE.UM", "market_type": None, "asset": "BTC", "symbol": None}]}
        )
        d1 = svc.refresh([BTC_SWAP, ETH_SPOT])
        assert d1.blacklisted_added == [BTC_SWAP]
        assert d1.blacklisted_removed == []
        # second refresh: BTC no longer blacklisted
        client.get.return_value = _resp({"entries": []})
        d2 = svc.refresh([BTC_SWAP, ETH_SPOT])
        assert d2.blacklisted_added == []
        assert d2.blacklisted_removed == [BTC_SWAP]

    def test_network_error_keeps_previous_cache(self):
        import httpx

        client = MagicMock()
        client.get.return_value = _resp(
            {"entries": [{"exchange": "BINANCE.UM", "market_type": None, "asset": "BTC", "symbol": None}]}
        )
        svc = HttpInstrumentService(base_url="http://svc", exchanges=["BINANCE.UM"], client=client)
        svc.refresh([BTC_SWAP, ETH_SPOT])
        assert svc.is_blacklisted(BTC_SWAP) is True
        # now the network fails -> cache must be preserved, diff empty
        client.get.side_effect = httpx.ConnectError("boom")
        diff = svc.refresh([BTC_SWAP, ETH_SPOT])
        assert diff.blacklisted_added == []
        assert diff.blacklisted_removed == []
        assert diff.entries_changed is False  # cache preserved -> no change
        assert svc.is_blacklisted(BTC_SWAP) is True

    def test_refresh_reports_entries_changed_for_off_universe_edit(self):
        # An entry for an instrument NOT in the known universe produces no universe diff,
        # but entries_changed must still flip so the manager can re-fit on add AND remove.
        client = MagicMock()
        svc = HttpInstrumentService(base_url="http://svc", exchanges=["BINANCE.UM"], client=client)
        universe = [ETH_SPOT]  # does not contain BTC
        client.get.return_value = _resp(
            {"entries": [{"exchange": "BINANCE.UM", "market_type": None, "asset": "BTC", "symbol": None}]}
        )
        d_add = svc.refresh(universe)
        assert d_add.blacklisted_added == [] and d_add.blacklisted_removed == []
        assert d_add.entries_changed is True
        # remove the off-universe entry: still no universe diff, but entries changed
        client.get.return_value = _resp({"entries": []})
        d_remove = svc.refresh(universe)
        assert d_remove.blacklisted_added == [] and d_remove.blacklisted_removed == []
        assert d_remove.entries_changed is True
        # a repeat refresh with identical entries is not a change
        client.get.return_value = _resp({"entries": []})
        d_same = svc.refresh(universe)
        assert d_same.entries_changed is False


from qubx.core.instrument_service import create_instrument_service


class TestFactory:
    def test_unset_env_returns_null_service(self, monkeypatch):
        monkeypatch.delenv("QUBX_INSTRUMENT_SERVICE_URL", raising=False)
        svc = create_instrument_service(["BINANCE.UM"])
        assert isinstance(svc, NullInstrumentService)

    def test_empty_env_returns_null_service(self, monkeypatch):
        monkeypatch.setenv("QUBX_INSTRUMENT_SERVICE_URL", "")
        svc = create_instrument_service(["BINANCE.UM"])
        assert isinstance(svc, NullInstrumentService)

    def test_set_env_returns_http_service(self, monkeypatch):
        monkeypatch.setenv("QUBX_INSTRUMENT_SERVICE_URL", "http://control-api:8080")
        svc = create_instrument_service(["BINANCE.UM"])
        assert isinstance(svc, HttpInstrumentService)
        assert svc.get_blacklist_entries() == []  # nothing fetched yet
