import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from qubx.core.basics import AssetKind, MarketType, Underlying
from qubx.core.lookups import InstrumentsLookupService, LookupsManager, listing_to_instrument

# Recorded from the dev /internal/instrument-service/snapshot (2026-10-02) and trimmed. The
# BINANCE.UM POLUSDT listing carries a synthesized MATICUSDT predecessor version + alias: dev
# had no linked rename yet.
SNAPSHOT = json.loads((Path(__file__).parents[2] / "data" / "instrument_service" / "snapshot.json").read_text())


class _Service:
    """Minimal /snapshot endpoint: ETag over the body, 304 on If-None-Match."""

    def __init__(self, body: dict):
        self.body = body
        self.requests: list[dict] = []
        self.fail = False
        svc = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                svc.requests.append({"path": self.path, "headers": dict(self.headers)})
                if svc.fail:
                    self.send_response(500)
                    self.end_headers()
                    return
                data = json.dumps(svc.body).encode()
                etag = f'"{hash(data)}"'
                if self.headers.get("If-None-Match") == etag:
                    self.send_response(304)
                    self.send_header("ETag", etag)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("ETag", etag)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/internal/instrument-service"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def service():
    svc = _Service(json.loads(json.dumps(SNAPSHOT)))
    yield svc
    svc.close()


def _listing(exchange: str, symbol: str) -> dict:
    return next(l for l in SNAPSHOT["listings"] if l["exchange"] == exchange and l["versions"][-1]["symbol"] == symbol)


class TestListingToInstrument:
    def test_maps_static_and_current_version_fields(self):
        i = listing_to_instrument(_listing("BINANCE.UM", "BTCUSDT"))
        assert str(i) == "BINANCE.UM:SWAP:BTCUSDT"
        assert i.market_type == MarketType.SWAP
        assert (i.base, i.quote, i.settle) == ("BTC", "USDT", "USDT")
        assert i.underlying == Underlying(AssetKind.CRYPTO, "BTC")
        assert i.listing_id == _listing("BINANCE.UM", "BTCUSDT")["id"]
        assert i.contract_size == 1.0 and i.quantity_multiplier == 1.0
        assert i.tick_size > 0 and i.lot_size > 0
        assert i.venue_attributes["contractType"] == "PERPETUAL"

    def test_uses_the_current_version(self):
        raw = _listing("BINANCE", "GTCUSDT")
        assert len(raw["versions"]) > 1
        i = listing_to_instrument(raw)
        assert i.tick_size == raw["versions"][-1]["tick_size"]
        assert i.tick_size != raw["versions"][0]["tick_size"]

    def test_okx_contract_size_is_the_quantity_multiplier(self):
        i = listing_to_instrument(_listing("OKX.F", "INITUSDT"))
        assert i.contract_size == 10.0
        assert i.quantity_multiplier == 10.0

    def test_bstock_points_at_the_stock(self):
        i = listing_to_instrument(_listing("BINANCE", "SNDKBUSDT"))
        assert i.market_type == MarketType.SPOT
        assert i.base == "SNDKB"
        assert i.underlying == Underlying(AssetKind.EQUITY, "SNDK.XNAS")
        assert i.asset == "SNDK.XNAS"
        assert i.margin_tradable is True

    def test_multiplier_contract_asset_is_the_underlying_code(self):
        i = listing_to_instrument(_listing("BINANCE.UM", "1000PEPEUSDT"))
        assert i.base == "1000PEPE"
        assert i.asset == "PEPE"

    def test_delisted_listing_keeps_its_dates(self):
        i = listing_to_instrument(_listing("BINANCE", "VOXELUSDT"))
        assert i.delisted_at == pd.Timestamp("2025-12-18")
        assert i.listed_at == pd.Timestamp("2021-12-14")
        assert i.delisted_at.tzinfo is None

    def test_venue_attributes_are_read_only(self):
        i = listing_to_instrument(_listing("BINANCE.UM", "HK0625USDT"))
        assert i.venue_attributes["contractType"] == "TRADIFI_PERPETUAL"
        with pytest.raises(TypeError):
            i.venue_attributes["contractType"] = "X"  # type: ignore[index]


class TestInstrumentsLookupService:
    def test_loads_the_snapshot(self, service):
        lookup = InstrumentsLookupService(service.url)
        instruments = lookup.get_lookup()
        assert len(instruments) == len(SNAPSHOT["listings"])
        assert service.requests[0]["path"].startswith("/internal/instrument-service/snapshot")
        btc = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert btc is not None and btc.underlying == Underlying(AssetKind.CRYPTO, "BTC")
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT", MarketType.SPOT) is None
        assert lookup.find_symbol("BINANCE", "BTCUSDT", MarketType.SPOT) is not None

    def test_exchange_filter_and_token(self, service):
        InstrumentsLookupService(service.url, token="secret", exchanges=["OKX.F", "BINANCE"])
        req = service.requests[0]
        assert "exchange=OKX.F" in req["path"] and "exchange=BINANCE" in req["path"]
        assert req["headers"]["Authorization"] == "Bearer secret"

    def test_no_auth_header_without_token(self, service):
        InstrumentsLookupService(service.url)
        assert "Authorization" not in service.requests[0]["headers"]

    def test_former_symbol_resolves_to_the_current_instrument(self, service):
        lookup = InstrumentsLookupService(service.url)
        current = lookup.find_symbol("BINANCE.UM", "POLUSDT")
        assert current is not None
        assert lookup.find_symbol("BINANCE.UM", "MATICUSDT") is current
        assert lookup.find_symbol("BINANCE.UM", "MATICUSDT", MarketType.SWAP) is current
        assert "BINANCE.UM:SWAP:MATICUSDT" not in lookup.get_lookup()

    def test_a_current_symbol_wins_over_an_alias(self, service):
        reused = json.loads(json.dumps(_listing("BINANCE.UM", "BTCUSDT")))
        reused["id"] = "00000000-0000-0000-0000-000000000001"
        for v in reused["versions"]:
            v["symbol"] = "MATICUSDT"
        service.body["listings"].append(reused)
        lookup = InstrumentsLookupService(service.url)
        assert lookup.find_symbol("BINANCE.UM", "MATICUSDT").listing_id == reused["id"]

    def test_an_active_listing_wins_over_a_delisted_one_with_the_same_symbol(self, service):
        old = json.loads(json.dumps(_listing("BINANCE.UM", "BTCUSDT")))
        old["id"] = "00000000-0000-0000-0000-000000000002"
        old["listed_at"], old["delisted_at"] = "2019-01-01T00:00:00Z", "2019-06-01T00:00:00Z"
        service.body["listings"].append(old)
        service.body["listings"].reverse()
        lookup = InstrumentsLookupService(service.url)
        current = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert current.listing_id == _listing("BINANCE.UM", "BTCUSDT")["id"]
        assert lookup.get_lookup()["BINANCE.UM:SWAP:BTCUSDT"] is current
        assert len(lookup.get_lookup()) == len(SNAPSHOT["listings"])

    def test_find_instruments_matches_the_underlying_code(self, service):
        lookup = InstrumentsLookupService(service.url)
        found = lookup.find_instruments("BINANCE.UM", base="PEPE")
        assert [i.symbol for i in found] == ["1000PEPEUSDT"]

    def test_find_instruments_as_of_excludes_delisted(self, service):
        lookup = InstrumentsLookupService(service.url)
        assert lookup.find_instruments("BINANCE", base="VOXEL", as_of="2025-01-01")
        assert not lookup.find_instruments("BINANCE", base="VOXEL", as_of="2026-01-01")

    def test_refresh_not_modified_keeps_the_copy(self, service):
        lookup = InstrumentsLookupService(service.url, reload_interval="1ms")
        before = lookup.get_lookup()
        lookup.refresh()
        assert service.requests[-1]["headers"]["If-None-Match"]
        assert lookup.get_lookup() is before

    def test_refresh_picks_up_changes(self, service):
        lookup = InstrumentsLookupService(service.url)
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT").tick_size != 0.5
        service.body["listings"] = [l for l in service.body["listings"] if l["exchange"] == "BINANCE.UM"]
        for l in service.body["listings"]:
            if l["versions"][-1]["symbol"] == "BTCUSDT":
                l["versions"][-1]["tick_size"] = 0.5
        lookup.refresh()
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT").tick_size == 0.5
        assert lookup.find_symbol("OKX.F", "BTCUSDT") is None

    def test_refresh_runs_on_access_after_the_interval(self, service):
        lookup = InstrumentsLookupService(service.url, reload_interval="1h")
        lookup.get_lookup()
        assert len(service.requests) == 1
        lookup._last_refresh = pd.Timestamp.now() - pd.Timedelta("2h")
        lookup.get_lookup()
        assert len(service.requests) == 2

    def test_failed_refresh_keeps_the_copy(self, service):
        lookup = InstrumentsLookupService(service.url)
        service.fail = True
        lookup.refresh()
        assert len(lookup.get_lookup()) == len(SNAPSHOT["listings"])
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT") is not None

    def test_startup_fails_when_unreachable(self, service):
        url = service.url
        service.close()
        with pytest.raises(RuntimeError, match="instrument service"):
            InstrumentsLookupService(url, timeout=2)

    def test_startup_fails_on_server_error(self, service):
        service.fail = True
        with pytest.raises(RuntimeError, match="instrument service"):
            InstrumentsLookupService(service.url)

    def test_unknown_kind_is_skipped(self, service):
        odd = json.loads(json.dumps(_listing("BINANCE.UM", "BTCUSDT")))
        odd["underlying"]["kind"] = "WEATHER"
        odd["versions"][-1]["symbol"] = "RAINUSDT"
        service.body["listings"].append(odd)
        lookup = InstrumentsLookupService(service.url)
        assert lookup.find_symbol("BINANCE.UM", "RAINUSDT") is None
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT") is not None


class TestLookupsManagerFactory:
    def test_service_type_builds_the_service_lookup(self, service):
        lookup = LookupsManager._get_instrument_lookup("service", url=service.url, path="/ignored")
        assert isinstance(lookup, InstrumentsLookupService)

    def test_service_type_requires_a_url(self):
        with pytest.raises(ValueError, match="url"):
            LookupsManager._get_instrument_lookup("service")

    def test_mongo_type_is_gone(self):
        with pytest.raises(ValueError, match="removed"):
            LookupsManager._get_instrument_lookup("mongo")
