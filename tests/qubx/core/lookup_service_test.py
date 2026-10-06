import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from qubx.core.basics import AssetKind, MarketType, Underlying
from qubx.core.lookups import InstrumentsLookupService, LookupsManager, _InstrumentMapper

# Recorded from the dev /internal/instrument-service/snapshot (2026-10-02) and trimmed. The
# BINANCE.UM POLUSDT listing carries a synthesized MATICUSDT predecessor version + alias: dev
# had no linked rename yet.
_DATA = Path(__file__).parents[2] / "data" / "instrument_service"
SNAPSHOT = json.loads((_DATA / "snapshot.json").read_text())
# Hand-built from the doc's mapping table (dev ingests none of these yet): a quarterly, an option,
# a COIN-M inverse perp, a TradFi perp with a calendar and a listing with no versions.
SYNTHETIC = json.loads((_DATA / "synthetic.json").read_text())


class _Service:
    """Minimal /snapshot endpoint: ETag over the body, 304 on If-None-Match."""

    def __init__(self, body: dict):
        self.body = body
        self.requests: list[dict] = []
        self.fail = False
        self.fail_first = 0
        self.delay = 0.0
        self.raw: bytes | None = None
        svc = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                svc.requests.append({"path": self.path, "headers": dict(self.headers)})
                time.sleep(svc.delay)
                if svc.fail or len(svc.requests) <= svc.fail_first:
                    self.send_response(500)
                    self.end_headers()
                    return
                data = svc.raw if svc.raw is not None else json.dumps(svc.body).encode()
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


@pytest.fixture
def lookups():
    made: list[InstrumentsLookupService] = []

    def make(*args, **kwargs) -> InstrumentsLookupService:
        kwargs.setdefault("retry_backoff", 0.01)
        made.append(InstrumentsLookupService(*args, **kwargs))
        return made[-1]

    yield make
    for lk in made:
        lk.close()


def _synthetic(listing_id_suffix: str) -> dict:
    return next(l for l in SYNTHETIC["listings"] if l["id"].endswith(listing_id_suffix))


def _listing(exchange: str, symbol: str) -> dict:
    return next(l for l in SNAPSHOT["listings"] if l["exchange"] == exchange and l["versions"][-1]["symbol"] == symbol)


class TestListingToInstrument:
    def test_maps_static_and_current_version_fields(self):
        i = _InstrumentMapper.from_listing(_listing("BINANCE.UM", "BTCUSDT"))
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
        i = _InstrumentMapper.from_listing(raw)
        assert i.tick_size == raw["versions"][-1]["tick_size"]
        assert i.tick_size != raw["versions"][0]["tick_size"]

    def test_okx_contract_size_is_the_quantity_multiplier(self):
        i = _InstrumentMapper.from_listing(_listing("OKX.F", "INITUSDT"))
        assert i.contract_size == 10.0
        assert i.quantity_multiplier == 10.0

    def test_bstock_points_at_the_stock(self):
        i = _InstrumentMapper.from_listing(_listing("BINANCE", "SNDKBUSDT"))
        assert i.market_type == MarketType.SPOT
        assert i.base == "SNDKB"
        assert i.underlying == Underlying(AssetKind.EQUITY, "SNDK.XNAS")
        assert i.exposure_code == "SNDK.XNAS"
        assert i.asset == "SNDKB"
        assert i.margin_tradable is True

    def test_multiplier_contract_asset_strips_the_multiplier(self):
        i = _InstrumentMapper.from_listing(_listing("BINANCE.UM", "1000PEPEUSDT"))
        assert i.base == "1000PEPE"
        assert i.asset == "PEPE"

    def test_delisted_listing_keeps_its_dates(self):
        i = _InstrumentMapper.from_listing(_listing("BINANCE", "VOXELUSDT"))
        assert i.delisted_at == pd.Timestamp("2025-12-18")
        assert i.listed_at == pd.Timestamp("2021-12-14")
        assert i.delisted_at.tzinfo is None

    def test_future_carries_its_expiry(self):
        i = _InstrumentMapper.from_listing(_synthetic("000000000001"))
        assert i.market_type == MarketType.FUTURE
        assert i.symbol == "BTCUSDT.20261225"
        assert i.expiry == pd.Timestamp("2026-12-25 08:00")
        assert i.strike is None and i.option_right is None

    def test_option_carries_strike_and_right(self):
        i = _InstrumentMapper.from_listing(_synthetic("000000000002"))
        assert i.market_type == MarketType.OPTION
        assert i.strike == 85000.0 and isinstance(i.strike, float)
        assert i.option_right == "CALL"
        assert i.contract_size == 0.1

    def test_coin_m_inverse_contract_size_is_in_quote_units(self):
        i = _InstrumentMapper.from_listing(_synthetic("000000000003"))
        assert i.inverse is True
        assert (i.quote, i.settle) == ("USD", "BTC")
        assert i.contract_size == 100.0 and i.quantity_multiplier == 100.0

    def test_tradfi_perp_carries_its_calendar_and_underlying(self):
        i = _InstrumentMapper.from_listing(_synthetic("000000000004"))
        assert i.calendar == "24/7"
        assert i.underlying == Underlying(AssetKind.EQUITY, "TSLA.XNAS")
        assert (i.asset, i.exposure_code) == ("TSLA", "TSLA.XNAS")

    def test_a_listing_without_versions_raises(self):
        with pytest.raises(IndexError):
            _InstrumentMapper.from_listing(_synthetic("000000000005"))

    def test_venue_attributes_are_read_only(self):
        i = _InstrumentMapper.from_listing(_listing("BINANCE.UM", "HK0625USDT"))
        assert i.venue_attributes["contractType"] == "TRADIFI_PERPETUAL"
        with pytest.raises(TypeError):
            i.venue_attributes["contractType"] = "X"  # type: ignore[index]


class TestInstrumentsLookupService:
    def test_loads_the_snapshot(self, service, lookups):
        lookup = lookups(service.url)
        instruments = lookup.get_lookup()
        assert len(instruments) == len(SNAPSHOT["listings"])
        assert service.requests[0]["path"].startswith("/internal/instrument-service/snapshot")
        btc = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert btc is not None and btc.underlying == Underlying(AssetKind.CRYPTO, "BTC")
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT", MarketType.SPOT) is None
        assert lookup.find_symbol("BINANCE", "BTCUSDT", MarketType.SPOT) is not None

    def test_exchange_filter_and_token(self, service, lookups):
        lookups(service.url, token="secret", exchanges=["OKX.F", "BINANCE"])
        req = service.requests[0]
        assert "exchange=OKX.F" in req["path"] and "exchange=BINANCE" in req["path"]
        assert req["headers"]["Authorization"] == "Bearer secret"

    def test_no_auth_header_without_token(self, service, lookups):
        lookups(service.url)
        assert "Authorization" not in service.requests[0]["headers"]

    def test_former_symbol_resolves_to_the_current_instrument(self, service, lookups):
        lookup = lookups(service.url)
        current = lookup.find_symbol("BINANCE.UM", "POLUSDT")
        assert current is not None
        assert lookup.find_symbol("BINANCE.UM", "MATICUSDT") is current
        assert lookup.find_symbol("BINANCE.UM", "MATICUSDT", MarketType.SWAP) is current
        assert "BINANCE.UM:SWAP:MATICUSDT" not in lookup.get_lookup()

    def test_a_current_symbol_wins_over_an_alias(self, service, lookups):
        reused = json.loads(json.dumps(_listing("BINANCE.UM", "BTCUSDT")))
        reused["id"] = "00000000-0000-0000-0000-000000000001"
        for v in reused["versions"]:
            v["symbol"] = "MATICUSDT"
        service.body["listings"].append(reused)
        lookup = lookups(service.url)
        assert lookup.find_symbol("BINANCE.UM", "MATICUSDT").listing_id == reused["id"]

    def test_an_active_listing_wins_over_a_delisted_one_with_the_same_symbol(self, service, lookups):
        old = json.loads(json.dumps(_listing("BINANCE.UM", "BTCUSDT")))
        old["id"] = "00000000-0000-0000-0000-000000000002"
        old["listed_at"], old["delisted_at"] = "2019-01-01T00:00:00Z", "2019-06-01T00:00:00Z"
        service.body["listings"].append(old)
        service.body["listings"].reverse()
        lookup = lookups(service.url)
        current = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert current.listing_id == _listing("BINANCE.UM", "BTCUSDT")["id"]
        assert lookup.get_lookup()["BINANCE.UM:SWAP:BTCUSDT"] is current
        assert len(lookup.get_lookup()) == len(SNAPSHOT["listings"])

    def test_a_relisted_symbol_keeps_every_listing_for_as_of_lookups(self, service, lookups):
        current_id = _listing("BINANCE.UM", "BTCUSDT")["id"]
        old = json.loads(json.dumps(_listing("BINANCE.UM", "BTCUSDT")))
        old["id"] = "00000000-0000-0000-0000-000000000003"
        old["listed_at"], old["delisted_at"] = "2015-01-01T00:00:00Z", "2016-01-01T00:00:00Z"
        service.body["listings"].append(old)
        lookup = lookups(service.url)

        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT").listing_id == current_id
        assert [i.listing_id for i in lookup.find_listings("BINANCE.UM", "BTCUSDT")] == [current_id, old["id"]]
        assert [i.listing_id for i in lookup.find_instruments("BINANCE.UM", base="BTC")] == [current_id]
        during_old = lookup.find_instruments("BINANCE.UM", base="BTC", quote="USDT", as_of="2015-06-01")
        assert [i.listing_id for i in during_old] == [old["id"]]
        at_delisting = lookup.find_instruments("BINANCE.UM", base="BTC", quote="USDT", as_of="2016-01-01")
        assert old["id"] not in [i.listing_id for i in at_delisting]
        assert [i.listing_id for i in lookup.find_instruments("BINANCE.UM", base="BTC", as_of="2026-01-01")] == [
            current_id
        ]

    def test_find_instruments_matches_the_coin_under_a_multiplier(self, service, lookups):
        lookup = lookups(service.url)
        found = lookup.find_instruments("BINANCE.UM", base="PEPE")
        assert [i.symbol for i in found] == ["1000PEPEUSDT"]

    def test_find_instruments_as_of_excludes_delisted(self, service, lookups):
        lookup = lookups(service.url)
        assert lookup.find_instruments("BINANCE", base="VOXEL", as_of="2025-01-01")
        assert not lookup.find_instruments("BINANCE", base="VOXEL", as_of="2026-01-01")

    def test_refresh_not_modified_keeps_the_copy(self, service, lookups):
        lookup = lookups(service.url)
        before = lookup.get_lookup()
        assert lookup.refresh() is False
        assert service.requests[-1]["headers"]["If-None-Match"]
        assert lookup.get_lookup() is before

    def test_refresh_picks_up_changes(self, service, lookups):
        lookup = lookups(service.url)
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT").tick_size != 0.5
        service.body["listings"] = [l for l in service.body["listings"] if l["exchange"] == "BINANCE.UM"]
        for l in service.body["listings"]:
            if l["versions"][-1]["symbol"] == "BTCUSDT":
                l["versions"][-1]["tick_size"] = 0.5
        lookup.refresh()
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT").tick_size == 0.5
        assert lookup.find_symbol("OKX.F", "BTCUSDT") is None

    def test_failed_refresh_keeps_the_copy(self, service, lookups):
        lookup = lookups(service.url)
        service.fail = True
        lookup.refresh()
        assert len(lookup.get_lookup()) == len(SNAPSHOT["listings"])
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT") is not None

    def test_startup_fails_when_unreachable(self, service, lookups):
        url = service.url
        service.close()
        with pytest.raises(RuntimeError, match="instrument service"):
            lookups(url, timeout=2)
        assert len(service.requests) == 0

    def test_startup_fails_on_server_error(self, service, lookups):
        service.fail = True
        with pytest.raises(RuntimeError, match="instrument service"):
            lookups(service.url)

    def test_unknown_kind_is_skipped(self, service, lookups):
        odd = json.loads(json.dumps(_listing("BINANCE.UM", "BTCUSDT")))
        odd["underlying"]["kind"] = "WEATHER"
        odd["versions"][-1]["symbol"] = "RAINUSDT"
        service.body["listings"].append(odd)
        lookup = lookups(service.url)
        assert lookup.find_symbol("BINANCE.UM", "RAINUSDT") is None
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT") is not None

    def test_first_load_retries_before_failing(self, service, lookups):
        service.fail_first = 2
        lookup = lookups(service.url)
        assert len(service.requests) == 3
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT") is not None

    def test_first_load_gives_up_after_three_attempts(self, service, lookups):
        service.fail = True
        with pytest.raises(RuntimeError):
            lookups(service.url)
        assert len(service.requests) == 3

    def test_startup_fails_on_timeout(self, service, lookups):
        service.delay = 1.0
        with pytest.raises(RuntimeError, match="instrument service"):
            lookups(service.url, timeout=0.2)

    def test_startup_fails_on_truncated_json(self, service, lookups):
        service.raw = json.dumps(SNAPSHOT).encode()[:500]
        with pytest.raises(RuntimeError, match="instrument service"):
            lookups(service.url)

    def test_startup_fails_on_a_body_without_listings(self, service, lookups):
        service.raw = b'{"items": []}'
        with pytest.raises(RuntimeError, match="no listings"):
            lookups(service.url)

    def test_startup_fails_on_an_empty_snapshot(self, service, lookups):
        service.body["listings"] = []
        with pytest.raises(RuntimeError, match="no usable listings"):
            lookups(service.url)
        assert len(service.requests) == InstrumentsLookupService.FIRST_LOAD_ATTEMPTS

    def test_startup_fails_when_no_listing_maps(self, service, lookups):
        service.body["listings"] = [_synthetic("000000000005")]
        with pytest.raises(RuntimeError, match="no usable listings"):
            lookups(service.url)

    def test_an_empty_refresh_keeps_the_copy(self, service, lookups):
        lookup = lookups(service.url)
        before = lookup.get_lookup()
        service.body["listings"] = []
        assert lookup.refresh() is False
        assert lookup.get_lookup() is before
        assert "If-None-Match" in service.requests[-1]["headers"]
        service.body = json.loads(json.dumps(SNAPSHOT))
        assert lookup.refresh() is False  # - the kept copy's etag still matches

    def test_a_truncated_refresh_keeps_the_copy(self, service, lookups):
        lookup = lookups(service.url)
        service.raw = json.dumps(SNAPSHOT).encode()[:500]
        assert lookup.refresh() is False
        assert len(lookup.get_lookup()) == len(SNAPSHOT["listings"])

    def test_a_concurrent_refresh_is_skipped(self, service, lookups):
        lookup = lookups(service.url)
        n = len(service.requests)
        with lookup._refresh_lock:
            assert lookup.refresh() is False
        assert len(service.requests) == n

    def test_background_refresh_swaps_in_changes(self, service, lookups):
        lookup = lookups(service.url, reload_interval="200ms")
        service.body["listings"] = [l for l in service.body["listings"] if l["exchange"] != "OKX.F"]
        deadline = time.monotonic() + 5
        while lookup.find_symbol("OKX.F", "BTCUSDT") is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert lookup.find_symbol("OKX.F", "BTCUSDT") is None
        assert lookup.find_symbol("BINANCE.UM", "BTCUSDT") is not None

    def test_accessors_never_block_on_a_slow_refresh(self, service, lookups):
        lookup = lookups(service.url, reload_interval="100ms")
        service.delay = 2.0
        time.sleep(0.3)  # - the background refresh is now stuck on the server
        t0 = time.monotonic()
        for _ in range(100):
            assert lookup.find_symbol("BINANCE.UM", "BTCUSDT") is not None
            assert lookup.get_lookup()
        assert time.monotonic() - t0 < 0.5

    def test_no_refresh_thread_without_an_interval(self, service, lookups):
        lookups(service.url)
        time.sleep(0.3)
        assert len(service.requests) == 1

    def test_a_listing_without_versions_is_skipped(self, service, lookups):
        service.body["listings"].extend(SYNTHETIC["listings"])
        lookup = lookups(service.url)
        assert lookup.find_symbol("BINANCE.UM", "EMPTYUSDT") is None
        assert lookup.find_symbol("BINANCE.UM", "TSLAUSDT") is not None
        assert len(lookup.get_lookup()) == len(SNAPSHOT["listings"]) + len(SYNTHETIC["listings"]) - 1


class TestLookupsManagerFactory:
    def test_service_type_builds_the_service_lookup(self, service):
        lookup = LookupsManager._get_instrument_lookup("service", url=service.url, path="/ignored", exchanges=["OKX.F"])
        assert isinstance(lookup, InstrumentsLookupService)
        assert "exchange=OKX.F" in service.requests[0]["path"]
        lookup.close()

    def test_service_type_requires_a_url(self):
        with pytest.raises(ValueError, match="url"):
            LookupsManager._get_instrument_lookup("service")

    def test_mongo_type_is_gone(self):
        with pytest.raises(ValueError, match="removed"):
            LookupsManager._get_instrument_lookup("mongo")


class TestLookupsManagerSingleton:
    def test_a_failed_build_leaves_no_instance_and_the_next_call_retries(self, monkeypatch):
        monkeypatch.delattr(LookupsManager, "instance", raising=False)
        real = LookupsManager._get_instrument_lookup
        calls = []

        def flaky(type: str, **kwargs):
            calls.append(type)
            if len(calls) == 1:
                raise RuntimeError("instrument service is unavailable")
            return real(type, **kwargs)

        monkeypatch.setattr(LookupsManager, "_get_instrument_lookup", staticmethod(flaky))
        with pytest.raises(RuntimeError, match="unavailable"):
            LookupsManager()
        assert not hasattr(LookupsManager, "instance")

        manager = LookupsManager()
        assert LookupsManager.instance is manager and manager._i_lookup is not None
        assert LookupsManager() is manager
        assert len(calls) == 2
