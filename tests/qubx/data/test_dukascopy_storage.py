"""
Dukascopy decoding, path building and caching. No network: the fetcher is replaced by a stub that
serves synthesised `.bi5` bodies.
"""

import json
import lzma
import struct
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd
import pytest

from qubx.data.registry import StorageRegistry
from qubx.data.storages import dukascopy as dukascopy_module
from qubx.data.storages.dukascopy import (
    BASE_URL,
    DATAFEED_HEADERS,
    DukascopyFetcher,
    DukascopyFetchReader,
    DukascopyStorage,
    FeedRefused,
    Instruments,
    _candle_paths,
    _resample,
    _tick_paths,
    decode_candles,
    decode_ticks,
    point_of,
)

DAY = datetime(2024, 3, 5)


def candle_body(records):
    """
    records: (offset_seconds, open, close, low, high, volume) as the feed stores them.
    """
    return b"".join(struct.pack(">5if", *r) for r in records)


def tick_body(records):
    """
    records: (offset_ms, ask, bid, ask_vol, bid_vol) — ask before bid, as the feed stores them.
    """
    return b"".join(struct.pack(">3i2f", *r) for r in records)


class TestDecoding:
    def test_candle_field_order_is_open_close_low_high(self):
        # - close is the second field; reading the record as OHLC swaps close and high
        frame = decode_candles(candle_body([(0, 108540, 108535, 108531, 108560, 58.31)]), DAY, 1e-5)
        row = frame.iloc[0]
        assert row["open"] == pytest.approx(1.08540)
        assert row["close"] == pytest.approx(1.08535)
        assert row["low"] == pytest.approx(1.08531)
        assert row["high"] == pytest.approx(1.08560)
        assert row["low"] <= min(row["open"], row["close"]) and row["high"] >= max(row["open"], row["close"])

    def test_candle_offsets_are_seconds_and_padding_is_dropped(self):
        frame = decode_candles(
            candle_body([(0, 1, 1, 1, 1, 1.0), (60, 2, 2, 2, 2, 1.0), (120, 0, 0, 0, 0, 0.0)]), DAY, 1e-5
        )
        assert list(frame.index) == [DAY, DAY + pd.Timedelta(minutes=1)]

    def test_closed_session_padding_goes_but_a_one_price_minute_stays(self):
        # - the feed fills closed hours with the last price at zero volume; a real quiet minute
        #   has the same flat shape but keeps its volume
        frame = decode_candles(
            candle_body([(0, 108540, 108540, 108540, 108540, 0.0), (60, 108540, 108540, 108540, 108540, 0.4)]),
            DAY,
            1e-5,
        )
        assert list(frame.index) == [DAY + pd.Timedelta(minutes=1)]

    def test_tick_field_order_is_ask_before_bid(self):
        # - reversed, every spread comes out negative
        frame = decode_ticks(tick_body([(47, 108513, 108512, 0.12, 3.15)]), DAY, 1e-5)
        row = frame.iloc[0]
        assert row["bid"] == pytest.approx(1.08512)
        assert row["ask"] == pytest.approx(1.08513)
        assert row["ask"] > row["bid"]
        assert row["bid_size"] == pytest.approx(3.15)
        assert frame.index[0] == DAY + pd.Timedelta(milliseconds=47)


class TestPoint:
    def test_symbol_override_beats_the_market_default(self):
        # - USDJPY under FX must not take FX's 1e-5: that is 100x off
        assert point_of("USDJPY", "FX") == 1e-3
        assert point_of("EURUSD", "FX") == 1e-5

    def test_unknown_instrument_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match="No price point known"):
            point_of("USA500", "INDEX")

    def test_explicit_override_wins(self):
        assert point_of("USA500", "INDEX", override=1e-3) == 1e-3


class TestPaths:
    def test_month_is_zero_indexed(self):
        # - March is 02 in the feed's paths; off by one reads the wrong month
        paths = [
            p
            for p, _ in _candle_paths(
                "EURUSD", "1Min", DAY.replace(tzinfo=timezone.utc), datetime(2024, 3, 6, tzinfo=timezone.utc), "bid"
            )
        ]
        assert paths == ["EURUSD/2024/02/05/BID_candles_min_1.bi5"]

    def test_file_granularity_per_timeframe(self):
        t0 = datetime(2024, 3, 5, tzinfo=timezone.utc)
        t1 = datetime(2024, 3, 7, tzinfo=timezone.utc)
        assert len(_candle_paths("EURUSD", "1Min", t0, t1, "bid")) == 2  # - a file a day
        assert len(_candle_paths("EURUSD", "1h", t0, t1, "bid")) == 1  # - a file a month
        assert len(_candle_paths("EURUSD", "1d", t0, t1, "bid")) == 1  # - a file a year
        assert len(_tick_paths("EURUSD", t0, t1)) == 48  # - a file an hour

    def test_side_selects_the_bid_or_ask_file(self):
        t0 = datetime(2024, 3, 5, tzinfo=timezone.utc)
        t1 = datetime(2024, 3, 6, tzinfo=timezone.utc)
        assert "ASK_candles_min_1" in _candle_paths("EURUSD", "1Min", t0, t1, "ask")[0][0]


class TestRateLimiterUse:
    def test_fetcher_paces_through_the_shared_token_bucket(self):
        """
        The bucket is shared by every reader of one storage, so several threads pace as one.
        """
        fetcher = DukascopyFetcher(requests_per_second=100.0, burst=2.0)
        assert fetcher._limiter.refill_rate == 100.0
        assert fetcher._limiter.capacity == 2.0

        # - burst of 2 is free, the third waits for a refill
        t0 = time.monotonic()
        for _ in range(3):
            fetcher._limiter.acquire_blocking()
        assert time.monotonic() - t0 >= 0.009


class StubFetcher:
    """
    Serves synthetic bodies and records the paths asked for.
    """

    def __init__(self):
        self.paths: list[str] = []

    def get(self, path: str) -> bytes | None:
        self.paths.append(path)
        if "ticks" in path:
            return tick_body([(i * 1000, 108513 + i, 108512 + i, 0.5, 1.5) for i in range(10)])
        # - the ask file sits 4 points above the bid one, so a mid bar is distinguishable from both
        up = 4 if path.startswith("ASK") or "/ASK_" in path else 0
        return candle_body(
            [(i * 60, 108540 + i + up, 108535 + i + up, 108530 + i + up, 108560 + i + up, 1.0) for i in range(60)]
        )


class NoCatalogue:
    """
    Stands in for the instrument catalogue so the tests never reach the network.
    """

    def symbols(self):
        return []

    def point(self, symbol):
        return None

    def history_start(self, symbol):
        return None


class TestStorage:
    def _storage(self, tmp_path):
        st = DukascopyStorage(str(tmp_path), catalogue=NoCatalogue())
        stub = StubFetcher()
        st.get_reader("DUKASCOPY", "FX")._inner._reader._fetcher = stub  # type: ignore[attr-defined]
        return st, stub

    def test_bars_come_from_candle_files_not_ticks(self, tmp_path):
        st, stub = self._storage(tmp_path)
        st["DUKASCOPY", "FX"].read("EURUSD", "ohlc(1Min)", "2024-03-05", "2024-03-06")
        assert stub.paths and all("candles" in p for p in stub.paths)

    def test_quotes_come_from_tick_files(self, tmp_path):
        st, stub = self._storage(tmp_path)
        df = st["DUKASCOPY", "FX"].read("EURUSD", "quote", "2024-03-05T00:00", "2024-03-05T01:00").to_pd()
        assert all("ticks" in p for p in stub.paths)
        assert list(df.columns) == ["bid", "ask", "bid_size", "ask_size"]

    def test_second_read_of_a_cached_window_does_not_refetch(self, tmp_path):
        st, stub = self._storage(tmp_path)
        reader = st["DUKASCOPY", "FX"]
        reader.read("EURUSD", "ohlc(1Min)", "2024-03-05", "2024-03-06")
        n = len(stub.paths)
        reader.read("EURUSD", "ohlc(1Min)", "2024-03-05T00:10", "2024-03-05T00:50")
        assert len(stub.paths) == n

    def test_non_native_timeframe_is_built_from_one_minute(self, tmp_path):
        st, stub = self._storage(tmp_path)
        df = st["DUKASCOPY", "FX"].read("EURUSD", "ohlc(15Min)", "2024-03-05", "2024-03-06", side="bid").to_pd()
        assert all("candles_min_1" in p for p in stub.paths)
        # - 60 synthetic minutes give four 15-minute bars; the first keeps the minute's open
        assert len(df) == 4
        assert df["open"].iloc[0] == pytest.approx(1.08540)
        assert df["high"].iloc[0] == pytest.approx(1.08574)

    def test_bid_and_ask_are_cached_apart(self, tmp_path):
        st, stub = self._storage(tmp_path)
        reader = st["DUKASCOPY", "FX"]
        reader.read("EURUSD", "ohlc(1Min)", "2024-03-05", "2024-03-06", side="bid")
        n = len(stub.paths)
        reader.read("EURUSD", "ohlc(1Min)", "2024-03-05", "2024-03-06", side="ask")
        assert len(stub.paths) > n, "the ask side must not be served from the bid cache"


class TestSides:
    def _storage(self, tmp_path):
        st = DukascopyStorage(str(tmp_path), catalogue=NoCatalogue())
        stub = StubFetcher()
        st.get_reader("DUKASCOPY", "FX")._inner._reader._fetcher = stub  # type: ignore[attr-defined]
        return st, stub

    def test_mid_is_the_default_and_averages_the_two_sides(self, tmp_path):
        st, _ = self._storage(tmp_path)
        reader = st["DUKASCOPY", "FX"]
        mid = reader.read("EURUSD", "ohlc(1Min)", "2024-03-05", "2024-03-06").to_pd()
        bid = reader.read("EURUSD", "ohlc(1Min)", "2024-03-05", "2024-03-06", side="bid").to_pd()
        ask = reader.read("EURUSD", "ohlc(1Min)", "2024-03-05", "2024-03-06", side="ask").to_pd()

        for col in ("open", "high", "low", "close"):
            assert mid[col].iloc[0] == pytest.approx((bid[col].iloc[0] + ask[col].iloc[0]) / 2)
        assert bid["close"].iloc[0] != ask["close"].iloc[0]
        assert mid["volume"].iloc[0] == pytest.approx(bid["volume"].iloc[0] + ask["volume"].iloc[0])

    def test_mid_reads_both_files_and_one_side_reads_one(self, tmp_path):
        st, stub = self._storage(tmp_path)
        reader = st["DUKASCOPY", "FX"]
        reader.read("EURUSD", "ohlc(1Min)", "2024-03-05", "2024-03-06", side="bid")
        one_side = len(stub.paths)
        assert all(p.count("BID") for p in stub.paths)

        st2, stub2 = self._storage(tmp_path / "other")
        st2["DUKASCOPY", "FX"].read("EURUSD", "ohlc(1Min)", "2024-03-05", "2024-03-06")
        assert len(stub2.paths) == 2 * one_side
        assert any("ASK" in p for p in stub2.paths) and any("BID" in p for p in stub2.paths)

    def test_quotes_ignore_side(self, tmp_path):
        st, stub = self._storage(tmp_path)
        st["DUKASCOPY", "FX"].read("EURUSD", "quote", "2024-03-05T00:00", "2024-03-05T01:00")
        assert all("ticks" in p for p in stub.paths)


class TestCatalogue:
    def _catalogue(self, tmp_path, payload):
        (tmp_path / "instruments.json").write_text(json.dumps(payload))
        return Instruments(tmp_path)

    def test_point_is_pipvalue_over_ten_and_survives_string_values(self, tmp_path):
        cat = self._catalogue(
            tmp_path,
            {
                "instruments": {
                    "EUR/USD": {"historical_filename": "EURUSD", "pipValue": 0.0001},
                    "USD/JPY": {"historical_filename": "USDJPY", "pipValue": "0.01"},
                }
            },
        )
        assert cat.point("EURUSD") == pytest.approx(1e-5)
        assert cat.point("USDJPY") == pytest.approx(1e-3)

    def test_history_start_is_epoch_milliseconds(self, tmp_path):
        cat = self._catalogue(
            tmp_path,
            {"instruments": {"EUR/USD": {"historical_filename": "EURUSD", "history_start_tick": 1167609605163}}},
        )
        assert cat.history_start("EURUSD").year == 2007
        assert cat.history_start("NOPE") is None

    def test_symbols_come_from_the_catalogue_not_the_cache(self, tmp_path):
        cat = self._catalogue(
            tmp_path,
            {
                "instruments": {
                    "EUR/USD": {"historical_filename": "EURUSD", "pipValue": 0.0001},
                    "0005.HK/HKD": {"historical_filename": "0005HKHKD", "pipValue": 0.01},
                }
            },
        )
        st = DukascopyStorage(str(tmp_path), catalogue=cat)
        assert st["DUKASCOPY", "FX"].get_data_id("ohlc(1Min)") == ["0005HKHKD", "EURUSD"]

    def test_point_prefers_the_explicit_table(self, tmp_path):
        cat = self._catalogue(
            tmp_path, {"instruments": {"USD/JPY": {"historical_filename": "USDJPY", "pipValue": 99.0}}}
        )
        assert point_of("USDJPY", "FX", catalogue=cat) == 1e-3


class TestDailyBoundary:
    def test_daily_buckets_start_at_utc_midnight(self):
        """
        Dukascopy's daily candle for EURUSD 2024-03-04 is open 1.08417, high 1.08667, low 1.08377,
        close 1.08541, and hourly bars bucketed at 00:00 UTC reproduce it. The 21:00/22:00 GMT
        overnight rollover applies to swap, not to the bar boundary.
        """
        hours = pd.date_range("2024-03-04", periods=48, freq="1h")
        frame = pd.DataFrame(
            {
                "open": [1.0 + i / 1000 for i in range(48)],
                "high": [1.1 + i / 1000 for i in range(48)],
                "low": [0.9 + i / 1000 for i in range(48)],
                "close": [1.05 + i / 1000 for i in range(48)],
                "volume": [1.0] * 48,
            },
            index=pd.DatetimeIndex(hours, name="timestamp"),
        )
        daily = _resample(frame, "1d")

        assert list(daily.index) == [pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")]
        assert daily["open"].iloc[0] == pytest.approx(1.0)
        assert daily["close"].iloc[0] == pytest.approx(1.05 + 23 / 1000)


class TestHeaders:
    def test_datafeed_requests_carry_a_browser_agent(self):
        """
        Measured one header at a time: none 429 at request 45, Referer only at 74, Accept only at
        60, User-Agent only 200 requests clean. The throttle keys on the User-Agent.
        """
        assert "Mozilla" in DATAFEED_HEADERS["User-Agent"]
        assert DATAFEED_HEADERS["Referer"].startswith("https://www.dukascopy.com")

    def test_the_request_is_built_with_them(self, monkeypatch):
        seen = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b""

        def fake_urlopen(request, timeout=None):
            seen["headers"] = dict(request.header_items())
            seen["url"] = request.full_url
            return Response()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        DukascopyFetcher(requests_per_second=1000.0).get("EURUSD/2024/02/05/10h_ticks.bi5")

        lowered = {k.lower(): v for k, v in seen["headers"].items()}
        assert "Mozilla" in lowered["User-agent".lower()]
        assert lowered["referer"].startswith("https://www.dukascopy.com")
        assert seen["url"].endswith("EURUSD/2024/02/05/10h_ticks.bi5")


class TestDefaultPath:
    def test_no_path_means_the_shared_cache_folder(self, tmp_path, monkeypatch):
        called = {}

        def fake_folder(name=""):
            called["name"] = name
            return str(tmp_path / name)

        monkeypatch.setattr(dukascopy_module, "get_local_data_cache_folder", fake_folder)
        st = StorageRegistry.get("dukascopy")

        assert called["name"] == "dukascopy"
        assert st._path == tmp_path / "dukascopy"

    def test_an_explicit_path_still_wins(self, tmp_path):
        assert StorageRegistry.get(f"dukascopy::{tmp_path}")._path == tmp_path


def test_registered_without_importing_the_module():
    """
    The @storage decorator only runs when the module is imported, so qubx.data must import it.
    Without that, StorageRegistry.get("dukascopy") raises for anyone who has not touched the module.
    """
    import qubx.data  # noqa: F401

    assert StorageRegistry.is_registered("dukascopy")
    assert StorageRegistry.is_registered("yahoo")


class TestFetcherErrors:
    def _fetcher_with(self, monkeypatch, codes, cooldown=0.0):
        """
        A fetcher whose urlopen raises the given HTTP codes in turn, then succeeds.
        """
        calls = {"n": 0}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return lzma.compress(b"payload")

        def fake_urlopen(request, timeout=None):
            i = calls["n"]
            calls["n"] += 1
            if i < len(codes):
                raise urllib.error.HTTPError(request.full_url, codes[i], "nope", {}, None)  # type: ignore[arg-type]
            return Response()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return DukascopyFetcher(requests_per_second=1e6, cooldown=cooldown), calls

    def test_503_is_retried_not_raised(self, monkeypatch):
        """
        A 503 is the feed pushing back, not a failure: the file is there and the retry gets it.
        """
        fetcher, calls = self._fetcher_with(monkeypatch, [503, 503])
        assert fetcher.get("EURUSD/2024/02/05/10h_ticks.bi5") == b"payload"
        assert calls["n"] == 3

    def test_404_is_a_missing_file_not_an_error(self, monkeypatch):
        fetcher, calls = self._fetcher_with(monkeypatch, [404])
        assert fetcher.get("EURUSD/2024/02/05/10h_ticks.bi5") is None
        assert calls["n"] == 1

    def test_giving_up_raises_so_no_hole_reaches_the_cache(self, monkeypatch):
        """
        Skipping the file would leave a gap in the frame, and CachedReader would then record the
        whole window as covered and serve that gap from then on.
        """
        fetcher, _ = self._fetcher_with(monkeypatch, [503] * 20)
        with pytest.raises(FeedRefused):
            fetcher.get("EURUSD/2024/02/05/10h_ticks.bi5")

    def test_other_codes_still_raise(self, monkeypatch):
        fetcher, _ = self._fetcher_with(monkeypatch, [403])
        with pytest.raises(urllib.error.HTTPError):
            fetcher.get("EURUSD/2024/02/05/10h_ticks.bi5")

    def test_repeated_refusals_wait_the_same_pause_each_time(self, monkeypatch):
        """
        Each retry waits one cooldown, not a growing multiple of it.
        """
        cooldown = 0.2
        fetcher, calls = self._fetcher_with(monkeypatch, [503, 503, 503], cooldown=cooldown)

        t0 = time.time()
        assert fetcher.get("EURUSD/2024/02/05/10h_ticks.bi5") == b"payload"
        elapsed = time.time() - t0

        assert calls["n"] == 4
        assert elapsed < 4 * cooldown  # - three pauses of 0.2, not 0.2 + 0.4 + 0.6

    def test_base_url_is_the_direct_host(self):
        # - www.dukascopy.com/datafeed 302s here; each hop is another request
        assert "datafeed.dukascopy.com" in BASE_URL


class TestConcurrentFetch:
    def _reader(self, bodies, workers):
        class Fetcher:
            def __init__(self):
                self.paths = []
                self.lock = threading.Lock()

            def get(self, path):
                with self.lock:
                    self.paths.append(path)
                time.sleep(0.05)  # - stand in for the ~8s the server takes
                return bodies(path)

        f = Fetcher()
        return DukascopyFetchReader(f, "FX", NoCatalogue(), workers=workers), f  # type: ignore[arg-type]

    def test_order_is_preserved_across_workers(self, tmp_path):
        """
        Workers finish out of order; the frames must still be concatenated in date order.
        """

        def body(path):
            day = int(path.split("/")[3])
            return candle_body([(0, 108000 + day, 108000 + day, 108000 + day, 108000 + day, 1.0)])

        reader, _ = self._reader(body, workers=8)
        raw = reader.read("EURUSD", "ohlc(1Min)", "2024-03-01", "2024-03-09", side="bid")
        opens = raw.data.to_pandas()["open"].tolist()

        assert opens == sorted(opens)
        assert len(opens) == 8

    def test_workers_overlap_the_waiting(self, tmp_path):
        def body(path):
            return candle_body([(0, 108000, 108000, 108000, 108000, 1.0)])

        serial, _ = self._reader(body, workers=1)
        t0 = time.monotonic()
        serial.read("EURUSD", "ohlc(1Min)", "2024-03-01", "2024-03-09", side="bid")
        one = time.monotonic() - t0

        parallel, _ = self._reader(body, workers=8)
        t0 = time.monotonic()
        parallel.read("EURUSD", "ohlc(1Min)", "2024-03-01", "2024-03-09", side="bid")
        many = time.monotonic() - t0

        assert many < one / 2, f"serial {one:.2f}s, parallel {many:.2f}s"

    def test_a_refusal_in_any_worker_propagates(self, tmp_path):
        """
        Swallowing it would hand a short frame to the cache, which then records the window covered.
        """

        def body(path):
            if path.endswith("05/BID_candles_min_1.bi5"):
                raise FeedRefused(path)
            return candle_body([(0, 108000, 108000, 108000, 108000, 1.0)])

        reader, _ = self._reader(body, workers=8)
        with pytest.raises(FeedRefused):
            reader.read("EURUSD", "ohlc(1Min)", "2024-03-01", "2024-03-09", side="bid")
