"""
Dukascopy decoding, path building and caching. No network: the fetcher is replaced by a stub that
serves synthesised `.bi5` bodies.
"""

import json
import struct
import time
from datetime import datetime, timezone

import pandas as pd
import pytest

from qubx.data.storages.dukascopy import (
    DukascopyFetcher,
    DukascopyStorage,
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
