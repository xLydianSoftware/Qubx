"""
Dukascopy decoding, path building and caching. No network: the fetcher is replaced by a stub that
serves synthesised `.bi5` bodies.
"""

import struct
from datetime import datetime, timezone

import pandas as pd
import pytest

from qubx.data.storages.dukascopy import (
    DukascopyStorage,
    RateLimiter,
    _candle_paths,
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


class TestRateLimiter:
    def test_penalty_grows_on_429_and_decays_on_success(self):
        limiter = RateLimiter(min_interval=0.01)
        first = limiter.on_429()
        second = limiter.on_429()
        assert second > first
        limiter.on_success()
        limiter.on_success()
        assert limiter._penalty < second


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
        # - a day of minutes; only the first few are filled
        return candle_body([(i * 60, 108540 + i, 108535 + i, 108530 + i, 108560 + i, 1.0) for i in range(60)])


class TestStorage:
    def _storage(self, tmp_path):
        st = DukascopyStorage(str(tmp_path))
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
        df = st["DUKASCOPY", "FX"].read("EURUSD", "ohlc(15Min)", "2024-03-05", "2024-03-06").to_pd()
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
