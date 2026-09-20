"""
Dukascopy storage: FX, metals, indices and CFDs from the free `.bi5` feed.

Bars come from the candle files, quotes from the tick files. Both are cached to parquet.

    storage = StorageRegistry.get("dukascopy::~/data/dukas/")
    reader = storage["DUKASCOPY", "FX"]
    reader.read("EURUSD", "ohlc(1h)", "2024-01-01", "2024-02-01").to_pd()             # - mid
    reader.read("EURUSD", "ohlc(1h)", "2024-01-01", "2024-02-01", side="bid")
    reader.read("EURUSD", "quote", "2024-03-05", "2024-03-06").to_pd()                # - ticks

`side` is "mid" by default, or "bid" / "ask". Mid is the average of the bid and ask candles, which
costs two requests per file and two cached copies; bid and ask are each cached once and the mid is
computed from them. The average is an approximation: a true mid high needs the ticks, because the
bid high and the ask high can fall on different ticks within the bar. Volume is the sum of the two
sides.

Feed layout. All times UTC. The month in the path is 0-indexed.

    {SYM}/{Y}/{M}/{D}/{H}h_ticks.bi5           one hour of ticks, 20 bytes each, >3i2f
    {SYM}/{Y}/{M}/{D}/BID_candles_min_1.bi5    one day of 1-minute bars, 24 bytes each, >5if
    {SYM}/{Y}/{M}/BID_candles_hour_1.bi5       one month of hourly bars
    {SYM}/{Y}/BID_candles_day_1.bi5            one year of daily bars

Candle record: offset, open, close, low, high, volume. Close is the second field, not the fourth.
Tick record: offset, ask, bid, ask volume, bid volume. Ask is the second field.
Prices are integers scaled by the instrument's point.

Bars use the candle files because a year of 1-minute bars is 365 requests there against 8,760 from
ticks; one day of EURUSD ticks measured 117 seconds. Ticks are fetched only for a `quote` read.

The feed returns 429 under load. Requests go through `TokenBucketRateLimiter`, shared by every
reader of one storage, so several threads pace as one. A 429 drains the bucket, which makes all of
them wait out the block. A wide first fetch takes hours.

The point is 1e-5 for FX, 1e-3 for JPY crosses and metals. Other instruments raise unless passed
`point=` or added to POINTS. A wrong point scales every price by 100.
"""

from __future__ import annotations

import json
import lzma
import os
import struct
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.cache import CachedReader, ParquetCache
from qubx.data.containers import RawData, RawMultiData
from qubx.data.registry import storage
from qubx.data.storage import IReader, IStorage, Transformable
from qubx.utils.rate_limiter import TokenBucketRateLimiter

BASE_URL = "http://www.dukascopy.com/datafeed"

TICK_FMT, TICK_SIZE = ">3i2f", 20
CANDLE_FMT, CANDLE_SIZE = ">5if", 24

EXCHANGE = "DUKASCOPY"

# - default price point per market type; None means it must be given explicitly
MARKET_POINTS: dict[str, float | None] = {"FX": 1e-5, "METAL": 1e-3, "INDEX": None, "STOCK": None, "CFD": None}
MARKET_TYPES = tuple(MARKET_POINTS)

# - per-symbol overrides. EURUSD and USDJPY checked back to 2004: the feed has no 4-digit era, so
# - one value per symbol covers the whole history.
POINTS: dict[str, float] = {
    "USDJPY": 1e-3,
    "EURJPY": 1e-3,
    "GBPJPY": 1e-3,
    "AUDJPY": 1e-3,
    "CADJPY": 1e-3,
    "CHFJPY": 1e-3,
    "NZDJPY": 1e-3,
    "XAUUSD": 1e-3,
    "XAGUSD": 1e-3,
}

# - timeframes published as candle files, and the period each file covers
NATIVE: dict[str, tuple[str, str]] = {
    "1Min": ("candles_min_1", "day"),
    "1h": ("candles_hour_1", "month"),
    "1d": ("candles_day_1", "year"),
}
NATIVE_STEP = {"1Min": timedelta(minutes=1), "1h": timedelta(hours=1), "1d": timedelta(days=1)}


CATALOGUE_URL = "https://freeserv.dukascopy.com/2.0/index.php?path=common/instruments"
CATALOGUE_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.dukascopy.com/"}


class Instruments:
    """
    Dukascopy's instrument catalogue: 1,604 entries keyed by the datafeed symbol.

    Fetched once and kept as JSON next to the parquet cache. Each entry carries `description`,
    `pipValue` and `history_start_tick` (epoch ms).

    `point` is `pipValue / 10`. That holds for EURUSD, USDJPY and XAUUSD, checked against prices
    decoded from the tick files; it is not checked on an index or a stock. POINTS still wins.
    """

    def __init__(self, path: Path, ttl_days: int = 30) -> None:
        self._file = path / "instruments.json"
        self._ttl = ttl_days * 86400
        self._by_symbol: dict[str, dict] | None = None

    def load(self) -> dict[str, dict]:
        if self._by_symbol is not None:
            return self._by_symbol
        raw = self._from_disk() or self._from_feed()
        self._by_symbol = {}
        for entry in (raw or {}).get("instruments", {}).values():
            name = entry.get("historical_filename")
            if name:
                self._by_symbol[name.upper()] = entry
        return self._by_symbol

    def _from_disk(self) -> dict | None:
        try:
            if time.time() - self._file.stat().st_mtime < self._ttl:
                return json.loads(self._file.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        return None

    def _from_feed(self) -> dict | None:
        try:
            request = urllib.request.Request(CATALOGUE_URL, headers=CATALOGUE_HEADERS)
            with urllib.request.urlopen(request, timeout=30) as response:
                text = response.read().decode()
            body = json.loads(text[text.index("(") + 1 : text.rindex(")")])
        except Exception as e:
            logger.warning(f"[Dukascopy] instrument catalogue unavailable: {e}")
            return None
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(json.dumps(body))
        except OSError:
            pass
        return body

    def symbols(self) -> list[str]:
        return sorted(self.load())

    def point(self, symbol: str) -> float | None:
        # - pipValue arrives as a string for some instruments
        entry = self.load().get(symbol.upper())
        try:
            pip = float(entry["pipValue"]) if entry and entry.get("pipValue") else 0.0
        except (TypeError, ValueError):
            return None
        return pip / 10.0 if pip else None

    def history_start(self, symbol: str) -> datetime | None:
        entry = self.load().get(symbol.upper())
        try:
            ms = int(entry["history_start_tick"]) if entry and entry.get("history_start_tick") else 0
        except (TypeError, ValueError):
            return None
        return datetime.fromtimestamp(ms / 1000, timezone.utc) if ms else None


def point_of(symbol: str, market: str, override: float | None = None, catalogue: Instruments | None = None) -> float:
    """
    Price multiplier for a symbol. POINTS first, then the catalogue's pipValue / 10, then the
    market default. Raises when none of them answers.
    """
    if override is not None:
        return override
    s = symbol.upper()
    if s in POINTS:
        return POINTS[s]
    if catalogue is not None:
        from_catalogue = catalogue.point(s)
        if from_catalogue:
            return from_catalogue
    default = MARKET_POINTS.get(market.upper())
    if default is not None:
        return default
    raise ValueError(
        f"No price point known for '{symbol}' under market type '{market}'. "
        f"Pass point=... to read(), or add it to qubx.data.storages.dukascopy.POINTS."
    )


class DukascopyFetcher:
    """
    Downloads and decompresses one `.bi5` file per call.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        retries: int = 4,
        requests_per_second: float = 2.0,
        burst: float = 20.0,
        cooldown: float = 150.0,
    ) -> None:
        self._timeout = timeout
        self._retries = retries
        self._cooldown = cooldown
        self._limiter = TokenBucketRateLimiter(capacity=burst, refill_rate=requests_per_second, name="dukascopy")

    def get(self, path: str) -> bytes | None:
        """
        Decompressed bytes, or None on 404. A weekend, a holiday or a pre-listing date all 404.
        """
        url = f"{BASE_URL}/{path}"
        for attempt in range(self._retries):
            self._limiter.acquire_blocking()
            try:
                with urllib.request.urlopen(url, timeout=self._timeout) as response:
                    raw = response.read()
                return lzma.decompress(raw) if raw else None
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code == 429:
                    # - drain the bucket so every thread waits out the block, not just this one
                    self._limiter.set_tokens(-self._cooldown * self._limiter.refill_rate)
                    logger.warning(f"[Dukascopy] 429 — pausing about {self._cooldown:.0f}s")
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionResetError, lzma.LZMAError) as e:
                if attempt == self._retries - 1:
                    logger.error(f"[Dukascopy] {path} failed after {self._retries} attempts: {e}")
                    return None
                time.sleep(2**attempt)
        return None


def decode_candles(body: bytes, base: datetime, point: float) -> pd.DataFrame:
    """
    Candle file to an OHLCV frame. Record: offset in seconds, open, close, low, high, volume.
    """
    rows = []
    for off in range(0, len(body) - CANDLE_SIZE + 1, CANDLE_SIZE):
        t, o, c, lo, hi, v = struct.unpack(CANDLE_FMT, body[off : off + CANDLE_SIZE])
        if o == 0 and c == 0 and hi == 0 and lo == 0:
            continue  # - minute with no trading
        rows.append((base + timedelta(seconds=t), o * point, hi * point, lo * point, c * point, float(v)))
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return frame.set_index("timestamp").sort_index()


def decode_ticks(body: bytes, base: datetime, point: float) -> pd.DataFrame:
    """
    Tick file to bid/ask quotes. Record: offset in ms, ask, bid, ask volume, bid volume.
    """
    n = len(body) // TICK_SIZE
    if n == 0:
        return pd.DataFrame(
            columns=["bid", "ask", "bid_size", "ask_size"], index=pd.DatetimeIndex([], name="timestamp")
        )
    raw = np.frombuffer(
        body[: n * TICK_SIZE],
        dtype=np.dtype([("ms", ">i4"), ("ask", ">i4"), ("bid", ">i4"), ("av", ">f4"), ("bv", ">f4")]),
    )
    frame = pd.DataFrame(
        {
            "bid": raw["bid"].astype("float64") * point,
            "ask": raw["ask"].astype("float64") * point,
            "bid_size": raw["bv"].astype("float64"),
            "ask_size": raw["av"].astype("float64"),
        },
        index=pd.DatetimeIndex(base + pd.to_timedelta(raw["ms"].astype("int64"), unit="ms"), name="timestamp"),
    )
    return frame.sort_index()


def _timeframe_of(dtype: DataType | str) -> str | None:
    s = str(dtype)
    if not s.lower().startswith("ohlc"):
        return None
    return s[s.index("(") + 1 : s.index(")")] if "(" in s else "1d"


def _candle_paths(symbol: str, tf: str, start: datetime, stop: datetime, side: str) -> list[tuple[str, datetime]]:
    """
    Candle files covering [start, stop), each with the timestamp its offsets count from.
    """
    name, period = NATIVE[tf]
    prefix = side.upper()
    out: list[tuple[str, datetime]] = []
    if period == "day":
        cur = start.replace(hour=0, minute=0, second=0, microsecond=0)
        while cur < stop:
            out.append((f"{symbol}/{cur.year}/{cur.month - 1:02d}/{cur.day:02d}/{prefix}_{name}.bi5", cur))
            cur += timedelta(days=1)
    elif period == "month":
        cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        while cur < stop:
            out.append((f"{symbol}/{cur.year}/{cur.month - 1:02d}/{prefix}_{name}.bi5", cur))
            cur = (cur + timedelta(days=32)).replace(day=1)
    else:
        for year in range(start.year, stop.year + 1):
            base = datetime(year, 1, 1, tzinfo=timezone.utc)
            if base < stop:
                out.append((f"{symbol}/{year}/{prefix}_{name}.bi5", base))
    return out


def _tick_paths(symbol: str, start: datetime, stop: datetime) -> list[tuple[str, datetime]]:
    cur = start.replace(minute=0, second=0, microsecond=0)
    out = []
    while cur < stop:
        out.append((f"{symbol}/{cur.year}/{cur.month - 1:02d}/{cur.day:02d}/{cur.hour:02d}h_ticks.bi5", cur))
        cur += timedelta(hours=1)
    return out


def _as_utc(t: str | pd.Timestamp | None, default: datetime) -> datetime:
    if t is None:
        return default
    ts = pd.Timestamp("now", tz="UTC") if str(t).lower() in ("now", "today") else pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


class DukascopyFetchReader(IReader):
    """
    Downloads and decodes. Bars from the candle files, ticks only for a `quote` read.
    """

    def __init__(
        self, fetcher: DukascopyFetcher | None = None, market: str = "FX", catalogue: Instruments | None = None
    ) -> None:
        self._fetcher = fetcher or DukascopyFetcher()
        self._market = market.upper()
        self._catalogue = catalogue
        self._seen: set[str] = set()

    def _read_one(self, data_id: str, dtype: DataType | str, start: str | None, stop: str | None, **kwargs) -> RawData:
        symbol = data_id.upper()
        point = point_of(symbol, self._market, kwargs.get("point"), self._catalogue)
        side = str(kwargs.get("side", "bid")).lower()
        if side not in ("bid", "ask"):
            raise ValueError(f"the fetch reader serves 'bid' or 'ask', not '{side}'")
        t0 = _as_utc(start, datetime(2003, 1, 1, tzinfo=timezone.utc))
        t1 = _as_utc(stop, datetime.now(timezone.utc))
        if self._catalogue is not None:
            # - clamp to the first tick Dukascopy has, else the walk requests years of 404s
            first = self._catalogue.history_start(symbol)
            if first is not None and first > t0:
                t0 = first

        tf = _timeframe_of(dtype)
        if tf is None:
            if str(dtype).lower().startswith("quote"):
                frame = self._ticks(symbol, t0, t1, point)
            else:
                raise ValueError(f"Dukascopy serves OHLC and quotes, not '{dtype}'")
        else:
            frame = self._bars(symbol, tf, t0, t1, point, side)

        if len(frame):
            self._seen.add(symbol)
            frame = frame[(frame.index >= t0.replace(tzinfo=None)) & (frame.index < t1.replace(tzinfo=None))]
        return RawData.from_pandas(data_id, dtype, frame)  # type: ignore[arg-type]

    def _bars(self, symbol: str, tf: str, t0: datetime, t1: datetime, point: float, side: str) -> pd.DataFrame:
        # - a non-native timeframe is built from the finest native one that divides it
        native = tf if tf in NATIVE else self._nearest_native(tf)
        parts = []
        for path, base in _candle_paths(symbol, native, t0, t1, side):
            body = self._fetcher.get(path)
            if body:
                parts.append(decode_candles(body, base.replace(tzinfo=None), point))
        if not parts:
            return _empty_ohlc()
        frame = pd.concat(parts).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        return frame if native == tf else _resample(frame, tf)

    def _ticks(self, symbol: str, t0: datetime, t1: datetime, point: float) -> pd.DataFrame:
        parts = []
        for path, base in _tick_paths(symbol, t0, t1):
            body = self._fetcher.get(path)
            if body:
                parts.append(decode_ticks(body, base.replace(tzinfo=None), point))
        if not parts:
            return pd.DataFrame(
                columns=["bid", "ask", "bid_size", "ask_size"], index=pd.DatetimeIndex([], name="timestamp")
            )
        return pd.concat(parts).sort_index()

    @staticmethod
    def _nearest_native(tf: str) -> str:
        delta = pd.Timedelta(tf.replace("Min", "min"))
        for name in ("1d", "1h", "1Min"):
            if NATIVE_STEP[name] <= delta:
                return name
        return "1Min"

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None = None,
        stop: str | None = None,
        chunksize: int = 0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        if isinstance(data_id, (list, tuple, set)):
            ids = list(data_id)
            if not ids:
                raise ValueError("Dukascopy cannot enumerate symbols; name the ones to read")
            return RawMultiData([self._read_one(i, dtype, start, stop, **kwargs) for i in ids])
        return self._read_one(data_id, dtype, start, stop, **kwargs)

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        # - this process only; the cache holds the persistent answer
        return sorted(self._seen)

    def get_data_types(self, data_id: str) -> list[DataType]:
        return [DataType.OHLC[tf] for tf in NATIVE] + [DataType.QUOTE]  # type: ignore[index]

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[Any, Any]:
        raw = self._read_one(data_id, dtype, None, None)
        s, e = raw.get_time_interval()
        return (np.datetime64(s, "ns"), np.datetime64(e, "ns"))

    def close(self) -> None:
        pass


def _empty_ohlc() -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in ("open", "high", "low", "close", "volume")},
        index=pd.DatetimeIndex([], name="timestamp"),
    )


def _resample(frame: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Build a coarser timeframe from a finer one.

    Buckets start at UTC midnight, which is what Dukascopy's own candle files use: their daily bar
    for EURUSD 2024-03-04 (1.08417 / 1.08667 / 1.08377 / 1.08541) is reproduced exactly by hourly
    bars resampled at 00:00 UTC, and not by 21:00 or 22:00.

    That is separate from the overnight policy, which rolls positions at 21:00/22:00 GMT for swap
    (19:00/18:00 for NZD pairs, except Fridays). The rollover does not move the bar boundary.
    """
    rule = tf.replace("Min", "min")
    out = frame.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["open"])


def _mid_one(bid: RawData, ask: RawData) -> RawData:
    """
    Average two candle frames into a mid one. Volume is summed.
    """
    b, a = bid.data.to_pandas(), ask.data.to_pandas()
    if not len(b):
        return ask
    if not len(a):
        return bid
    b, a = b.set_index("timestamp"), a.set_index("timestamp")
    common = b.index.intersection(a.index)
    b, a = b.loc[common], a.loc[common]
    out = (b[["open", "high", "low", "close"]] + a[["open", "high", "low", "close"]]) / 2.0
    out["volume"] = b["volume"] + a["volume"]
    return RawData.from_pandas(bid.data_id, bid.dtype, out)


def _mid(bid: Any, ask: Any) -> Any:
    if isinstance(bid, RawData) and isinstance(ask, RawData):
        return _mid_one(bid, ask)
    if isinstance(bid, RawMultiData) and isinstance(ask, RawMultiData):
        by_id = {r.data_id: r for r in ask.data}
        return RawMultiData([_mid_one(r, by_id[r.data_id]) for r in bid.data if r.data_id in by_id])
    return bid


class DukascopyReader(IReader):
    """
    Reads through the parquet cache and answers `get_data_id` from it.
    """

    def __init__(self, inner: IReader, cache: ParquetCache | None = None, catalogue: Instruments | None = None) -> None:
        self._inner = inner
        self._cache = cache
        self._catalogue = catalogue

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None = None,
        stop: str | None = None,
        chunksize: int = 0,
        side: str = "mid",
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        if str(side).lower() != "mid" or not str(dtype).lower().startswith("ohlc"):
            s = "bid" if str(side).lower() == "mid" else str(side).lower()
            return self._inner.read(data_id, dtype, start, stop, chunksize, side=s, **kwargs)
        bid = self._inner.read(data_id, dtype, start, stop, chunksize, side="bid", **kwargs)
        ask = self._inner.read(data_id, dtype, start, stop, chunksize, side="ask", **kwargs)
        return _mid(bid, ask)

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        """
        Every symbol in Dukascopy's instrument catalogue. Falls back to what the local cache holds
        when the catalogue cannot be fetched.
        """
        if self._catalogue is not None:
            symbols = self._catalogue.symbols()
            if symbols:
                return symbols
        if self._cache is None:
            return self._inner.get_data_id(dtype)
        keys = [f"ohlc({tf})" for tf in NATIVE] + ["quote"] if str(dtype) == str(DataType.ALL) else [str(dtype)]
        found: set[str] = set()
        for k in keys:
            found.update(self._cache.get_stored_ids(k))
        return sorted(found)

    def get_data_types(self, data_id: str) -> list[DataType]:
        return self._inner.get_data_types(data_id)

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[Any, Any]:
        return self._inner.get_time_range(data_id, dtype)

    def close(self) -> None:
        self._inner.close()


@storage("dukascopy")
class DukascopyStorage(IStorage):
    """
    Reader chain:

        DukascopyReader -> CachedReader(ParquetCache) -> DukascopyFetchReader(.bi5 over HTTP)

    The market type supplies the default price point.
    """

    def __init__(
        self,
        path: str = "~/.qubx/dukascopy",
        prefetch_period: str | None = None,
        requests_per_second: float = 2.0,
        catalogue: Instruments | None = None,
        **kwargs,
    ) -> None:
        self._path = Path(os.path.expanduser(path))
        self._path.mkdir(parents=True, exist_ok=True)
        self._prefetch_period = prefetch_period
        self._requests_per_second = requests_per_second
        self._fetcher_kwargs = kwargs
        self._cache: ParquetCache | None = None
        self._catalogue: Instruments | None = catalogue
        self._own_catalogue = catalogue is None
        self._readers: dict[str, DukascopyReader] = {}

    def get_exchanges(self) -> list[str]:
        return [EXCHANGE]

    def get_market_types(self, exchange: str) -> list[str]:
        return list(MARKET_TYPES) if exchange.upper() == EXCHANGE else []

    def get_reader(self, exchange: str, market: str) -> IReader:
        if exchange.upper() != EXCHANGE:
            raise ValueError(f"Dukascopy storage has one exchange, '{EXCHANGE}', not '{exchange}'")
        m = market.upper()
        if m not in MARKET_TYPES:
            raise ValueError(f"Unknown market type '{market}' for Dukascopy; one of {MARKET_TYPES}")
        if self._cache is None:
            self._cache = ParquetCache(self._path)
            if self._own_catalogue:
                self._catalogue = Instruments(self._path)
        if m not in self._readers:
            fetch = DukascopyFetchReader(
                DukascopyFetcher(requests_per_second=self._requests_per_second, **self._fetcher_kwargs),
                m,
                self._catalogue,
            )
            self._readers[m] = DukascopyReader(
                CachedReader(fetch, self._cache, self._prefetch_period), self._cache, self._catalogue
            )
        return self._readers[m]

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        self._cache = None
        if self._own_catalogue:
            self._catalogue = None

    def __repr__(self) -> str:
        return f"DukascopyStorage({self._path})"
