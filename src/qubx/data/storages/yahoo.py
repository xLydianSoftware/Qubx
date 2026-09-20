"""
Yahoo Finance storage — daily and coarser bars.

Bars are cached on disk as parquet through the existing caching layer, so a window already on disk
is not requested again.

    storage = StorageRegistry.get("yahoo::~/data/yahoo/")
    reader = storage["YAHOO", "STOCK"]
    reader.read("SPY", "ohlc(1d)", "2010-01-01", "now").to_pd()            # - adjusted (default)
    reader.read("SPY", "ohlc(1d)", "2010-01-01", "now", adjusted=False)    # - as traded

Yahoo's symbol carries the instrument class, so every market type is served by the same reader and
the label is only there to make the call read correctly. Measured 2026-09-20, all with daily history
back to 2002 apart from the newer ones:

    STOCK    SPY AAPL VFIAX        FX       EURUSD=X USDJPY=X
    INDEX    ^GSPC ^NDX ^VIX ^TNX  FUTURE   ES=F ZN=F CL=F GC=F
    ETF      TLT IEF AGG           CRYPTO   BTC-USD ETH-USD

One copy per symbol is cached: the unadjusted `open/high/low/close/volume` plus Yahoo's `adjclose`.
`adjusted=True` is applied by this reader after the cache, by scaling OHLC with `adjclose / close`,
so the two views never occupy two copies on disk.

Requires the `yahoo` extra: `pip install qubx[yahoo]`. Yahoo rejects plain HTTP clients — a request
from `httpx` with browser headers and a session cookie returns 429, and so does the crumb endpoint,
because the TLS fingerprint is checked as well. `yfinance` carries the machinery that gets past
this; that is the only reason it is a dependency here.

Limits worth knowing before relying on this:

- Yahoo serves about one month of 1-minute bars and about two years of hourly bars. Only `1d`, `1wk`
  and `1mo` have long history, so those are the only timeframes offered here.
- The endpoint returns **currently listed** symbols. Companies that were delisted or renamed are
  gone, so an index study built from a present-day constituent list is biased upward. A
  point-in-time constituent list has to come from somewhere else.
- Yahoo's terms do not permit redistribution. This is a local research cache.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
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

# - Yahoo's own interval spellings for the timeframes with usable history
_INTERVALS: dict[str, str] = {"1d": "1d", "1w": "1wk", "1M": "1mo"}

COLUMNS = ("open", "high", "low", "close", "adjclose", "volume")

EXCHANGE = "YAHOO"
# - labels only: the class is in the symbol, and one reader serves all of them
MARKET_TYPES = ("STOCK", "ETF", "FUND", "INDEX", "FUTURE", "FX", "CRYPTO")


def _yfinance():
    try:
        import yfinance  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Yahoo storage needs the 'yahoo' extra: pip install qubx[yahoo]") from e
    return yfinance


def _timeframe_of(dtype: DataType | str) -> str:
    """
    Pull the timeframe out of `ohlc(1d)` and check it against what Yahoo keeps history for.
    """
    s = str(dtype)
    if not s.lower().startswith("ohlc"):
        raise ValueError(f"Yahoo storage serves OHLC only, got '{dtype}'")
    tf = s[s.index("(") + 1 : s.index(")")] if "(" in s else "1d"
    tf = {"1day": "1d", "d": "1d", "1D": "1d", "1week": "1w", "1month": "1M"}.get(tf, tf)
    if tf not in _INTERVALS:
        raise ValueError(f"Yahoo storage serves {sorted(_INTERVALS)} only, got '{tf}'. Intraday history is too short.")
    return tf


def _epoch(t: str | pd.Timestamp | None, default: int) -> int:
    if t is None:
        return default
    ts = pd.Timestamp("now", tz="UTC") if str(t).lower() in ("now", "today") else pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS}, index=pd.DatetimeIndex([], name="timestamp"))


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance's frame → the column names and index this storage caches.

    Columns arrive capitalised and, for a single ticker, under a MultiIndex level naming it. The
    ticker level is dropped, names are lowercased, `adj close` becomes `adjclose`, and the index is
    made tz-naive so parquet round-trips it unchanged.
    """
    if frame is None or not len(frame):
        return empty_frame()
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).lower().replace(" ", "") for c in out.columns]
    out = out.rename(columns={"adjclose": "adjclose", "adj_close": "adjclose"})
    if "adjclose" not in out.columns and "close" in out.columns:
        out["adjclose"] = out["close"]
    keep = [c for c in COLUMNS if c in out.columns]
    out = out[keep]
    idx = pd.DatetimeIndex(out.index)
    out.index = idx.tz_localize(None) if idx.tz is not None else idx
    out.index.name = "timestamp"
    # - a session still in progress comes through with a null close
    return out.dropna(subset=["close"]).astype("float64")


class YahooFetcher:
    """
    One `yfinance` download per symbol.

    Kept separate from the reader so a different transport (a proxy, a recorded fixture, another
    vendor) can be substituted without touching the caching or the adjustment.
    """

    def __init__(self, timeout: float = 30.0, retries: int = 3, pause: float = 1.0) -> None:
        self._timeout = timeout
        self._retries = retries
        self._pause = pause

    def fetch(self, symbol: str, interval: str, start: int, stop: int) -> pd.DataFrame:
        yf = _yfinance()
        t0 = pd.Timestamp(start, unit="s")
        t1 = pd.Timestamp(stop, unit="s")
        last: Exception | None = None
        for attempt in range(self._retries):
            try:
                # - auto_adjust=False keeps the traded prices; this storage adjusts after the cache
                frame = yf.download(
                    symbol,
                    start=t0,
                    end=t1,
                    interval=interval,
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=False,
                    timeout=self._timeout,
                )
                return normalize(frame)
            except Exception as e:  # - yfinance raises its own types; treat them all as retryable
                last = e
                if attempt < self._retries - 1:
                    time.sleep(self._pause * (attempt + 1))
        logger.error(f"[YahooFetcher] '{symbol}' failed after {self._retries} attempts: {last}")
        return empty_frame()


class YahooFetchReader(IReader):
    """
    The uncached half: calls Yahoo on every read and returns bars as traded, with `adjclose` kept as
    its own column. Nothing here adjusts prices.
    """

    def __init__(self, fetcher: YahooFetcher | None = None) -> None:
        self._fetcher = fetcher or YahooFetcher()
        self._seen: set[str] = set()

    def _read_one(self, data_id: str, dtype: DataType | str, start: str | None, stop: str | None) -> RawData:
        interval = _INTERVALS[_timeframe_of(dtype)]
        frame = self._fetcher.fetch(data_id.upper(), interval, _epoch(start, 0), _epoch(stop, int(time.time())))
        if len(frame):
            self._seen.add(data_id.upper())
        return RawData.from_pandas(data_id, dtype, frame)  # type: ignore[arg-type]

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
                raise ValueError("Yahoo storage cannot enumerate symbols; name the ones to read")
            return RawMultiData([self._read_one(i, dtype, start, stop) for i in ids])
        return self._read_one(data_id, dtype, start, stop)

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        # - only what this process has asked for; the cache knows the persistent answer
        return sorted(self._seen)

    def get_data_types(self, data_id: str) -> list[DataType]:
        return [DataType.OHLC[tf] for tf in _INTERVALS]  # type: ignore[index]

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[Any, Any]:
        raw = self._read_one(data_id, dtype, None, None)
        s, e = raw.get_time_interval()
        return (np.datetime64(s, "ns"), np.datetime64(e, "ns"))

    def close(self) -> None:
        pass


class YahooReader(IReader):
    """
    The reader handed to callers. Reads through the parquet cache and applies the price adjustment.

    `read(..., adjusted=True)` — the default — scales open, high and low by `adjclose / close`, puts
    `adjclose` into `close` and drops the extra column, so a split reads as a continuous series.
    `adjusted=False` returns the bars as traded. Volume is left as Yahoo reports it, already split
    adjusted on their side.
    """

    def __init__(self, inner: IReader, cache: ParquetCache | None = None) -> None:
        self._inner = inner
        self._cache = cache

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None = None,
        stop: str | None = None,
        chunksize: int = 0,
        adjusted: bool = True,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        result = self._inner.read(data_id, dtype, start, stop, chunksize, **kwargs)
        return _adjust(result, adjusted)

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        """
        What the local cache holds, not what Yahoo carries.

        Yahoo has no endpoint that enumerates symbols, so the only answer that can be given is the
        set of symbols already downloaded into this cache directory. It survives restarts, because
        it is read from the on-disk index rather than from memory.
        """
        if self._cache is None:
            return self._inner.get_data_id(dtype)
        keys = [f"ohlc({tf})" for tf in _INTERVALS] if str(dtype) == str(DataType.ALL) else [str(dtype)]
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


def _adjust_one(raw: RawData, adjusted: bool) -> RawData:
    if "adjclose" not in raw.names:
        return raw
    frame = raw.data.to_pandas()
    if adjusted and len(frame):
        ratio = (frame["adjclose"] / frame["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for col in ("open", "high", "low"):
            if col in frame:
                frame[col] = frame[col] * ratio
        frame["close"] = frame["adjclose"]
    return RawData.from_pandas(raw.data_id, raw.dtype, frame.drop(columns=["adjclose"]))


def _adjust(result: Any, adjusted: bool) -> Any:
    if isinstance(result, RawData):
        return _adjust_one(result, adjusted)
    if isinstance(result, RawMultiData):
        return RawMultiData([_adjust_one(r, adjusted) for r in result.data])
    if isinstance(result, Iterator):
        return (_adjust(chunk, adjusted) for chunk in result)
    return result


@storage("yahoo")
class YahooStorage(IStorage):
    """
    Yahoo bars with a parquet cache underneath.

    The reader chain is built here rather than by the caller, so one copy of each symbol lives on
    disk and the adjusted view is derived from it:

        YahooReader(adjust) → CachedReader(ParquetCache) → YahooFetchReader(yfinance)
    """

    def __init__(self, path: str = "~/.qubx/yahoo", prefetch_period: str | None = None, **kwargs) -> None:
        self._path = Path(os.path.expanduser(path))
        self._path.mkdir(parents=True, exist_ok=True)
        self._prefetch_period = prefetch_period
        self._fetcher_kwargs = kwargs
        self._reader: YahooReader | None = None

    def get_exchanges(self) -> list[str]:
        return [EXCHANGE]

    def get_market_types(self, exchange: str) -> list[str]:
        return list(MARKET_TYPES) if exchange.upper() == EXCHANGE else []

    def get_reader(self, exchange: str, market: str) -> IReader:
        if exchange.upper() != EXCHANGE:
            raise ValueError(f"Yahoo storage has one exchange, '{EXCHANGE}', not '{exchange}'")
        if market.upper() not in MARKET_TYPES:
            raise ValueError(f"Unknown market type '{market}' for Yahoo; one of {MARKET_TYPES}")
        # - the same reader and the same cache serve every market type
        if self._reader is None:
            fetch = YahooFetchReader(YahooFetcher(**self._fetcher_kwargs))
            cache = ParquetCache(self._path)
            self._reader = YahooReader(CachedReader(fetch, cache, self._prefetch_period), cache)
        return self._reader

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __repr__(self) -> str:
        return f"YahooStorage({self._path})"
