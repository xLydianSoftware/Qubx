"""
Yahoo Finance storage — daily and coarser bars.

Bars are cached on disk as parquet through the existing caching layer, so a window already on disk
is not requested again.

    storage = StorageRegistry.get("yahoo::~/data/yahoo/")
    reader = storage["YAHOO", "STOCK"]
    reader.read("SPY", "ohlc(1d)", "2010-01-01", "now").to_pd()            # - adjusted (default)
    reader.read("SPY", "ohlc(1d)", "2010-01-01", "now", adjusted=False)    # - as traded

The market type supplies Yahoo's instrument mark, so callers pass a plain name. Checked
2026-09-20, all with daily history back to 2002 apart from the newer ones:

    STOCK    SPY AAPL VFIAX        FX       EURUSD=X USDJPY=X
    INDEX    ^GSPC ^NDX ^VIX ^TNX  FUTURE   ES=F ZN=F CL=F GC=F
    ETF      TLT IEF AGG           CRYPTO   BTC-USD ETH-USD

One copy per symbol is cached: the unadjusted `open/high/low/close/volume` plus Yahoo's `adjclose`.
`adjusted=True` is applied by this reader after the cache, by scaling OHLC with `adjclose / close`,
so the two views never occupy two copies on disk.

Requires the `yahoo` extra: `pip install qubx[yahoo]`. Yahoo returns 429 to httpx even with
browser headers and a session cookie, and to the crumb endpoint as well: the TLS fingerprint is
checked. yfinance handles that, which is why it is the dependency.

Yahoo serves about one month of 1-minute bars and two years of hourly. Only `1d`, `1wk` and `1mo`
have long history, so those are the timeframes offered here.

The endpoint returns currently listed symbols only. Delisted and renamed companies are absent, so
an index study built from a present-day constituent list is biased upward.

Yahoo's terms do not permit redistribution. This is a local research cache.
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
from qubx.utils.misc import get_local_data_cache_folder

# - Yahoo's own interval spellings for the timeframes with usable history
_INTERVALS: dict[str, str] = {"1d": "1d", "1w": "1wk", "1M": "1mo"}

COLUMNS = ("open", "high", "low", "close", "adjclose", "volume")

EXCHANGE = "YAHOO"

# - Yahoo marks the class on the symbol. The market type supplies the mark, so callers pass a plain
# - name: ("FX", "EURUSD") not "EURUSD=X", ("INDEX", "GSPC") not "^GSPC".
# - Each entry is (prefix, suffix). An already-marked symbol is left alone.
MARKET_AFFIXES: dict[str, tuple[str, str]] = {
    "STOCK": ("", ""),
    "ETF": ("", ""),
    "FUND": ("", ""),
    "INDEX": ("^", ""),
    "FUTURE": ("", "=F"),
    "FX": ("", "=X"),
    "CRYPTO": ("", "-USD"),
}
MARKET_TYPES = tuple(MARKET_AFFIXES)


class YahooFetcher:
    """
    One `yfinance` download per symbol. Separate from the reader so the transport can be replaced
    without touching the caching or the adjustment.
    """

    def __init__(self, timeout: float = 30.0, retries: int = 3, pause: float = 1.0) -> None:
        self._timeout = timeout
        self._retries = retries
        self._pause = pause

    def fetch(self, symbol: str, interval: str, start: int, stop: int) -> pd.DataFrame:
        yf = self._yfinance()
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
                return self.normalize(frame)
            except Exception as e:  # - yfinance raises its own types; treat them all as retryable
                last = e
                if attempt < self._retries - 1:
                    time.sleep(self._pause * (attempt + 1))
        logger.error(f"[YahooFetcher] '{symbol}' failed after {self._retries} attempts: {last}")
        return self.empty_frame()

    @staticmethod
    def _yfinance():
        try:
            import yfinance  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("Yahoo storage needs the 'yahoo' extra: pip install qubx[yahoo]") from e
        return yfinance

    @staticmethod
    def empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in COLUMNS}, index=pd.DatetimeIndex([], name="timestamp")
        )

    @staticmethod
    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        """
        yfinance's frame to the column names and index this storage caches.

        Columns arrive capitalised and, for a single ticker, under a MultiIndex level naming it. The
        ticker level is dropped, names lowercased, `adj close` renamed to `adjclose`, and the index made
        tz-naive so parquet round-trips it.
        """
        if frame is None or not len(frame):
            return YahooFetcher.empty_frame()
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


class YahooFetchReader(IReader):
    """
    Calls Yahoo on every read. Returns bars as traded with `adjclose` as its own column.
    """

    def __init__(self, fetcher: YahooFetcher | None = None) -> None:
        self._fetcher = fetcher or YahooFetcher()
        self._seen: set[str] = set()

    def _read_one(self, data_id: str, dtype: DataType | str, start: str | None, stop: str | None) -> RawData:
        interval = _INTERVALS[self._timeframe_of(dtype)]
        frame = self._fetcher.fetch(
            data_id.upper(), interval, self._epoch(start, 0), self._epoch(stop, int(time.time()))
        )
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

    @staticmethod
    def _timeframe_of(dtype: DataType | str) -> str:
        """
        The Yahoo interval for a dtype, or a ValueError naming what it does serve.

        This parses the string itself instead of calling `DataType.from_str`, which cannot be used
        here: it normalises the timeframe through `to_timedelta`, and pandas rejects `M` and `Y` as
        durations, so `DataType.from_str("ohlc(1M)")` raises. Monthly is one of the three intervals
        Yahoo keeps long history for.
        """
        s = str(dtype)
        if not s.lower().startswith("ohlc"):
            raise ValueError(f"Yahoo storage serves OHLC only, got '{dtype}'")
        tf = s[s.index("(") + 1 : s.index(")")] if "(" in s else "1d"
        tf = {"1day": "1d", "d": "1d", "1D": "1d", "1week": "1w", "1month": "1M"}.get(tf, tf)
        if tf not in _INTERVALS:
            raise ValueError(
                f"Yahoo storage serves {sorted(_INTERVALS)} only, got '{tf}'. Intraday history is too short."
            )
        return tf

    @staticmethod
    def _epoch(t: str | pd.Timestamp | None, default: int) -> int:
        if t is None:
            return default
        ts = pd.Timestamp("now", tz="UTC") if str(t).lower() in ("now", "today") else pd.Timestamp(t)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return int(ts.timestamp())


class YahooReader(IReader):
    """
    Reads through the parquet cache and applies the price adjustment.

    `read(..., adjusted=True)`, the default, scales open, high and low by `adjclose / close`, puts
    `adjclose` into `close` and drops the extra column. `adjusted=False` returns bars as traded.
    Volume is left as Yahoo reports it, already split adjusted.
    """

    def __init__(self, inner: IReader, cache: ParquetCache | None = None, market: str = "STOCK") -> None:
        self._inner = inner
        self._cache = cache
        self._market = market.upper()

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
        if isinstance(data_id, (list, tuple, set)):
            requested = {self.to_yahoo(d, self._market): d for d in data_id}
            ids: str | list[str] = list(requested)
        else:
            requested = {self.to_yahoo(data_id, self._market): data_id}
            ids = next(iter(requested))
        result = self._inner.read(ids, dtype, start, stop, chunksize, **kwargs)
        return self._adjust(result, adjusted, requested)

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        """
        Symbols held in the local cache. Yahoo has no endpoint that enumerates symbols.

        Read from the on-disk index, so it survives a restart.
        """
        if self._cache is None:
            return self._inner.get_data_id(dtype)
        keys = [f"ohlc({tf})" for tf in _INTERVALS] if str(dtype) == str(DataType.ALL) else [str(dtype)]
        found: set[str] = set()
        for k in keys:
            found.update(self._cache.get_stored_ids(k))
        return sorted(self.from_yahoo(f, self._market) for f in found if self.belongs_to(f, self._market))

    def get_data_types(self, data_id: str) -> list[DataType]:
        return self._inner.get_data_types(data_id)

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[Any, Any]:
        return self._inner.get_time_range(self.to_yahoo(data_id, self._market), dtype)

    def close(self) -> None:
        self._inner.close()

    @staticmethod
    def to_yahoo(symbol: str, market: str) -> str:
        """
        Plain name to Yahoo's spelling. An already-marked symbol passes through unchanged.
        """
        prefix, suffix = MARKET_AFFIXES[market.upper()]
        s = symbol.upper()
        if market.upper() == "CRYPTO":
            return s if "-" in s else s + suffix
        if prefix and not s.startswith(prefix):
            s = prefix + s
        if suffix and not s.endswith(suffix):
            s = s + suffix
        return s

    @staticmethod
    def from_yahoo(symbol: str, market: str) -> str:
        """
        Yahoo's spelling to the plain name the caller used.
        """
        prefix, suffix = MARKET_AFFIXES[market.upper()]
        s = symbol
        if prefix and s.startswith(prefix):
            s = s[len(prefix) :]
        if suffix and s.endswith(suffix):
            s = s[: -len(suffix)]
        return s

    @staticmethod
    def belongs_to(symbol: str, market: str) -> bool:
        """
        Whether a Yahoo symbol belongs to this market type, used to filter the cache listing.
        """
        m = market.upper()
        if m == "CRYPTO":
            return "-" in symbol
        if m == "INDEX":
            return symbol.startswith("^")
        if m == "FUTURE":
            return symbol.endswith("=F")
        if m == "FX":
            return symbol.endswith("=X")
        # - STOCK / ETF / FUND carry no mark of their own
        return not (symbol.startswith("^") or symbol.endswith(("=F", "=X")) or "-" in symbol)

    @staticmethod
    def _adjust_one(raw: RawData, adjusted: bool, names: dict[str, str]) -> RawData:
        """
        Apply the price adjustment and restore the caller's symbol on the result.
        """
        display = names.get(raw.data_id, raw.data_id)
        if "adjclose" not in raw.names:
            return raw if display == raw.data_id else RawData.from_pandas(display, raw.dtype, raw.data.to_pandas())
        frame = raw.data.to_pandas()
        if adjusted and len(frame):
            ratio = (frame["adjclose"] / frame["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            for col in ("open", "high", "low"):
                if col in frame:
                    frame[col] = frame[col] * ratio
            frame["close"] = frame["adjclose"]
        return RawData.from_pandas(display, raw.dtype, frame.drop(columns=["adjclose"]))

    @staticmethod
    def _adjust(result: Any, adjusted: bool, names: dict[str, str] | None = None) -> Any:
        names = names or {}
        if isinstance(result, RawData):
            return YahooReader._adjust_one(result, adjusted, names)
        if isinstance(result, RawMultiData):
            return RawMultiData([YahooReader._adjust_one(r, adjusted, names) for r in result.data])
        if isinstance(result, Iterator):
            return (YahooReader._adjust(chunk, adjusted, names) for chunk in result)
        return result


@storage("yahoo")
class YahooStorage(IStorage):
    """
    Reader chain, built here so one copy of each symbol is cached and the adjusted view derives
    from it:

        YahooReader(adjust) -> CachedReader(ParquetCache) -> YahooFetchReader(yfinance)
    """

    def __init__(self, path: str | None = None, prefetch_period: str | None = None, **kwargs) -> None:
        self._path = Path(os.path.expanduser(path)) if path else Path(get_local_data_cache_folder("yahoo"))
        self._path.mkdir(parents=True, exist_ok=True)
        self._prefetch_period = prefetch_period
        self._fetcher_kwargs = kwargs
        self._cached: CachedReader | None = None
        self._cache: ParquetCache | None = None
        self._readers: dict[str, YahooReader] = {}

    def get_exchanges(self) -> list[str]:
        return [EXCHANGE]

    def get_market_types(self, exchange: str) -> list[str]:
        return list(MARKET_TYPES) if exchange.upper() == EXCHANGE else []

    def get_reader(self, exchange: str, market: str) -> IReader:
        if exchange.upper() != EXCHANGE:
            raise ValueError(f"Yahoo storage has one exchange, '{EXCHANGE}', not '{exchange}'")
        m = market.upper()
        if m not in MARKET_TYPES:
            raise ValueError(f"Unknown market type '{market}' for Yahoo; one of {MARKET_TYPES}")
        # - one cache for the storage: Yahoo symbols are unique across classes, so EURUSD=X and
        # - ZN=F cannot collide. Each market type gets a reader that translates to and from them.
        if self._cached is None:
            self._cache = ParquetCache(self._path)
            self._cached = CachedReader(
                YahooFetchReader(YahooFetcher(**self._fetcher_kwargs)), self._cache, self._prefetch_period
            )
        if m not in self._readers:
            self._readers[m] = YahooReader(self._cached, self._cache, m)
        return self._readers[m]

    def close(self) -> None:
        if self._cached is not None:
            self._cached.close()
        self._cached = None
        self._cache = None
        self._readers.clear()

    def __repr__(self) -> str:
        return f"YahooStorage({self._path})"
