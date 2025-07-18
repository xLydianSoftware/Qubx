from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import delayed

from qubx import logger
from qubx.core.basics import DataType, ITimeProvider
from qubx.core.series import TimeSeries
from qubx.data.readers import (
    DataReader,
    DataTransformer,
    InMemoryDataFrameReader,
    _list_to_chunked_iterator,
)
from qubx.data.registry import ReaderRegistry
from qubx.pandaz.utils import OhlcDict, generate_equal_date_ranges, ohlc_resample, srows
from qubx.utils.misc import ProgressParallel
from qubx.utils.time import handle_start_stop


class InMemoryCachedReader(InMemoryDataFrameReader):
    """
    A class for caching and reading financial data from memory.

    This class extends InMemoryDataFrameReader to provide efficient data caching and retrieval
    for financial data from a specific exchange and timeframe.
    """

    exchange: str
    _data_timeframe: str
    _reader: DataReader
    _n_jobs: int
    _start: pd.Timestamp | None = None
    _stop: pd.Timestamp | None = None
    _symbols: list[str]

    # - external data
    _external: dict[str, pd.DataFrame | pd.Series]

    def __init__(
        self,
        exchange: str,
        reader: DataReader,
        base_timeframe: str,
        n_jobs: int = -1,
        **kwargs,
    ) -> None:
        self._reader = reader
        self._n_jobs = n_jobs
        self._data_timeframe = base_timeframe
        self.exchange = exchange
        self._external = {}
        self._symbols = []

        # - copy external data
        for k, v in kwargs.items():
            if isinstance(v, (pd.DataFrame, pd.Series)):
                self._external[k] = v

        super().__init__({}, exchange)

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        # timeframe: str | None = None,
        **kwargs,
    ) -> Iterable | list:
        _s_path = data_id
        if not data_id.startswith(self.exchange):
            _s_path = f"{self.exchange}:{data_id}"
        _, symb = _s_path.split(":")

        _start = str(self._start) if start is None and self._start is not None else start
        _stop = str(self._stop) if stop is None and self._stop is not None else stop
        if _start is None or _stop is None:
            raise ValueError("Start and stop date must be provided")

        # - refresh symbol's data
        self._handle_symbols_data_from_to([symb], _start, _stop)

        # - super InMemoryDataFrameReader supports chunked reading now
        return super().read(_s_path, start, stop, transform, chunksize=chunksize, **kwargs)

    def __getitem__(self, keys) -> dict[str, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series:
        """
        This helper mostly for using in research notebooks
        """
        _start: str | None = None
        _stop: str | None = None
        _instruments: list[str] = []
        _as_dict = False

        if isinstance(keys, (tuple)):
            for k in keys:
                if isinstance(k, slice):
                    _start, _stop = k.start, k.stop
                if isinstance(k, (list, tuple, set)):
                    _instruments = list(k)
                    _as_dict = True
                if isinstance(k, str):
                    _instruments.append(k)
        else:
            if isinstance(keys, (list, tuple)):
                _instruments.extend(keys)
                _as_dict = True
            elif isinstance(keys, slice):
                _start, _stop = keys.start, keys.stop
            else:
                _instruments.append(keys)
        _as_dict |= len(_instruments) > 1

        if not _instruments:
            _instruments = list(self._data.keys())

        if not _instruments:
            raise ValueError("No instruments provided")

        if (_start is None and self._start is None) or (_stop is None and self._stop is None):
            raise ValueError("Start and stop date must be provided")

        _start = str(self._start) if _start is None else _start
        _stop = str(self._stop) if _stop is None else _stop

        _r = self._handle_symbols_data_from_to(_instruments, _start, _stop)
        if not _as_dict and len(_instruments) == 1:
            return _r.get(_instruments[0], pd.DataFrame())
        return _r

    def _load_candle_data(
        self, symbols: list[str], start: str | pd.Timestamp, stop: str | pd.Timestamp, timeframe: str
    ) -> dict[str, pd.DataFrame | pd.Series]:
        _ohlcs = defaultdict(list)
        _chunk_size_id_days = 30 * (4 if pd.Timedelta(timeframe) >= pd.Timedelta("1h") else 1)
        _ranges = list(generate_equal_date_ranges(str(start), str(stop), _chunk_size_id_days, "D"))

        # - for timeframes less than 1d generate_equal_date_ranges may skip days
        # so we need to fix intervals
        _es = list(zip(_ranges[:], _ranges[1:]))
        _es = [(start, end[0]) for (start, _), end in _es]
        _es.append((_ranges[-1][0], str(stop)))

        if self._n_jobs > 1:
            _results = ProgressParallel(n_jobs=self._n_jobs, silent=True, total=len(_ranges))(
                delayed(self._reader.get_aux_data)(
                    "candles", exchange=self.exchange, symbols=symbols, start=s, stop=e, timeframe=timeframe
                )
                for s, e in _es
            )
        else:
            _results = [
                self._reader.get_aux_data(
                    "candles", exchange=self.exchange, symbols=symbols, start=s, stop=e, timeframe=timeframe
                )
                for s, e in _es
            ]

        for (s, e), data in zip(_ranges, _results):
            assert isinstance(data, pd.DataFrame)
            try:
                # - some periods of data may be empty so just skipping it to avoid error log
                if not data.empty:
                    data_symbols = data.index.get_level_values(1).unique()
                    for smb in data_symbols:
                        _ohlcs[smb].append(data.loc[pd.IndexSlice[:, smb], :].droplevel(1))
            except Exception as exc:
                logger.warning(f"(InMemoryCachedReader) Failed to load data for {s} - {e} : {str(exc)}")

        ohlc = {smb.upper(): srows(*vs, keep="first") for smb, vs in _ohlcs.items() if len(vs) > 0}
        return ohlc

    def _handle_symbols_data_from_to(
        self, symbols: list[str], start: str, stop: str
    ) -> dict[str, pd.DataFrame | pd.Series]:
        def convert_to_timestamp(x):
            return pd.Timestamp(x)

        _start, _stop = map(convert_to_timestamp, handle_start_stop(start, stop))

        # - full interval
        _new_symbols = list(set([s for s in symbols if s not in self._data]))
        if _new_symbols:
            _s_req = min(_start, self._start if self._start else _start)
            _e_req = max(_stop, self._stop if self._stop else _stop)
            logger.debug(f"(InMemoryCachedReader) Loading all data {_s_req} - {_e_req} for {','.join(_new_symbols)} ")
            # _new_data = self._load_candle_data(_new_symbols, _s_req, _e_req + _dtf, self._data_timeframe)
            _new_data = self._load_candle_data(_new_symbols, _s_req, _e_req, self._data_timeframe)
            self._data |= _new_data

        # - pre intervals
        if self._start and _start < self._start:
            _smbs = list(self._data.keys())
            logger.debug(f"(InMemoryCachedReader) Updating {len(_smbs)} symbols pre interval {_start} : {self._start}")
            # _before = self._load_candle_data(_smbs, _start, self._start + _dtf, self._data_timeframe)
            _before = self._load_candle_data(_smbs, _start, self._start, self._data_timeframe)
            for k, c in _before.items():
                # self._data[k] = srows(c, self._data[k], keep="first")
                self._data[k] = srows(c, self._data[k], keep="last")

        # - post intervals
        if self._stop and _stop > self._stop:
            _smbs = list(self._data.keys())
            logger.debug(f"(InMemoryCachedReader) Updating {len(_smbs)} symbols post interval {self._stop} : {_stop}")
            # _after = self._load_candle_data(_smbs, self._stop - _dtf, _stop, self._data_timeframe)
            _after = self._load_candle_data(_smbs, self._stop, _stop, self._data_timeframe)
            for k, c in _after.items():
                self._data[k] = srows(self._data[k], c, keep="last")

        self._start = min(_start, self._start if self._start else _start)
        self._stop = max(_stop, self._stop if self._stop else _stop)

        return OhlcDict({s: self._data[s].loc[_start:_stop] for s in symbols if s in self._data})

    def get_aux_data_ids(self) -> set[str]:
        return self._reader.get_aux_data_ids() | set(self._external.keys())

    def get_aux_data(self, data_id: str, **kwargs) -> Any:
        _exch = kwargs.pop("exchange") if "exchange" in kwargs else None
        if _exch and _exch != self.exchange:
            raise ValueError(f"Exchange mismatch: expected {self.exchange}, got {_exch}")

        match data_id:
            # - special case for candles - it builds them from loaded ohlc data
            case "candles":
                return self._get_candles(**kwargs)

            # - only symbols in cache
            case "symbols":
                return list(self._data.keys())

        if data_id not in self._external:
            self._external[data_id] = self._reader.get_aux_data(data_id, exchange=self.exchange, **kwargs)

        _ext_data = self._external.get(data_id)
        if _ext_data is not None:
            _s, _e = kwargs.pop("start", None), kwargs.pop("stop", None)
            # if isinstance(_s, str):
            #     _s = pd.Timestamp(_s)
            # if isinstance(_e, str):
            #     _e = pd.Timestamp(_e)

            _get_idx_at = lambda x, n: x.index[n][0] if isinstance(x.index, pd.MultiIndex) else x.index[n]

            # - extends actual data if need
            try:
                _ds, _de = _get_idx_at(_ext_data, 0), _get_idx_at(_ext_data, -1)
                if (_s and _ds > _s) or (_e and _de < _e):
                    self._external[data_id] = (
                        _ext_data := self._reader.get_aux_data(
                            data_id,
                            exchange=self.exchange,
                            start=min(_s, _ds) if _s else _ds,
                            stop=max(_e, _de) if _e else _de,
                            **kwargs,
                        )
                    )
                    # print(f"reloading -> {_get_idx_at(_ext_data, 0)} : {_get_idx_at(_ext_data, -1)}")
            except Exception as exc:
                # - if failed to extend data - just return actual data
                logger.warning(f"(InMemoryCachedReader) Failed to extend aux data for {data_id} : {str(exc)}")

            _ext_data = _ext_data[:_e] if _e else _ext_data
            _ext_data = _ext_data[_s:] if _s else _ext_data
        return _ext_data

    def _get_candles(
        self,
        symbols: list[str],
        start: str | pd.Timestamp,
        stop: str | pd.Timestamp,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        _xd: dict[str, pd.DataFrame] = self[symbols, start:stop]
        _xd = ohlc_resample(_xd, timeframe) if timeframe else _xd
        _r = [x.assign(symbol=s.upper(), timestamp=x.index) for s, x in _xd.items()]
        return srows(*_r).set_index(["timestamp", "symbol"])

    def get_names(self, **kwargs) -> list[str]:
        return self._reader.get_names(**kwargs)

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        if not self._symbols:
            self._symbols = self._reader.get_symbols(self.exchange, DataType.OHLC)
        return self._symbols

    def get_time_ranges(self, symbol: str, dtype: DataType) -> tuple[Any, Any]:
        _id = f"{self.exchange}:{symbol}" if not symbol.startswith(self.exchange) else symbol
        return self._reader.get_time_ranges(_id, dtype)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(exchange={self.exchange},timeframe={self._data_timeframe})"


class TimeGuardedWrapper(DataReader):
    # - currently 'known' time, can be used for limiting data
    _time_guard_provider: ITimeProvider
    _reader: InMemoryCachedReader

    def __init__(
        self,
        reader: InMemoryCachedReader,
        time_guard: ITimeProvider | None = None,
    ) -> None:
        # - if no time provider is provided, use stub
        class _NoTimeGuard(ITimeProvider):
            def time(self) -> np.datetime64 | None:
                return None

        self._time_guard_provider = time_guard if time_guard is not None else _NoTimeGuard()
        self._reader = reader

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        # timeframe: str | None = None,
        **kwargs,
    ) -> Iterable | list:
        xs = self._time_guarded_data(
            self._reader.read(data_id, start=start, stop=stop, transform=transform, chunksize=0, **kwargs),  # type: ignore
            prev_bar=True,
        )
        return _list_to_chunked_iterator(xs, chunksize) if chunksize > 0 else xs

    def get_aux_data(self, data_id: str, **kwargs) -> Any:
        return self._time_guarded_data(self._reader.get_aux_data(data_id, **kwargs))

    def __getitem__(self, keys):
        return self._time_guarded_data(self._reader.__getitem__(keys), prev_bar=True)

    def _time_guarded_data(
        self, data: pd.DataFrame | pd.Series | dict[str, pd.DataFrame | pd.Series] | list, prev_bar: bool = False
    ) -> pd.DataFrame | pd.Series | dict[str, pd.DataFrame | pd.Series] | list:
        """
        This function is responsible for limiting the data based on a given time guard.

        Parameters:
        - data (pd.DataFrame | pd.Series | Dict[str, pd.DataFrame | pd.Series] | List): The data to be limited.
        - prev_bar (bool, optional): If True, the time guard is applied to the previous bar. Defaults to False.

        Returns:
        - pd.DataFrame | pd.Series | Dict[str, pd.DataFrame | pd.Series] | List: The limited data.
        """
        # - when no any limits - just returns it as is
        if (_c_time := self._time_guard_provider.time()) is None:
            return data

        def cut_dict(xs, t):
            return OhlcDict({s: v.loc[:t] for s, v in xs.items()})

        def cut_list_of_timestamped(xs, t):
            return list(filter(lambda x: x.time <= t, xs))

        def cut_list_raw(xs, t):
            return list(filter(lambda x: x[0] <= t, xs))

        def cut_time_series(ts, t):
            return ts.loc[: str(t)]

        def cut_dataframe(ts: pd.DataFrame | pd.Series, t):
            if isinstance(ts, pd.DataFrame):
                # - special case for slicing some fundamental data
                if isinstance(ts.index, pd.MultiIndex):
                    return ts.loc[:_c_time, :]
            return ts.loc[:t]

        if prev_bar:
            _c_time = _c_time - pd.Timedelta(self._reader._data_timeframe)

        match data:
            # - input is Dict[str, pd.DataFrame]
            case dict():
                return cut_dict(data, _c_time)

            # - input is List[(time, *data)] or List[Quote | Trade | Bar]
            case list():
                if isinstance(data[0], (list, tuple, np.ndarray)):
                    return cut_list_raw(data, _c_time)
                else:
                    return cut_list_of_timestamped(data, _c_time.asm8.item())

            # - input is TimeSeries
            case TimeSeries():
                return cut_time_series(data, _c_time)

            # - input is frame or series
            case pd.DataFrame() | pd.Series():
                return cut_dataframe(data, _c_time)

        raise ValueError(f"Unsupported data type {type(data)} !")

    def __str__(self) -> str:
        return f"TimeGuarded @ {str(self._reader)}"


def loader(
    exchange: str, timeframe: str | None, *symbols: list[str], source: str = "mqdb::localhost", no_cache=False, **kwargs
) -> DataReader:
    """
    Create and initialize an InMemoryCachedReader for a specific exchange and timeframe.

    This function sets up a cached reader for financial data, optionally pre-loading
    data for specified symbols from the beginning of time until now.

    Args:
        exchange (str): The name of the exchange to load data from.
        timeframe (str): The timeframe to use for the data (e.g., "1d", "4h").
        *symbols (List[str]): Optional list of symbols to pre-load.
        source (str): The data source to use, in the format "reader_type::connection_string".
        no_cache (bool): If True, don't cache the data in memory.
        **kwargs: Additional arguments to pass to the InMemoryCachedReader.

    Returns:
        DataReader: A configured data reader.

    Examples:
        >>> d = loader("BINANCE", "1d", "BTCUSDT", "ETHUSDT", source="mqdb::localhost")
        >>> d["BTCUSDT"].close.plot()
        >>> d["ETHUSDT", "BTCUSDT"].close.corr()
        >>> print(d('1d'))
        >>> d("4h").close.pct_change(fill_method=None).cov()
    """
    if not source:
        raise ValueError("Source parameter must be provided")

    try:
        # Use ReaderRegistry to get the reader instance
        reader_object = ReaderRegistry.get(source)
    except ValueError as e:
        raise ValueError(f"Failed to create reader from source '{source}': {e}")

    inmcr = reader_object
    # - if not need to cache data
    if not no_cache:
        inmcr = InMemoryCachedReader(exchange, reader_object, timeframe, **kwargs)
        if symbols:
            # by default slicing from 1970-01-01 until now
            inmcr[list(symbols), slice("1970-01-01", str(pd.Timestamp("now")))]

    return inmcr


class CachedPrefetchReader(DataReader):
    """
    A caching wrapper for any DataReader that supports prefetching of auxiliary data.

    This class wraps any DataReader implementation and provides:
    - Caching of auxiliary data with configurable prefetch period
    - Pass-through of read operations to the underlying reader
    - Memory management with configurable cache size limits

    Args:
        reader: The DataReader to wrap
        prefetch_period: Period to prefetch ahead (default "1w")
        cache_size_mb: Maximum cache size in MB (default 1000)
    """

    def __init__(self, reader: DataReader, prefetch_period: str = "1w", cache_size_mb: int = 1000, **kwargs) -> None:
        self._reader = reader
        self._prefetch_period = pd.Timedelta(prefetch_period)
        self._aux_cache = {}  # Cache for aux data only
        self._aux_cache_ranges = {}  # Track cached time ranges as list of (start, stop) tuples
        self._cache_size_mb = cache_size_mb
        self._cache_stats = {"hits": 0, "misses": 0}

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize: int = 0,
        data_type: str = "candles",
        **kwargs,
    ) -> Iterable | list:
        """
        Read operation - currently passes through to underlying reader.

        Args:
            data_id: The identifier for the data to be read
            start: Start time for the data range
            stop: Stop time for the data range
            transform: Data transformer instance
            chunksize: Size of data chunks (0 for all data)
            data_type: Type of data to read (OHLC, QUOTE, TRADE, etc.)
            **kwargs: Additional parameters

        Returns:
            Data as list (chunksize=0) or iterator (chunksize>0)
        """
        # PASS-THROUGH: Direct delegation to base reader
        return self._reader.read(data_id, start, stop, transform, chunksize, data_type=data_type, **kwargs)

    def get_aux_data(self, data_id: str, **kwargs) -> Any:
        """
        Get auxiliary data with caching and prefetch support.

        Args:
            data_id: Identifier for the auxiliary data
            **kwargs: Additional parameters including exchange, symbols, start, stop, etc.

        Returns:
            The auxiliary data
        """
        # Generate cache key for this request
        cache_key = self._generate_aux_cache_key(data_id, **kwargs)

        # Check if we have cached data that covers the requested range
        if cache_key in self._aux_cache:
            cached_ranges = self._aux_cache_ranges.get(cache_key, [])
            requested_start = kwargs.get("start")
            requested_stop = kwargs.get("stop")

            # If no time range requested, or cached ranges cover requested range
            if self._cache_covers_range(cached_ranges, requested_start, requested_stop):
                self._cache_stats["hits"] += 1
                return self._filter_aux_data_to_requested_range(self._aux_cache[cache_key], kwargs)

        # Try to find a compatible cache entry (broader scope) that can satisfy the request
        compatible_cache_key = self._find_compatible_cache_entry(data_id, **kwargs)
        if compatible_cache_key:
            self._cache_stats["hits"] += 1
            cached_data = self._aux_cache[compatible_cache_key]
            # Filter by symbols first, then by time range
            filtered_data = self._filter_cached_data_by_symbols(cached_data, kwargs.get("symbols", []))
            return self._filter_aux_data_to_requested_range(filtered_data, kwargs)

        # Cache miss - fetch with prefetch
        self._cache_stats["misses"] += 1
        return self._fetch_and_cache_aux_data(data_id, cache_key, **kwargs)

    def _generate_aux_cache_key(self, data_id: str, **kwargs) -> str:
        """
        Generate a cache key for auxiliary data requests.

        For time-based requests, we use a base key (without time parameters)
        to allow cache reuse across overlapping time ranges.

        Args:
            data_id: Identifier for the auxiliary data
            **kwargs: Additional parameters

        Returns:
            A unique cache key string
        """
        # Start with data_id
        key_parts = [data_id]

        # Add all kwargs except time-related ones for better cache reuse
        for k, v in sorted(kwargs.items()):
            # Skip time-related parameters for cache key
            if k in ["start", "stop"]:
                continue

            # Handle list/tuple parameters specially
            if isinstance(v, (list, tuple)):
                v_str = ",".join(str(item) for item in v)
            else:
                v_str = str(v)
            key_parts.extend([k, v_str])

        return "|".join(key_parts)

    def _cache_covers_range(
        self, cached_ranges: list[tuple] | None, requested_start: str | None, requested_stop: str | None
    ) -> bool:
        """
        Check if cached time ranges cover the requested range.

        Args:
            cached_ranges: List of (start, stop) tuples for cached data, or None
            requested_start: Requested start time
            requested_stop: Requested stop time

        Returns:
            True if cached ranges fully cover requested range
        """
        # If no time range requested, any cached data is valid
        if not requested_start and not requested_stop:
            return True

        # If no cached range info, assume no coverage for time-based requests
        if not cached_ranges:
            return False

        # Convert requested times to timestamps
        try:
            req_start = pd.Timestamp(requested_start) if requested_start else None
            req_stop = pd.Timestamp(requested_stop) if requested_stop else None
            
            # Check if the requested range is covered by any combination of cached ranges
            # First, merge overlapping/adjacent cached ranges
            merged_ranges = self._merge_time_ranges(cached_ranges)
            
            # Check if any merged range covers the requested range
            for cached_start, cached_stop in merged_ranges:
                cached_start_ts = pd.Timestamp(cached_start) if cached_start else None
                cached_stop_ts = pd.Timestamp(cached_stop) if cached_stop else None
                
                # Check if this cached range covers the requested range
                covers_start = not req_start or (cached_start_ts and cached_start_ts <= req_start)
                covers_stop = not req_stop or (cached_stop_ts and cached_stop_ts >= req_stop)
                
                if covers_start and covers_stop:
                    return True
                    
            return False
        except Exception:
            # If timestamp conversion fails, assume no coverage
            return False
            
    def _merge_time_ranges(self, ranges: list[tuple]) -> list[tuple]:
        """
        Merge overlapping or adjacent time ranges.
        
        Args:
            ranges: List of (start, stop) tuples
            
        Returns:
            List of merged (start, stop) tuples
        """
        if not ranges:
            return []
            
        # Convert to timestamps and sort by start time
        ts_ranges = []
        for start, stop in ranges:
            try:
                ts_start = pd.Timestamp(start) if start else pd.Timestamp.min
                ts_stop = pd.Timestamp(stop) if stop else pd.Timestamp.max
                ts_ranges.append((ts_start, ts_stop))
            except Exception:
                continue
                
        if not ts_ranges:
            return []
            
        ts_ranges.sort(key=lambda x: x[0])
        
        # Merge overlapping/adjacent ranges
        merged = [ts_ranges[0]]
        for start, stop in ts_ranges[1:]:
            last_start, last_stop = merged[-1]
            
            # Check if ranges overlap or are adjacent (within 1 day)
            if start <= last_stop + pd.Timedelta(days=1):
                # Merge ranges
                merged[-1] = (last_start, max(last_stop, stop))
            else:
                # Add as new range
                merged.append((start, stop))
                
        # Convert back to string format
        return [(str(start) if start != pd.Timestamp.min else None,
                 str(stop) if stop != pd.Timestamp.max else None)
                for start, stop in merged]

    def _find_compatible_cache_entry(self, data_id: str, **kwargs) -> str | None:
        """
        Find a compatible cache entry that can satisfy the request.
        
        Looks for cache entries with broader scope (e.g., all symbols or superset of symbols) 
        that contain the requested data and can be filtered to satisfy the request.
        
        Args:
            data_id: Identifier for the auxiliary data
            **kwargs: Request parameters
            
        Returns:
            Compatible cache key if found, None otherwise
        """
        requested_symbols = kwargs.get("symbols")
        requested_start = kwargs.get("start")
        requested_stop = kwargs.get("stop")
        
        # Only try to find compatible entry if specific symbols are requested
        if not requested_symbols:
            return None
            
        # Option 1: Look for cache entry without symbols (for "all symbols" lookup)
        kwargs_without_symbols = {k: v for k, v in kwargs.items() if k != "symbols"}
        broader_cache_key = self._generate_aux_cache_key(data_id, **kwargs_without_symbols)
        
        # Check if broader cache entry exists and covers the request
        if broader_cache_key in self._aux_cache:
            broader_ranges = self._aux_cache_ranges.get(broader_cache_key, [])
            if self._cache_covers_range(broader_ranges, requested_start, requested_stop):
                cached_data = self._aux_cache[broader_cache_key]
                if self._can_filter_by_symbols(cached_data, requested_symbols):
                    return broader_cache_key
                    
        # Option 2: Look for cache entries with symbols that contain the requested symbols
        # Check all existing cache entries for potential matches
        for cache_key in self._aux_cache:
            # Skip the exact cache key (already checked in main method)
            if cache_key == self._generate_aux_cache_key(data_id, **kwargs):
                continue
                
            # Check if this cache entry covers the time range
            cache_ranges = self._aux_cache_ranges.get(cache_key, [])
            if not self._cache_covers_range(cache_ranges, requested_start, requested_stop):
                continue
                
            # Check if the cached data can be filtered by the requested symbols
            cached_data = self._aux_cache[cache_key]
            if self._can_filter_by_symbols(cached_data, requested_symbols):
                return cache_key
                
        return None

    def _can_filter_by_symbols(self, cached_data: Any, requested_symbols: list) -> bool:
        """
        Check if cached data can be filtered by the requested symbols.
        
        Args:
            cached_data: The cached data
            requested_symbols: List of symbols to filter by
            
        Returns:
            True if data can be filtered by symbols, False otherwise
        """
        # Only DataFrames with MultiIndex can be filtered by symbols
        if not isinstance(cached_data, pd.DataFrame) or not isinstance(cached_data.index, pd.MultiIndex):
            return False
            
        # Find symbol level in MultiIndex
        symbol_level = None
        for i, name in enumerate(cached_data.index.names):
            if name in ["symbol", "ticker", "instrument"]:
                symbol_level = i
                break
                
        if symbol_level is None:
            return False
            
        # Check if all requested symbols are present in the cached data
        try:
            cached_symbols = set(cached_data.index.get_level_values(symbol_level).unique())
            requested_symbols_set = set(requested_symbols)
            return requested_symbols_set.issubset(cached_symbols)
        except Exception:
            return False

    def _filter_cached_data_by_symbols(self, cached_data: Any, requested_symbols: list) -> Any:
        """
        Filter cached data by the requested symbols.
        
        Args:
            cached_data: The cached data (should be MultiIndex DataFrame)
            requested_symbols: List of symbols to filter by
            
        Returns:
            Filtered data containing only the requested symbols
        """
        # Only filter DataFrames with MultiIndex
        if not isinstance(cached_data, pd.DataFrame) or not isinstance(cached_data.index, pd.MultiIndex):
            return cached_data
            
        # Find symbol level in MultiIndex
        symbol_level = None
        for i, name in enumerate(cached_data.index.names):
            if name in ["symbol", "ticker", "instrument"]:
                symbol_level = i
                break
                
        if symbol_level is None:
            return cached_data
            
        # Filter by symbols using IndexSlice
        try:
            idx = pd.IndexSlice
            if symbol_level == 0:
                # Symbol is first level: (symbol, timestamp) or (symbol, other)
                filtered_data = cached_data.loc[idx[requested_symbols, :], :]
            elif symbol_level == 1:
                # Symbol is second level: (timestamp, symbol) or (other, symbol)
                filtered_data = cached_data.loc[idx[:, requested_symbols], :]
            else:
                # Symbol is in higher level - more complex slicing needed
                # For now, return as-is (can be extended later if needed)
                return cached_data
                
            # Sort the filtered data to ensure proper MultiIndex ordering
            # This is important for subsequent time-based slicing operations
            return filtered_data.sort_index()
        except Exception as e:
            # If filtering fails, return original data
            logger.warning(f"Symbol filtering failed: {e}")
            return cached_data

    def _filter_aux_data_to_requested_range(self, cached_data: Any, kwargs: dict) -> Any:
        """
        Filter cached auxiliary data to the requested range.

        Args:
            cached_data: The cached data (potentially with extended range)
            kwargs: Original request parameters

        Returns:
            Data filtered to the requested range
        """
        # Extract time range parameters
        start = kwargs.get("start")
        stop = kwargs.get("stop")

        # If no time filtering requested, return as-is
        if not start and not stop:
            return cached_data

        # Handle pandas DataFrame/Series with time-based index
        if isinstance(cached_data, (pd.DataFrame, pd.Series)):
            # Handle MultiIndex with 'timestamp' level
            if isinstance(cached_data.index, pd.MultiIndex):
                if "timestamp" in cached_data.index.names:
                    # Use pandas IndexSlice for MultiIndex slicing on timestamp level
                    idx = pd.IndexSlice
                    if start and stop:
                        return cached_data.loc[idx[start:stop, :]]
                    elif start:
                        return cached_data.loc[idx[start:, :]]
                    elif stop:
                        return cached_data.loc[idx[:stop, :]]
            # Handle regular time-based index (DatetimeIndex or other time-based indices)
            elif isinstance(cached_data.index, pd.DatetimeIndex) or hasattr(cached_data.index, "to_timestamp"):
                # Handle time-based filtering
                if start and stop:
                    return cached_data.loc[start:stop]
                elif start:
                    return cached_data.loc[start:]
                elif stop:
                    return cached_data.loc[:stop]

        # For other data types, return as-is for now
        return cached_data

    def _fetch_and_cache_aux_data(self, data_id: str, cache_key: str, **kwargs) -> Any:
        """
        Fetch auxiliary data with prefetch and cache it, merging with existing cached data.

        Args:
            data_id: Identifier for the auxiliary data
            cache_key: Cache key for this request
            **kwargs: Request parameters

        Returns:
            The auxiliary data (filtered to requested range)
        """
        # Extract time range parameters
        start = kwargs.get("start")
        stop = kwargs.get("stop")

        if start and stop:
            # Calculate extended range with prefetch
            try:
                extended_stop = pd.Timestamp(stop) + self._prefetch_period
                extended_kwargs = kwargs.copy()
                extended_kwargs["stop"] = str(extended_stop)

                # Fetch extended range
                new_data = self._reader.get_aux_data(data_id, **extended_kwargs)
                fetch_range = (start, str(extended_stop))

                # Merge with existing cached data if any
                if cache_key in self._aux_cache:
                    merged_data = self._merge_aux_data(self._aux_cache[cache_key], new_data)
                    self._aux_cache[cache_key] = merged_data
                    # Add the new range to existing ranges
                    self._aux_cache_ranges[cache_key].append(fetch_range)
                else:
                    # First time caching this key
                    self._aux_cache[cache_key] = new_data
                    self._aux_cache_ranges[cache_key] = [fetch_range]

                # Return only requested range
                return self._filter_aux_data_to_requested_range(self._aux_cache[cache_key], kwargs)

            except Exception as e:
                # If prefetch fails, fall back to exact range
                logger.warning(f"Prefetch failed for {data_id}: {e}, falling back to exact range")

        # No time range or prefetch failed - fetch and cache as-is
        new_data = self._reader.get_aux_data(data_id, **kwargs)
        
        # Merge with existing cached data if any
        if cache_key in self._aux_cache:
            if start and stop:
                # Time-based data - merge
                merged_data = self._merge_aux_data(self._aux_cache[cache_key], new_data)
                self._aux_cache[cache_key] = merged_data
                # Add the new range to existing ranges
                fetch_range = (start, stop)
                self._aux_cache_ranges[cache_key].append(fetch_range)
            else:
                # Non-time-based data - overwrite
                self._aux_cache[cache_key] = new_data
                self._aux_cache_ranges[cache_key] = [None]
        else:
            # First time caching this key
            self._aux_cache[cache_key] = new_data
            if start and stop:
                self._aux_cache_ranges[cache_key] = [(start, stop)]
            else:
                self._aux_cache_ranges[cache_key] = [None]

        return new_data
        
    def _merge_aux_data(self, existing_data: Any, new_data: Any) -> Any:
        """
        Merge new auxiliary data with existing cached data.
        
        Args:
            existing_data: Previously cached data
            new_data: New data to merge
            
        Returns:
            Merged data
        """
        # Handle pandas DataFrames/Series
        if isinstance(existing_data, (pd.DataFrame, pd.Series)) and isinstance(new_data, (pd.DataFrame, pd.Series)):
            try:
                # Concatenate and remove duplicates, keeping the most recent data
                combined = pd.concat([existing_data, new_data])
                
                # Remove duplicates based on index, keeping last occurrence (most recent)
                if hasattr(combined, 'index'):
                    # For DataFrames/Series, drop duplicates on index
                    combined = combined[~combined.index.duplicated(keep='last')]
                    
                    # Sort by index to maintain order
                    combined = combined.sort_index()
                    
                return combined
            except Exception:
                # If merging fails, return new data
                return new_data
        
        # For other data types, return new data (overwrite)
        return new_data

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        """Delegate to underlying reader."""
        return self._reader.get_symbols(exchange, dtype)

    def get_time_ranges(self, symbol: str, dtype: str) -> tuple[Any, Any]:
        """Delegate to underlying reader."""
        return self._reader.get_time_ranges(symbol, dtype)

    def get_names(self, **kwargs) -> list[str]:
        """Delegate to underlying reader."""
        return self._reader.get_names(**kwargs)

    def get_aux_data_ids(self) -> set[str]:
        """Delegate to underlying reader."""
        return self._reader.get_aux_data_ids()

    def get_cache_stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hits, misses, and other stats
        """
        return self._cache_stats.copy()

    def clear_cache(self, data_id: str | None = None) -> None:
        """
        Clear the cache.

        Args:
            data_id: If provided, clear only entries for this data_id.
                    If None, clear entire cache.
        """
        if data_id is None:
            self._aux_cache.clear()
            self._aux_cache_ranges.clear()
        else:
            # Clear entries that start with the data_id
            keys_to_remove = [k for k in self._aux_cache.keys() if k.startswith(data_id)]
            for key in keys_to_remove:
                del self._aux_cache[key]
                if key in self._aux_cache_ranges:
                    del self._aux_cache_ranges[key]

    def prefetch_aux_data(
        self,
        aux_data_names: list[str],
        start: str,
        stop: str,
        exchange: str | None = None,
        symbols: list[str] | None = None,
        **kwargs
    ) -> dict[str, int]:
        """
        Prefetch multiple auxiliary data types into the cache.
        
        This method is useful for warming up the cache and debugging. It loads
        the specified auxiliary data types for the given time range and parameters,
        updating the cache with the fetched data.
        
        Args:
            aux_data_names: List of auxiliary data identifiers to prefetch (e.g., ["candles", "funding"])
            start: Start time for the data range (required)
            stop: Stop time for the data range (required)
            exchange: Exchange identifier (optional)
            symbols: List of symbols to fetch (optional)
            **kwargs: Additional parameters to pass to get_aux_data
            
        Returns:
            Dictionary mapping data type names to the number of elements fetched
            
        Example:
            >>> reader = CachedPrefetchReader(base_reader)
            >>> results = reader.prefetch_aux_data(
            ...     ["candles", "funding"],
            ...     start="2023-01-01",
            ...     stop="2023-01-10",
            ...     exchange="BINANCE.UM",
            ...     symbols=["BTCUSDT", "ETHUSDT"]
            ... )
            >>> print(results)
            {'candles': 20, 'funding': 240}
        """
        results = {}
        
        # Build base parameters
        base_params = kwargs.copy()
        if exchange is not None:
            base_params["exchange"] = exchange
        if symbols is not None:
            base_params["symbols"] = symbols
        base_params["start"] = start
        base_params["stop"] = stop
        
        # Track initial cache stats to measure what was actually fetched
        initial_cache_keys = set(self._aux_cache.keys())
        
        # Fetch each auxiliary data type
        for data_name in aux_data_names:
            try:
                # Get the data (this will populate the cache)
                data = self.get_aux_data(data_name, **base_params)
                
                # Count the elements in the fetched data
                if data is None:
                    count = 0
                elif hasattr(data, '__len__'):
                    count = len(data)
                elif hasattr(data, 'shape'):
                    # For numpy arrays or similar
                    count = data.shape[0] if len(data.shape) > 0 else 1
                else:
                    # For scalar values or unknown types
                    count = 1
                    
                results[data_name] = count
                
            except Exception as e:
                logger.warning(f"Failed to prefetch {data_name}: {e}")
                results[data_name] = 0
                
        # Log cache statistics for debugging
        new_cache_keys = set(self._aux_cache.keys())
        added_keys = new_cache_keys - initial_cache_keys
        
        if added_keys:
            logger.info(f"Prefetch added {len(added_keys)} new cache entries: {list(added_keys)}")
        else:
            logger.info("Prefetch completed using existing cache entries")
            
        return results

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(reader={self._reader}, prefetch_period={self._prefetch_period})"
