import ast
import re
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
    _reader: DataReader

    def __init__(
        self,
        reader: DataReader,
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
            timeframe=kwargs.get("timeframe", "1m"),
            prev_bar=True,
        )
        return _list_to_chunked_iterator(xs, chunksize) if chunksize > 0 else xs

    def get_aux_data(self, data_id: str, **kwargs) -> Any:
        aux_data = self._reader.get_aux_data(data_id, **kwargs)
        if data_id == "candles":
            return self._time_guarded_data(aux_data, timeframe=kwargs.get("timeframe", "1d"), prev_bar=True)
        else:
            return self._time_guarded_data(aux_data)

    def __getitem__(self, keys):
        if hasattr(self._reader, "__getitem__"):
            return self._time_guarded_data(getattr(self._reader, "__getitem__")(keys), prev_bar=True)
        raise NotImplementedError("__getitem__ is not implemented for this reader")

    def _time_guarded_data(
        self,
        data: pd.DataFrame | pd.Series | dict[str, pd.DataFrame | pd.Series] | list,
        timeframe: str | None = None,
        prev_bar: bool = False,
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

        if prev_bar and timeframe:
            _c_time = _c_time - pd.Timedelta(timeframe)

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
        self._read_cache = {}  # Cache for read data (raw records)
        self._read_cache_ranges = {}  # Track cached time ranges for read data
        self._read_cache_columns = {}  # Cache for column names corresponding to read cache keys
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
        Read operation with caching and prefetch support.

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
        if "timeframe" in kwargs:
            if kwargs["timeframe"] is not None:
                kwargs["timeframe"] = kwargs["timeframe"].lower()
            else:
                kwargs.pop("timeframe")

        # Normalize ohlc to candles
        if DataType.OHLC == data_type:
            _, _kwargs = DataType.from_str(data_type)
            if "timeframe" in _kwargs:
                kwargs["timeframe"] = _kwargs["timeframe"].lower()
            data_type = "candles"

        if start is not None and stop is not None and pd.Timestamp(start) > pd.Timestamp(stop):
            start, stop = stop, start

        # Prepare kwargs for cache key generation
        cache_kwargs = {"start": start, "stop": stop, "data_type": data_type}
        cache_kwargs.update(kwargs)

        # Generate cache key for this read request
        cache_key = self._generate_cache_key("read", data_id, **cache_kwargs)

        # Check if we have cached read data that covers the requested range
        cached_data = None
        cached_columns = None
        if cache_key in self._read_cache and cache_key in self._read_cache_columns:
            cached_ranges = self._read_cache_ranges.get(cache_key, [])

            # Check if cached ranges cover the requested range
            if self._cache_covers_range(cached_ranges, start, stop):
                self._cache_stats["hits"] += 1
                cached_data = self._filter_read_data_to_requested_range(self._read_cache[cache_key], cache_kwargs)
                cached_columns = self._read_cache_columns[cache_key]
                logger.debug(f"Read cache hit for {data_id} with {len(cached_data)} records")

        # If no cached data, try to find aux data overlap
        aux_columns = None
        if cached_data is None:
            # Remove data_type from kwargs to avoid duplicate argument
            aux_kwargs = {k: v for k, v in cache_kwargs.items() if k != "data_type"}
            aux_overlap = self._detect_aux_data_overlap(data_id, data_type, **aux_kwargs)
            if aux_overlap:
                aux_data_type, aux_data = aux_overlap
                # Convert aux data to read format
                records, columns = self._convert_aux_data_to_read_format(aux_data, aux_data_type)
                if records:
                    self._cache_stats["hits"] += 1
                    cached_data = records
                    aux_columns = columns  # Store column names from aux data
                    logger.debug(
                        f"Aux data overlap found for {data_id}: using {aux_data_type} with {len(records)} records"
                    )

        # Handle chunked vs non-chunked reading
        if chunksize > 0:
            # For chunked reading, create an iterator that reads and caches on-the-fly
            if cached_data is None:
                self._cache_stats["misses"] += 1
                return self._create_chunked_caching_iterator(data_id, cache_key, transform, chunksize, **cache_kwargs)
            else:
                # We have cached data, create iterator over it
                # Use cached_columns (from read cache) or aux_columns (from aux data overlap)
                column_names = cached_columns or aux_columns
                if column_names is None:
                    raise ValueError(
                        f"No column names provided for cached data iterator for {data_id}. For read cache hits, "
                        "column names should be available from read cache or aux data overlap detection."
                    )
                return self._create_cached_data_iterator(
                    cached_data, data_id, transform, chunksize, column_names, **cache_kwargs
                )
        else:
            # For non-chunked reading, fetch all data and cache it
            if cached_data is None:
                self._cache_stats["misses"] += 1
                cached_data = self._fetch_and_cache_read_data(data_id, cache_key, chunksize=0, **cache_kwargs)
                try:
                    record_count = len(cached_data) if cached_data else 0
                except TypeError:
                    # Handle Mock objects in tests
                    record_count = "unknown"
                logger.debug(f"Read cache miss for {data_id}: fetched {record_count} records")
                # After fetching, get the cached column names
                cached_columns = self._read_cache_columns.get(cache_key)

            # Return all data transformed
            # Use cached_columns (from read cache) or aux_columns (from aux data overlap)
            column_names = cached_columns or aux_columns
            return self._apply_transform_to_cached_data(cached_data, data_id, transform, column_names, **cache_kwargs)

    def get_aux_data(self, data_id: str, **kwargs) -> Any:
        """
        Get auxiliary data with caching and prefetch support.

        Args:
            data_id: Identifier for the auxiliary data
            **kwargs: Additional parameters including exchange, symbols, start, stop, etc.

        Returns:
            The auxiliary data
        """
        if data_id == "candles" and "timeframe" not in kwargs:
            kwargs["timeframe"] = "1d"

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

    def _generate_cache_key(self, cache_type: str, data_id: str, **kwargs) -> str:
        """
        Generate a cache key for requests (shared utility for aux and read caching).

        For time-based requests, we use a base key (without time parameters)
        to allow cache reuse across overlapping time ranges.

        Args:
            cache_type: Type of cache ("aux" or "read")
            data_id: Identifier for the data
            **kwargs: Additional parameters

        Returns:
            A unique cache key string
        """
        # Start with cache_type and data_id
        key_parts = [cache_type, data_id]

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

    def _generate_aux_cache_key(self, data_id: str, **kwargs) -> str:
        """
        Generate a cache key for auxiliary data requests.

        Args:
            data_id: Identifier for the auxiliary data
            **kwargs: Additional parameters

        Returns:
            A unique cache key string
        """
        return self._generate_cache_key("aux", data_id, **kwargs)

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
        for range_item in ranges:
            # Skip None ranges
            if range_item is None:
                continue

            try:
                start, stop = range_item
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
        return [
            (str(start) if start != pd.Timestamp.min else None, str(stop) if stop != pd.Timestamp.max else None)
            for start, stop in merged
        ]

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

    def _merge_cached_data(self, existing_data: Any, new_data: Any, data_type: str) -> Any:
        """
        Merge new data with existing cached data (shared utility for aux and read caching).

        Args:
            existing_data: Previously cached data
            new_data: New data to merge
            data_type: Type of data being merged ("dataframe", "list", etc.)

        Returns:
            Merged data
        """
        if data_type == "dataframe":
            # Handle pandas DataFrames/Series
            if isinstance(existing_data, (pd.DataFrame, pd.Series)) and isinstance(new_data, (pd.DataFrame, pd.Series)):
                try:
                    # Concatenate and remove duplicates, keeping the most recent data
                    combined = pd.concat([existing_data, new_data])

                    # Remove duplicates based on index, keeping last occurrence (most recent)
                    if hasattr(combined, "index"):
                        # Keep last occurrence of duplicate index values
                        combined = combined[~combined.index.duplicated(keep="last")]

                        # Sort by index to maintain order
                        combined = combined.sort_index(kind="stable")

                    return combined
                except Exception:
                    # If merging fails, return new data
                    return new_data
        elif data_type == "list":
            # Handle list of records (for read caching)
            if isinstance(existing_data, list) and isinstance(new_data, list):
                try:
                    # For lists of records, we need to merge and deduplicate
                    # Assuming each record is a list/tuple with timestamp as first element
                    combined = existing_data + new_data

                    # Remove duplicates based on timestamp (first element)
                    # Keep the later occurrence (most recent)
                    seen_timestamps = set()
                    merged_data = []

                    # Process in reverse order to keep latest records
                    for record in reversed(combined):
                        if len(record) > 0:
                            try:
                                timestamp = record[0]
                                if timestamp not in seen_timestamps:
                                    seen_timestamps.add(timestamp)
                                    merged_data.append(record)
                            except (IndexError, TypeError) as e:
                                # Skip invalid records
                                logger.debug(f"Skipping invalid record during merge {record}: {e}")
                                continue

                    # Sort by timestamp (first element)
                    merged_data.sort(key=lambda x: x[0] if len(x) > 0 else 0)

                    return merged_data
                except Exception:
                    # If merging fails, return new data
                    return new_data

        # For other data types, return new data (overwrite)
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
        return self._merge_cached_data(existing_data, new_data, "dataframe")

    def _dataframe_to_records(self, df: pd.DataFrame) -> tuple[list[list], list[str]]:
        """
        Convert a pandas DataFrame to list of records format compatible with DataTransformer.

        Args:
            df: DataFrame to convert

        Returns:
            Tuple of (list of records, column names)
            - records: List of lists where each inner list is a row
            - columns: List of column names (including timestamp if it was index)
        """
        if df.empty:
            return [], []

        # Reset index to make timestamp a regular column
        # This ensures timestamp is part of the record data
        df_with_timestamp = df.reset_index()

        # Ensure we have at least one column
        if len(df_with_timestamp.columns) == 0:
            return [], []

        # Rename the index column to "timestamp"
        if df_with_timestamp.columns[0] == "index":
            df_with_timestamp.columns = ["timestamp"] + list(df_with_timestamp.columns[1:])

        # Convert to list of lists (records)
        records = df_with_timestamp.values.tolist()

        # Get column names (including timestamp column from index)
        columns = list(df_with_timestamp.columns)

        # Ensure all records have the expected number of columns
        expected_len = len(columns)
        for i, record in enumerate(records):
            if len(record) != expected_len:
                # Pad with None values if record is too short
                records[i] = list(record) + [None] * (expected_len - len(record))

        return records, columns

    def _fetch_and_cache_read_data(self, data_id: str, cache_key: str, chunksize: int = 0, **kwargs) -> Any:
        """
        Fetch read data with prefetch and cache it, merging with existing cached data.

        Args:
            data_id: Identifier for the data
            cache_key: Cache key for this request
            chunksize: Chunk size for the request
            **kwargs: Request parameters

        Returns:
            The raw data (list of records) that can be processed by DataTransformer
        """
        # Extract time range parameters
        start = kwargs.get("start")
        stop = kwargs.get("stop")

        # For read() method, don't extend the range - use exact range requested
        # Prefetching should only be used for aux data, not for main data streams
        # Fetch data using default DataTransformer to get raw records
        # Force chunksize=0 to get all data at once for caching
        raw_data_transformer = DataTransformer()
        raw_data = self._reader.read(
            data_id,
            start,
            stop,
            transform=raw_data_transformer,  # Default transformer returns list of records
            chunksize=0,  # Force all data for caching
            **{k: v for k, v in kwargs.items() if k not in ["start", "stop"]},
        )

        # Cache the column names from the transformer
        if hasattr(raw_data_transformer, "_column_names") and raw_data_transformer._column_names:
            self._read_cache_columns[cache_key] = raw_data_transformer._column_names

        # Cache the raw data
        if cache_key in self._read_cache:
            # Merge with existing cached data
            existing_data = self._read_cache[cache_key]
            merged_data = self._merge_cached_data(existing_data, raw_data, "list")
            self._read_cache[cache_key] = merged_data
        else:
            # First time caching this data
            self._read_cache[cache_key] = raw_data

        # Update cache ranges
        if start and stop:
            try:
                start_ts = pd.Timestamp(start)
                stop_ts = pd.Timestamp(stop)
                new_range = (start_ts, stop_ts)

                if cache_key in self._read_cache_ranges:
                    existing_ranges = self._read_cache_ranges[cache_key]
                    existing_ranges.append(new_range)
                    self._read_cache_ranges[cache_key] = self._merge_time_ranges(existing_ranges)
                else:
                    self._read_cache_ranges[cache_key] = [new_range]

                logger.debug(f"Updated read cache ranges for {cache_key}: {self._read_cache_ranges[cache_key]}")
            except Exception:
                # If range handling fails, mark as no time info
                self._read_cache_ranges[cache_key] = [None]
        else:
            # No time range specified
            self._read_cache_ranges[cache_key] = [None]

        # Filter to requested range and return
        if start and stop:
            return self._filter_read_data_to_requested_range(self._read_cache[cache_key], kwargs)
        else:
            return self._read_cache[cache_key]

    def _filter_read_data_to_requested_range(self, cached_data: list, kwargs: dict) -> list:
        """
        Filter cached read data to the requested time range.

        Args:
            cached_data: List of records (cached data)
            kwargs: Original request parameters

        Returns:
            Filtered list of records
        """
        # Extract time range parameters
        start = kwargs.get("start")
        stop = kwargs.get("stop")

        # If no time filtering requested, return as-is
        if not start and not stop:
            return cached_data

        # Filter records by timestamp (assuming first element is timestamp)
        if not cached_data:
            return cached_data

        try:
            start_ts = pd.Timestamp(start) if start else None
            stop_ts = pd.Timestamp(stop) if stop else None

            filtered_data = []
            for record in cached_data:
                if len(record) > 0:
                    try:
                        # Assume first element is timestamp
                        record_ts = pd.Timestamp(record[0])

                        # Check if record is within requested range
                        if start_ts and record_ts < start_ts:
                            continue
                        if stop_ts and record_ts > stop_ts:
                            continue

                        filtered_data.append(record)
                    except (ValueError, TypeError, IndexError) as e:
                        # Skip invalid records
                        logger.debug(f"Skipping invalid record {record}: {e}")
                        continue

            return filtered_data
        except Exception:
            # If filtering fails, return original data
            return cached_data

    def _apply_transform_to_cached_data(
        self, cached_data: list, data_id: str, transform: DataTransformer, aux_columns: list = None, **kwargs
    ) -> Any:
        """
        Apply the provided transform to cached data.

        Args:
            cached_data: List of records (cached data)
            data_id: Data identifier
            transform: Transform to apply
            aux_columns: Column names from aux data (if applicable)
            **kwargs: Additional parameters

        Returns:
            Transformed data
        """
        # if not cached_data:
        #     return cached_data

        # Get column names from aux_columns if available, otherwise get from underlying reader
        if aux_columns:
            column_names = aux_columns
        # If there is data we do need some column names
        elif cached_data:
            # For read cache hits, we need to get column names by calling the underlying reader
            # We'll call it with a small range just to get the column names
            raw_data_transformer = DataTransformer()

            # Call underlying reader with a minimal request to get column names
            try:
                # Try to get just one record to establish column names
                sample_data = self._reader.read(
                    data_id,
                    kwargs.get("start"),
                    kwargs.get("start"),  # Same start and stop to get minimal data
                    transform=raw_data_transformer,
                    chunksize=0,
                    **{k: v for k, v in kwargs.items() if k not in ["start", "stop"]},
                )

                # Check if transformer now has column names
                if hasattr(raw_data_transformer, "_column_names") and raw_data_transformer._column_names:
                    column_names = raw_data_transformer._column_names
                else:
                    raise ValueError(f"Could not get column names for {data_id}")
            except Exception as e:
                raise ValueError(f"Failed to get column names for cached data transformation: {e}")
        else:
            column_names = []

        try:
            # Apply transform
            # Filter out parameters that shouldn't be passed to start_transform
            transform_kwargs = {k: v for k, v in kwargs.items() if k not in ["start", "stop", "data_type"]}
            transform.start_transform(data_id, column_names, **transform_kwargs)
            transform.process_data(cached_data)
            result = transform.collect()
            return result
        except Exception as e:
            # If transform fails, return raw data
            logger.debug(f"Transform failed: {e}, returning raw data")
            return cached_data

    def _create_chunked_caching_iterator(
        self, data_id: str, cache_key: str, transform: DataTransformer, chunksize: int, **kwargs
    ) -> Iterable:
        """
        Create an iterator that reads chunks from the underlying reader and caches them on-the-fly.

        Args:
            data_id: Data identifier
            cache_key: Cache key for this request
            transform: Transform to apply to chunks
            chunksize: Size of chunks for iteration
            **kwargs: Additional parameters

        Returns:
            Iterator over transformed chunks
        """
        # Prepare parameters for underlying reader
        reader_kwargs = kwargs.copy()

        # Calculate extended range with prefetch if applicable
        start = kwargs.get("start")
        stop = kwargs.get("stop")

        # Create the DataTransformer that will be used to read raw data
        raw_data_transformer = DataTransformer()

        # Create iterator from underlying reader
        underlying_iterator = self._reader.read(
            data_id,
            start,
            stop,
            transform=raw_data_transformer,  # Use default transformer to get raw records
            chunksize=chunksize,
            **{k: v for k, v in reader_kwargs.items() if k not in ["start", "stop"]},
        )

        # Initialize cache for this key if not exists
        if cache_key not in self._read_cache:
            self._read_cache[cache_key] = []
            self._read_cache_ranges[cache_key] = []

        # Create generator that caches chunks as they're read
        def chunked_caching_generator():
            all_cached_data = []
            for chunk in underlying_iterator:
                if chunk:
                    # Cache this chunk
                    all_cached_data.extend(chunk)
                    self._read_cache[cache_key].extend(chunk)

                    # Apply transform to this chunk and yield
                    if transform:
                        # Use the raw data transformer to get column names
                        transformed_chunk = self._apply_transform_to_chunk(
                            chunk, data_id, transform, raw_data_transformer, **kwargs
                        )
                        yield transformed_chunk
                    else:
                        yield chunk

            # After all chunks are processed, update cache ranges
            self._update_read_cache_ranges(cache_key, all_cached_data, **kwargs)

            # Cache the column names from the raw data transformer
            if hasattr(raw_data_transformer, "_column_names") and raw_data_transformer._column_names:
                self._read_cache_columns[cache_key] = raw_data_transformer._column_names

            logger.debug(f"Cached {len(all_cached_data)} records for {cache_key}")

        return chunked_caching_generator()

    def _apply_transform_to_chunk(
        self,
        chunk_data: list,
        data_id: str,
        transform: DataTransformer,
        raw_data_transformer: DataTransformer,
        **kwargs,
    ) -> Any:
        """
        Apply transform to a single chunk of data.

        Args:
            chunk_data: List of records for this chunk
            data_id: Data identifier
            transform: Transform to apply
            **kwargs: Additional parameters

        Returns:
            Transformed chunk
        """
        if not chunk_data:
            return chunk_data

        try:
            # Get column names from the raw data transformer
            if not hasattr(raw_data_transformer, "_column_names") or not raw_data_transformer._column_names:
                raise ValueError(f"Raw data transformer does not have column names for {data_id}")

            column_names = raw_data_transformer._column_names

            # Apply transform to this chunk
            # Filter out parameters that shouldn't be passed to start_transform
            transform_kwargs = {k: v for k, v in kwargs.items() if k not in ["start", "stop", "data_type"]}

            # Create a new instance of the same transform type for this chunk
            chunk_transform = type(transform)()
            chunk_transform.start_transform(data_id, column_names, **transform_kwargs)
            chunk_transform.process_data(chunk_data)
            result = chunk_transform.collect()
            return result
        except Exception as e:
            logger.debug(f"Transform failed for chunk: {e}, returning raw data")
            return chunk_data

    def _update_read_cache_ranges(self, cache_key: str, cached_data: list, **kwargs):
        """
        Update cache ranges after caching data.

        Args:
            cache_key: Cache key
            cached_data: Data that was cached
            **kwargs: Request parameters
        """
        start = kwargs.get("start")
        stop = kwargs.get("stop")

        if start and stop and cached_data:
            try:
                # Get actual range from cached data
                first_record = cached_data[0]
                last_record = cached_data[-1]

                if (
                    len(first_record) > 0
                    and len(last_record) > 0
                    and first_record[0] is not None
                    and last_record[0] is not None
                ):
                    actual_start = pd.Timestamp(first_record[0])
                    actual_stop = pd.Timestamp(last_record[0])
                    new_range = (actual_start, actual_stop)

                    # Merge with existing ranges
                    existing_ranges = self._read_cache_ranges[cache_key]
                    combined_ranges = existing_ranges + [new_range]
                    self._read_cache_ranges[cache_key] = self._merge_time_ranges(combined_ranges)

                    logger.debug(f"Updated read cache ranges for {cache_key}: {self._read_cache_ranges[cache_key]}")
            except Exception as e:
                # If range handling fails, mark as no time info
                logger.debug(f"Exception in _update_read_cache_ranges: {e}")
                self._read_cache_ranges[cache_key] = [None]
        else:
            # No time range specified
            self._read_cache_ranges[cache_key] = [None]

    def _create_cached_data_iterator(
        self,
        cached_data: list,
        data_id: str,
        transform: DataTransformer,
        chunksize: int,
        aux_columns: list = None,
        **kwargs,
    ) -> Iterable:
        """
        Create an iterator over cached data with chunking support.

        Args:
            cached_data: List of records (cached data)
            data_id: Data identifier
            transform: Transform to apply
            chunksize: Size of chunks for iteration
            **kwargs: Additional parameters

        Returns:
            Iterator over transformed chunks
        """
        if not cached_data:
            return iter([])

        # Get column names from aux_columns if available, otherwise get from underlying reader
        if aux_columns:
            column_names = aux_columns
        else:
            # For read cache hits, we need to get column names by calling the underlying reader
            # We'll call it with a small range just to get the column names
            raw_data_transformer = DataTransformer()

            # Call underlying reader with a minimal request to get column names
            try:
                # Try to get just one record to establish column names
                sample_data = self._reader.read(
                    data_id,
                    kwargs.get("start"),
                    kwargs.get("start"),  # Same start and stop to get minimal data
                    transform=raw_data_transformer,
                    chunksize=0,
                    **{k: v for k, v in kwargs.items() if k not in ["start", "stop"]},
                )

                # Check if transformer now has column names
                if hasattr(raw_data_transformer, "_column_names") and raw_data_transformer._column_names:
                    column_names = raw_data_transformer._column_names
                else:
                    raise ValueError(f"Could not get column names for {data_id}")
            except Exception as e:
                raise ValueError(f"Failed to get column names for cached data iterator: {e}")

        def _chunk_generator():
            # Use the existing _list_to_chunked_iterator helper
            for chunk in _list_to_chunked_iterator(cached_data, chunksize):
                # Apply transform to each chunk
                chunk_transform = type(transform)()  # Create new instance of the same transform type
                # Filter out parameters that shouldn't be passed to start_transform
                transform_kwargs = {k: v for k, v in kwargs.items() if k not in ["start", "stop", "data_type"]}
                chunk_transform.start_transform(data_id, column_names, **transform_kwargs)
                chunk_transform.process_data(chunk)
                yield chunk_transform.collect()

        return _chunk_generator()

    def _extract_symbol_from_data_id(self, data_id: str) -> str | None:
        """
        Extract symbol from data_id.

        Args:
            data_id: Data identifier (e.g., "BINANCE.UM:BTCUSDT")

        Returns:
            Symbol if found, None otherwise
        """
        if ":" in data_id:
            return data_id.split(":")[1]
        return None

    def _filter_aux_data_by_symbol(self, aux_data: Any, symbol: str) -> Any:
        """
        Filter aux data to only include the specified symbol.

        Args:
            aux_data: The aux data (typically DataFrame)
            symbol: Symbol to filter by

        Returns:
            Filtered aux data
        """
        if not isinstance(aux_data, pd.DataFrame):
            return aux_data

        # Check if DataFrame has a symbol column or symbol index level
        if isinstance(aux_data.index, pd.MultiIndex):
            # Look for symbol level in MultiIndex
            symbol_level = None
            for i, name in enumerate(aux_data.index.names):
                if name in ["symbol", "ticker", "instrument"]:
                    symbol_level = i
                    break

            if symbol_level is not None:
                # Filter by symbol using IndexSlice
                try:
                    idx = pd.IndexSlice
                    if symbol_level == 0:
                        # Symbol is first level: (symbol, timestamp) or (symbol, other)
                        filtered_data = aux_data.loc[idx[symbol, :], :]
                    elif symbol_level == 1:
                        # Symbol is second level: (timestamp, symbol) or (other, symbol)
                        filtered_data = aux_data.loc[idx[:, symbol], :]
                    else:
                        # Symbol is in higher level - return as-is for now
                        return aux_data

                    # Drop the symbol level from the MultiIndex since we're filtering to one symbol
                    filtered_data = filtered_data.droplevel(symbol_level)

                    # Ensure we have valid data after filtering
                    if filtered_data.empty:
                        return filtered_data

                    return filtered_data
                except Exception:
                    # If filtering fails, return original data
                    return aux_data
        elif "symbol" in aux_data.columns:
            # Filter by symbol column
            filtered_data = aux_data[aux_data["symbol"] == symbol].copy()
            # Drop the symbol column since it's now redundant
            filtered_data = filtered_data.drop(columns=["symbol"])
            return filtered_data

        return aux_data

    def _detect_aux_data_overlap(self, data_id: str, data_type: str, **kwargs) -> tuple[str, Any] | None:
        """
        Detect if a read request can be satisfied by existing aux data.

        Args:
            data_id: Data identifier for the read request
            data_type: Type of data being requested
            **kwargs: Additional parameters (start, stop, symbols, etc.)

        Returns:
            Tuple of (aux_data_type, aux_data) if overlap found, None otherwise
        """
        # Check if the requested data type exactly matches any aux data type
        AUX_DATA_MAPPING = {
            "ohlc": "candles",
        }
        aux_data_type = AUX_DATA_MAPPING.get(data_type, data_type)
        if not aux_data_type:
            return None

        # Extract exchange and symbol from data_id for aux data lookup
        # data_id format is typically "EXCHANGE.TYPE:SYMBOL"
        exchange = None
        symbol = self._extract_symbol_from_data_id(data_id)

        if ":" in data_id:
            exchange_part = data_id.split(":")[0]
            if "." in exchange_part:
                exchange = exchange_part
            else:
                exchange = exchange_part

        # Add exchange to kwargs for aux data lookup if not already present
        aux_kwargs = dict(kwargs)
        if exchange and "exchange" not in aux_kwargs:
            aux_kwargs["exchange"] = exchange

        # For candles/ohlc data, ensure timeframe is included in cache key
        if aux_data_type in ["candles", "ohlc"] and "timeframe" in kwargs:
            aux_kwargs["timeframe"] = kwargs["timeframe"]

        # Remove data_type from aux_kwargs to avoid duplication in cache key
        aux_kwargs = {k: v for k, v in aux_kwargs.items() if k != "data_type"}

        # Remove timeframe=None to avoid cache key mismatches for funding_payment
        if aux_kwargs.get("timeframe") is None:
            aux_kwargs.pop("timeframe", None)

        # Generate aux cache key for the same request parameters
        aux_cache_key = self._generate_cache_key("aux", aux_data_type, **aux_kwargs)

        # Check if we have cached aux data that covers the requested range
        if aux_cache_key in self._aux_cache:
            cached_ranges = self._aux_cache_ranges.get(aux_cache_key, [])
            requested_start = aux_kwargs.get("start")
            requested_stop = aux_kwargs.get("stop")

            # Check if cached ranges cover the requested range
            if self._cache_covers_range(cached_ranges, requested_start, requested_stop):
                # Get the cached aux data
                cached_aux_data = self._aux_cache[aux_cache_key]

                # Filter aux data to requested range
                filtered_aux_data = self._filter_aux_data_to_requested_range(cached_aux_data, aux_kwargs)

                # Filter by symbol if specified in data_id
                if symbol:
                    filtered_aux_data = self._filter_aux_data_by_symbol(filtered_aux_data, symbol)

                    # If filtering results in empty data, don't use aux data overlap
                    if isinstance(filtered_aux_data, pd.DataFrame) and filtered_aux_data.empty:
                        return None

                return (aux_data_type, filtered_aux_data)

        # Try to find a compatible aux cache entry (broader scope)
        compatible_aux_cache_key = self._find_compatible_cache_entry(aux_data_type, **aux_kwargs)
        if compatible_aux_cache_key:
            cached_aux_data = self._aux_cache[compatible_aux_cache_key]

            # Filter by symbols first, then by time range
            filtered_data = self._filter_cached_data_by_symbols(cached_aux_data, aux_kwargs.get("symbols", []))
            final_filtered_data = self._filter_aux_data_to_requested_range(filtered_data, aux_kwargs)

            # Filter by symbol if specified in data_id
            if symbol:
                final_filtered_data = self._filter_aux_data_by_symbol(final_filtered_data, symbol)

                # If filtering results in empty data, don't use aux data overlap
                if isinstance(final_filtered_data, pd.DataFrame) and final_filtered_data.empty:
                    return None

            return (aux_data_type, final_filtered_data)

        return None

    def _convert_aux_data_to_read_format(self, aux_data: Any, aux_data_type: str) -> tuple[list[list], list[str]]:
        """
        Convert cached aux data to read format (list of records).

        Args:
            aux_data: Cached aux data (typically DataFrame)
            aux_data_type: Type of aux data

        Returns:
            Tuple of (list of records, column names)
        """
        if isinstance(aux_data, pd.DataFrame):
            return self._dataframe_to_records(aux_data)
        elif isinstance(aux_data, pd.Series):
            # Convert Series to DataFrame first
            df = aux_data.to_frame()
            return self._dataframe_to_records(df)
        elif isinstance(aux_data, list):
            # Already in record format
            if aux_data and isinstance(aux_data[0], (list, tuple)):
                # Assume it's already list of records
                sample_record = aux_data[0]
                columns = [f"col_{i}" for i in range(len(sample_record))]
                if len(columns) > 0:
                    columns[0] = "timestamp"
                return aux_data, columns

        # If conversion fails, return empty
        return [], []

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        """Delegate to underlying reader."""
        return self._reader.get_symbols(exchange, dtype)

    def get_time_ranges(self, symbol: str, dtype: str) -> tuple[Any, Any]:
        """Delegate to underlying reader, with fallback for CSV reader."""
        try:
            return self._reader.get_time_ranges(symbol, dtype)
        except NotImplementedError:
            # CSV reader doesn't implement get_time_ranges, so we need to read some data
            # to determine the time ranges
            try:
                # Try to read a small amount of data to get time ranges
                data = self._reader.read(symbol, chunksize=1)
                if isinstance(data, list) and len(data) > 0:
                    # Get timestamp from first record
                    first_record = data[0]
                    if isinstance(first_record, (list, tuple, np.ndarray)) and len(first_record) > 0:
                        first_timestamp = first_record[0]
                        if isinstance(first_timestamp, (pd.Timestamp, np.datetime64)):
                            # For now, return the first timestamp as both start and end
                            # This is a minimal implementation to make the sniffer work
                            return (first_timestamp, first_timestamp)
                return (None, None)
            except Exception as e:
                logger.debug(f"Exception in fallback get_time_ranges: {e}")
                return (None, None)

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
            self._read_cache.clear()
            self._read_cache_ranges.clear()
            self._read_cache_columns.clear()
        else:
            # Clear entries that start with the data_id
            aux_keys_to_remove = [k for k in self._aux_cache.keys() if k.startswith(data_id)]
            for key in aux_keys_to_remove:
                del self._aux_cache[key]
                if key in self._aux_cache_ranges:
                    del self._aux_cache_ranges[key]

            read_keys_to_remove = [k for k in self._read_cache.keys() if k.startswith(data_id)]
            for key in read_keys_to_remove:
                del self._read_cache[key]
                if key in self._read_cache_ranges:
                    del self._read_cache_ranges[key]
                if key in self._read_cache_columns:
                    del self._read_cache_columns[key]

    def _parse_aux_data_name(self, aux_data_name: str) -> tuple[str, dict[str, Any]]:
        """
        Parse aux data name string with optional kwargs.

        Examples:
            "candles" -> ("candles", {})
            "candles(timeframe=1d)" -> ("candles", {"timeframe": "1d"})
            "candles(timeframe=1d,symbols=BTCUSDT)" -> ("candles", {"timeframe": "1d", "symbols": ["BTCUSDT"]})
            "candles(timeframe=1d, symbols=['BTCUSDT', 'ETHUSDT'])" -> ("candles", {"timeframe": "1d", "symbols": ["BTCUSDT", "ETHUSDT"]})

        Args:
            aux_data_name: Aux data name string, optionally with kwargs in parentheses

        Returns:
            Tuple of (data_name, kwargs_dict)
        """
        # Match pattern like "name(arg1=val1,arg2=val2)" or "name()"
        match = re.match(r"^([^(]+)\(([^)]*)\)$", aux_data_name.strip())
        if not match:
            # No parentheses, just return the name as-is
            return aux_data_name.strip(), {}

        data_name = match.group(1).strip()
        kwargs_str = match.group(2).strip()

        # Parse kwargs string
        kwargs_dict = {}
        if kwargs_str and kwargs_str.strip():
            # Split by comma, but handle nested brackets for list values
            # Try to parse as a simple dict-like string
            try:
                # Wrap in braces to make it a valid dict literal
                dict_str = f"{{{kwargs_str}}}"
                # Use ast.literal_eval for safe evaluation
                parsed_dict = ast.literal_eval(dict_str)

                # Convert any string values that look like lists
                for key, value in parsed_dict.items():
                    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                        try:
                            # Try to parse as a list
                            parsed_value = ast.literal_eval(value)
                            if isinstance(parsed_value, list):
                                kwargs_dict[key] = parsed_value
                            else:
                                kwargs_dict[key] = value
                        except (ValueError, SyntaxError):
                            kwargs_dict[key] = value
                    else:
                        kwargs_dict[key] = value

            except (ValueError, SyntaxError):
                # Fallback to manual parsing for simple cases
                pairs = []
                current_pair = ""
                bracket_count = 0

                for char in kwargs_str:
                    if char == "[":
                        bracket_count += 1
                    elif char == "]":
                        bracket_count -= 1
                    elif char == "," and bracket_count == 0:
                        pairs.append(current_pair.strip())
                        current_pair = ""
                        continue
                    current_pair += char

                if current_pair.strip():
                    pairs.append(current_pair.strip())

                # Parse each key=value pair
                for pair in pairs:
                    if "=" not in pair:
                        continue

                    key, value = pair.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]

                    # Try to parse as literal (for numbers, booleans, lists)
                    try:
                        parsed_value = ast.literal_eval(value)
                        kwargs_dict[key] = parsed_value
                    except (ValueError, SyntaxError):
                        # Keep as string if parsing fails
                        kwargs_dict[key] = value

        return data_name, kwargs_dict

    def prefetch_aux_data(
        self,
        aux_data_names: list[str],
        start: str,
        stop: str,
        exchange: str | None = None,
        symbols: list[str] | None = None,
        **kwargs,
    ) -> dict[str, int]:
        """
        Prefetch multiple auxiliary data types into the cache.

        This method is useful for warming up the cache and debugging. It loads
        the specified auxiliary data types for the given time range and parameters,
        updating the cache with the fetched data.

        Args:
            aux_data_names: List of auxiliary data identifiers to prefetch. Can include
                           simple names like "candles" or names with kwargs like "candles(timeframe=1d)".
                           Examples:
                           - ["candles", "funding"]
                           - ["candles(timeframe=1d)", "funding(interval=8h)"]
                           - ["candles(timeframe=1h,symbols=['BTCUSDT'])"]
            start: Start time for the data range (required)
            stop: Stop time for the data range (required)
            exchange: Exchange identifier (optional)
            symbols: List of symbols to fetch (optional)
            **kwargs: Additional parameters to pass to get_aux_data

        Returns:
            Dictionary mapping data type names (with kwargs) to the number of elements fetched

        Examples:
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

            >>> # With kwargs in aux data names
            >>> results = reader.prefetch_aux_data(
            ...     ["candles(timeframe=1d)", "funding(interval=8h)"],
            ...     start="2023-01-01",
            ...     stop="2023-01-10",
            ...     exchange="BINANCE.UM"
            ... )
            >>> print(results)
            {'candles(timeframe=1d)': 10, 'funding(interval=8h)': 240}
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
        for aux_data_name in aux_data_names:
            try:
                # Parse the aux data name to extract data name and additional kwargs
                data_name, parsed_kwargs = self._parse_aux_data_name(aux_data_name)

                # Merge base params with parsed kwargs (parsed kwargs take precedence)
                fetch_params = base_params.copy()
                fetch_params.update(parsed_kwargs)

                # Get the data (this will populate the cache)
                data = self.get_aux_data(data_name, **fetch_params)

                # Count the elements in the fetched data
                if data is None:
                    count = 0
                elif hasattr(data, "__len__"):
                    count = len(data)
                elif hasattr(data, "shape"):
                    # For numpy arrays or similar
                    count = data.shape[0] if len(data.shape) > 0 else 1
                else:
                    # For scalar values or unknown types
                    count = 1

                # Use the original aux_data_name (with kwargs) as the key for results
                results[aux_data_name] = count

            except Exception as e:
                logger.warning(f"Failed to prefetch {aux_data_name}: {e}")
                results[aux_data_name] = 0

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
