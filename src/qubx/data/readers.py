import itertools
import os
import re
from functools import wraps
from os.path import exists, join
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import psycopg as pg
import pyarrow as pa
from pyarrow import csv, table

from qubx import logger
from qubx.core.basics import DataType, FundingPayment, TimestampedDict, dt_64
from qubx.core.series import OHLCV, Bar, OrderBook, Quote, Trade, TradeArray
from qubx.data.registry import reader
from qubx.pandaz.utils import ohlc_resample, srows
from qubx.utils.time import handle_start_stop, infer_series_frequency


def convert_timedelta_to_numpy(x: str) -> int:
    return pd.Timedelta(x).to_numpy().item()


D1, H1 = convert_timedelta_to_numpy("1D"), convert_timedelta_to_numpy("1h")
MS1 = 1_000_000
S1 = 1000 * MS1
M1 = 60 * S1

DEFAULT_DAILY_SESSION = (convert_timedelta_to_numpy("00:00:00.100"), convert_timedelta_to_numpy("23:59:59.900"))
STOCK_DAILY_SESSION = (convert_timedelta_to_numpy("9:30:00.100"), convert_timedelta_to_numpy("15:59:59.900"))
CME_FUTURES_DAILY_SESSION = (convert_timedelta_to_numpy("8:30:00.100"), convert_timedelta_to_numpy("15:14:59.900"))


def _recognize_t(t: int | str, defaultvalue, timeunit) -> int:
    if isinstance(t, (str, pd.Timestamp)):
        try:
            return np.datetime64(t, timeunit)
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to convert time {t} to datetime64: {e}")
    return defaultvalue


def _time(t, timestamp_units: str) -> int:
    t = int(t) if isinstance(t, float) or isinstance(t, np.int64) else t  # type: ignore
    if timestamp_units == "ns":
        return np.datetime64(t, "ns").item()
    return np.datetime64(t, timestamp_units).astype("datetime64[ns]").item()


def _find_column_index_in_list(xs, *args):
    xs = [x.lower() for x in xs]
    for a in args:
        ai = a.lower()
        if ai in xs:
            return xs.index(ai)
    raise IndexError(f"Can't find any specified columns from [{args}] in provided list: {xs}")


def _list_to_chunked_iterator(data: list[Any], chunksize: int) -> Iterable:
    it = iter(data)
    chunk = list(itertools.islice(it, chunksize))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, chunksize))


def _find_time_col_idx(column_names):
    return _find_column_index_in_list(column_names, "time", "timestamp", "datetime", "date", "open_time", "ts")


def _get_volume_block_indexes(
    column_names: list[str],
) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    def _safe_find_col(*args):
        try:
            return _find_column_index_in_list(column_names, *args)
        except Exception:
            return None

    _volume_idx = _safe_find_col("volume", "vol")
    _b_volume_idx = _safe_find_col("bought_volume", "taker_buy_volume", "taker_bought_volume")
    _volume_quote_idx = _safe_find_col("volume_quote", "quote_volume")
    _b_volume_quote_idx = _safe_find_col("bought_volume_quote", "taker_buy_quote_volume", "taker_bought_quote_volume")
    _trade_count_idx = _safe_find_col("trade_count", "count")

    return _volume_idx, _b_volume_idx, _volume_quote_idx, _b_volume_quote_idx, _trade_count_idx


class DataTransformer:
    def __init__(self) -> None:
        self.buffer = []
        self._column_names = []

    def start_transform(
        self,
        name: str,
        column_names: list[str],
        start: str | None = None,
        stop: str | None = None,
    ):
        self._column_names = column_names
        self.buffer = []

    def process_data(self, rows_data: Iterable) -> Any:
        if rows_data is not None:
            self.buffer.extend(rows_data)

    def collect(self) -> Any:
        return self.buffer


class DataReader:
    def get_names(self, **kwargs) -> list[str]:
        """
        TODO: not sure we really need this !
        """
        raise NotImplementedError("get_names() method is not implemented")

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        **kwargs,
    ) -> Iterator | list:
        raise NotImplementedError("read() method is not implemented")

    def get_aux_data_ids(self) -> set[str]:
        """
        Returns list of all auxiliary data IDs available for this data reader
        """

        def _list_methods(cls):
            _meth = []
            for k, s in cls.__dict__.items():
                if (
                    k.startswith("get_")
                    and k not in ["get_names", "get_symbols", "get_time_ranges", "get_aux_data_ids", "get_aux_data"]
                    and callable(s)
                ):
                    _meth.append(k[4:])
            return _meth

        _d_ids = _list_methods(self.__class__)
        for bc in self.__class__.__bases__:
            _d_ids.extend(_list_methods(bc))
        return set(_d_ids)

    def get_aux_data(self, data_id: str, **kwargs) -> Any:
        """
        Returns auxiliary data for the specified data ID
        """
        if hasattr(self, f"get_{data_id}"):
            return getattr(self, f"get_{data_id}")(**kwargs)
        raise ValueError(
            f"{self.__class__.__name__} doesn't have getter for '{data_id}' auxiliary data. Available data: {self.get_aux_data_ids()}"
        )

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        raise NotImplementedError("get_symbols() method is not implemented")

    def get_time_ranges(self, symbol: str, dtype: str) -> tuple[np.datetime64, np.datetime64]:
        """
        Returns first and last time for the specified symbol and data type in the reader's storage
        """
        raise NotImplementedError("get_time_ranges() method is not implemented")


@reader("csv")
class CsvStorageDataReader(DataReader):
    """
    Data reader for timeseries data stored as csv files in the specified directory
    """

    def __init__(self, path: str) -> None:
        _path = os.path.expanduser(path)
        if not exists(_path):
            raise ValueError(f"Folder is not found at {path}")
        self.path = _path

    def __find_time_idx(self, arr: pa.ChunkedArray, v) -> int:
        ix = arr.index(v).as_py()
        if ix < 0:
            for c in arr.iterchunks():
                a = c.to_numpy()
                ix = np.searchsorted(a, v, side="right")
                if ix > 0 and ix < len(c):
                    ix = arr.index(a[ix]).as_py() - 1
                    break
        return ix

    def __check_file_name(self, name: str) -> str | None:
        _f = join(self.path, name.replace(":", os.sep))
        for sfx in [".csv", ".csv.gz", ""]:
            if exists(p := (_f + sfx)):
                return p
        return None

    def __try_read_data(
        self, data_id: str, start: str | None = None, stop: str | None = None, timestamp_formatters=None
    ) -> tuple[table, np.ndarray, Any, list[str], int, int]:
        f_path = self.__check_file_name(data_id)
        if not f_path:
            ValueError(f"Can't find any csv data for {data_id} in {self.path} !")

        convert_options = None
        if timestamp_formatters is not None:
            convert_options = csv.ConvertOptions(timestamp_parsers=timestamp_formatters)

        table = csv.read_csv(
            f_path,
            parse_options=csv.ParseOptions(ignore_empty_lines=True),
            convert_options=convert_options,
        )
        fieldnames = table.column_names

        # - try to find range to load
        start_idx, stop_idx = 0, table.num_rows
        try:
            _time_field_idx = _find_time_col_idx(fieldnames)
            _time_type = table.field(_time_field_idx).type
            _time_unit = _time_type.unit if hasattr(_time_type, "unit") else "ms"
            _time_data = table[_time_field_idx]

            # - check if need convert time to primitive types (i.e. Date32 -> timestamp[x])
            _time_cast_function = lambda xs: xs
            if _time_type != pa.timestamp(_time_unit):
                _time_cast_function = lambda xs: xs.cast(pa.timestamp(_time_unit))
                _time_data = _time_cast_function(_time_data)

            # - preprocessing start and stop
            t_0, t_1 = handle_start_stop(start, stop, convert=lambda x: _recognize_t(x, None, _time_unit))

            # - check requested range
            if t_0:
                start_idx = self.__find_time_idx(_time_data, t_0)
                if start_idx >= table.num_rows:
                    # - no data for requested start date
                    return table, _time_data, _time_unit, fieldnames, -1, -1

            if t_1:
                stop_idx = self.__find_time_idx(_time_data, t_1)
                if stop_idx < 0 or stop_idx < start_idx:
                    stop_idx = table.num_rows

        except Exception as exc:
            logger.warning(f"exception [{exc}] during preprocessing '{f_path}'")

        return table, _time_data, _time_unit, fieldnames, start_idx, stop_idx

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        timestamp_formatters=None,
        timeframe=None,
        **kwargs,
    ) -> Iterable | Any:
        table, _, _, fieldnames, start_idx, stop_idx = self.__try_read_data(data_id, start, stop, timestamp_formatters)
        if start_idx < 0 or stop_idx < 0:
            return None
        length = stop_idx - start_idx + 1
        selected_table = table.slice(start_idx, length)

        # - in this case we want to return iterable chunks of data
        if chunksize > 0:

            def _iter_chunks():
                for n in range(0, length // chunksize + 1):
                    transform.start_transform(data_id, fieldnames, start=start, stop=stop)
                    raw_data = selected_table[n * chunksize : min((n + 1) * chunksize, length)].to_pandas().to_numpy()
                    transform.process_data(raw_data)
                    yield transform.collect()

            return _iter_chunks()

        transform.start_transform(data_id, fieldnames, start=start, stop=stop)
        raw_data = selected_table.to_pandas().to_numpy()
        transform.process_data(raw_data)
        return transform.collect()

    def get_candles(
        self,
        exchange: str,
        symbols: list[str],
        start: str | pd.Timestamp,
        stop: str | pd.Timestamp,
        timeframe: str | None = None,
    ) -> pd.DataFrame:
        """
        Returns pandas DataFrame of candles for given exchange and symbols within specified time range and timeframe
        """
        _r = []
        for symbol in symbols:
            x = self.read(
                f"{exchange}:{symbol}", start=start, stop=stop, timeframe=timeframe, transform=AsPandasFrame()
            )
            if x is not None:
                if timeframe is not None:
                    x = ohlc_resample(x, timeframe)
                _r.append(x.assign(symbol=symbol.upper(), timestamp=x.index))  # type: ignore
        return srows(*_r).set_index(["timestamp", "symbol"]) if _r else pd.DataFrame()

    def get_names(self, **kwargs) -> list[str]:
        _n = []
        for root, _, files in os.walk(self.path):
            path = root.split(os.sep)
            for file in files:
                if re.match(r"(.*)\.csv(.gz)?$", file):
                    f = path[-1]
                    n = file.split(".")[0]
                    if f == self.path:
                        name = n
                    else:
                        name = f"{f}:{n}" if f else n
                    _n.append(name)
        return _n

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        return self.get_names()

    def get_time_ranges(self, symbol: str, dtype: DataType) -> tuple[Any, Any]:
        """
        Get the time range for a symbol.

        Args:
            symbol: The symbol to get the time range for
            dtype: The data type to get the time range for

        Returns:
            A tuple of (start_time, end_time)
        """
        _, _time_data, _time_unit, _, start_idx, stop_idx = self.__try_read_data(symbol, None, None, None)
        return (
            np.datetime64(_time_data[start_idx].value, _time_unit),
            np.datetime64(_time_data[stop_idx - 1].value, _time_unit),
        )


class InMemoryDataFrameReader(DataReader):
    """
    Data reader for pandas DataFrames
    """

    exchange: str | None
    _data: dict[str, pd.DataFrame | pd.Series]

    def __init__(self, data: dict[str, pd.DataFrame | pd.Series], exchange: str | None = None) -> None:
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary of pandas DataFrames")
        self._data = data
        self.exchange = exchange

    def get_names(self, **kwargs) -> list[str]:
        keys = list(self._data.keys())
        if self.exchange:
            return [f"{self.exchange}:{k}" for k in keys]
        return keys

    def _get_data_by_key(self, data_id: str) -> tuple[str, pd.DataFrame | pd.Series]:
        if data_id not in self._data:
            if self.exchange and data_id.startswith(self.exchange):
                data_id = data_id.split(":")[1]
        if (d := self._data.get(data_id)) is None:
            raise ValueError(f"No data found for {data_id}")
        return data_id, d

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        **kwargs,
    ) -> Iterable | list:
        """
        Read and transform data for a given data_id within a specified time range.

        Parameters:
        -----------
        data_id : str
            The identifier for the data to be read.
        start : str | None, optional
            The start time for the data range (inclusive). If None, start from the earliest available data.
        stop : str | None, optional
            The stop time for the data range (inclusive). If None, include data up to the latest available.
        transform : DataTransformer, optional
            An instance of DataTransformer to process the retrieved data. Defaults to DataTransformer().
        chunksize : int, optional
            The size of data chunks to process at a time. If 0, process all data at once. Defaults to 0.
        **kwargs : dict
            Additional keyword arguments for future extensions.

        Returns:
        --------
        Iterable | list
            The processed and transformed data, either as an iterable (if chunksize > 0) or as a list.

        Raises:
        -------
        ValueError
            If no data is found for the given data_id.
        """
        start, stop = handle_start_stop(start, stop)
        data_id, _stored_data = self._get_data_by_key(data_id)

        _sliced_data = _stored_data.loc[start:stop].copy()
        if _tf := kwargs.get("timeframe"):
            _sliced_data = ohlc_resample(_sliced_data, _tf)
            assert isinstance(_sliced_data, pd.DataFrame), "Resampled data should be a DataFrame"
        _sliced_data = _sliced_data.reset_index()

        def _do_transform(values: Iterable, columns: list[str]) -> Iterable:
            transform.start_transform(data_id, columns, start=start, stop=stop)
            transform.process_data(values)
            return transform.collect()

        if chunksize > 0:
            # returns chunked frames
            def _chunked_dataframe(data: np.ndarray, columns: list[str], chunksize: int) -> Iterable:
                it = iter(data)
                chunk = list(itertools.islice(it, chunksize))
                while chunk:
                    yield _do_transform(chunk, columns)
                    chunk = list(itertools.islice(it, chunksize))

            return _chunked_dataframe(_sliced_data.values, list(_sliced_data.columns), chunksize)

        return _do_transform(_sliced_data.values, list(_sliced_data.columns))

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        return self.get_names()

    def get_time_ranges(self, symbol: str, dtype: DataType) -> tuple[np.datetime64 | None, np.datetime64 | None]:
        try:
            _, _stored_data = self._get_data_by_key(symbol)
            return _stored_data.index[0], _stored_data.index[-1]
        except ValueError:
            return None, None


class AsPandasFrame(DataTransformer):
    """
    List of records to pandas dataframe transformer
    """

    def __init__(self, timestamp_units=None) -> None:
        self.timestamp_units = timestamp_units

    def start_transform(self, name: str, column_names: list[str], **kwargs):
        self._time_idx = _find_time_col_idx(column_names)
        self._column_names = column_names
        self._frame = pd.DataFrame()

    def process_data(self, rows_data: Iterable) -> Any:
        self._frame
        p = pd.DataFrame.from_records(rows_data, columns=self._column_names)
        p.set_index(self._column_names[self._time_idx], drop=True, inplace=True)
        p.index = pd.to_datetime(p.index, unit=self.timestamp_units) if self.timestamp_units else p.index
        p.index.rename("timestamp", inplace=True)
        p.sort_index(inplace=True)
        self._frame = pd.concat((self._frame, p), axis=0, sort=True)
        return p

    def collect(self) -> Any:
        return self._frame


class AsOhlcvSeries(DataTransformer):
    """
    Convert incoming data into OHLCV series.

    Incoming data may have one of the following structures:

        ```
        ohlcv:        time,open,high,low,close,volume|quote_volume,(buy_volume)
        quotes:       time,bid,ask,bidsize,asksize
        trades (TAS): time,price,size,(is_taker)
        ```
    """

    timeframe: str | None
    _series: OHLCV | None
    _data_type: str | None

    def __init__(self, timeframe: str | None = None, timestamp_units="ns") -> None:
        super().__init__()
        self.timeframe = timeframe
        self._series = None
        self._data_type = None
        self.timestamp_units = timestamp_units

    def start_transform(self, name: str, column_names: list[str], **kwargs):
        self._time_idx = _find_time_col_idx(column_names)
        self._volume_idx = None
        self._b_volume_idx = None
        try:
            self._close_idx = _find_column_index_in_list(column_names, "close")
            self._open_idx = _find_column_index_in_list(column_names, "open")
            self._high_idx = _find_column_index_in_list(column_names, "high")
            self._low_idx = _find_column_index_in_list(column_names, "low")
            (
                self._volume_idx,
                self._b_volume_idx,
                self._volume_quote_idx,
                self._b_volume_quote_idx,
                self._trade_count_idx,
            ) = _get_volume_block_indexes(column_names)

            self._data_type = "ohlc"
        except:
            try:
                self._ask_idx = _find_column_index_in_list(column_names, "ask")
                self._bid_idx = _find_column_index_in_list(column_names, "bid")
                self._data_type = "quotes"
            except:
                try:
                    self._price_idx = _find_column_index_in_list(column_names, "price")
                    self._size_idx = _find_column_index_in_list(
                        column_names, "quote_qty", "qty", "size", "amount", "volume"
                    )
                    self._taker_idx = None
                    try:
                        self._taker_idx = _find_column_index_in_list(
                            column_names,
                            "is_buyer_maker",
                            "side",
                            "aggressive",
                            "taker",
                            "is_taker",
                        )
                    except:
                        pass

                    self._data_type = "trades"
                except:
                    raise ValueError(f"Can't recognize data for update from header: {column_names}")

        self._column_names = column_names
        self._name = name
        if self.timeframe:
            self._series = OHLCV(self._name, self.timeframe)

    def _proc_ohlc(self, rows_data: list[list]):
        for d in rows_data:
            self._series.update_by_bar(
                _time(d[self._time_idx], self.timestamp_units),
                d[self._open_idx],
                d[self._high_idx],
                d[self._low_idx],
                d[self._close_idx],
                d[self._volume_idx] if self._volume_idx else 0,
                d[self._b_volume_idx] if self._b_volume_idx else 0,
            )

    def _proc_quotes(self, rows_data: list[list]):
        for d in rows_data:
            self._series.update(
                _time(d[self._time_idx], self.timestamp_units),
                (d[self._ask_idx] + d[self._bid_idx]) / 2,
            )

    def _proc_trades(self, rows_data: list[list]):
        for d in rows_data:
            a = d[self._taker_idx] if self._taker_idx else 0
            s = d[self._size_idx]
            b = s if a else 0
            self._series.update(_time(d[self._time_idx], self.timestamp_units), d[self._price_idx], s, b)

    def process_data(self, rows_data: list[list]) -> Any:
        if self._series is None:
            ts = [t[self._time_idx] for t in rows_data[:100]]
            self.timeframe = pd.Timedelta(infer_series_frequency(ts)).asm8.item()

            # - create instance after first data received if
            self._series = OHLCV(self._name, self.timeframe)

        match self._data_type:
            case "ohlc":
                self._proc_ohlc(rows_data)
            case "quotes":
                self._proc_quotes(rows_data)
            case "trades":
                self._proc_trades(rows_data)

        return None

    def collect(self) -> Any:
        return self._series


class AsBars(AsOhlcvSeries):
    """
    Convert incoming data into Bars sequence.

    Incoming data may have one of the following structures:

        ```
        ohlcv:        time,open,high,low,close,volume|quote_volume,(buy_volume)
        quotes:       time,bid,ask,bidsize,asksize
        trades (TAS): time,price,size,(is_taker)
        ```
    """

    def collect(self) -> Any:
        return self._series[::-1] if self._series is not None else None


class AsQuotes(DataTransformer):
    """
    Tries to convert incoming data to list of Quote's
    Data must have appropriate structure: bid, ask, bidsize, asksize and time
    """

    def start_transform(self, name: str, column_names: list[str], **kwargs):
        self.buffer = list()
        self._time_idx = _find_time_col_idx(column_names)
        self._bid_idx = _find_column_index_in_list(column_names, "bid")
        self._ask_idx = _find_column_index_in_list(column_names, "ask")
        self._bidvol_idx = _find_column_index_in_list(column_names, "bidvol", "bid_vol", "bidsize", "bid_size")
        self._askvol_idx = _find_column_index_in_list(column_names, "askvol", "ask_vol", "asksize", "ask_size")

    def process_data(self, rows_data: Iterable) -> Any:
        if rows_data is not None:
            for d in rows_data:
                t = d[self._time_idx]
                b = d[self._bid_idx]
                a = d[self._ask_idx]
                bv = d[self._bidvol_idx]
                av = d[self._askvol_idx]
                self.buffer.append(Quote(_time(t, "ns"), b, a, bv, av))


class AsOrderBook(DataTransformer):
    """
    Tries to convert incoming data to list of OrderBook objects
    Data must have appropriate structure: bids, asks, top_bid, top_ask, tick_size and time
    """

    def __init__(self, timestamp_units="ns") -> None:
        super().__init__()
        self.timestamp_units = timestamp_units

    def start_transform(self, name: str, column_names: list[str], **kwargs):
        self.buffer = list()
        self._time_idx = _find_time_col_idx(column_names)
        self._top_bid_idx = _find_column_index_in_list(column_names, "top_bid")
        self._top_ask_idx = _find_column_index_in_list(column_names, "top_ask")
        self._tick_size_idx = _find_column_index_in_list(column_names, "tick_size")
        self._bids_idx = _find_column_index_in_list(column_names, "bids")
        self._asks_idx = _find_column_index_in_list(column_names, "asks")

    def process_data(self, rows_data: Iterable) -> Any:
        if rows_data is not None:
            for d in rows_data:
                t = d[self._time_idx]
                top_bid = d[self._top_bid_idx]
                top_ask = d[self._top_ask_idx]
                tick_size = d[self._tick_size_idx]
                bids = d[self._bids_idx]
                asks = d[self._asks_idx]
                timestamp_ns = _time(t, self.timestamp_units)
                self.buffer.append(OrderBook(timestamp_ns, top_bid, top_ask, tick_size, bids, asks))


class AsTrades(DataTransformer):
    """
    Tries to convert incoming data to list of Trades
    Data must have appropriate structure: price, size, market_maker (optional).
    Market maker column specifies if buyer is a maker or taker.
    """

    def start_transform(self, name: str, column_names: list[str], **kwargs):
        self.buffer: list[Trade | TradeArray] = list()
        self._time_idx = _find_time_col_idx(column_names)
        self._price_idx = _find_column_index_in_list(column_names, "price")
        self._size_idx = _find_column_index_in_list(column_names, "size", "amount")
        self._side_idx = _find_column_index_in_list(column_names, "side")
        try:
            self._array_id_idx = _find_column_index_in_list(column_names, "array_id")
        except:
            self._array_id_idx = None

    def process_data(self, rows_data: Iterable) -> Any:
        if rows_data is None:
            return

        if self._array_id_idx is None:
            # return trades if no array_id column is present
            for d in rows_data:
                t = d[self._time_idx]
                price = d[self._price_idx]
                size = d[self._size_idx]
                side = d[self._side_idx] if self._side_idx else 0
                self.buffer.append(Trade(_time(t, "ns"), price, size, side))
        elif isinstance(rows_data, np.ndarray):
            # return TradeArray if array_id column is present
            # Split the trades array into subarrays based on array_id
            unique_array_ids = np.unique(rows_data["array_id"])
            for array_id in unique_array_ids:
                # Get all trades with the current array_id
                mask = rows_data["array_id"] == array_id
                trade_array = TradeArray(rows_data[mask][["timestamp", "price", "size", "side"]])
                self.buffer.append(trade_array)
        else:
            raise NotImplementedError("Unsupported transform")


class AsTimestampedRecords(DataTransformer):
    """
    Convert incoming data to list or dictionaries with preprocessed timestamps ('timestamp_ns' and 'timestamp')
    ```
    [
        {
            'open_time': 1711944240000.0,
            'open': 203.219,
            'high': 203.33,
            'low': 203.134,
            'close': 203.175,
            'volume': 10060.0,
            ....
            'timestamp_ns': 1711944240000000000,
            'timestamp': Timestamp('2024-04-01 04:04:00')
        },
        ...
    ] ```
    """

    def __init__(self, timestamp_units: str | None = None) -> None:
        self.timestamp_units = timestamp_units

    def start_transform(self, name: str, column_names: list[str], **kwargs):
        self.buffer = list()
        self._time_idx = _find_time_col_idx(column_names)
        self._column_names = column_names

    def process_data(self, rows_data: Iterable) -> Any:
        self.buffer.extend(rows_data)

    def collect(self) -> Any:
        res = []
        for r in self.buffer:
            t = r[self._time_idx]
            if self.timestamp_units:
                t = _time(t, self.timestamp_units)
            di = dict(zip(self._column_names, r)) | {
                "timestamp_ns": t,
                "timestamp": pd.Timestamp(t),
            }
            res.append(di)
        return res


class RestoredEmulatorHelper(DataTransformer):
    _freq: np.timedelta64 | None = None
    _t_start: int
    _t_mid1: int
    _t_mid2: int
    _t_end: int
    _open_close_time_shift_secs: int

    def __init__(self, daily_session_start_end: tuple, timestamp_units: str, open_close_time_shift_secs: int):
        super().__init__()
        self._d_session_start = daily_session_start_end[0]
        self._d_session_end = daily_session_start_end[1]
        self._timestamp_units = timestamp_units
        self._open_close_time_shift_secs = open_close_time_shift_secs  # type: ignore

    def _detect_emulation_timestamps(self, rows_data: list[list]):
        if self._freq is None:
            ts = [t[self._time_idx] for t in rows_data]
            try:
                self._freq = infer_series_frequency(ts)
            except ValueError:
                logger.warning("Can't determine frequency of incoming data")
                return

            # - timestamps when we emit simulated quotes
            dt = self._freq.astype("timedelta64[ns]").item()
            dt10 = dt // 10

            # - adjust open-close time shift to avoid overlapping timestamps
            if self._open_close_time_shift_secs * S1 >= (dt // 2 - dt10):
                self._open_close_time_shift_secs = (dt // 2 - 2 * dt10) // S1

            if dt < D1:
                self._t_start = self._open_close_time_shift_secs * S1
                self._t_mid1 = dt // 2 - dt10
                self._t_mid2 = dt // 2 + dt10
                self._t_end = dt - self._open_close_time_shift_secs * S1
            else:
                self._t_start = self._d_session_start + self._open_close_time_shift_secs * S1
                self._t_mid1 = dt // 2 - H1
                self._t_mid2 = dt // 2 + H1
                self._t_end = self._d_session_end - self._open_close_time_shift_secs * S1

    def start_transform(self, name: str, column_names: list[str], **kwargs):
        self.buffer = []
        # - it will fail if receive data doesn't look as ohlcv
        self._time_idx = _find_time_col_idx(column_names)
        self._open_idx = _find_column_index_in_list(column_names, "open")
        self._high_idx = _find_column_index_in_list(column_names, "high")
        self._low_idx = _find_column_index_in_list(column_names, "low")
        self._close_idx = _find_column_index_in_list(column_names, "close")
        self._volume_idx = None
        self._b_volume_idx = None
        self._volume_quote_idx = None
        self._b_volume_quote_idx = None
        self._trade_count_idx = None
        self._freq = None
        (
            self._volume_idx,
            self._b_volume_idx,
            self._volume_quote_idx,
            self._b_volume_quote_idx,
            self._trade_count_idx,
        ) = _get_volume_block_indexes(column_names)


class RestoreTicksFromOHLC(RestoredEmulatorHelper):
    """
    Emulates quotes (and trades) from OHLC bars
    """

    def __init__(
        self,
        trades: bool = False,  # if we also wants 'trades'
        default_bid_size=1e9,  # default bid/ask is big
        default_ask_size=1e9,  # default bid/ask is big
        daily_session_start_end=DEFAULT_DAILY_SESSION,
        timestamp_units="ns",
        spread=0.0,
        open_close_time_shift_secs=1.0,
        quotes=True,
    ):
        super().__init__(daily_session_start_end, timestamp_units, open_close_time_shift_secs)
        assert trades or quotes or trades and trades, "Either trades or quotes or both must be enabled"
        self._trades = trades
        self._quotes = quotes
        self._bid_size = default_bid_size
        self._ask_size = default_ask_size
        self._s2 = spread / 2.0

    def start_transform(self, name: str, column_names: list[str], **kwargs):
        super().start_transform(name, column_names, **kwargs)
        # -  disable trades if no volume information is available
        if self._volume_idx is None and self._trades:
            logger.warning("Input OHLC data doesn't contain volume information so trades can't be emulated !")
            self._trades = False

    def process_data(self, rows_data: list[list]) -> Any:
        if rows_data is None:
            return

        s2 = self._s2
        if self._freq is None:
            self._detect_emulation_timestamps(rows_data[:100])

        # - input data
        for data in rows_data:
            # ti = pd.Timestamp(data[self._time_idx]).as_unit("ns").asm8.item()
            ti = _time(data[self._time_idx], self._timestamp_units)
            o = data[self._open_idx]
            h = data[self._high_idx]
            l = data[self._low_idx]
            c = data[self._close_idx]
            rv = data[self._volume_idx] if self._volume_idx else 0
            rv = rv / (h - l) if h > l else rv

            # - opening quote
            if self._quotes:
                self.buffer.append(Quote(ti + self._t_start, o - s2, o + s2, self._bid_size, self._ask_size))

            if c >= o:
                if self._trades:
                    self.buffer.append(Trade(ti + self._t_start, o - s2, rv * (o - l)))  # sell 1

                if self._quotes:
                    self.buffer.append(
                        Quote(
                            ti + self._t_mid1,
                            l - s2,
                            l + s2,
                            self._bid_size,
                            self._ask_size,
                        )
                    )

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid1, l + s2, rv * (c - o)))  # buy 1

                if self._quotes:
                    self.buffer.append(
                        Quote(
                            ti + self._t_mid2,
                            h - s2,
                            h + s2,
                            self._bid_size,
                            self._ask_size,
                        )
                    )

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid2, h - s2, rv * (h - c)))  # sell 2
            else:
                if self._trades:
                    self.buffer.append(Trade(ti + self._t_start, o + s2, rv * (h - o)))  # buy 1

                if self._quotes:
                    self.buffer.append(
                        Quote(
                            ti + self._t_mid1,
                            h - s2,
                            h + s2,
                            self._bid_size,
                            self._ask_size,
                        )
                    )

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid1, h - s2, rv * (o - c)))  # sell 1

                if self._quotes:
                    self.buffer.append(
                        Quote(
                            ti + self._t_mid2,
                            l - s2,
                            l + s2,
                            self._bid_size,
                            self._ask_size,
                        )
                    )

                if self._trades:
                    self.buffer.append(Trade(ti + self._t_mid2, l + s2, rv * (c - l)))  # buy 2

            # - closing quote
            if self._quotes:
                self.buffer.append(Quote(ti + self._t_end, c - s2, c + s2, self._bid_size, self._ask_size))


class RestoreQuotesFromOHLC(RestoreTicksFromOHLC):
    """
    Restore (emulate) quotes from OHLC bars
    """

    def __init__(
        self,
        default_bid_size=1e9,  # default bid/ask is big
        default_ask_size=1e9,  # default bid/ask is big
        daily_session_start_end=DEFAULT_DAILY_SESSION,
        timestamp_units="ns",
        spread=0.0,
        open_close_time_shift_secs=1.0,
    ):
        super().__init__(
            trades=False,
            default_bid_size=default_bid_size,
            default_ask_size=default_ask_size,
            daily_session_start_end=daily_session_start_end,
            timestamp_units=timestamp_units,
            spread=spread,
            open_close_time_shift_secs=open_close_time_shift_secs,
            quotes=True,
        )


class RestoreTradesFromOHLC(RestoreTicksFromOHLC):
    """
    Restore (emulate) trades from OHLC bars
    """

    def __init__(
        self,
        daily_session_start_end=DEFAULT_DAILY_SESSION,
        timestamp_units="ns",
        open_close_time_shift_secs=1.0,
    ):
        super().__init__(
            trades=True,
            default_bid_size=0,
            default_ask_size=0,
            daily_session_start_end=daily_session_start_end,
            timestamp_units=timestamp_units,
            spread=0,
            open_close_time_shift_secs=open_close_time_shift_secs,
            quotes=False,
        )


class RestoredBarsFromOHLC(RestoredEmulatorHelper):
    """
    Transforms OHLC data into a sequence of bars trying to mimic real-world market data updates
    """

    def __init__(
        self, daily_session_start_end=DEFAULT_DAILY_SESSION, timestamp_units="ns", open_close_time_shift_secs=1.0
    ):
        super().__init__(daily_session_start_end, timestamp_units, open_close_time_shift_secs)

    def process_data(self, rows_data: list[list]) -> Any:
        if rows_data is None:
            return

        if self._freq is None:
            self._detect_emulation_timestamps(rows_data[:100])

        # - input data
        # fmt: off
        for data in rows_data:
            ti = _time(data[self._time_idx], self._timestamp_units)
            o = data[self._open_idx]
            h = data[self._high_idx]
            l = data[self._low_idx]
            c = data[self._close_idx]

            # - volumes data
            vol = data[self._volume_idx] if self._volume_idx is not None else 0
            bvol = data[self._b_volume_idx] if self._b_volume_idx is not None else 0
            vol_q = data[self._volume_quote_idx] if self._volume_quote_idx is not None else 0
            bvol_q = data[self._b_volume_quote_idx] if self._b_volume_quote_idx is not None else 0

            # - trade count data
            tcount = data[self._trade_count_idx] if self._trade_count_idx is not None else 0

            # rvol = vol / (h - l) if h > l else vol
            # - opening bar (o,h,l,c=o, v=0, bv=0)
            self.buffer.append(
                Bar(
                    ti + self._t_start, o, o, o, o, volume=0, bought_volume=0, volume_quote=0, bought_volume_quote=0, trade_count=0,
                )
            )

            if c >= o:
                # v1 = rvol * (o - l)
                self.buffer.append(Bar(ti + self._t_mid1, o, o, l, l, volume=0))

                # v2 = v1 + rvol * (c - o)
                self.buffer.append(Bar(ti + self._t_mid2, o, h, l, h, volume=0))

            else:
                # v1 = rvol * (h - o)
                self.buffer.append(Bar(ti + self._t_mid1, o, h, o, h, volume=0))

                # v2 = v1 + rvol * (o - c)
                self.buffer.append(Bar(ti + self._t_mid2, o, h, l, l, volume=0))

            # - final bar - propagate full data
            self.buffer.append(
                Bar(
                    ti + self._t_end, o, h, l, c, 
                    volume=vol, bought_volume=bvol, volume_quote=vol_q, bought_volume_quote=bvol_q, trade_count=tcount,
                )
            )
        # fmt: on


class AsDict(DataTransformer):
    """
    Tries to keep incoming data as list of dictionaries with preprocessed time
    """

    def start_transform(self, name: str, column_names: list[str], **kwargs):
        self.buffer = list()
        self._time_idx = _find_time_col_idx(column_names)
        self._column_names = column_names
        self._time_name = column_names[self._time_idx]

    def process_data(self, rows_data: Iterable):
        if rows_data is not None:
            for d in rows_data:
                _r_dict = dict(zip(self._column_names, d))
                self.buffer.append(TimestampedDict(_time(d[self._time_idx], "ns"), _r_dict))  # type: ignore


class AsFundingPayments(DataTransformer):
    """
    Tries to convert incoming data to list of FundingPayment objects.
    Data must have structure: timestamp, symbol, funding_rate, funding_interval_hours
    """

    def start_transform(self, name: str, column_names: list[str], **kwargs):
        self.buffer = list()
        self._time_idx = _find_time_col_idx(column_names)
        self._funding_rate_idx = _find_column_index_in_list(column_names, "funding_rate")
        self._funding_interval_idx = _find_column_index_in_list(column_names, "funding_interval_hours")

    def process_data(self, rows_data: Iterable) -> Any:
        if rows_data is not None:
            for d in rows_data:
                t = d[self._time_idx]
                funding_rate = d[self._funding_rate_idx]
                funding_interval_hours = d[self._funding_interval_idx]
                self.buffer.append(FundingPayment(_time(t, "ns"), funding_rate, funding_interval_hours))

    def collect(self) -> Any:
        return self.buffer


def _retry(fn):
    @wraps(fn)
    def wrapper(*args, **kw):
        cls = args[0]
        for x in range(cls._reconnect_tries):
            # print(x, cls._reconnect_tries)
            try:
                return fn(*args, **kw)
            except (pg.InterfaceError, pg.OperationalError, AttributeError):
                logger.debug("Database Connection [InterfaceError or OperationalError]")
                # print ("Idle for %s seconds" % (cls._reconnect_idle))
                # time.sleep(cls._reconnect_idle)
                cls._connect()

    return wrapper


def _calculate_max_candles_chunksize(timeframe: str | None) -> int:
    """
    Calculate maximum chunksize for candles data based on timeframe.
    Limits to at most 1 week of data for candles_1m data type.

    Args:
        timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")

    Returns:
        Maximum number of candles representing 1 week of data
    """
    if not timeframe:
        return 10080  # Default to 1 week of 1-minute candles

    try:
        # Convert timeframe to pandas Timedelta
        tf_delta = pd.Timedelta(timeframe)

        if timeframe == "1d":
            one_month = pd.Timedelta("30d")
            max_candles = int(one_month / tf_delta)
        else:
            # Calculate how many candles fit in 1 week
            one_week = pd.Timedelta("7d")
            max_candles = int(one_week / tf_delta)

        # Ensure we don't return 0 or negative values
        return max(1, max_candles)
    except (ValueError, TypeError):
        # If timeframe can't be parsed, default to 1 week of 1-minute candles
        return 10080


def _calculate_time_windows_for_chunking(
    start: str | None, end: str | None, timeframe: str, chunksize: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Calculate time windows for efficient chunking based on timeframe and chunksize.

    Args:
        start: Start time string
        end: End time string
        timeframe: Timeframe string (e.g., "1m", "5m", "1h")
        chunksize: Number of candles per chunk

    Returns:
        List of (start_time, end_time) tuples for each chunk
    """
    if not start or not end:
        return []

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    try:
        # Calculate time period per chunk based on timeframe and chunksize
        tf_delta = pd.Timedelta(timeframe)
        chunk_duration = tf_delta * chunksize

        windows = []
        current_start = start_dt

        while current_start < end_dt:
            current_end = min(current_start + chunk_duration, end_dt)
            windows.append((current_start, current_end))
            current_start = current_end

        # If last window is less than half of the window before it, then merge together
        if len(windows) > 1 and windows[-1][0] - windows[-1][1] < chunk_duration / 2:
            windows[-2] = (windows[-2][0], windows[-1][1])
            windows.pop()

        return windows
    except (ValueError, TypeError):
        # If timeframe can't be parsed, fall back to single window
        return [(start_dt, end_dt)]


class QuestDBSqlBuilder:
    """
    Generic sql builder for QuestDB data
    """

    _aliases = {"um": "umswap", "cm": "cmswap", "f": "futures"}

    def get_table_name(self, data_id: str, sfx: str = "") -> str:
        """
        Get table name for data_id
        data_id can have format <exchange>.<type>:<symbol>
        for example:
            BINANCE.UM:BTCUSDT or BINANCE:BTCUSDT for spot
        """
        sfx = sfx or "candles_1m"
        table_name = data_id
        _exch, _symb, _mktype = self._get_exchange_symbol_market_type(data_id)
        if _exch and _symb:
            parts = [_exch.lower(), _mktype]
            if "candles" not in sfx:
                parts.append(_symb)
            parts.append(sfx)
            table_name = ".".join(filter(lambda x: x, parts))

        return table_name

    def _get_exchange_symbol_market_type(self, data_id: str) -> tuple[str | None, str | None, str | None]:
        _ss = data_id.split(":")
        if len(_ss) > 1:
            _exch, symb = _ss
            _mktype = "spot"
            _ss = _exch.split(".")
            if len(_ss) > 1:
                _exch = _ss[0]
                _mktype = _ss[1]
            _mktype = _mktype.lower()
            return _exch.lower(), symb.upper(), self._aliases.get(_mktype, _mktype)
        return None, None, None

    def prepare_data_sql(
        self,
        data_id: str,
        start: str | None,
        end: str | None,
        resample: str | None,
        data_type: str,
    ) -> str | None:
        pass

    def prepare_names_sql(self) -> str:
        return "select table_name from tables()"

    def prepare_symbols_sql(self, exchange: str, dtype: str) -> str:
        _table = self.get_table_name(f"{exchange}:BTCUSDT", dtype)
        return f"select distinct(symbol) from {_table}"

    def prepare_data_ranges_sql(self, data_id: str) -> str:
        raise NotImplementedError()


class QuestDBSqlCandlesBuilder(QuestDBSqlBuilder):
    """
    Sql builder for candles data
    """

    def prepare_names_sql(self) -> str:
        return "select table_name from tables() where table_name like '%candles%'"

    @staticmethod
    def _convert_time_delta_to_qdb_resample_format(c_tf: str):
        if c_tf:
            _t = re.match(r"(\d+)(\w+)", c_tf)
            if _t and len(_t.groups()) > 1:
                c_tf = f"{_t[1]}{_t[2][0].lower()}"
        return c_tf

    def prepare_data_sql(
        self,
        data_id: str,
        start: str | None,
        end: str | None,
        resample: str | None,
        data_type: str,
    ) -> str:
        _exch, _symb, _mktype = self._get_exchange_symbol_market_type(data_id)
        if _symb is None:
            _symb = data_id

        _symb = _symb.upper()

        where = f"where symbol = '{_symb}'"
        w0 = f"timestamp >= '{start}'" if start else ""
        w1 = f"timestamp < '{end}'" if end else ""

        # - fix: when no data ranges are provided we must skip empy where keyword
        if w0 or w1:
            where = f"{where} and {w0} and {w1}" if (w0 and w1) else f"{where} and {(w0 or w1)}"

        # - filter out candles without any volume
        where = f"{where} and volume > 0"

        # - check resample format
        resample = (
            QuestDBSqlCandlesBuilder._convert_time_delta_to_qdb_resample_format(resample)
            if resample
            else "1m"  # if resample is empty let's use 1 minute timeframe
        )
        _rsmpl = f"SAMPLE by {resample} FILL(NONE)" if resample else ""

        table_name = self.get_table_name(data_id, data_type)
        return f"""
                select timestamp, 
                first(open) as open, max(high) as high, min(low) as low, last(close) as close,
                sum(volume) as volume,
                sum(quote_volume) as quote_volume,
                sum(count) as count,
                sum(taker_buy_volume) as taker_buy_volume,
                sum(taker_buy_quote_volume) as taker_buy_quote_volume
                from "{table_name}" {where} {_rsmpl};
            """

    def prepare_data_ranges_sql(self, data_id: str) -> str:
        _exch, _symb, _mktype = self._get_exchange_symbol_market_type(data_id)
        if _exch is None:
            raise ValueError(f"Can't get exchange name from data id: {data_id} !")
        return f"""(SELECT timestamp FROM "{_exch}.{_mktype}.candles_1m" WHERE symbol='{_symb}' ORDER BY timestamp ASC LIMIT 1)
                        UNION
                   (SELECT timestamp FROM "{_exch}.{_mktype}.candles_1m" WHERE symbol='{_symb}' ORDER BY timestamp DESC LIMIT 1)
                """


class QuestDBSqlTOBBilder(QuestDBSqlBuilder):
    def prepare_data_ranges_sql(self, data_id: str) -> str:
        _exch, _symb, _mktype = self._get_exchange_symbol_market_type(data_id)
        if _exch is None:
            raise ValueError(f"Can't get exchange name from data id: {data_id} !")
        # TODO: ????
        return f"""(SELECT timestamp FROM "{_exch}.{_mktype}.{_symb}.orderbook" ORDER BY timestamp ASC LIMIT 1)
                        UNION
                   (SELECT timestamp FROM "{_exch}.{_mktype}.{_symb}.orderbook" ORDER BY timestamp DESC LIMIT 1)
                """


@reader("qdb")
class QuestDBConnector(DataReader):
    """
    Data connector for QuestDB which provides access to following data types:
      - candles
      - trades
      - orderbook snapshots
      - liquidations
      - funding rate

    Examples:
    1. Retrieving trades:
        qdb.read(
            "BINANCE.UM:BTCUSDT",
            "2023-01-01 00:00",
            transform=AsPandasFrame(),
            data_type="trade"
        )
    """

    _reconnect_tries = 5
    _reconnect_idle = 0.1  # wait seconds before retying
    _builder: QuestDBSqlBuilder

    def __init__(
        self,
        builder: QuestDBSqlBuilder = QuestDBSqlCandlesBuilder(),
        host="localhost",
        user="admin",
        password="quest",
        port=8812,
        timeframe: str = "1m",
    ) -> None:
        self._connection = None
        self._host = host
        self._port = port
        self.connection_url = f"user={user} password={password} host={host} port={port}"
        self._builder = builder
        self._default_timeframe = timeframe
        self._connect()

    def __getstate__(self):
        if self._connection:
            self._connection.close()
            self._connection = None
        state = self.__dict__.copy()
        return state

    def _connect(self):
        self._connection = pg.connect(self.connection_url, autocommit=True)
        logger.debug(f"Connected to QuestDB at {self._host}:{self._port}")

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.debug(f"Disconnected from QuestDB at {self._host}:{self._port}")

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        timeframe: str | None = "1m",
        data_type="candles_1m",
    ) -> Any:
        return self._read(
            data_id,
            start,
            stop,
            transform,
            chunksize,
            timeframe,
            data_type,
            self._builder,
        )

    def get_candles(
        self,
        exchange: str,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        stop: str | pd.Timestamp | None = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        # Use any symbol to get the table name (candles are in a single table per exchange)
        dummy_symbol = "BTCUSDT" if not symbols or len(symbols) == 0 else symbols[0]
        table_name = QuestDBSqlCandlesBuilder().get_table_name(f"{exchange}:{dummy_symbol}")

        # Build WHERE conditions
        conditions = []

        # Add symbol filtering if symbols are provided
        if symbols and len(symbols) > 0:
            quoted_symbols = [f"'{s.upper()}'" for s in symbols]
            conditions.append(f"symbol in ({', '.join(quoted_symbols)})")

        # Add time filtering if provided
        if start:
            conditions.append(f"timestamp >= '{start}'")
        if stop:
            conditions.append(f"timestamp < '{stop}'")

        # Build WHERE clause
        where_clause = f"where {' and '.join(conditions)}" if conditions else ""

        _rsmpl = f"sample by {QuestDBSqlCandlesBuilder._convert_time_delta_to_qdb_resample_format(timeframe)}"

        query = f"""
        select timestamp, 
        upper(symbol) as symbol,
        first(open) as open, 
        max(high) as high,
        min(low) as low,
        last(close) as close,
        sum(volume) as volume,
        sum(quote_volume) as quote_volume,
        sum(count) as count,
        sum(taker_buy_volume) as taker_buy_volume,
        sum(taker_buy_quote_volume) as taker_buy_quote_volume
        from "{table_name}" {where_clause} {_rsmpl};
        """
        res = self.execute(query)
        if res.empty:
            return res
        return res.set_index(["timestamp", "symbol"])

    def get_average_quote_volume(
        self,
        exchange: str,
        start: str | pd.Timestamp,
        stop: str | pd.Timestamp,
        timeframe: str = "1d",
    ) -> pd.Series:
        table_name = QuestDBSqlCandlesBuilder().get_table_name(f"{exchange}:BTCUSDT")
        query = f"""
        WITH sampled as (
            select timestamp, symbol, sum(quote_volume) as qvolume 
            from "{table_name}"
            where timestamp >= '{start}' and timestamp < '{stop}'
            SAMPLE BY {QuestDBSqlCandlesBuilder._convert_time_delta_to_qdb_resample_format(timeframe)}
        )
        select upper(symbol) as symbol, avg(qvolume) as quote_volume from sampled
        group by symbol
        order by quote_volume desc;
        """
        vol_stats = self.execute(query)
        if vol_stats.empty:
            return pd.Series()
        return vol_stats.set_index("symbol")["quote_volume"]

    def get_fundamental_data(
        self,
        exchange: str,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        stop: str | pd.Timestamp | None = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        # TODO: fix this to just fundamental
        table_name = {"BINANCE.UM": "coingecko.fundamental"}[exchange]
        query = f"select timestamp, asset, metric, last(value) as value from {table_name}"
        # TODO: fix handling without start/stop, where needs to be added
        if start or stop:
            conditions = []
            if start:
                conditions.append(f"timestamp >= '{start}'")
            if stop:
                conditions.append(f"timestamp < '{stop}'")
            query += " where " + " and ".join(conditions)
        if symbols:
            # py < 3.12 doesn't recognize f-string double quotes properly
            quoted_symbols = [f"'{s.upper()}'" for s in symbols]
            query += f" and asset in ({', '.join(quoted_symbols)})"
        _rsmpl = f"sample by {QuestDBSqlCandlesBuilder._convert_time_delta_to_qdb_resample_format(timeframe)}"
        query += f" {_rsmpl}"
        df = self.execute(query)
        if df.empty:
            return pd.DataFrame()
        return df.set_index(["timestamp", "asset", "metric"]).value.unstack("metric")

    def get_names(self) -> list[str]:
        return self._get_names(self._builder)

    @_retry
    def execute(self, query: str) -> pd.DataFrame:
        _cursor = self._connection.cursor()  # type: ignore
        _cursor.execute(query)  # type: ignore
        names = [d.name for d in _cursor.description]  # type: ignore
        records = _cursor.fetchall()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records, columns=names)

    @_retry
    def _read(
        self,
        data_id: str,
        start: str | None,
        stop: str | None,
        transform: DataTransformer,
        chunksize: int,
        timeframe: str | None,
        data_type: str,
        builder: QuestDBSqlBuilder,
    ) -> Any:
        # Apply maximum chunksize limits for candles_1m data type when timeframe is provided
        if chunksize > 0 and data_type == "candles_1m" and timeframe:
            max_chunksize = _calculate_max_candles_chunksize(timeframe)
            chunksize = min(chunksize, max_chunksize)

        start, end = handle_start_stop(start, stop)
        # If timeframe is not specified, assume 1 minute as default
        effective_timeframe = timeframe or "1m"

        if chunksize > 0:
            # Use efficient chunking with multiple smaller queries
            def _iter_efficient_chunks():
                time_windows = _calculate_time_windows_for_chunking(start, end, effective_timeframe, chunksize)
                if self._connection is None:
                    self._connect()
                    if self._connection is None:
                        raise ConnectionError("Failed to connect to QuestDB")

                _cursor = self._connection.cursor()  # type: ignore

                try:
                    for window_start, window_end in time_windows:
                        _req = builder.prepare_data_sql(
                            data_id, str(window_start), str(window_end), effective_timeframe, data_type
                        )
                        # logger.debug(f"Executing query: {_req}")

                        _cursor.execute(_req)  # type: ignore
                        names = [d.name for d in _cursor.description]  # type: ignore
                        records = _cursor.fetchall()

                        if records:
                            transform.start_transform(data_id, names, start=start, stop=stop)
                            transform.process_data(records)
                            yield transform.collect()
                finally:
                    _cursor.close()

            return _iter_efficient_chunks()

        # No chunking requested - return all data at once
        _req = builder.prepare_data_sql(data_id, start, end, effective_timeframe, data_type)
        # logger.debug(f"Executing query: {_req}")

        _cursor = self._connection.cursor()  # type: ignore
        try:
            _cursor.execute(_req)  # type: ignore
            names = [d.name for d in _cursor.description]  # type: ignore
            records = _cursor.fetchall()
            if not records:
                return None
            transform.start_transform(data_id, names, start=start, stop=stop)
            transform.process_data(records)
            return transform.collect()
        finally:
            _cursor.close()

    @_retry
    def _get_names(self, builder: QuestDBSqlBuilder) -> list[str]:
        _cursor = None
        try:
            _cursor = self._connection.cursor()  # type: ignore
            _cursor.execute(builder.prepare_names_sql())  # type: ignore
            records = _cursor.fetchall()
        finally:
            if _cursor:
                _cursor.close()
        return [r[0] for r in records]

    @_retry
    def _get_symbols(self, builder: QuestDBSqlBuilder, exchange: str, dtype: str) -> list[str]:
        _cursor = None
        try:
            _cursor = self._connection.cursor()  # type: ignore
            _cursor.execute(builder.prepare_symbols_sql(exchange, dtype))  # type: ignore
            records = _cursor.fetchall()
        finally:
            if _cursor:
                _cursor.close()
        return [f"{exchange}:{r[0].upper()}" for r in records]

    @_retry
    def _get_range(self, builder: QuestDBSqlBuilder, data_id: str) -> tuple[Any] | None:
        _cursor = None
        try:
            _cursor = self._connection.cursor()  # type: ignore
            _cursor.execute(builder.prepare_data_ranges_sql(data_id))  # type: ignore
            return tuple([np.datetime64(r[0]) for r in _cursor.fetchall()])
        finally:
            if _cursor:
                _cursor.close()

    def __del__(self):
        try:
            if self._connection is not None:
                logger.debug("Closing connection")
                self._connection.close()
        except:  # noqa: E722
            pass


class QuestDBSqlOrderBookBuilder(QuestDBSqlCandlesBuilder):
    """
    Sql builder for snapshot data
    """

    SNAPSHOT_DELTA = pd.Timedelta("1h")
    MIN_DELTA = pd.Timedelta("1s")

    def prepare_data_sql(
        self,
        data_id: str,
        start: str | None,
        end: str | None,
        resample: str,
        data_type: str,
    ) -> str:
        if not start or not end:
            raise ValueError("Start and end dates must be provided for orderbook data!")
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)

        raw_start_dt = start_dt.floor(self.SNAPSHOT_DELTA) - self.MIN_DELTA

        table_name = self.get_table_name(data_id, data_type)
        query = f"""
SELECT * FROM {table_name}
WHERE timestamp BETWEEN '{raw_start_dt}' AND '{end_dt}'
"""
        return query


class TradeSql(QuestDBSqlCandlesBuilder):
    def prepare_data_sql(
        self,
        data_id: str,
        start: str | None,
        end: str | None,
        resample: str,
        data_type: str,
    ) -> str:
        table_name = self.get_table_name(data_id, data_type)
        where = ""
        w0 = f"timestamp >= '{start}'" if start else ""
        w1 = f"timestamp <= '{end}'" if end else ""

        # - fix: when no data ranges are provided we must skip empy where keyword
        if w0 or w1:
            where = f"where {w0} and {w1}" if (w0 and w1) else f"where {(w0 or w1)}"

        resample = (
            QuestDBSqlCandlesBuilder._convert_time_delta_to_qdb_resample_format(resample) if resample else resample
        )
        if resample:
            sql = f"""
                select timestamp, first(price) as open, max(price) as high, min(price) as low, last(price) as close, 
                sum(size) as volume from "{table_name}" {where} SAMPLE by {resample};"""
        else:
            sql = f"""select timestamp, price, size, market_maker from "{table_name}" {where};"""

        return sql

    def prepare_symbols_sql(self, exchange: str, dtype: str) -> str:
        # TODO:
        raise NotImplementedError("Not implemented yet")


class QuestDBSqlFundingBuilder(QuestDBSqlBuilder):
    """
    SQL builder for funding payment data.

    Handles queries for funding payment data from QuestDB tables with schema:
    timestamp, symbol, funding_rate, funding_interval_hours
    """

    def prepare_data_sql(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        resample: str | None = None,
        data_type: str = "funding_payment",
    ) -> str:
        """
        Prepare SQL query for funding payment data.

        Args:
            data_id: Data identifier (e.g., 'BINANCE.UM:BTCUSDT')
            start: Start time string
            stop: Stop time string
            resample: Not used for funding payments
            data_type: Data type (should be 'funding_payment')

        Returns:
            SQL query string
        """
        _exch, _symb, _mktype = self._get_exchange_symbol_market_type(data_id)
        table_name = self.get_table_name(data_id, data_type)

        # Build WHERE clauses
        where_clauses = []

        if _symb:
            # Properly escape single quotes in symbol to prevent SQL injection
            escaped_symbol = _symb.upper().replace("'", "''")
            where_clauses.append(f"symbol = '{escaped_symbol}'")

        if start:
            where_clauses.append(f"timestamp >= '{start}'")

        if stop:
            where_clauses.append(f"timestamp <= '{stop}'")

        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        return f"""
        SELECT timestamp, symbol, funding_rate, funding_interval_hours
        FROM {table_name}
        {where_clause}
        ORDER BY timestamp ASC
        """.strip()

    def prepare_data_ranges_sql(self, data_id: str) -> str:
        """
        Prepare SQL to get time ranges for funding payment data.

        Args:
            data_id: Data identifier

        Returns:
            SQL query to get min/max timestamps
        """
        _exch, _symb, _mktype = self._get_exchange_symbol_market_type(data_id)
        table_name = self.get_table_name(data_id, "funding_payment")

        if _symb:
            escaped_symbol = _symb.upper().replace("'", "''")
            where_clause = f"WHERE symbol = '{escaped_symbol}'"
        else:
            where_clause = ""

        return f"""(SELECT timestamp FROM "{table_name}" {where_clause} ORDER BY timestamp ASC LIMIT 1)
                        UNION
                   (SELECT timestamp FROM "{table_name}" {where_clause} ORDER BY timestamp DESC LIMIT 1)
                """

    def get_table_name(self, data_id: str, sfx: str = "") -> str:
        """
        Get table name for funding payment data.

        For funding payments, we use a single table per exchange/market type:
        e.g., 'binance.umswap.funding_payment'

        Args:
            data_id: Data identifier
            sfx: Suffix (data type)

        Returns:
            Table name string
        """
        _exch, _symb, _mktype = self._get_exchange_symbol_market_type(data_id)

        # For funding payments, use a single aggregated table
        parts = [_exch.lower(), _mktype, sfx if sfx else "funding_payment"]
        return ".".join(filter(lambda x: x, parts))


@reader("mqdb")
@reader("multi")
@reader("questdb")
class MultiQdbConnector(QuestDBConnector):
    """
    Data connector for QuestDB which provides access to following data types:
      - candles
      - trades
      - orderbook snapshots
      - liquidations
      - funding rate
      - funding payments
    """

    _TYPE_TO_BUILDER = {
        "candles_1m": QuestDBSqlCandlesBuilder(),
        "tob": QuestDBSqlTOBBilder(),
        "trade": TradeSql(),
        "agg_trade": TradeSql(),
        "orderbook": QuestDBSqlOrderBookBuilder(),
        "funding_payment": QuestDBSqlFundingBuilder(),
    }

    _TYPE_MAPPINGS = {
        "candles": "candles_1m",
        "ohlc": "candles_1m",
        "trades": "trade",
        "ob": "orderbook",
        "trd": "trade",
        "td": "trade",
        "quote": "tob",
        "aggTrade": "agg_trade",
        "agg_trades": "agg_trade",
        "aggTrades": "agg_trade",
        "funding": "funding_payment",
        "funding_payment": "funding_payment",
        "funding_payments": "funding_payment",
    }

    def __init__(
        self,
        host="localhost",
        user="admin",
        password="quest",
        port=8812,
        timeframe: str = "1m",
    ) -> None:
        self._connection = None
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._default_timeframe = timeframe
        self._connect()

    @property
    def connection_url(self):
        return " ".join(
            [
                f"user={self._user}",
                f"password={self._password}",
                f"host={self._host}",
                f"port={self._port}",
            ]
        )

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize: int = 0,
        timeframe: str | None = None,
        data_type: str = "candles",
    ) -> Any:
        if timeframe is None:
            timeframe = self._default_timeframe

        _mapped_data_type = self._TYPE_MAPPINGS.get(data_type, data_type)
        return self._read(
            data_id,
            start,
            stop,
            transform,
            chunksize,
            timeframe,
            _mapped_data_type,
            self._TYPE_TO_BUILDER[_mapped_data_type],
        )

    def get_names(self, data_type: str) -> list[str]:
        return self._get_names(self._TYPE_TO_BUILDER[self._TYPE_MAPPINGS.get(data_type, data_type)])

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        return self._get_symbols(
            self._TYPE_TO_BUILDER[self._TYPE_MAPPINGS.get(dtype, dtype)],
            exchange,
            self._TYPE_MAPPINGS.get(dtype, dtype),
        )

    def get_time_ranges(self, symbol: str, dtype: str) -> tuple[np.datetime64, np.datetime64]:
        try:
            _xr = self._get_range(self._TYPE_TO_BUILDER[self._TYPE_MAPPINGS.get(dtype, dtype)], symbol)
            return (None, None) if not _xr else _xr  # type: ignore
        except Exception:
            return (None, None)  # type: ignore

    def get_funding_payment(
        self,
        exchange: str,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        stop: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Returns pandas DataFrame of funding payments for given exchange and symbols within specified time range.

        Args:
            exchange: Exchange identifier (e.g., "BINANCE.UM")
            symbols: List of symbols to filter by. If None or empty, returns all symbols.
            start: Start time for filtering. If None, no start time filter.
            stop: Stop time for filtering. If None, no stop time filter.

        Returns:
            DataFrame with MultiIndex [timestamp, symbol] and columns [funding_rate, funding_interval_hours]
        """
        # Use any symbol to get the table name (funding payments are in a single table per exchange)
        dummy_symbol = "BTCUSDT" if not symbols or len(symbols) == 0 else symbols[0]
        table_name = QuestDBSqlFundingBuilder().get_table_name(f"{exchange}:{dummy_symbol}")

        # Build WHERE conditions
        conditions = []

        # Add symbol filtering if symbols are provided
        if symbols and len(symbols) > 0:
            quoted_symbols = [f"'{s.upper()}'" for s in symbols]
            conditions.append(f"symbol in ({', '.join(quoted_symbols)})")

        # Add time filtering if provided
        if start:
            conditions.append(f"timestamp >= '{start}'")
        if stop:
            conditions.append(f"timestamp <= '{stop}'")

        # Build WHERE clause
        where_clause = f"where {' and '.join(conditions)}" if conditions else ""

        query = f"""
        select timestamp, 
        upper(symbol) as symbol,
        funding_rate,
        funding_interval_hours
        from "{table_name}" {where_clause}
        order by timestamp asc;
        """

        res = self.execute(query)
        if res.empty:
            return res
        return res.set_index(["timestamp", "symbol"])


class MultiTypeReader(DataReader):
    """
    Data reader for multiple data types. Can be used in simulation environment to provide data for multiple custom types.
    Example:
    ```
    idx = pd.date_range(start="2023-06-01 00:00", end="2023-06-01 01:00", freq="1s", name="timestamp")

    qts_raw = [TimestampedDict(t, {"bid": i, "ask": i, "bid_vol": 100, "ask_vol": 200}) for i, t in enumerate(idx)]
    trades_raw = [TimestampedDict(t, {"price": i, "size": 100}) for i, t in enumerate(idx)]
    ob_raw = [
        TimestampedDict(t, {
            "asks": [(100 + i/100, 100) for i in range(100)],
            "bids": [(100 - i/100, 100) for i in range(100)],
        }) for t in idx
    ]
    custom_raw = [
        TimestampedDict(t, {"what_to_do": np.random.choice(["buy", "sell"]), "how_much": i}) for i, t in enumerate(idx)
    ]

    reader = MultiTypeReader(
        {
            "BINANCE.UM:BTCUSDT": {
                "quote":     qts_raw,
                "trade":     trades_raw,
                "orderbook": ob_raw,
                "CUSTOM":    custom_raw,
            }
        }
    )

    class TestMulti(IStrategy):
        def on_market_data(self, ctx: IStrategyContext, data: MarketEvent) -> list[Signal] | Signal | None:
            logger.info(f'{data.instrument} market event ::: <g>{data.type}</g> ::: -> {data.data}')

    r = simulate({'Test multi': TestMulti()},
        {
            'quote':     reader,
            'trade':     reader,
            'orderbook': reader,
            'CUSTOM':    reader,
        },
        1000, ['BINANCE.UM:BTCUSDT'], "vip0_usdt", "2023-06-01 00:00", "2023-06-01 00:01", debug="DEBUG"
    )
    ```

    """

    _data: dict[str, dict[str, list[TimestampedDict]]]
    _columns_info: dict[str, dict[str, list[str]]]

    def __init__(self, data: dict[str, dict[str, list[TimestampedDict]]]) -> None:
        super().__init__()
        self._data = data

        # - extract fields names for each symbol and data type
        self._columns_info = {
            s: {t: list(records[0].data.keys()) for t, records in tps.items()} for s, tps in data.items()
        }

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        return list(self._data.keys())

    def _get_data_type(self, data_id: str, data_type: str) -> list[TimestampedDict]:
        if (_symb_cont := self._data.get(data_id)) is None:
            raise ValueError(f"No data for {data_id} symbol")

        raw_data = _symb_cont.get(data_type)
        if raw_data is None:
            raise ValueError(f"No data for {data_type} type")

        return raw_data

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        data_type: str | None = None,
        chunksize: int = 0,
        transform: DataTransformer = DataTransformer(),
        **kwargs,
    ) -> Iterable | list:
        start, stop = handle_start_stop(start, stop)
        raw_data = self._get_data_type(data_id, data_type)

        _find_idx = lambda xs, t: next((i for i, item in enumerate(xs) if item.time >= pd.Timestamp(t)), -1)
        _tsd2rec = lambda td: [td.time, *td.data.values()]

        _s0 = _find_idx(raw_data, start)
        _s1 = _find_idx(raw_data, stop)
        _sliced_data = [_tsd2rec(x) for x in raw_data[_s0:_s1]]
        columns = ["timestamp", *self._columns_info[data_id][data_type]]

        def _do_transform(values: Iterable, columns: list[str]) -> Iterable:
            transform.start_transform(data_id, columns, start=start, stop=stop)
            transform.process_data(values)
            return transform.collect()

        if chunksize > 0:

            def _chunked_dataframe(data: list[TimestampedDict], columns: list[str], chunksize: int) -> Iterable:
                it = iter(data)
                chunk = list(itertools.islice(it, chunksize))
                while chunk:
                    yield _do_transform(chunk, columns)
                    chunk = list(itertools.islice(it, chunksize))

            return _chunked_dataframe(_sliced_data, columns, chunksize)

        return _do_transform(_sliced_data, columns)

    def get_time_ranges(self, symbol: str, dtype: DataType) -> tuple[np.datetime64 | None, np.datetime64 | None]:
        raw_data = self._get_data_type(symbol, dtype)
        return raw_data[0].time, raw_data[-1].time
