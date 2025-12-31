import os
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.containers import IteratorsMaster, RawData, RawMultiData
from qubx.data.registry import storage
from qubx.data.storage import IReader, IStorage, Transformable
from qubx.data.storages.utils import find_time_col_idx, recognize_t
from qubx.utils.time import handle_start_stop


class CsvReader(IReader):
    """
    TODO: add docstring here
    """

    _reader_path: Path
    _dtyped_symbols: dict[DataType, list[tuple[str, str]]]

    def __init__(self, path: Path, dtypes: dict[DataType, list[tuple[str, str]]]):
        self._reader_path = path
        self._dtyped_symbols = dtypes

    def _find_time_idx(self, arr: pa.ChunkedArray, v) -> int:
        # - Try exact match first
        ix = arr.index(v).as_py()
        if ix >= 0:
            return ix

        # - Value not found exactly, find insertion point using searchsorted
        offset = 0
        for c in arr.iterchunks():
            a = c.to_numpy()
            local_ix = np.searchsorted(a, v, side="right")

            if local_ix < len(c):
                # - Found insertion point in this chunk
                global_ix = offset + local_ix
                # - Return index of element just before insertion point
                return max(global_ix - 1, 0)

            offset += len(c)

        # - Value is after all data, return last index
        return offset - 1

    def _get_file_name(self, dtype: DataType, data_id: str) -> Path | None:
        for _s, _fi in self._dtyped_symbols.get(dtype, []):
            if _s == data_id.upper():
                if ".csv" in (_ff := (self._reader_path / _fi)).suffixes and _ff.exists():
                    return _ff
        return None

    def _try_read_data(
        self,
        data_id: str,
        dtype: DataType,
        start: str | None = None,
        stop: str | None = None,
        timestamp_formatters=None,
    ) -> tuple[pa.table, np.ndarray, Any, list[str], int, int]:
        f_path = self._get_file_name(dtype, data_id)
        if not f_path:
            raise ValueError(f"Can't find any csv data for '{data_id}' of '{dtype}' in {self._reader_path} !")

        convert_options = None
        if timestamp_formatters is not None:
            convert_options = pa.csv.ConvertOptions(timestamp_parsers=timestamp_formatters)

        table = pa.csv.read_csv(
            f_path, parse_options=pa.csv.ParseOptions(ignore_empty_lines=True), convert_options=convert_options
        )
        fieldnames = table.column_names

        # - try to find range to load
        start_idx, stop_idx = 0, table.num_rows
        try:
            _time_field_idx = find_time_col_idx(fieldnames)
            _time_type = table.field(_time_field_idx).type
            _time_unit = _time_type.unit if hasattr(_time_type, "unit") else "ms"
            _time_data = table[_time_field_idx]

            # - check if need convert time to primitive types (i.e. Date32 -> timestamp[x])
            _time_cast_function = lambda xs: xs
            if _time_type != pa.timestamp(_time_unit):
                _time_cast_function = lambda xs: xs.cast(pa.timestamp(_time_unit))
                _time_data = _time_cast_function(_time_data)

            # - preprocessing start and stop
            t_0, t_1 = handle_start_stop(start, stop, convert=lambda x: recognize_t(x, None, _time_unit))

            # - check requested range
            if t_0:
                start_idx = self._find_time_idx(_time_data, t_0)
                if start_idx >= table.num_rows:
                    # - no data for requested start date
                    return table, _time_data, _time_unit, fieldnames, -1, -1

            if t_1:
                stop_idx = self._find_time_idx(_time_data, t_1)
                if stop_idx < 0 or stop_idx < start_idx:
                    stop_idx = table.num_rows

        except Exception as exc:
            logger.warning(f"exception [{exc}] during preprocessing '{f_path}'")

        return table, _time_data, _time_unit, fieldnames, start_idx, stop_idx

    def get_time_range(self, data_id: str, dtype: DataType) -> tuple[Any, Any]:
        _, _time_data, _time_unit, _, start_idx, stop_idx = self._try_read_data(data_id, dtype, None, None, None)
        return (
            np.datetime64(_time_data[start_idx].value, _time_unit),
            np.datetime64(_time_data[max(stop_idx - 1, 0)].value, _time_unit),
        )

    def get_data_id(self, dtype: DataType = DataType.ALL) -> list[str] | dict[DataType, list[str]]:
        if dtype == DataType.ALL:
            r = []
            for vs in self._dtyped_symbols.values():
                r.extend(v[0] for v in vs)
            return list(set(r))

        return [v[0] for v in self._dtyped_symbols.get(dtype, [])]

    def get_data_types(self, data_id: str) -> list[DataType]:
        _du = data_id.upper()
        return [k for k, vs in self._dtyped_symbols.items() if _du in [x[0] for x in vs]]

    def _read_single_data_id(
        self,
        data_id: str,
        dtype: DataType,
        start: str | None = None,
        stop: str | None = None,
        chunksize=0,
        **kwargs,
    ) -> RawData | Iterator[RawData]:
        table, _, _, fieldnames, start_idx, stop_idx = self._try_read_data(
            data_id, dtype, start, stop, kwargs.get("timestamp_formatters")
        )
        if start_idx < 0 or stop_idx < 0:
            logger.warning(f"Can't detect proper start and stop timestamps for {data_id} of {dtype} !")
            return iter([])

        length = stop_idx - start_idx + 1
        selected_table = table.slice(start_idx, length)

        # - in this case we want to return iterable chunks of data
        if chunksize > 0:

            def _chunks_iterator():
                for n in range(0, length // chunksize + 1):
                    raw_data = selected_table[n * chunksize : min((n + 1) * chunksize, length)].to_pandas().to_numpy()
                    yield RawData(data_id, fieldnames, dtype, raw_data)

            return _chunks_iterator()

        raw_data = selected_table.to_pandas().to_numpy()
        return RawData(data_id, fieldnames, dtype, raw_data)

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType,
        start: str | None = None,
        stop: str | None = None,
        chunksize=0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        if isinstance(data_id, (list, tuple)):
            multi = [self._read_single_data_id(d_id, dtype, start, stop, chunksize, **kwargs) for d_id in data_id]
            return IteratorsMaster(multi) if chunksize > 0 else RawMultiData(multi)  # type: ignore
        return self._read_single_data_id(data_id, dtype, start, stop, chunksize, **kwargs)


@storage("csv")
class CsvStorage(IStorage):
    _path: str
    _exchanges: dict[str, dict[str, dict[DataType, list[tuple[str, str]]]]]

    def __init__(self, path: str):
        if not os.path.exists(_path := os.path.expanduser(path)):
            raise ValueError(f"Specified path not found at '{path}'")
        self._path = _path
        self._read_data_structure()

    def _recognize_mkt_type(self, mtype: str) -> DataType:
        """
        Recognize specified data type
            - 1H -> DataType.OHLC["1h"]
            - trades or trade -> DataType.TRADE
            - quotes or quote -> DataType.QUOTE
            - orderbook or ob -> DataType.ORDERBOOK
        etc
        """
        match mtype_lower := mtype.lower():
            case _ if re.match(r"^\d+[hdw]$", mtype_lower):
                return DataType.OHLC[mtype_lower]  # type: ignore

            case _ if re.match(r"^\d+m$", mtype_lower):
                return DataType.OHLC[mtype_lower[:-1] + "Min"]  # type: ignore

            case _ if re.match(r"^\d+min$", mtype_lower):
                return DataType.OHLC[mtype_lower]  # type: ignore

            case "trades" | "trade":
                return DataType.TRADE

            case "quotes" | "quote":
                return DataType.QUOTE

            case "orderbook" | "ob":  # - TODO: do we need to provide/check OB parameters here ?
                return DataType.ORDERBOOK

            case "liquidation" | "lq":
                return DataType.LIQUIDATION

            case "funding_rate" | "fr":
                return DataType.FUNDING_RATE

            case "funding_payment" | "fp":
                return DataType.FUNDING_PAYMENT

            case "open_interest" | "oi":
                return DataType.OPEN_INTEREST

            case _:
                raise ValueError(f"Unrecognized datatype: '{mtype}'")

    def _read_data_structure(self):
        _exchanges = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for root, _, files in os.walk(self._path):
            path = root.split(os.sep)
            for f_name in files:
                if re.match(r"(.*)\.csv(.gz)?$", f_name):
                    # - file name must have type of data: (quotes, trades, ohlc etc)
                    # - example: BTCUSDT.1H.csv or BTCUSDT.trades.csv.gz etc
                    if len(sn := f_name.split(".")) < 3 or len(path) < 2:
                        continue

                    # - exchange and market type
                    ex, mt = path[-2], path[-1]
                    dt = self._recognize_mkt_type(sn[1])
                    # name = f"{ex}:{mt}:{sn[0]} of {dt} -> {f_name}"
                    _exchanges[ex.upper()][mt.upper()][dt].append((sn[0], f_name))
        # - convert to ordinary dict to prevent adding something
        self._exchanges = {
            s0: {s1: {s2: v for s2, v in v1.items()} for s1, v1 in v0.items()} for s0, v0 in _exchanges.items()
        }

    def get_exchanges(self) -> list[str]:
        return list(self._exchanges.keys())

    def get_market_types(self, exchange: str) -> list[str]:
        return list(self._exchanges[exchange.upper()].keys())

    def get_reader(self, exchange: str, market: str) -> IReader:
        if (_e := exchange.upper()) not in self._exchanges:
            raise ValueError(f"Data for exchange {_e} not found in this storage")

        if (_m := market.upper()) not in self._exchanges[_e]:
            raise ValueError(f"Data for {_m} is not found for exchange {_e} in this storage")

        return CsvReader(Path(self._path) / _e / _m, self._exchanges[_e][_m])
