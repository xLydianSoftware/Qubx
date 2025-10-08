from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from qubx.core.basics import DataType
from qubx.core.series import OHLCV
from qubx.data.storage import IDataTransformer, Transformable
from qubx.data.storages.utils import find_time_col_idx
from qubx.data.transformers import OHLCVSeries, PandasFrame, TickSeries, TypedRecords


class TransformableWithHelpers(Transformable):
    """
    Handy transformer helpers to avoid typing x.transform(PandasFrame()) etc
    """

    def to_pd(self, id_in_index=False) -> pd.DataFrame:
        return self.transform(PandasFrame(id_in_index))

    def to_ohlc(self, timestamp_units="ns", max_length=np.inf) -> OHLCV:
        return self.transform(OHLCVSeries(timestamp_units, max_length))

    def to_ticks(
        self,
        trades: bool = False,  # if we also wants 'trades'
        default_bid_size=1e9,  # default bid/ask is big
        default_ask_size=1e9,  # default bid/ask is big
        daily_session_start_end=TickSeries.DEFAULT_DAILY_SESSION,
        timestamp_units="ns",
        spread=0.0,
        open_close_time_shift_secs=1.0,
        quotes=True,
    ) -> TypedRecords:
        return self.transform(
            TickSeries(
                trades,
                default_bid_size,
                default_ask_size,
                daily_session_start_end,
                timestamp_units,
                spread,
                open_close_time_shift_secs,
                quotes,
            )
        )

    def to_records(self, timestamp_units="ns") -> TypedRecords:
        return self.transform(TypedRecords(timestamp_units=timestamp_units))


class RawData(TransformableWithHelpers):
    """
    Container for raw market data from a single instrument/symbol.

    Holds untransformed data as returned by IReader.read() along with metadata needed
    for further processing. The raw data is stored as a list/array with named columns.

    Attributes:
        data_id: Symbol or instrument identifier (e.g., 'BTCUSDT', 'ETHUSDT')
        names: Column names from the data source (e.g., ['time', 'price', 'size'])
        dtype: Type of market data (DataType.TRADE, DataType.QUOTE, DataType.OHLC, etc.)
        raw: Raw data as numpy array or list of rows

    Usage:
        # - Read raw data from storage
        raw = reader.read("BTCUSDT", DataType.OHLC["1h"])

        # - Transform to pandas DataFrame
        df = raw.transform(PandasFrame())

        # - Transform to typed records (Trade, Quote, OHLCV objects)
        records = raw.transform(TypedRecords())

        # - Transform to OHLCV series
        ohlcv = raw.transform(OHLCVSeries())

        # - Transform to simulated tick data
        ticks = raw.transform(TickSeries(trades=True, quotes=True))

        # - Get time range
        start, end = raw.get_time_interval()

    The transform() method delegates to IDataTransformer implementations for
    converting raw data into various formats (DataFrames, typed objects, series, etc.).
    """

    data_id: str
    names: list[str]
    dtype: DataType
    raw: list
    _index: int

    def __init__(self, data_id: str, field_names: list[str], dtype: DataType, data: list):
        self.data_id = data_id
        self.names = field_names
        self.dtype = dtype
        self.raw = data
        self._index = find_time_col_idx(field_names)

    def __len__(self) -> int:
        return len(self.raw)

    def get_time_interval(self) -> tuple:
        """
        Returns start and end timestamp from raw data
        """
        return (self.raw[0][self._index], self.raw[-1][self._index]) if len(self) > 0 else (None, None)

    def transform(self, transformer: IDataTransformer) -> Any:
        return transformer.process_data(self.data_id, self.dtype, self.raw, self.names, self._index)

    def __repr__(self) -> str:
        s, e = self.get_time_interval()
        _range = f"{s} : {e}" if s and e else "EMPTY"
        return f"{self.data_id}({self.dtype})[{_range}]"


class RawMultiData(TransformableWithHelpers):
    """
    Container for raw market data from multiple instruments/symbols.

    Aggregates multiple RawData instances for batch processing and multi-instrument
    transformations. All contained RawData must have the same underlying data type
    (same raw data structure) to ensure consistent processing.

    Attributes:
        raws: Dictionary mapping data_id (symbol) to RawData instances

    Usage:
        # - Read data for multiple symbols
        btc_raw = reader.read("BTCUSDT", DataType.OHLC["1h"])
        eth_raw = reader.read("ETHUSDT", DataType.OHLC["1h"])

        # - Combine into multi-data container
        multi = RawMultiData([btc_raw, eth_raw])

        # - Transform to multi-index DataFrame (symbol as level)
        df = multi.transform(PandasFrame(symbol_as_index=True))
        # Returns DataFrame with MultiIndex (timestamp, symbol)

        # - Transform to dict of DataFrames
        dfs = multi.transform(PandasFrame(symbol_as_index=False))
        # Returns {'BTCUSDT': df1, 'ETHUSDT': df2}


    The transform() method applies the transformer to each contained RawData and
    then calls combine_data() to merge results according to transformer logic.

    Raises:
        ValueError: If attempting to add RawData with incompatible raw data type
    """

    raws: dict[str, RawData]

    def __init__(self, data: list[RawData]):
        self.raws = {}

        _t = None
        for r in data:
            if not _t:
                _t = type(r.raw)
            elif _t is not type(r.raw):
                raise ValueError(f"RawMultiData container may contain only single data type {_t.__name__}")

            self.raws[r.data_id] = r

    def pop(self, data_id: str) -> RawData | None:
        if data_id in self.raws:
            return self.raws.pop(data_id)
        return None

    def add(self, r: RawData):
        self.raws[r.data_id] = r

    def get_time_interval(self, data_id: str) -> tuple:
        return self.raws[data_id].get_time_interval()

    def get_data_ids(self) -> list[str]:
        return list(self.raws.keys())

    def transform(self, transformer: IDataTransformer) -> Any:
        return transformer.combine_data({k: r.transform(transformer) for k, r in self.raws.items()})

    def __getitem__(self, data_id: str) -> RawData:
        return self.raws[data_id]

    def __len__(self) -> int:
        return len(self.raws)

    def __repr__(self) -> str:
        return "-[MULTI DATA]-\n" + "\n".join(["\t - " + repr(s) for s in self.raws.values()])


class IteratorsMaster:
    """
    Manages multiple iterators and advances them in parallel.
    """

    iterators: list[Iterator]

    def __init__(self, iterators: list[Iterator]):
        self.iterators = iterators

    def __iter__(self):
        return self

    def __next__(self) -> RawMultiData:
        """
        Advance all iterators and return their values as a list.

        Returns:
            List of values in the same order as the dictionary keys.

        Raises:
            StopIteration: When all underlying iterators are exhausted.
        """
        result = []
        all_exhausted = True

        for iter in self.iterators:
            try:
                value = next(iter)
                result.append(value)
                all_exhausted = False
            except StopIteration:
                pass

        if all_exhausted:
            raise StopIteration

        # - return multi data
        return RawMultiData(result)
