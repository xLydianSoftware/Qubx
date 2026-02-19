from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from qubx.core.basics import DataType, Timestamped
from qubx.core.series import OHLCV, ColumnarSeries
from qubx.data.storage import IDataTransformer, IRawContainer, Transformable
from qubx.data.storages.utils import find_time_col_idx
from qubx.data.transformers import ColumnarSeriesTransformer, OHLCVSeries, PandasFrame, TypedGenericSeries, TypedRecords


class TransformableWithHelpers(Transformable):
    """
    Handy basic transformer helpers to avoid typing x.transform(PandasFrame()) etc
    """

    def to_pd(self, id_in_index=False) -> pd.DataFrame:
        """
        Transform raw data into a pandas DataFrame.

        Args:
            id_in_index: if True, includes data_id (symbol) as a column in the result
        """
        return self.transform(PandasFrame(id_in_index))

    def to_ohlc(self, timestamp_units="ns", max_length=np.inf) -> OHLCV:
        """
        Transform raw data into an OHLCV series (streaming-compatible).

        Args:
            timestamp_units: time resolution for timestamps (default "ns")
            max_length: maximum number of bars to keep in the series
        """
        return self.transform(OHLCVSeries(timestamp_units, max_length))

    def to_records(self, timestamp_units="ns") -> list[Timestamped]:
        """
        Transform raw data into typed record objects (Trade, Quote, Bar, etc.).

        Args:
            timestamp_units: time resolution for timestamps (default "ns")
        """
        return self.transform(TypedRecords(timestamp_units=timestamp_units))

    def to_series(self, timestamp_units="ns") -> TypedGenericSeries:
        """
        Transform raw data into a generic typed series.

        Args:
            timestamp_units: time resolution for timestamps (default "ns")
        """
        return self.transform(TypedGenericSeries(timestamp_units=timestamp_units))

    def to_columnar_series(self, timestamp_units="ns", max_length=np.inf) -> ColumnarSeries:
        """
        Transform raw data into a ColumnarSeries with individual TimeSeries for each column.

        Unlike to_series() which stores complete objects, ColumnarSeries decomposes data
        into separate TimeSeries for each column. This allows attaching indicators to
        individual columns and proper indicator update propagation.

        Args:
            timestamp_units: time resolution for timestamps (default "ns")
            max_length: maximum number of items to keep in the series

        Example:
            cs = raw_data.to_columnar_series()
            ratio = cs.taker_buy_sell_ratio  # Returns TimeSeries
            sma = ta.Sma.wrap(ratio, 20)  # Attach indicator
        """
        return self.transform(ColumnarSeriesTransformer(timestamp_units=timestamp_units, max_length=max_length))


class RawData(IRawContainer, TransformableWithHelpers):
    """
    Container for raw market data from a single instrument/symbol.

    Holds untransformed data as returned by IReader.read() along with metadata needed
    for further processing. The raw data is stored as pyarrow RecordBatch object

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
        ticks = raw.transform(EmulatedTickSequence(trades=True, quotes=True))

        # - Get time range
        start, end = raw.get_time_interval()

    The transform() method delegates to IDataTransformer implementations for
    converting raw data into various formats (DataFrames, typed objects, series, etc.).
    """

    _create_key = object()
    __slots__ = ("data_id", "dtype", "_raw", "_index")

    def __init__(self, data_id: str, dtype: DataType, record_batch: pa.RecordBatch, *, _key=None):
        if _key is not RawData._create_key:
            raise TypeError("Do not call RawData() directly, use RawData.from_*() methods")
        self.data_id = data_id
        self.dtype = dtype
        self._raw = record_batch
        self._index = find_time_col_idx(self.names)

    @property
    def names(self) -> list[str]:
        return list(self._raw.schema.names)

    @property
    def data(self) -> pa.RecordBatch:
        return self._raw

    @property
    def index(self) -> int:
        return self._index

    @classmethod
    def from_record_batch(cls, data_id: str, dtype: DataType, data: pa.RecordBatch) -> "RawData":
        return cls(data_id, dtype, data, _key=cls._create_key)

    @classmethod
    def from_table(cls, data_id: str, dtype: DataType, data: pa.Table) -> "RawData":
        batches = data.combine_chunks().to_batches()
        batch = batches[0] if batches else pa.RecordBatch.from_pydict({n: [] for n in data.schema.names})
        return cls(data_id, dtype, batch, _key=cls._create_key)

    @classmethod
    def from_pandas(cls, data_id: str, dtype: DataType, data: pd.DataFrame | pd.Series) -> "RawData":
        if isinstance(data, pd.Series):
            _data = data.reset_index(name=data.name or "value")
            _data.columns = ["time", _data.columns[1]]
        else:
            _data = data.reset_index() if data.index.name else data
        return cls(data_id, dtype, pa.RecordBatch.from_pandas(_data), _key=cls._create_key)

    def __len__(self) -> int:
        return self._raw.num_rows

    def get_time_interval(self) -> tuple[int | None, int | None]:
        """
        Returns start and end timestamp from raw data
        """
        if len(self) == 0:
            return (None, None)
        _idx_col = self._raw.column(self._index)
        return (_idx_col[0].as_py(), _idx_col[-1].as_py())

    def transform(self, transformer: IDataTransformer) -> object:
        return transformer.process_data(self)

    def __repr__(self) -> str:
        s, e = self.get_time_interval()
        _range = f"{s} : {e}" if s is not None else "EMPTY"
        return f"{self.data_id}({self.dtype})[{_range}] : ({len(self)} x {len(self.names)})"


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
                _t = type(r.data)
            elif _t is not type(r.data):
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
        return transformer.combine_data(self.raws)

    def to_pd(self, id_in_index: bool = False) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.

        Delegates to ``transform(PandasFrame(id_in_index))``.  PandasFrame uses a
        bulk Arrow concat path for ``id_in_index=True`` (significantly faster than
        N per-symbol conversions + ``pd.concat``), and an auto-pivot path for
        ``id_in_index=False`` when the data is in long format (e.g. FUNDAMENTAL).
        """
        return self.transform(PandasFrame(id_in_index))

    def __getitem__(self, data_id: str) -> RawData:
        return self.raws[data_id]

    def __iter__(self) -> Iterator[RawData]:
        return iter(self.raws.values())

    def __len__(self) -> int:
        return len(self.raws)

    def __repr__(self) -> str:
        return "-[MULTI DATA]-\n" + "\n".join([" | " + repr(s) for s in self.raws.values()])


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
