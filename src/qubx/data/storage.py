#
# New experimental data reading interface. We need to deprecate old DataReader approach after this new one will be finished and approved
#
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd

from qubx.core.basics import DataType
from qubx.core.interfaces import Timestamped
from qubx.core.series import OHLCV, Bar, OrderBook, Quote, Trade, TradeArray
from qubx.data.storages.utils import find_column_index_in_list, find_time_col_idx, recognize_t
from qubx.utils.time import handle_start_stop, infer_series_frequency


class IDataTransformer:
    def process_data(
        self, data_id: str, dtype: DataType, raw_data: Iterable[np.ndarray], names: list[str], index: int
    ) -> Any: ...

class RawData:
    """
    Data container that holds raw output from IReader.read() for single data_id.
    """

    data_id: str
    names: list[str]
    dtype: DataType
    raw: Iterable
    _index: int

    def __init__(self, data_id: str, field_names: list[str], dtype: DataType, data: Iterable):
        self.data_id = data_id
        self.names = field_names
        self.dtype = dtype
        self.raw = data
        self._index = find_time_col_idx(field_names)

    def __len__(self) -> int:
        return len(self.raw)

    def transform(self, transformer: IDataTransformer) -> Any:
        return transformer.process_data(self.data_id, self.dtype, self.raw, self.names, self._index)


class IReader:
    def read(
        self, data_id: str, dtype: DataType | str, start: str | None, stop: str | None, chunksize=0, **kwargs
    ) -> Iterator[RawData] | RawData: ...

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str] | dict[DataType, list[str]]:
        """
        Returns data id this reader provides for specified data type (or all types ?).
        """
        ...

    def get_data_types(self, data_id: str) -> list[DataType]:
        """
        Returns what data types this reader provides for specified data_id.
        """
        ...

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[np.datetime64, np.datetime64]:
        """
        Returns first and last time for the specified data_id and type in this reader
        """
        ...


class IStorage:
    """
    Generic interface for storage
    """

    def get_exchanges(self) -> list[str]: ...

    def get_market_types(self, exchange: str) -> list[str]: ...

    def get_reader(self, exchange: str, market: str) -> IReader:
        """
        Returns data reader for specified exchange and market type. For example storage.get_reader("BINANCE.UM", "SWAP")
        """
        ...

    def __getitem__(self, key: tuple[str, str]) -> IReader:
        """
        Just shorthand for get_reader() method
        """
        return self.get_reader(*key)
