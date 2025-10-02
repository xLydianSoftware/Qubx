#
# New experimental data reading interface. We need to deprecate old DataReader approach after this new one will be finished and approved
#
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from qubx.core.basics import DataType
from qubx.data.storages.utils import find_time_col_idx


class IDataTransformer:
    def process_data(
        self, data_id: str, dtype: DataType, raw_data: Iterable[np.ndarray], names: list[str], index: int
    ) -> Any: ...

    def combine_data(self, transformed: dict[str, Any]) -> Any:
        return transformed


class Transformable:
    def transform(self, transformer: IDataTransformer) -> Any: ...


class RawData(Transformable):
    """
    Data container that holds raw output from IReader.read() for single data_id.
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


class RawMultiData(Transformable):
    """
    Data container that holds raw outputs from IReader.read() for multiple data_id.
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

    def transform(self, transformer: IDataTransformer) -> Any:
        return transformer.combine_data({k: r.transform(transformer) for k, r in self.raws.items()})

    def __getitem__(self, data_id: str) -> RawData:
        return self.raws[data_id]

    def __len__(self) -> int:
        return len(self.raws)


class IReader:
    def read(
        self, data_id: str, dtype: DataType | str, start: str | None, stop: str | None, chunksize=0, **kwargs
    ) -> Iterator[Transformable] | Transformable: ...

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
        Just shorthand for the get_reader() method
        """
        return self.get_reader(*key)
