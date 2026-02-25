from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa

from qubx.core.basics import DataType


class IRawContainer:
    @property
    def data_id(self) -> str: ...

    @property
    def names(self) -> list[str]: ...

    @property
    def data(self) -> pa.RecordBatch: ...

    @property
    def dtype(self) -> DataType: ...

    @property
    def index(self) -> int: ...


class Transformable:
    def transform(self, transformer: "IDataTransformer") -> Any: ...


class IDataTransformer:
    def process_data(self, data: Transformable) -> Any: ...

    def combine_data(self, transformed: dict[str, Any]) -> Any:
        """
        Merge per-symbol results into a single output.

        Default: when raw IRawContainer objects are passed (e.g. from
        RawMultiData.transform), apply process_data() to each one first,
        then return the resulting dict.  Subclasses override for richer merging
        (e.g. PandasFrame uses a fast Arrow concat for the id_in_index=True path).
        """
        if transformed:
            first = next(iter(transformed.values()))
            if isinstance(first, IRawContainer):
                return {k: self.process_data(v) for k, v in transformed.items()}
        return transformed


class IReader:
    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
        chunksize=0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        """
        Read data for given symbol(s) and data type.

        ``data_id`` may be:
          - a single symbol string (e.g. ``"BTCUSDT"``) → returns ``RawData``
          - a list of symbol strings → returns ``RawMultiData``
          - an empty list ``[]`` or empty set ``set()`` → reads every symbol
            available for the requested ``dtype`` without a separate
            ``get_data_id()`` call; always returns ``RawMultiData``
        """
        ...

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
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

    def close(self) -> None:
        """
        If reader provides close operation
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
