#
# New experimental data reading interface. We need to deprecate old DataReader approach after this new one will be finished and approved
#
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from qubx.core.basics import DataType


class IDataTransformer:
    def process_data(
        self, data_id: str, dtype: DataType, raw_data: Iterable[np.ndarray], names: list[str], index: int
    ) -> Any: ...

    def combine_data(self, transformed: dict[str, Any]) -> Any:
        return transformed


class Transformable:
    def transform(self, transformer: IDataTransformer) -> Any: ...


class IReader:
    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
        chunksize=0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable: ...

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
