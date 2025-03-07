from typing import Any, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.readers import DataReader, DataTransformer


# TODO: implement properly
class CompositeReader(DataReader):
    """
    A composite reader that tries to read data from multiple readers in order.

    This reader will try to read data from each reader in the order they were provided.
    If a reader fails to provide data (returns None or raises an exception), the next reader is tried.

    Args:
        readers: A list of DataReader instances to use for reading data.
    """

    def __init__(self, readers: List[DataReader]) -> None:
        """Initialize the composite reader with a list of readers."""
        self._readers = readers

    def get_names(self, **kwargs) -> List[str]:
        """Get all available data names from all readers."""
        names = set()
        for reader in self._readers:
            try:
                reader_names = reader.get_names(**kwargs)
                names.update(reader_names)
            except Exception as e:
                logger.warning(f"Failed to get names from reader {reader}: {e}")
        return list(names)

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        **kwargs,
    ) -> Iterator | List:
        """
        Read data from the first reader that successfully returns data.

        Args:
            data_id: The data identifier to read.
            start: The start time for the data.
            stop: The end time for the data.
            transform: A transformer to apply to the data.
            chunksize: The chunk size for reading data.
            **kwargs: Additional arguments to pass to the readers.

        Returns:
            The data from the first reader that successfully returns data.

        Raises:
            Exception: If all readers fail to provide data.
        """
        last_error = None

        for i, reader in enumerate(self._readers):
            try:
                logger.debug(f"Trying to read {data_id} from reader {i + 1}/{len(self._readers)}")
                result = reader.read(
                    data_id=data_id,
                    start=start,
                    stop=stop,
                    transform=transform,
                    chunksize=chunksize,
                    **kwargs,
                )
                if result is not None:
                    logger.debug(f"Successfully read {data_id} from reader {i + 1}/{len(self._readers)}")
                    return result
            except Exception as e:
                logger.debug(f"Failed to read {data_id} from reader {i + 1}/{len(self._readers)}: {e}")
                last_error = e

        if last_error:
            raise last_error
        else:
            raise ValueError(f"No reader could provide data for {data_id}")

    def get_aux_data_ids(self) -> Set[str]:
        """Get all available auxiliary data IDs from all readers."""
        aux_data_ids = set()
        for reader in self._readers:
            try:
                reader_aux_data_ids = reader.get_aux_data_ids()
                aux_data_ids.update(reader_aux_data_ids)
            except Exception as e:
                logger.warning(f"Failed to get aux data IDs from reader {reader}: {e}")
        return aux_data_ids

    def get_aux_data(self, data_id: str, **kwargs) -> Any:
        """Get auxiliary data from the first reader that successfully returns data."""
        last_error = None

        for i, reader in enumerate(self._readers):
            try:
                logger.debug(f"Trying to get aux data {data_id} from reader {i + 1}/{len(self._readers)}")
                result = reader.get_aux_data(data_id=data_id, **kwargs)
                if result is not None:
                    logger.debug(f"Successfully got aux data {data_id} from reader {i + 1}/{len(self._readers)}")
                    return result
            except Exception as e:
                logger.debug(f"Failed to get aux data {data_id} from reader {i + 1}/{len(self._readers)}: {e}")
                last_error = e

        if last_error:
            raise last_error
        else:
            raise ValueError(f"No reader could provide aux data for {data_id}")

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        """Get all available symbols from all readers."""
        symbols = set()
        for reader in self._readers:
            try:
                reader_symbols = reader.get_symbols(exchange=exchange, dtype=dtype)
                symbols.update(reader_symbols)
            except Exception as e:
                logger.warning(f"Failed to get symbols from reader {reader}: {e}")
        return list(symbols)

    def get_time_ranges(self, symbol: str, dtype: DataType) -> tuple[np.datetime64 | None, np.datetime64 | None]:
        """Get the time range from the first reader that successfully returns a range."""
        last_error = None

        for i, reader in enumerate(self._readers):
            try:
                logger.debug(f"Trying to get time range for {symbol} from reader {i + 1}/{len(self._readers)}")
                result = reader.get_time_ranges(symbol=symbol, dtype=dtype)
                if result is not None and result[0] is not None and result[1] is not None:
                    logger.debug(f"Successfully got time range for {symbol} from reader {i + 1}/{len(self._readers)}")
                    return result
            except Exception as e:
                logger.debug(f"Failed to get time range for {symbol} from reader {i + 1}/{len(self._readers)}: {e}")
                last_error = e

        if last_error:
            raise last_error
        else:
            return np.datetime64(None), np.datetime64(None)
