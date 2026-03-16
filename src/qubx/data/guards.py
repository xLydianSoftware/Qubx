"""
Time-guarded wrappers for IReader and IStorage.

Prevents look-ahead bias in simulation by clamping the `stop` parameter
at the read level — before any SQL/fetch happens. This is much more efficient
than the old TimeGuardedWrapper approach (fetch all, then truncate in pandas).

TimeGuardedReader wraps an IReader and clamps stop to the current simulation time.
TimeGuardedStorage wraps an IStorage and returns TimeGuardedReader instances
from get_reader(), injecting the shared ITimeProvider.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd

from qubx.core.basics import DataType, ITimeProvider
from qubx.data.storage import IReader, IStorage, Transformable


class TimeGuardedReader(IReader):
    """
    Wraps an IReader and clamps the `stop` parameter to the current simulation time.

    All data types are clamped the same way — stop is set to current sim time.
    This ensures no data from the future is visible, regardless of data type.

    This operates at the query level (before data is fetched), which is much more
    efficient than post-fetch truncation.
    """

    _reader: IReader
    _time_provider: ITimeProvider

    def __init__(self, reader: IReader, time_provider: ITimeProvider) -> None:
        self._reader = reader
        self._time_provider = time_provider

    def _clamp_stop(self, stop: str | None) -> str | None:
        """
        Clamp stop to the current simulation time.

        If the caller already provided a stop that is earlier than simulation
        time, the caller's stop is preserved (we never widen the range).
        """
        current_time = self._time_provider.time()
        if current_time is None:
            return stop

        # - convert to Timestamp for comparison
        guard_time = pd.Timestamp(current_time)

        # - if caller provided stop, use the earlier of the two
        if stop is not None:
            caller_stop = pd.Timestamp(stop)
            if caller_stop < guard_time:
                return stop

        return str(guard_time)

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None = None,
        stop: str | None = None,
        chunksize: int = 0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        clamped_stop = self._clamp_stop(stop)
        # - if start is beyond clamped stop, entire range is in the future — force empty result
        #   (without this, handle_start_stop would swap start/stop and return past data)
        if clamped_stop is not None and start is not None:
            if pd.Timestamp(start) >= pd.Timestamp(clamped_stop):
                start = clamped_stop
        return self._reader.read(data_id, dtype, start=start, stop=clamped_stop, chunksize=chunksize, **kwargs)

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        return self._reader.get_data_id(dtype)

    def get_data_types(self, data_id: str) -> list[DataType]:
        return self._reader.get_data_types(data_id)

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[np.datetime64, np.datetime64]:
        return self._reader.get_time_range(data_id, dtype)

    def close(self) -> None:
        self._reader.close()

    def __repr__(self) -> str:
        return f"TimeGuardedReader({self._reader!r})"


class TimeGuardedStorage(IStorage):
    """
    Wraps an IStorage and returns TimeGuardedReader instances from get_reader().

    All readers produced by this storage will have their `stop` parameter clamped
    to the current simulation time, preventing look-ahead bias.

    Readers are cached per (exchange, market) key so that repeated get_reader()
    calls return the same TimeGuardedReader instance.
    """

    _storage: IStorage
    _time_provider: ITimeProvider
    _readers: dict[str, TimeGuardedReader]

    def __init__(self, storage: IStorage, time_provider: ITimeProvider) -> None:
        self._storage = storage
        self._time_provider = time_provider
        self._readers = {}

    def get_exchanges(self) -> list[str]:
        return self._storage.get_exchanges()

    def get_market_types(self, exchange: str) -> list[str]:
        return self._storage.get_market_types(exchange)

    def get_reader(self, exchange: str, market: str) -> IReader:
        _key = f"{exchange}:{market}"
        if _key not in self._readers:
            inner_reader = self._storage.get_reader(exchange, market)
            self._readers[_key] = TimeGuardedReader(inner_reader, self._time_provider)
        return self._readers[_key]

    def __repr__(self) -> str:
        return f"TimeGuardedStorage({self._storage!r})"
