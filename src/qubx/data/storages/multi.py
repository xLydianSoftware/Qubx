from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.storage import IReader, IStorage


class MultiReader(IReader):
    """
    An IReader that combines data from multiple IReader instances for the same
    exchange and market type.

    Reads from all underlying readers, merges results by sorting on the time
    column and deduplicating exact-timestamp matches.  When timestamps collide
    the last reader in the list wins (higher-priority readers go last).

    For chunked reading (chunksize > 0) all data is collected from each reader
    first, merged in memory, then re-yielded in chunks of the requested size.
    """

    def __init__(self, readers: list[IReader]):
        self._readers = readers

    # ------------------------------------------------------------------
    # IReader.read — public entry point
    # ------------------------------------------------------------------

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None = None,
        stop: str | None = None,
        chunksize: int = 0,
        **kwargs,
    ) -> "Iterator | Any":
        if isinstance(data_id, list):
            return self._read_multi(data_id, dtype, start, stop, chunksize, **kwargs)
        return self._read_single(data_id, dtype, start, stop, chunksize, **kwargs)

    # ------------------------------------------------------------------
    # single-symbol path
    # ------------------------------------------------------------------

    def _read_single(
        self,
        data_id: str,
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
        chunksize: int,
        **kwargs,
    ) -> "Any":
        from qubx.data.containers import RawData

        batches, time_idx, schema, dtype_ref = self._collect_single(data_id, dtype, start, stop, **kwargs)

        if not batches:
            raise ValueError(f"No data for '{data_id}' ({dtype}) in any of the {len(self._readers)} readers")

        merged = self._merge_batches(batches, time_idx)
        raw = RawData.from_record_batch(data_id, dtype_ref, merged)

        if chunksize == 0:
            return raw

        return self._chunk_iter(raw, chunksize)

    def _collect_single(
        self,
        data_id: str,
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
        **kwargs,
    ) -> "tuple[list[pa.RecordBatch], int, pa.Schema | None, DataType | str]":
        from qubx.data.containers import RawData

        batches: list[pa.RecordBatch] = []
        time_idx = 0
        schema: pa.Schema | None = None
        dtype_ref: DataType | str = dtype

        for reader in self._readers:
            try:
                result = reader.read(data_id, dtype, start, stop, chunksize=0, **kwargs)
                if result is None:
                    continue
                if not isinstance(result, RawData):
                    # - reader returned iterator despite chunksize=0; drain it
                    for chunk in result:
                        if isinstance(chunk, RawData) and len(chunk) > 0:
                            if schema is None:
                                schema = chunk.data.schema
                                time_idx = chunk.index
                                dtype_ref = chunk.dtype
                            batches.append(chunk.data)
                elif len(result) > 0:
                    if schema is None:
                        schema = result.data.schema
                        time_idx = result.index
                        dtype_ref = result.dtype
                    batches.append(result.data)
            except Exception as e:
                logger.warning(f"{reader.__class__.__name__}.read() failed for '{data_id}': {e}")

        return batches, time_idx, schema, dtype_ref

    # ------------------------------------------------------------------
    # multi-symbol path
    # ------------------------------------------------------------------

    def _read_multi(
        self,
        data_ids: list[str],
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
        chunksize: int,
        **kwargs,
    ) -> "Any":
        from qubx.data.containers import RawData, RawMultiData

        batches_per_id: dict[str, list[pa.RecordBatch]] = {}
        time_idx = 0
        schema: pa.Schema | None = None
        dtype_ref: DataType | str = dtype

        for reader in self._readers:
            try:
                result = reader.read(data_ids, dtype, start, stop, chunksize=0, **kwargs)
                if result is None:
                    continue
                if isinstance(result, RawData):
                    batches_per_id.setdefault(result.data_id, []).append(result.data)
                    if schema is None:
                        schema = result.data.schema
                        time_idx = result.index
                        dtype_ref = result.dtype
                elif isinstance(result, RawMultiData):
                    for raw in result:
                        batches_per_id.setdefault(raw.data_id, []).append(raw.data)
                        if schema is None:
                            schema = raw.data.schema
                            time_idx = raw.index
                            dtype_ref = raw.dtype
            except Exception as e:
                logger.warning(f"{reader.__class__.__name__}.read() failed for multi data_ids: {e}")

        if not batches_per_id:
            return RawMultiData([])

        merged_raws = [
            RawData.from_record_batch(did, dtype_ref, self._merge_batches(blist, time_idx))
            for did, blist in batches_per_id.items()
        ]
        multi = RawMultiData(merged_raws)

        if chunksize == 0:
            return multi

        # - TODO: proper streaming chunked multi-data iterator; for now single chunk
        return iter([multi])

    # ------------------------------------------------------------------
    # Arrow merge / chunk helpers
    # ------------------------------------------------------------------

    def _merge_batches(self, batches: list[pa.RecordBatch], time_idx: int) -> pa.RecordBatch:
        """
        \n
        Concatenate Arrow RecordBatches, sort ascending by the time column at
        time_idx, then deduplicate exact-timestamp rows keeping the last
        occurrence (so the last reader in self._readers wins on collision).
        \n
        """
        if len(batches) == 1:
            return batches[0]

        # - pa.Table.from_batches() concatenates multiple RecordBatches into one table
        table = pa.Table.from_batches(batches)
        if table.num_rows == 0:
            return batches[0]

        time_col_name = table.schema.names[time_idx]

        # - stable sort preserves reader order within equal timestamps
        table = table.sort_by(time_col_name)

        # - keep mask: True for the last occurrence of each unique timestamp
        times_np = table.column(time_col_name).to_numpy()
        n = len(times_np)
        keep = np.ones(n, dtype=bool)
        if n > 1:
            # - current row is NOT the last of its group when the next row has the same time
            keep[:-1] = times_np[:-1] != times_np[1:]

        deduped = table.take(pa.array(np.where(keep)[0]))

        # - combine_chunks() ensures a single RecordBatch after Arrow slicing
        out = deduped.combine_chunks().to_batches()
        if out:
            return out[0]
        return pa.RecordBatch.from_pydict({col: [] for col in table.schema.names}, schema=table.schema)

    def _chunk_iter(self, raw: Any, chunksize: int) -> "Iterator":
        """
        \n
        Yield slices of a RawData in chunks of chunksize rows.
        \n
        """
        from qubx.data.containers import RawData

        total = len(raw)
        for offset in range(0, total, chunksize):
            length = min(chunksize, total - offset)
            yield RawData.from_record_batch(raw.data_id, raw.dtype, raw.data.slice(offset, length))

    # ------------------------------------------------------------------
    # IReader metadata methods
    # ------------------------------------------------------------------

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        result: set[str] = set()
        for reader in self._readers:
            try:
                result.update(reader.get_data_id(dtype))
            except Exception:
                pass
        return sorted(result)

    def get_data_types(self, data_id: str) -> list[DataType]:
        result: set[DataType] = set()
        for reader in self._readers:
            try:
                result.update(reader.get_data_types(data_id))
            except Exception:
                pass
        return list(result)

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[np.datetime64 | None, np.datetime64 | None]:
        mins: list = []
        maxes: list = []
        for reader in self._readers:
            try:
                t0, t1 = reader.get_time_range(data_id, dtype)
                if t0 is not None:
                    mins.append(t0)
                if t1 is not None:
                    maxes.append(t1)
            except Exception:
                pass
        return (min(mins) if mins else None, max(maxes) if maxes else None)

    def close(self) -> None:
        for reader in self._readers:
            reader.close()


class MultiStorage(IStorage):
    """
    An IStorage that combines multiple IStorage instances.

    get_reader(exchange, market) returns a MultiReader that merges data from
    every underlying storage supporting that exchange + market combination.
    Readers are cached so repeated calls for the same key return the same object.
    """

    def __init__(self, storages: list[IStorage]):
        self._storages = storages
        self._reader_cache: dict[tuple[str, str], IReader] = {}

    def get_exchanges(self) -> list[str]:
        result: set[str] = set()
        for s in self._storages:
            try:
                result.update(s.get_exchanges())
            except Exception:
                pass
        return sorted(result)

    def get_market_types(self, exchange: str) -> list[str]:
        result: set[str] = set()
        for s in self._storages:
            try:
                result.update(s.get_market_types(exchange))
            except Exception:
                pass
        return sorted(result)

    def get_reader(self, exchange: str, market: str) -> IReader:
        key = (exchange, market)
        if key not in self._reader_cache:
            readers: list[IReader] = []
            for s in self._storages:
                try:
                    readers.append(s.get_reader(exchange, market))
                except Exception:
                    pass
            if not readers:
                raise ValueError(
                    f"No reader available for {exchange}:{market} in any of the {len(self._storages)} storages"
                )
            # - single reader: return directly; multiple: wrap in MultiReader
            self._reader_cache[key] = MultiReader(readers) if len(readers) > 1 else readers[0]
        return self._reader_cache[key]
