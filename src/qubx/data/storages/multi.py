from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
from numba import njit

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.containers import RawData, RawMultiData
from qubx.data.registry import storage
from qubx.data.storage import IReader, IStorage
from qubx.utils.time import to_timedelta


@njit
def _tol_mask_nb(times_ns: np.ndarray, tolerance_ns: int, keep_last: bool) -> np.ndarray:
    """
    Numba-compiled greedy tolerance-grouping mask.

    Scans `times_ns` (sorted int64 nanosecond timestamps) left-to-right,
    extending the current group while successive elements fall within `tolerance_ns` of the group's first element.
    Exactly one element per group is marked True in the returned boolean array - the last element of
    the group when ``keep_last=True``, otherwise the first.
    """
    n = len(times_ns)
    keep = np.zeros(n, dtype=np.bool_)
    i = 0
    while i < n:
        j = i + 1
        while j < n and times_ns[j] - times_ns[i] <= tolerance_ns:
            j += 1
        # - group spans [i, j); mark the chosen representative
        keep[j - 1 if keep_last else i] = True
        i = j
    return keep


class MultiReader(IReader):
    """
    An IReader that combines data from multiple IReader instances for the same exchange and market type.

    Reads from all underlying readers, merges results by sorting on the time column and deduplicating timestamp collisions
    according to the configured strategy.

    Deduplication modes (controlled by ``tolerance`` and ``keep``):

    * ``tolerance=None`` (default) — **exact** dedup: rows with identical
      timestamps are collapsed; the last reader in the list wins.
    * ``tolerance="<timedelta>"`` — **tolerance-based** dedup: rows whose
      timestamps fall within the given window are treated as the same event
      and collapsed to one representative (first or last, per ``keep``).
      Typical use-case: combining two funding-rate feeds where one source
      lags the other by ~30 s.

    For chunked reading (chunksize > 0) all data is collected from every
    reader first, merged in memory, then re-yielded in chunks of the
    requested size.
    """

    def __init__(
        self,
        readers: list[IReader],
        tolerance: str | None = None,
        keep: str = "last",
    ):
        """
        Args:
            readers:   underlying IReader instances to merge
            tolerance: optional pandas-style timedelta string (e.g. ``"1min"``, ``"30s"``) for near-duplicate grouping; ``None`` means exact dedup
            keep:      which row to retain per duplicate group – ``"last"`` (default, last reader wins) or ``"first"``
        """
        self._readers = readers
        self._tolerance = tolerance
        self._keep = keep

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None = None,
        stop: str | None = None,
        chunksize: int = 0,
        **kwargs,
    ) -> Iterator | Any:
        if isinstance(data_id, list):
            return self._read_multi(data_id, dtype, start, stop, chunksize, **kwargs)
        return self._read_single(data_id, dtype, start, stop, chunksize, **kwargs)

    def _read_single(
        self,
        data_id: str,
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
        chunksize: int,
        **kwargs,
    ) -> Any:
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
    ) -> tuple[list[pa.RecordBatch], int, pa.Schema | None, DataType | str]:
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

    def _read_multi(
        self,
        data_ids: list[str],
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
        chunksize: int,
        **kwargs,
    ) -> Any:
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

    def _normalize_batches(self, batches: list[pa.RecordBatch]) -> tuple[list[pa.RecordBatch], int]:
        """
        Project all batches to their common column intersection and unify timestamp precision
        to the finest unit seen across all sources (e.g. ``ms`` + ``us`` → ``us``).

        Returns ``(normalized_batches, time_idx)`` where ``time_idx`` is the index of
        the timestamp column in the output schema.

        This is needed when different storages produce different schemas for the same
        logical data type — e.g. CCXT yields ``timestamp[ms]`` with 7 columns while
        QuestDB yields ``timestamp[us]`` with 10 columns.
        """
        if len(batches) == 1:
            schema = batches[0].schema
            time_idx = next((i for i, f in enumerate(schema) if pa.types.is_timestamp(f.type)), 0)
            return batches, time_idx

        # - common columns (intersection), preserving order from the first batch
        common: set[str] = set(batches[0].schema.names)
        for b in batches[1:]:
            common &= set(b.schema.names)
        ordered = [n for n in batches[0].schema.names if n in common]

        if not ordered:
            raise ValueError("MultiReader: no common columns across schemas — cannot merge")

        # - find timestamp column and the finest precision across all batches
        # - ts_unit is initialised lazily from the first batch so that batches
        #   with coarser precision (e.g. "s") are not silently upcast to "ms"
        _precision_rank = {"s": 0, "ms": 1, "us": 2, "ns": 3}
        ts_name: str | None = None
        ts_unit: str | None = None
        for b in batches:
            for f in b.schema:
                if f.name in common and pa.types.is_timestamp(f.type):
                    if ts_name is None:
                        ts_name = f.name
                        ts_unit = f.type.unit
                    elif f.name == ts_name and _precision_rank.get(f.type.unit, 0) > _precision_rank.get(ts_unit or "s", 0):
                        ts_unit = f.type.unit
        # - fallback: if no timestamp column found, default to "ms"
        ts_unit = ts_unit or "ms"

        # - build target schema from first batch's common fields with unified timestamp type
        target_fields = []
        for name in ordered:
            f = batches[0].schema.field(name)
            if name == ts_name:
                f = pa.field(name, pa.timestamp(ts_unit))
            target_fields.append(f)
        target_schema = pa.schema(target_fields)

        # - project and cast each batch to target schema
        normalized: list[pa.RecordBatch] = []
        for b in batches:
            arrays = []
            for name in ordered:
                col = b.column(name)
                tgt_type = target_schema.field(name).type
                if col.type != tgt_type:
                    col = col.cast(tgt_type)
                arrays.append(col)
            normalized.append(pa.RecordBatch.from_arrays(arrays, schema=target_schema))

        time_idx = ordered.index(ts_name) if ts_name is not None else 0
        return normalized, time_idx

    def _merge_batches(self, batches: list[pa.RecordBatch], time_idx: int) -> pa.RecordBatch:
        """
        Concatenate Arrow RecordBatches, sort ascending by the time column at
        ``time_idx``, then deduplicate according to ``self._tolerance`` and
        ``self._keep``:

        * ``tolerance=None`` — exact dedup: the last reader's row wins on an
          identical timestamp.
        * ``tolerance=<str>`` — tolerance-based dedup: timestamps within the
          given window are grouped; ``keep`` controls which row survives.
        """
        if len(batches) == 1:
            return batches[0]

        # - unify schemas (column intersection + timestamp precision) before concatenation
        batches, time_idx = self._normalize_batches(batches)

        # - pa.Table.from_batches() concatenates multiple RecordBatches into one table
        table = pa.Table.from_batches(batches)
        if table.num_rows == 0:
            return batches[0]

        time_col_name = table.schema.names[time_idx]

        # - stable sort preserves reader order within equal timestamps
        table = table.sort_by(time_col_name)

        times_np = table.column(time_col_name).to_numpy()
        n = len(times_np)

        if self._tolerance is None:
            keep = self._exact_mask(times_np, n)
        else:
            keep = self._tolerance_mask(times_np, n)

        deduped = table.take(pa.array(np.where(keep)[0]))

        # - combine_chunks() ensures a single RecordBatch after Arrow slicing
        out = deduped.combine_chunks().to_batches()
        if out:
            return out[0]
        return pa.RecordBatch.from_pydict({col: [] for col in table.schema.names}, schema=table.schema)

    def _exact_mask(self, times_np: np.ndarray, n: int) -> np.ndarray:
        """
        Keep mask for exact-timestamp dedup.
        Marks the last occurrence of each unique timestamp as True.
        """
        keep = np.ones(n, dtype=bool)
        if n > 1:
            # - current row is NOT the last of its group when the next row has the same time
            keep[:-1] = times_np[:-1] != times_np[1:]
        return keep

    def _tolerance_mask(self, times_np: np.ndarray, n: int) -> np.ndarray:
        """
        Keep mask for tolerance-based dedup — thin wrapper around the
        Numba-compiled ``_tol_mask_nb`` kernel.

        Timestamps are normalised to contiguous int64 nanoseconds before being
        passed to the kernel so it works for both ``int64`` (ns-epoch) and
        any ``datetime64[*]`` Arrow column.
        """
        # - normalise to int64 nanoseconds; handles both int64 and datetime64[*]
        if np.issubdtype(times_np.dtype, np.datetime64):
            times_ns = np.ascontiguousarray(times_np.astype("datetime64[ns]").view(np.int64))
        else:
            times_ns = np.ascontiguousarray(times_np.astype(np.int64))

        tolerance_ns = to_timedelta(self._tolerance).value
        return _tol_mask_nb(times_ns, tolerance_ns, self._keep == "last")

    def _chunk_iter(self, raw: Any, chunksize: int) -> Iterator:
        """
        Yield slices of a RawData in chunks of chunksize rows.
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


@storage("multi")
@storage("multistorage")
class MultiStorage(IStorage):
    """
    An IStorage that combines multiple IStorage instances.

    get_reader(exchange, market) returns a MultiReader that merges data from
    every underlying storage supporting that exchange + market combination.
    Readers are cached so repeated calls for the same key return the same object.

    ``tolerance`` and ``keep`` are forwarded to every created MultiReader — see MultiReader for their semantics.
    """

    def __init__(
        self,
        storages: list[IStorage],
        tolerance: str | None = None,
        keep: str = "last",
    ):
        self._storages = storages
        self._tolerance = tolerance
        self._keep = keep
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
            self._reader_cache[key] = (
                MultiReader(readers, tolerance=self._tolerance, keep=self._keep) if len(readers) > 1 else readers[0]
            )
        return self._reader_cache[key]

    def close(self) -> None:
        for reader in self._reader_cache.values():
            reader.close()
        self._reader_cache.clear()
        for s in self._storages:
            try:
                s.close()
            except Exception:
                pass
