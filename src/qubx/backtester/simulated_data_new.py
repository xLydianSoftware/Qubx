from collections import deque
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa

from qubx import logger
from qubx.backtester.sentinels import NoDataContinue
from qubx.backtester.simulated_data import (
    EmulatedBarSequence,
    EmulatedTickSequence,
    EmulatedUpdatesFromOHLC,
)
from qubx.core.basics import DataType, Instrument, MarketType, Timestamped
from qubx.core.exceptions import SimulationError
from qubx.data.composite import IteratedDataStreamsSlicer
from qubx.data.containers import RawData, RawMultiData
from qubx.data.storage import IDataTransformer, IReader, Transformable
from qubx.data.transformers import TypedRecords


# ---------------------------------------------------------------------------
# RawSymbolBuffer — per-symbol raw data storage with dedup
# ---------------------------------------------------------------------------
class RawSymbolBuffer:
    """
    Per-symbol buffer holding raw RecordBatch chunks (pre-transformation).

    Stores RawData objects in a FIFO queue. On append, deduplicates against
    already-stored data using a watermark (latest timestamp seen).
    This allows safe restart of the backend reader without losing or
    duplicating data for continuing symbols.
    """

    __slots__ = ("_batches", "_watermark", "_active", "_symbol")

    def __init__(self, symbol: str):
        self._symbol = symbol
        self._batches: deque[RawData] = deque()
        self._watermark: int = 0  # - latest timestamp (ns) stored in buffer
        self._active: bool = True

    @property
    def has_data(self) -> bool:
        return len(self._batches) > 0

    @property
    def watermark(self) -> int:
        return self._watermark

    @property
    def active(self) -> bool:
        return self._active

    @staticmethod
    def _to_ns(value) -> int:
        """
        Normalize timestamp value to integer nanoseconds.
        Handles pd.Timestamp, np.datetime64, datetime, and int.
        """
        if isinstance(value, int):
            return value
        if isinstance(value, pd.Timestamp):
            return value.value
        if isinstance(value, np.datetime64):
            return pd.Timestamp(value).value
        # - fallback: try to convert via pd.Timestamp
        return pd.Timestamp(value).value

    def activate(self):
        self._active = True

    def deactivate(self):
        """
        Mark buffer as inactive. Pump will skip this symbol when distributing chunks.
        """
        self._active = False

    def clear(self):
        """
        Clear all buffered data and reset watermark.
        """
        self._batches.clear()
        self._watermark = 0

    def append(self, raw: RawData):
        """
        Append raw data chunk, filtering out rows with timestamp <= watermark.

        Uses pyarrow vectorized filter for dedup — no Python loops.
        """
        if not self._active:
            return

        batch = raw.data
        if batch.num_rows == 0:
            return

        time_col = batch.column(raw.index)

        # - dedup: filter out rows already in buffer using native pyarrow comparison
        if self._watermark > 0:
            if pa.types.is_timestamp(time_col.type):
                wm_scalar = pa.scalar(pd.Timestamp(self._watermark, unit="ns"), type=time_col.type)
            else:
                wm_scalar = pa.scalar(self._watermark, type=time_col.type)
            mask = pa.compute.greater(time_col, wm_scalar)
            batch = batch.filter(mask)
            if batch.num_rows == 0:
                return
            # - rebuild RawData with filtered batch
            raw = RawData.from_record_batch(raw.data_id, raw.dtype, batch)
            time_col = batch.column(raw.index)

        # - update watermark from last row (normalize to int nanoseconds)
        self._watermark = self._to_ns(time_col[-1].as_py())
        self._batches.append(raw)

    def pop(self) -> RawData:
        """
        Pop the oldest raw chunk from the buffer.

        Raises:
            IndexError: if buffer is empty
        """
        return self._batches.popleft()

    def __len__(self) -> int:
        return len(self._batches)

    def __repr__(self) -> str:
        _wm = pd.Timestamp(self._watermark, unit="ns") if self._watermark else "N/A"
        return f"RawSymbolBuffer({self._symbol}, chunks={len(self)}, wm={_wm}, active={self._active})"


# ---------------------------------------------------------------------------
# MemReader — per-symbol iterator for IteratedDataStreamsSlicer
# ---------------------------------------------------------------------------
class MemReader(Iterator):
    """
    Per-symbol lazy iterator that reads from RawSymbolBuffer and transforms on-demand.

    When the slicer pulls data from this iterator:
      1. Check if symbol buffer has raw data
      2. If empty → ask pump to advance backend reader (fills ALL symbol buffers)
      3. Pop raw chunk → transform via IDataTransformer → return list[Timestamped]

    This is what gets registered in IteratedDataStreamsSlicer as a per-symbol stream.
    """

    __slots__ = ("_symbol", "_buffer", "_pump", "_transformer")

    def __init__(
        self,
        symbol: str,
        buffer: RawSymbolBuffer,
        pump: "DataPumpV2",
        transformer: IDataTransformer,
    ):
        self._symbol = symbol
        self._buffer = buffer
        self._pump = pump
        self._transformer = transformer

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> list[Timestamped]:
        # - pull from pump until our buffer has data
        while not self._buffer.has_data:
            if not self._buffer.active:
                raise StopIteration
            # - advance pump → reads next chunk for ALL symbols
            self._pump.advance()

        raw = self._buffer.pop()
        return self._transformer.process_data(raw)

    def __repr__(self) -> str:
        return f"MemReader({self._symbol}, buffer={self._buffer})"


# ---------------------------------------------------------------------------
# DataPumpV2 — orchestrator for batched IReader reads
# ---------------------------------------------------------------------------
class DataPumpV2:
    """
    Orchestrates batched reads from IReader, distributes raw chunks to per-symbol buffers.

    Manages a single backend reader iterator at a time. On universe change,
    restarts the reader with the new symbol set. Dedup in RawSymbolBuffer
    handles overlap for continuing symbols.

    Handles warmup: newly added symbols get extended start time on first read.

    Usage:
        pump = DataPumpV2(reader, "ohlc(1h)", warmup_period=pd.Timedelta("30d"))

        # - add initial symbols
        pump.attach_instrument(btc_instrument)
        readers = pump.start_read(start, end)  # returns dict for slicer.put()

        # - rebalance: add new symbols
        pump.attach_instrument(eth_instrument)
        pump.attach_instrument(sol_instrument)
        new_readers = pump.restart_read(current_time, end)  # returns dict for slicer.put()

        # - remove symbols
        keys = pump.remove_instrument(xrp_instrument)  # returns keys for slicer.remove()
    """

    _reader: IReader
    _transformer: IDataTransformer
    _requested_data_type: str
    _producing_data_type: str
    _warmup_period: pd.Timedelta | None
    _chunksize: int

    # - per-symbol state
    _buffers: dict[str, RawSymbolBuffer]
    _mem_readers: dict[str, MemReader]
    _instruments: dict[str, Instrument]  # - symbol -> Instrument
    _warmed: set[str]  # - symbols that already received warmup data

    # - backend reader state
    _b_iter: (
        Iterator[Transformable] | Transformable | None
    )  # Transformable just for avoid type checker. reader always returns Iterator
    _end: pd.Timestamp | None

    def __init__(
        self,
        reader: IReader,
        subscription_type: str,
        warmup_period: pd.Timedelta | None = None,
        chunksize: int = 5000,
        open_close_time_indent_secs: float = 1.0,
        trading_session: str | tuple[int, int] = EmulatedUpdatesFromOHLC.DEFAULT_DAILY_SESSION,
    ) -> None:
        self._reader = reader
        self._warmup_period = warmup_period
        self._chunksize = chunksize

        self._buffers = {}
        self._mem_readers = {}
        self._instruments = {}
        self._warmed = set()
        self._b_iter = None
        self._end = None

        # - determine transformer and data types based on subscription
        match subscription_type:
            case DataType.OHLC:
                _, _p = DataType.from_str(subscription_type)
                if "timeframe" not in _p or not (_tf := _p.get("timeframe")):
                    raise SimulationError(
                        f"OHLC subscription must contain timeframe parameter ! Received {subscription_type}"
                    )
                self._transformer = EmulatedBarSequence(
                    daily_session_start_end=trading_session,
                    open_close_time_shift_secs=open_close_time_indent_secs,
                )
                self._requested_data_type = f"ohlc({_tf.lower()})"
                self._producing_data_type = "ohlc"

            case DataType.OHLC_QUOTES:
                _, _p = DataType.from_str(subscription_type)
                if "timeframe" not in _p or not (_tf := _p.get("timeframe")):
                    raise SimulationError(
                        f"OHLC_QUOTES subscription must contain timeframe parameter ! Received {subscription_type}"
                    )
                self._transformer = EmulatedTickSequence(
                    quotes=True,
                    trades=False,
                    daily_session_start_end=trading_session,
                    open_close_time_shift_secs=open_close_time_indent_secs,
                )
                self._requested_data_type = f"ohlc({_tf.lower()})"
                self._producing_data_type = "quote"

            case DataType.OHLC_TRADES:
                _, _p = DataType.from_str(subscription_type)
                if "timeframe" not in _p or not (_tf := _p.get("timeframe")):
                    raise SimulationError(
                        f"OHLC_TRADES subscription must contain timeframe parameter ! Received {subscription_type}"
                    )
                self._transformer = EmulatedTickSequence(
                    quotes=False,
                    trades=True,
                    daily_session_start_end=trading_session,
                    open_close_time_shift_secs=open_close_time_indent_secs,
                )
                self._requested_data_type = f"ohlc({_tf.lower()})"
                self._producing_data_type = "trade"

            case _:
                self._requested_data_type = subscription_type.lower()
                self._producing_data_type = subscription_type.lower()
                self._transformer = TypedRecords()

    @property
    def producing_data_type(self) -> str:
        return self._producing_data_type

    # -----------------------------------------------------------------------
    # Instrument management
    # -----------------------------------------------------------------------

    def has_instrument(self, instrument: Instrument) -> bool:
        return instrument.symbol in self._instruments

    def attach_instrument(self, instrument: Instrument):
        """
        Register instrument for data pumping.
        Creates fresh RawSymbolBuffer. Cleans stale MemReader if any (from previous remove).
        Duplicate attach on already-active symbol is a no-op.
        Does NOT trigger a read — call start_read() or restart_read() after.
        """
        sym = instrument.symbol
        if sym in self._instruments:
            return  # - already attached, no-op

        self._instruments[sym] = instrument
        self._buffers[sym] = RawSymbolBuffer(sym)
        # - clean stale MemReader from previous remove (if cleanup_inactive wasn't called)
        self._mem_readers.pop(sym, None)

    def remove_instrument(self, instrument: Instrument) -> str | None:
        """
        Deactivate instrument. Buffer stops accepting data, MemReader will StopIteration.

        Returns:
            Slicer key for this instrument (for slicer.remove()) or None if not found.
        """
        sym = instrument.symbol
        if sym not in self._instruments:
            return None

        # - deactivate buffer (pump stops filling it)
        self._buffers[sym].deactivate()

        # - forget warmup so re-add gets fresh warmup
        self._warmed.discard(sym)

        # - remove from active instruments
        del self._instruments[sym]

        # - return slicer key
        return self._make_slicer_key(sym)

    def get_instruments(self) -> list[Instrument]:
        return list(self._instruments.values())

    def _make_slicer_key(self, symbol: str) -> str:
        return f"{self._requested_data_type}.{symbol}"

    def _active_symbols(self) -> list[str]:
        return [s for s, buf in self._buffers.items() if buf.active]

    # -----------------------------------------------------------------------
    # Backend reader management
    # -----------------------------------------------------------------------

    def start_read(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> dict[str, MemReader]:
        """
        Initial read for all attached instruments.

        Starts backend reader, creates MemReaders for all symbols.
        Returns dict suitable for slicer.put().
        """
        self._end = pd.Timestamp(end)
        symbols = self._active_symbols()
        if not symbols:
            return {}

        # - all symbols need warmup on first read
        read_start = pd.Timestamp(start)
        if self._warmup_period:
            read_start = read_start - self._warmup_period
            self._warmed.update(symbols)

        self._b_iter = self._reader.read(
            symbols, self._requested_data_type, str(read_start), str(end), chunksize=self._chunksize
        )

        # - create MemReaders for all symbols
        result = {}
        for sym in symbols:
            mem_reader = MemReader(sym, self._buffers[sym], self, self._transformer)
            self._mem_readers[sym] = mem_reader
            result[self._make_slicer_key(sym)] = mem_reader

        return result

    def restart_read(
        self, current_time: str | pd.Timestamp, end: str | pd.Timestamp | None = None
    ) -> dict[str, MemReader]:
        """
        Restart backend reader with current symbol set.

        Called after universe change (attach/remove). Only creates NEW MemReaders
        for symbols that don't have one yet. Continuing symbols keep their existing
        MemReaders — dedup in RawSymbolBuffer handles overlap.

        Args:
            current_time: current simulation time (start of new read)
            end: end time (if None, uses the end from start_read)

        Returns:
            dict of NEW MemReaders only (for slicer.put())
        """
        if end is not None:
            self._end = pd.Timestamp(end)

        symbols = self._active_symbols()
        if not symbols:
            self._b_iter = None
            return {}

        # - determine start time: extend into past if new symbols need warmup
        new_symbols = [s for s in symbols if s not in self._warmed]
        read_start = pd.Timestamp(current_time)

        if new_symbols and self._warmup_period:
            read_start = read_start - self._warmup_period
            self._warmed.update(new_symbols)

        # - restart backend reader with full current symbol set
        logger.debug(
            f"[<c>DataPumpV2</c>] :: Restarting read for {len(symbols)} symbols "
            f"({len(new_symbols)} new) from {read_start}"
        )
        self._b_iter = self._reader.read(
            symbols, self._requested_data_type, str(read_start), str(self._end), chunksize=self._chunksize
        )

        # - create MemReaders only for NEW symbols (continuing symbols already have them)
        result = {}
        for sym in new_symbols:
            if sym not in self._mem_readers:
                mem_reader = MemReader(sym, self._buffers[sym], self, self._transformer)
                self._mem_readers[sym] = mem_reader
                result[self._make_slicer_key(sym)] = mem_reader

        return result

    def advance(self):
        """
        Pull next chunk from backend reader, distribute raw data to symbol buffers.

        Called by MemReader when its buffer is empty.

        Raises:
            StopIteration: when backend reader is exhausted
        """
        if self._b_iter is None:
            raise StopIteration

        try:
            chunk = next(self._b_iter)  # type: ignore
        except StopIteration:
            self._b_iter = None
            raise

        # - chunk can be RawMultiData (multi-symbol) or RawData (single-symbol)
        if isinstance(chunk, RawMultiData):
            for raw in chunk:
                sym = raw.data_id
                if sym in self._buffers:
                    self._buffers[sym].append(raw)
        elif isinstance(chunk, RawData):
            sym = chunk.data_id
            if sym in self._buffers:
                self._buffers[sym].append(chunk)
        else:
            logger.warning(f"[DataPumpV2] :: Unexpected chunk type: {type(chunk)}")

    def cleanup_inactive(self):
        """
        Remove fully inactive symbols (buffer deactivated + MemReader removed from slicer).
        Call periodically to free memory.
        """
        to_remove = [sym for sym, buf in self._buffers.items() if not buf.active and sym not in self._instruments]
        for sym in to_remove:
            self._buffers.pop(sym, None)
            self._mem_readers.pop(sym, None)
            logger.debug(f"[DataPumpV2] :: Cleaned up inactive symbol buffer: {sym}")

    def __repr__(self) -> str:
        active = self._active_symbols()
        warmed = len(self._warmed)
        return (
            f"DataPumpV2({self._requested_data_type} -> {self._producing_data_type}, "
            f"symbols={len(active)}, warmed={warmed}, chunksize={self._chunksize})"
        )


# ---------------------------------------------------------------------------
# IterableSimulationDataV2 — replacement for IterableSimulationData
# ---------------------------------------------------------------------------
class IterableSimulationDataV2(Iterator):
    """
    Simulation data source using IReader/IStorage with shared memory buffers.

    Replaces IterableSimulationData. Uses DataPumpV2 + RawSymbolBuffer + MemReader
    architecture for efficient batched reads with dynamic universe support.

    Old approach: DataFetcher issues one SQL query per symbol per read → O(N) queries.
    New approach: DataPumpV2 issues one batched IReader.read(all_symbols) → O(1) query.

    Data flow:
        IReader.read([symbols], dtype, start, end)
            → RawMultiData chunk (all symbols in one batch)
            → distribute by data_id to per-symbol RawSymbolBuffer (raw RecordBatch, pre-transform)
            → MemReader pops buffer on demand, transforms lazily → list[Timestamped]
            → IteratedDataStreamsSlicer merges all streams by timestamp

        DataPumpV2 (one per subscription, e.g. "ohlc.1h")
            ├── RawSymbolBuffer("BTC")  →  MemReader("BTC")  ─┐
            ├── RawSymbolBuffer("ETH")  →  MemReader("ETH")  ─┤  → Slicer → yield events
            └── RawSymbolBuffer("SOL")  →  MemReader("SOL")  ─┘

    Universe changes mid-sim:
        - Subscribe: pump.restart_read() with full symbol set, new MemReaders added to slicer
        - Unsubscribe: buffer deactivated → MemReader StopIteration → slicer removes stream
        - Dedup: RawSymbolBuffer watermark filters overlapping rows on restart (pyarrow vectorized)
        - Warmup: new symbols get extended start time, already-warmed symbols dedup via watermark

    Key features:
        - Batched multi-symbol reads (one SQL per chunk regardless of symbol count)
        - Dynamic subscribe/unsubscribe mid-simulation
        - Warmup support for newly added symbols
        - Dedup via pyarrow watermark filter (no data loss/duplication on restarts)
        - Lazy transformation (raw → Timestamped only when slicer pulls)
    """

    _readers: dict[str, IReader]
    _pumps: dict[str, DataPumpV2]  # - keyed by subscription access key (e.g. "ohlc.1h")
    _instruments: dict[str, tuple[Instrument, DataPumpV2, str]]  # - slicer_key -> (instrument, pump, subscription)
    _warmups: dict[str, pd.Timedelta]

    _slicer: IteratedDataStreamsSlicer | None
    _slicing_iterator: Iterator | None
    _current_time: int | None
    _start: pd.Timestamp | None
    _stop: pd.Timestamp | None
    _open_close_time_indent_secs: float

    def __init__(
        self,
        readers: dict[str, IReader],
        open_close_time_indent_secs: float = 1.0,
        trading_session: str | tuple[int, int] = EmulatedUpdatesFromOHLC.DEFAULT_DAILY_SESSION,
        chunksize: int = 5000,
    ):
        self._readers = dict(readers)
        self._pumps = {}
        self._instruments = {}
        self._warmups = {}
        self._open_close_time_indent_secs = open_close_time_indent_secs
        self._trading_session = trading_session
        self._chunksize = chunksize

        self._slicer = None
        self._slicing_iterator = None
        self._current_time = None
        self._start = None
        self._stop = None

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    def set_typed_reader(self, data_type: str, reader: IReader):
        self._readers[data_type] = reader

    def set_warmup_period(self, subscription: str, warmup_period: str | None = None):
        if warmup_period:
            access_key, _, _ = self._parse_subscription_spec(subscription)
            self._warmups[access_key] = pd.Timedelta(warmup_period)

    def _parse_subscription_spec(self, subscription: str) -> tuple[str, str, dict[str, object]]:
        _subtype, _params = DataType.from_str(subscription)
        match _subtype:
            case DataType.OHLC | DataType.OHLC_QUOTES | DataType.OHLC_TRADES:
                _timeframe = _params.get("timeframe", "1Min")
                access_key = f"{_subtype}.{_timeframe}"
            case DataType.TRADE | DataType.QUOTE | DataType.ORDERBOOK:
                access_key = f"{_subtype}"
            case _:
                _params = {}
                _subtype = subscription
                access_key = f"{_subtype}"
        return access_key, _subtype, _params

    # -----------------------------------------------------------------------
    # Instrument filtering (e.g. funding payments only for SWAP)
    # -----------------------------------------------------------------------

    def _filter_instruments_for_subscription(self, data_type: str, instruments: list[Instrument]) -> list[Instrument]:
        if data_type == DataType.FUNDING_PAYMENT:
            filtered = [i for i in instruments if i.market_type == MarketType.SWAP]
            if len(filtered) < len(instruments):
                logger.debug(
                    f"Filtered {len(instruments) - len(filtered)} non-SWAP instruments from funding payment subscription"
                )
            return filtered
        return instruments

    # -----------------------------------------------------------------------
    # Subscribe / Unsubscribe
    # -----------------------------------------------------------------------

    def _get_or_create_pump(self, access_key: str, subscription: str, data_type: str) -> DataPumpV2:
        """
        Get existing pump or create new one for given subscription type.

        Args:
            access_key: unique key for this subscription (e.g. "ohlc.1h")
            subscription: full subscription string (e.g. "ohlc(1h)") — passed to DataPumpV2
            data_type: base data type for reader lookup (e.g. "ohlc")
        """
        if access_key in self._pumps:
            return self._pumps[access_key]

        reader = self._readers.get(data_type)
        if reader is None:
            raise SimulationError(f"No reader configured for data type: {data_type}")

        pump = DataPumpV2(
            reader=reader,
            subscription_type=subscription,
            warmup_period=self._warmups.get(access_key),
            chunksize=self._chunksize,
            open_close_time_indent_secs=self._open_close_time_indent_secs,
            trading_session=self._trading_session,
        )
        self._pumps[access_key] = pump
        return pump

    def add_instruments_for_subscription(self, subscription: str, instruments: list[Instrument] | Instrument):
        instruments = instruments if isinstance(instruments, list) else [instruments]
        access_key, data_type, _params = self._parse_subscription_spec(subscription)

        # - filter instruments based on subscription type
        instruments = self._filter_instruments_for_subscription(data_type, instruments)
        if not instruments:
            return

        pump = self._get_or_create_pump(access_key, subscription, data_type)

        new_instruments = []
        for i in instruments:
            if not pump.has_instrument(i):
                pump.attach_instrument(i)
                slicer_key = pump._make_slicer_key(i.symbol)
                self._instruments[slicer_key] = (i, pump, subscription)
                new_instruments.append(i)

        # - if simulation is running, restart read to include new symbols
        if self.is_running and new_instruments:
            new_mem_readers = pump.restart_read(
                pd.Timestamp(self._current_time, unit="ns"),
                self._stop,
            )
            if new_mem_readers and self._slicer is not None:
                self._slicer.put(new_mem_readers)

    def remove_instruments_from_subscription(self, subscription: str, instruments: list[Instrument] | Instrument):
        instruments = instruments if isinstance(instruments, list) else [instruments]

        def _remove_from_pump(access_key: str, instruments: list[Instrument]):
            pump = self._pumps.get(access_key)
            if not pump:
                logger.warning(f"No configured data pump for '{access_key}' subscription !")
                return

            keys_to_remove = []
            for i in instruments:
                slicer_key = pump.remove_instrument(i)
                if slicer_key:
                    self._instruments.pop(slicer_key, None)
                    keys_to_remove.append(slicer_key)

            if self.is_running and keys_to_remove and self._slicer is not None:
                self._slicer.remove(keys_to_remove)

            # - cleanup inactive buffers
            pump.cleanup_inactive()

        if subscription == DataType.ALL:
            for key in list(self._pumps.keys()):
                _remove_from_pump(key, instruments)
            return

        access_key, _, _ = self._parse_subscription_spec(subscription)
        _remove_from_pump(access_key, instruments)

    # -----------------------------------------------------------------------
    # Query methods
    # -----------------------------------------------------------------------

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        for i, p, s in self._instruments.values():
            if i == instrument and s == subscription_type:
                return True
        return False

    def get_instruments_for_subscription(self, subscription: str) -> list[Instrument]:
        if subscription == DataType.ALL:
            return list(i for i, _, _ in self._instruments.values())

        access_key, _, _ = self._parse_subscription_spec(subscription)
        pump = self._pumps.get(access_key)
        if pump is not None:
            return pump.get_instruments()
        return []

    def get_subscriptions_for_instrument(self, instrument: Instrument | None) -> list[str]:
        result = []
        for i, p, s in self._instruments.values():
            if instrument is None or i == instrument:
                result.append(s)
        return list(set(result))

    def peek_historical_data(self, instrument: Instrument, subscription: str) -> list[Timestamped]:
        """
        Retrieve historical data for instrument up to current simulation time.
        """
        if not self.has_subscription(instrument, subscription):
            raise SimulationError(
                f"Instrument: {instrument} has no subscription: {subscription} in this simulation data provider"
            )

        if not self.is_running or self._slicer is None or self._current_time is None:
            return []

        access_key, _, _ = self._parse_subscription_spec(subscription)
        pump = self._pumps.get(access_key)
        if pump is None:
            return []

        slicer_key = pump._make_slicer_key(instrument.symbol)
        return self._slicer.fetch_before_time(slicer_key, self._current_time)

    # -----------------------------------------------------------------------
    # Iteration
    # -----------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._current_time is not None

    def create_iterable(self, start: str | pd.Timestamp, stop: str | pd.Timestamp) -> Iterator:
        self._start = pd.Timestamp(start)
        self._stop = pd.Timestamp(stop)
        self._current_time = None
        self._slicer = IteratedDataStreamsSlicer()
        return self

    def __iter__(self) -> Iterator:
        assert self._start is not None and self._stop is not None
        assert self._slicer is not None
        self._current_time = int(pd.Timestamp(self._start).timestamp() * 1e9)

        # - initial read for all pumps
        for pump in self._pumps.values():
            mem_readers = pump.start_read(self._start, self._stop)
            if mem_readers:
                self._slicer.put(mem_readers)

        self._slicing_iterator = iter(self._slicer)
        return self

    def __next__(self) -> tuple[Instrument | None, str, Timestamped | NoDataContinue, bool]:
        assert self._slicing_iterator is not None
        try:
            while data := next(self._slicing_iterator):
                k, t, v = data

                # - handle NoDataContinue sentinel
                if isinstance(v, NoDataContinue):
                    return None, "", v, False

                instr, pump, subt = self._instruments[k]
                data_type = pump.producing_data_type
                is_historical = False
                if t < self._current_time:
                    is_historical = True
                else:
                    self._current_time = t

                return instr, data_type, v, is_historical

            # - while loop exited without return (empty data from slicer)
            raise StopIteration
        except StopIteration:
            raise StopIteration
