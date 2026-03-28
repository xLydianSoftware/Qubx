from collections import deque
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa

from qubx import logger
from qubx.backtester.iteratedstream import IteratedDataStreamsSlicer
from qubx.backtester.sentinels import NoDataContinue
from qubx.backtester.utils import DEFAULT_DAILY_SESSION
from qubx.core.basics import Bar, DataType, Instrument, MarketType, Quote, Timestamped, Trade
from qubx.core.exceptions import SimulationError
from qubx.core.series import time_as_nsec
from qubx.data.containers import IRawContainer, RawData, RawMultiData
from qubx.data.storage import IDataTransformer, IReader, IStorage, Transformable
from qubx.data.storages.utils import find_column_index_in_list
from qubx.data.transformers import TypedRecords
from qubx.utils.time import convert_times_to_ns, infer_series_frequency, to_timedelta, to_timestamp


class EmulatedUpdatesFromOHLC(IDataTransformer):
    """
    Generic class for help emulating partial updates from OHLC data
    """

    @staticmethod
    def timedelta_to_numpy(x: str) -> int:
        return to_timedelta(x).to_numpy().item()

    D1, H1 = timedelta_to_numpy("1D"), timedelta_to_numpy("1h")
    MS1 = 1_000_000
    S1 = 1000 * MS1
    M1 = 60 * S1

    def __init__(
        self,
        daily_session_start_end: tuple[int, int] = DEFAULT_DAILY_SESSION,
        timestamp_units="ns",
        open_close_time_shift_secs=1.0,
    ) -> None:
        self._d_session_start = daily_session_start_end[0]
        self._d_session_end = daily_session_start_end[1]
        self._timestamp_units = timestamp_units
        self.set_emulation_adjustment_time(open_close_time_shift_secs)

    def set_emulation_adjustment_time(self, open_close_time_shift_secs):
        self._open_close_time_shift_secs = open_close_time_shift_secs

    def _detect_emulation_timestamps(self, times: np.ndarray):
        try:
            self._freq = infer_series_frequency(times[:100])
        except ValueError:
            logger.warning("Can't determine frequency for incoming data")
            return

        # - timestamps when we emit simulated quotes
        dt = self._freq.astype("timedelta64[ns]").item()
        dt10 = dt // 10

        # - adjust open-close time shift to avoid overlapping timestamps
        if self._open_close_time_shift_secs * self.S1 >= (dt // 2 - dt10):
            self._open_close_time_shift_secs = (dt // 2 - 2 * dt10) // self.S1

        if dt < self.D1:
            self._t_start = self._open_close_time_shift_secs * self.S1
            self._t_mid1 = dt // 2 - dt10
            self._t_mid2 = dt // 2 + dt10
            self._t_end = dt - self._open_close_time_shift_secs * self.S1
        else:
            self._t_start = self._d_session_start + self._open_close_time_shift_secs * self.S1
            self._t_mid1 = dt // 2 - self.H1
            self._t_mid2 = dt // 2 + self.H1
            self._t_end = self._d_session_end - self._open_close_time_shift_secs * self.S1


class EmulatedTickSequence(EmulatedUpdatesFromOHLC):
    """
    Emulate ticks (Quotes or Trades) updates from OHLC raw data
    """

    def __init__(
        self,
        trades: bool = False,  # if we also wants 'trades'
        default_bid_size=1e9,  # default bid/ask is big
        default_ask_size=1e9,  # default bid/ask is big
        daily_session_start_end: tuple[int, int] = DEFAULT_DAILY_SESSION,
        timestamp_units="ns",
        spread=0.0,
        open_close_time_shift_secs=1.0,
        quotes=True,
    ) -> None:
        # - check trading sessions
        super().__init__(daily_session_start_end, timestamp_units, open_close_time_shift_secs)

        self._trades = trades
        self._quotes = quotes
        self._bid_size = default_bid_size
        self._ask_size = default_ask_size
        self._s2 = spread / 2.0

    def process_data(self, raw_data: IRawContainer) -> list[Timestamped]:
        _data = raw_data.data
        names = raw_data.names
        index = raw_data.index
        n_rows = _data.num_rows

        if n_rows < 2:
            raise ValueError("Input data must contain at least two records for ticks simulation !")

        try:
            _close_idx = find_column_index_in_list(names, "close")
            _open_idx = find_column_index_in_list(names, "open")
            _high_idx = find_column_index_in_list(names, "high")
            _low_idx = find_column_index_in_list(names, "low")
        except:
            raise ValueError(
                f"Incoming data must be presented as OHLC bars and contains open, high, low, close fields, passed '{names}' !"
            )

        # - extract columns as numpy arrays
        times = convert_times_to_ns(_data.column(index).to_numpy(zero_copy_only=False), self._timestamp_units)
        opens = _data.column(_open_idx).to_numpy(zero_copy_only=False)
        highs = _data.column(_high_idx).to_numpy(zero_copy_only=False)
        lows = _data.column(_low_idx).to_numpy(zero_copy_only=False)
        closes = _data.column(_close_idx).to_numpy(zero_copy_only=False)

        # - for trades we need volumes
        volumes = None
        if self._trades:
            _volume_idx = find_column_index_in_list(names, "vol", "volume")
            volumes = _data.column(_volume_idx).to_numpy(zero_copy_only=False)

        # - detect parameters for transformation
        self._detect_emulation_timestamps(times)
        s2 = self._s2

        buffer = []
        for i in range(n_rows):
            ti = times[i]
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            rv = volumes[i] if volumes is not None else 0
            rv = rv / (h - l) if h > l else rv

            # - opening quote
            if self._quotes:
                buffer.append(Quote(ti + self._t_start, o - s2, o + s2, self._bid_size, self._ask_size))

            if c >= o:
                if self._trades:
                    buffer.append(Trade(ti + self._t_start, o - s2, rv * (o - l)))  # sell 1

                if self._quotes:
                    buffer.append(Quote(ti + self._t_mid1, l - s2, l + s2, self._bid_size, self._ask_size))

                if self._trades:
                    buffer.append(Trade(ti + self._t_mid1, l + s2, rv * (c - o)))  # buy 1

                if self._quotes:
                    buffer.append(Quote(ti + self._t_mid2, h - s2, h + s2, self._bid_size, self._ask_size))

                if self._trades:
                    buffer.append(Trade(ti + self._t_mid2, h - s2, rv * (h - c)))  # sell 2
            else:
                if self._trades:
                    buffer.append(Trade(ti + self._t_start, o + s2, rv * (h - o)))  # buy 1

                if self._quotes:
                    buffer.append(Quote(ti + self._t_mid1, h - s2, h + s2, self._bid_size, self._ask_size))

                if self._trades:
                    buffer.append(Trade(ti + self._t_mid1, h - s2, rv * (o - c)))  # sell 1

                if self._quotes:
                    buffer.append(Quote(ti + self._t_mid2, l - s2, l + s2, self._bid_size, self._ask_size))

                if self._trades:
                    buffer.append(Trade(ti + self._t_mid2, l + s2, rv * (c - l)))  # buy 2

            # - closing quote
            if self._quotes:
                buffer.append(Quote(ti + self._t_end, c - s2, c + s2, self._bid_size, self._ask_size))
        return buffer


class EmulatedBarSequence(EmulatedUpdatesFromOHLC):
    """
    Emulate bar updates (Bar) from OHLC raw data.

    Transforms each OHLC record into 4 progressive Bar updates mimicking real-world
    market data arrival:
      1. Opening bar  (t_start): o,o,o,o  - bar just opened
      2. Mid1 bar     (t_mid1):  partial price movement
      3. Mid2 bar     (t_mid2):  further price movement
      4. Final bar    (t_end):   o,h,l,c with full volume data

    The direction of mid-bar updates depends on whether close >= open (bullish)
    or close < open (bearish), replicating the logic from RestoredBarsFromOHLC.
    """

    def __init__(
        self,
        daily_session_start_end: tuple[int, int] = DEFAULT_DAILY_SESSION,
        timestamp_units="ns",
        open_close_time_shift_secs=1.0,
    ) -> None:
        super().__init__(daily_session_start_end, timestamp_units, open_close_time_shift_secs)

    def process_data(self, raw_data: IRawContainer) -> list[Bar]:
        _data = raw_data.data
        names = raw_data.names
        index = raw_data.index
        n_rows = _data.num_rows

        if n_rows < 1:
            return []

        try:
            _open_idx = find_column_index_in_list(names, "open")
            _high_idx = find_column_index_in_list(names, "high")
            _low_idx = find_column_index_in_list(names, "low")
            _close_idx = find_column_index_in_list(names, "close")
        except Exception:
            raise ValueError(
                f"Incoming data must be presented as OHLC bars and contains open, high, low, close fields, passed '{names}' !"
            )

        # - extract OHLC columns as numpy arrays
        _to_np = lambda col_idx: _data.column(col_idx).to_numpy(zero_copy_only=False)
        times = convert_times_to_ns(_to_np(index), self._timestamp_units)
        opens = _to_np(_open_idx)
        highs = _to_np(_high_idx)
        lows = _to_np(_low_idx)
        closes = _to_np(_close_idx)

        # - optional volume columns: extract and fill NaN/None with 0 once (avoid per-row checks)
        _lnames = [n.lower() for n in names]

        def _safe_col(*col_names: str, dtype=np.float64) -> np.ndarray:
            for cn in col_names:
                if cn.lower() in _lnames:
                    arr = _to_np(_lnames.index(cn.lower()))
                    if arr.dtype == object:
                        # - object array may contain None values
                        return np.array([v if v is not None else 0 for v in arr], dtype=dtype)
                    return np.nan_to_num(arr, nan=0).astype(dtype, copy=False)
            return np.zeros(n_rows, dtype=dtype)

        vols = _safe_col("volume", "vol")
        bvols = _safe_col("bought_volume", "taker_buy_volume", "taker_bought_volume")
        volqs = _safe_col("volume_quote", "quote_volume")
        bvolqs = _safe_col("bought_volume_quote", "taker_buy_quote_volume", "taker_bought_quote_volume")
        tcounts = _safe_col("trade_count", "count", dtype=np.int64)

        # - detect emulation timestamp offsets
        self._detect_emulation_timestamps(times)

        # - pre-allocate buffer (4 bars per row)
        buffer: list[Bar] = [None] * (n_rows * 4)  # type: ignore[list-item]
        t_start, t_mid1, t_mid2, t_end = self._t_start, self._t_mid1, self._t_mid2, self._t_end
        pos = 0

        # fmt: off
        for i in range(n_rows):
            ti = times[i]
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]

            # - opening bar (o,o,o,o, v=0)
            buffer[pos] = Bar(ti + t_start, o, o, o, o, 0)

            if c >= o:
                # - bullish: open -> low -> high -> close
                buffer[pos + 1] = Bar(ti + t_mid1, o, o, l, l, 0)
                buffer[pos + 2] = Bar(ti + t_mid2, o, h, l, h, 0)
            else:
                # - bearish: open -> high -> low -> close
                buffer[pos + 1] = Bar(ti + t_mid1, o, h, o, h, 0)
                buffer[pos + 2] = Bar(ti + t_mid2, o, h, l, l, 0)

            # - final bar with full OHLCV data
            buffer[pos + 3] = Bar(ti + t_end, o, h, l, c, vols[i], bvols[i], volqs[i], bvolqs[i], tcounts[i])
            pos += 4
        # fmt: on

        return buffer[:pos]


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
            return to_timestamp(value).value
        # - fallback: try to convert via pd.Timestamp
        return to_timestamp(value).value

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
                wm_scalar = pa.scalar(to_timestamp(self._watermark, unit="ns"), type=time_col.type)
            elif pa.types.is_date(time_col.type):
                # - date32 stores days since epoch; watermark is ns, convert to datetime.date
                wm_scalar = pa.scalar(to_timestamp(self._watermark, unit="ns").date(), type=time_col.type)
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
        _wm = to_timestamp(self._watermark, unit="ns") if self._watermark else "N/A"
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
        pump: "DataPump",
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
# DataPump — orchestrator for batched IReader reads
# ---------------------------------------------------------------------------
class DataPump:
    """
    Orchestrates batched reads from IReader, distributes raw chunks to per-symbol buffers.

    Scoped to a single (subscription, exchange, market_type). Manages a single backend
    reader iterator at a time. On universe change, restarts the reader with the new
    symbol set. Dedup in RawSymbolBuffer handles overlap for continuing symbols.

    Slicer keys include exchange:market_type to avoid cross-exchange symbol collisions.

    Usage:
        pump = DataPump(reader, "ohlc(1h)", "BINANCE.UM", "SWAP",
                        warmup_period=pd.Timedelta("30d"))

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

    # - backend reader state | Transformable just for avoid type checker. reader always returns Iterator
    _b_iter: Iterator[Transformable] | Transformable | None
    _end: pd.Timestamp | None

    def __init__(
        self,
        reader: IReader,
        subscription_type: str,
        exchange: str,
        market_type: str,
        warmup_period: pd.Timedelta | None = None,
        chunksize: int = 5000,
        open_close_time_indent_secs: float = 1.0,
        trading_session: tuple[int, int] = DEFAULT_DAILY_SESSION,
    ) -> None:
        self._reader = reader
        self._exchange = exchange
        self._market_type = market_type
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

    def update_emulation_time_indent_seconds(self, time_indent_seconds: float):
        if isinstance(self._transformer, EmulatedUpdatesFromOHLC):
            self._transformer.set_emulation_adjustment_time(time_indent_seconds)

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
        return f"{self._producing_data_type}.{self._requested_data_type}.{self._exchange}:{self._market_type}:{symbol}"

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
        self._end = to_timestamp(end)
        symbols = self._active_symbols()
        if not symbols:
            return {}

        # - all symbols need warmup on first read
        read_start = to_timestamp(start)
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
            self._end = to_timestamp(end)

        symbols = self._active_symbols()
        if not symbols:
            self._b_iter = None
            return {}

        # - determine start time: extend into past if new symbols need warmup
        new_symbols = [s for s in symbols if s not in self._warmed]
        read_start = to_timestamp(current_time)

        if new_symbols and self._warmup_period:
            read_start = read_start - self._warmup_period
            self._warmed.update(new_symbols)

        # - restart backend reader with full current symbol set
        logger.debug(
            f"[<c>DataPump</c>] :: Restarting read for {len(symbols)} symbols "
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
            logger.warning(f"[DataPump] :: Unexpected chunk type: {type(chunk)}")

    def cleanup_inactive(self):
        """
        Remove fully inactive symbols (buffer deactivated + MemReader removed from slicer).
        Call periodically to free memory.
        """
        to_remove = [sym for sym, buf in self._buffers.items() if not buf.active and sym not in self._instruments]
        for sym in to_remove:
            self._buffers.pop(sym, None)
            self._mem_readers.pop(sym, None)
            logger.debug(f"[DataPump] :: Cleaned up inactive symbol buffer: {sym}")

    def __repr__(self) -> str:
        active = self._active_symbols()
        warmed = len(self._warmed)
        return (
            f"DataPump({self._requested_data_type} -> {self._producing_data_type}, "
            f"{self._exchange}:{self._market_type}, "
            f"symbols={len(active)}, warmed={warmed}, chunksize={self._chunksize})"
        )


# ---------------------------------------------------------------------------
# SimulatedDataIterator — simulated data manager
# ---------------------------------------------------------------------------
class SimulatedDataIterator(Iterator):
    """
    General manager for iterating through simulated data source using IStorage/IReader with shared memory buffers.

    Accepts IStorage (+ optional custom storages per data type) and resolves IReaders
    lazily on first subscription. One DataPump per (subscription, exchange, market_type).

    Data flow:
        IStorage.get_reader(exchange, market_type) → IReader (cached)
        IReader.read([symbols], dtype, start, end)
            → RawMultiData chunk (all symbols in one batch)
            → distribute by data_id to per-symbol RawSymbolBuffer (raw RecordBatch, pre-transform)
            → MemReader pops buffer on demand, transforms lazily → list[Timestamped]
            → IteratedDataStreamsSlicer merges all streams by timestamp

        DataPump (one per subscription + exchange scope, e.g. "ohlc.1h.BINANCE.UM:SWAP")
            ├── RawSymbolBuffer("BTCUSDT")  →  MemReader("BTCUSDT")  ─┐
            ├── RawSymbolBuffer("ETHUSDT")  →  MemReader("ETHUSDT")  ─┤  → Slicer → yield events
            └── RawSymbolBuffer("SOLUSDT")  →  MemReader("SOLUSDT")  ─┘

    Multi-exchange support:
        - Instruments from different exchanges get separate pumps (separate IReaders)
        - Slicer keys include exchange:market_type to avoid symbol collisions
        - Custom storages can override specific data types (e.g. HFT storage for quotes)

    Universe changes mid-sim:
        - Subscribe: pump.restart_read() with full symbol set, new MemReaders added to slicer
        - Unsubscribe: buffer deactivated → MemReader StopIteration → slicer removes stream
        - Dedup: RawSymbolBuffer watermark filters overlapping rows on restart (pyarrow vectorized)
        - Warmup: new symbols get extended start time, already-warmed symbols dedup via watermark
    """

    _storage: IStorage
    _custom_storages: dict[str, IStorage]

    _readers: dict[str, IReader]  # - cached readers: cache_key -> IReader
    _pumps: dict[str, DataPump]  # - keyed by "{access_key}.{exchange}:{market_type}"
    _instruments: dict[str, tuple[Instrument, DataPump, str]]  # - slicer_key -> (instrument, pump, subscription)
    _warmups: dict[str, pd.Timedelta]

    _slicer: IteratedDataStreamsSlicer | None
    _slicing_iterator: Iterator | None
    _current_time: int | None
    _start: pd.Timestamp | None
    _stop: pd.Timestamp | None
    _open_close_time_indent_secs: float
    _trading_session: dict[str, tuple[int, int]]
    _default_trading_session: tuple[int, int]

    def __init__(
        self,
        storage: IStorage,
        custom_types_storages: dict[str, IStorage] | None = None,
        open_close_time_indent_secs: float = 1.0,
        chunksize: int = 5000,
        trading_session: dict[str, tuple[int, int]] | None = None,
        default_trading_session: tuple[int, int] = DEFAULT_DAILY_SESSION,
    ):
        self._readers = {}
        self._storage = storage
        self._custom_storages = dict(custom_types_storages or {})
        self._pumps = {}
        self._instruments = {}
        self._warmups = {}
        self._open_close_time_indent_secs = open_close_time_indent_secs
        self._chunksize = chunksize
        self._trading_session = trading_session or {}
        self._default_trading_session = default_trading_session
        self._slicer = None
        self._slicing_iterator = None
        self._current_time = None
        self._start = None
        self._stop = None

    @property
    def emulation_time_indent_seconds(self) -> float:
        """
        What time indent is used for emulated market data updates
        """
        return self._open_close_time_indent_secs

    def close(self):
        """
        Close all cached IReader instances to release underlying resources (DB connections, file handles, etc).
        """
        for reader in self._readers.values():
            try:
                reader.close()
            except Exception as e:
                logger.warning(f"Failed to close reader {reader}: {e}")
        self._readers.clear()

    def update_emulation_time_indent_seconds(self, time_indent_seconds: float):
        self._open_close_time_indent_secs = time_indent_seconds
        # - update transformers if there are any
        for p in self._pumps.values():
            p.update_emulation_time_indent_seconds(time_indent_seconds)

    def set_warmup_period(self, subscription: str, warmup_period: str | None = None):
        if warmup_period:
            access_key, _, _ = self._parse_subscription_spec(subscription)
            self._warmups[access_key] = to_timedelta(warmup_period)

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
    # Reader resolution (lazy, cached)
    # -----------------------------------------------------------------------

    def _get_or_create_reader(self, data_type: str, exchange: str, market_type: str) -> IReader:
        """
        Get or create cached IReader for given (data_type, exchange, market_type).

        Checks custom storage first (keyed by data_type), falls back to main storage.
        Custom storage readers are cached with data_type prefix to avoid collisions.
        Main storage readers are shared across data types for same (exchange, market_type).
        """
        # - check custom storage first
        custom_storage = self._custom_storages.get(data_type)
        if custom_storage is not None:
            cache_key = f"{data_type}:{exchange}:{market_type}"
            if cache_key not in self._readers:
                self._readers[cache_key] = custom_storage.get_reader(exchange, market_type)
            return self._readers[cache_key]

        # - fallback to main storage
        cache_key = f"{exchange}:{market_type}"
        if cache_key not in self._readers:
            self._readers[cache_key] = self._storage.get_reader(exchange, market_type)
        return self._readers[cache_key]

    # -----------------------------------------------------------------------
    # Subscribe / Unsubscribe
    # -----------------------------------------------------------------------

    def _get_or_create_pump(
        self, access_key: str, subscription: str, data_type: str, exchange: str, market_type: str
    ) -> DataPump:
        """
        Get existing pump or create new one for given (subscription, exchange, market_type).

        Pump key includes exchange scope so instruments from different exchanges
        get separate pumps (each with its own IReader).

        Args:
            access_key: subscription access key (e.g. "ohlc.1h")
            subscription: full subscription string (e.g. "ohlc(1h)") — passed to DataPump
            data_type: base data type for reader lookup (e.g. "ohlc")
            exchange: exchange identifier (e.g. "BINANCE.UM")
            market_type: market type (e.g. "SWAP")
        """
        pump_key = f"{access_key}.{exchange}:{market_type}"

        if pump_key in self._pumps:
            return self._pumps[pump_key]

        reader = self._get_or_create_reader(data_type, exchange, market_type)

        pump = DataPump(
            reader=reader,
            subscription_type=subscription,
            exchange=exchange,
            market_type=market_type,
            warmup_period=self._warmups.get(access_key),
            chunksize=self._chunksize,
            open_close_time_indent_secs=self._open_close_time_indent_secs,
            trading_session=self._trading_session.get(exchange, self._default_trading_session),
        )
        self._pumps[pump_key] = pump
        return pump

    def add_instruments_for_subscription(self, subscription: str, instruments: list[Instrument] | Instrument):
        instruments = instruments if isinstance(instruments, list) else [instruments]
        access_key, data_type, _params = self._parse_subscription_spec(subscription)

        # - filter instruments based on subscription type
        instruments = self._filter_instruments_for_subscription(data_type, instruments)
        if not instruments:
            return

        # - group instruments by (exchange, market_type) — each group gets its own pump
        groups: dict[tuple[str, str], list[Instrument]] = {}
        for i in instruments:
            groups.setdefault((i.exchange, str(i.market_type)), []).append(i)

        for (exchange, market_type), group_instruments in groups.items():
            pump = self._get_or_create_pump(access_key, subscription, data_type, exchange, market_type)

            new_instruments = []
            for i in group_instruments:
                if not pump.has_instrument(i):
                    pump.attach_instrument(i)
                    slicer_key = pump._make_slicer_key(i.symbol)
                    self._instruments[slicer_key] = (i, pump, subscription)
                    new_instruments.append(i)

            # - if simulation is running, restart read to include new symbols
            if self.is_running and new_instruments:
                new_mem_readers = pump.restart_read(to_timestamp(self._current_time, unit="ns"), self._stop)  # type: ignore
                if new_mem_readers and self._slicer is not None:
                    self._slicer.put(new_mem_readers)

    def remove_instruments_from_subscription(self, subscription: str, instruments: list[Instrument] | Instrument):
        instruments = instruments if isinstance(instruments, list) else [instruments]

        def _remove_from_pump(pump: DataPump, instruments: list[Instrument]):
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
            for pump in list(self._pumps.values()):
                _remove_from_pump(pump, instruments)
            return

        access_key, _, _ = self._parse_subscription_spec(subscription)

        # - group by (exchange, market_type) to find correct pump
        groups: dict[tuple[str, str], list[Instrument]] = {}
        for i in instruments:
            groups.setdefault((i.exchange, str(i.market_type)), []).append(i)

        for (exchange, market_type), group_instruments in groups.items():
            pump_key = f"{access_key}.{exchange}:{market_type}"
            pump = self._pumps.get(pump_key)
            if not pump:
                logger.warning(f"No configured data pump for '{pump_key}' subscription !")
                continue

            _remove_from_pump(pump, group_instruments)

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
        # - find all pumps matching this access_key (there can be multiple for different exchanges)
        access_prefix = access_key + "."
        result = []
        for pump_key, pump in self._pumps.items():
            if pump_key.startswith(access_prefix):
                result.extend(pump.get_instruments())
        return result

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
        pump_key = f"{access_key}.{instrument.exchange}:{instrument.market_type}"
        pump = self._pumps.get(pump_key)
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
        self._start = to_timestamp(start)
        self._stop = to_timestamp(stop)
        self._current_time = None
        self._slicer = IteratedDataStreamsSlicer()
        return self

    def __iter__(self) -> Iterator:
        assert self._start is not None and self._stop is not None
        assert self._slicer is not None
        self._current_time = int(to_timestamp(self._start).timestamp() * 1e9)

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

    def get_ohlc(self, instrument: Instrument, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Bar]:
        # - get reader for this instrument
        _reader = self._get_or_create_reader(DataType.OHLC[timeframe], instrument.exchange, instrument.market_type)
        if _reader is None:
            logger.error(f"Can't find or create reader for {DataType.OHLC[timeframe]} data")
            return []

        # - get ohlc(timeframe) data
        t_records = _reader.read(
            data_id=instrument.symbol, dtype=DataType.OHLC[timeframe], start=str(start), stop=str(end)
        )
        assert isinstance(t_records, Transformable)

        return self._process_bar_records(
            t_records.transform(TypedRecords()), time_as_nsec(start), to_timedelta(timeframe).asm8.item()
        )

    def _process_bar_records(self, records: list[Bar], cut_time_ns: int, timeframe_ns: int) -> list[Bar]:
        """
        Convert records to bars and we need to cut last bar up to the cut_time_ns
        """
        bars = []

        # - what indent time is used in data source
        _open_close_time_indent_ns = int(self.emulation_time_indent_seconds * 1_000_000_000)

        # - if no records, return empty list to avoid exception from infer_series_frequency
        if not records or records is None:
            return bars

        if len(records) > 1:
            _data_tf = infer_series_frequency([r.time for r in records[:50]])
            timeframe_ns = _data_tf.item()

        for r in records:
            _b_ts_0 = r.time
            _b_ts_1 = _b_ts_0 + timeframe_ns - _open_close_time_indent_ns

            if _b_ts_0 <= cut_time_ns and cut_time_ns < _b_ts_1:
                break

            # - handle None values in OHLC data
            open_price = r.open
            high_price = r.high
            low_price = r.low
            close_price = r.close

            # Skip this record if any OHLC value is None
            if open_price is None or high_price is None or low_price is None or close_price is None:
                continue

            bars.append(r)

        return bars
