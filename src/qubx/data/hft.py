import queue
from collections import defaultdict
from multiprocessing import Event, Process, Queue
from pathlib import Path
from threading import Thread
from typing import Any, Iterable, Optional, TypeAlias, TypeVar

import numpy as np
import pandas as pd

from qubx.data.registry import reader

try:
    from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, ROIVectorMarketDepthBacktest
    from hftbacktest.binding import ROIVectorMarketDepthBacktest as IStrategyContext
    from hftbacktest.types import BUY_EVENT, SELL_EVENT, TRADE_EVENT

    HFTBACKTEST_AVAILABLE = True
except ImportError:
    HFTBACKTEST_AVAILABLE = False

    # Define placeholder classes/constants for type checking
    class BacktestAsset:
        def data(self, *args):
            return self

        def tick_size(self, *args):
            return self

        def lot_size(self, *args):
            return self

        def last_trades_capacity(self, *args):
            return self

        def roi_lb(self, *args):
            return self

        def roi_ub(self, *args):
            return self

    class HashMapMarketDepthBacktest:
        pass

    class ROIVectorMarketDepthBacktest:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class IStrategyContext:
        pass

    BUY_EVENT = 1
    SELL_EVENT = 2
    TRADE_EVENT = 4


from numba import njit

from qubx import logger
from qubx.core.basics import Instrument
from qubx.core.lookups import lookup
from qubx.core.utils import recognize_timeframe
from qubx.utils.time import handle_start_stop

from .readers import DataReader, DataTransformer

HFT_EXCHANGE_MAPPERS = {
    "bitfinex.f": "bitfinex-derivatives",
    "binance.um": "binance-futures",
}

T = TypeVar("T")


class HftChunkPrefetcher:
    """Prefetches HFT data chunks in a separate process while maintaining context"""

    def __init__(
        self,
        max_prefetch: int = 3,
        enable_quote: bool = False,
        enable_trade: bool = False,
        enable_orderbook: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the prefetcher.

        Args:
            max_prefetch: Maximum number of chunks to prefetch and store in the queue
            enable_quote: Whether to process quote data
            enable_trade: Whether to process trade data
            enable_orderbook: Whether to process orderbook data
        """
        self.queues = {}
        if enable_quote:
            self.queues["quote"] = Queue(maxsize=max_prefetch)
        if enable_trade:
            self.queues["trade"] = Queue(maxsize=max_prefetch)
        if enable_orderbook:
            self.queues["orderbook"] = Queue(maxsize=max_prefetch)
        self.stop_event = Event()
        self.worker: Optional[Process | Thread] = None
        self.error_queue = Queue()  # type: Queue[Exception]
        self._mp_factory_factory = Thread if debug else Process

    def start(
        self,
        path: Path,
        data_id: str,
        start: pd.Timestamp,
        stop: pd.Timestamp,
        chunk_args: dict,
    ) -> None:
        """
        Start background prefetching in a separate process.

        Args:
            path: Path to HFT data directory
            data_id: Data identifier in format "exchange:symbol"
            start: Start timestamp
            stop: Stop timestamp
            chunk_args: Arguments for _next_batch method
        """

        def _prefetch_worker(
            path: Path,
            data_id: str,
            start: pd.Timestamp,
            stop: pd.Timestamp,
            chunk_args: dict,
            queues: dict[str, Queue],
            error_queue: Queue,
            stop_event: Event,  # type: ignore
        ) -> None:
            ctx = None
            try:
                # Create a new reader and context in this process
                reader = HftDataReader(
                    path=path,
                    quote_interval=chunk_args.get("quote_interval", "1s"),
                    orderbook_interval=chunk_args.get("orderbook_interval", "1s"),
                    trade_capacity=chunk_args.get("trade_capacity", 30_000),
                    enable_quote="quote" in queues,
                    enable_trade="trade" in queues,
                    enable_orderbook="orderbook" in queues,
                )
                ctx = reader._get_or_create_context(data_id, start.floor("d"), stop)
                instrument = reader._data_id_to_instrument[data_id]

                # Initialize buffers only for enabled data types
                reader._create_buffer_if_needed(
                    "ob_timestamp", instrument, (chunk_args["chunksize"],), np.dtype(np.int64)
                )
                reader._create_buffer_if_needed(
                    "bid_price", instrument, (chunk_args["chunksize"],), np.dtype(np.float64)
                )
                reader._create_buffer_if_needed(
                    "ask_price", instrument, (chunk_args["chunksize"],), np.dtype(np.float64)
                )
                reader._create_buffer_if_needed(
                    "bid_size",
                    instrument,
                    (chunk_args["chunksize"], chunk_args["depth"]),
                    np.dtype(np.float64),
                )
                reader._create_buffer_if_needed(
                    "ask_size",
                    instrument,
                    (chunk_args["chunksize"], chunk_args["depth"]),
                    np.dtype(np.float64),
                )
                reader._create_buffer_if_needed(
                    "tick_size", instrument, (chunk_args["chunksize"],), np.dtype(np.float64)
                )

                # Create trade buffer if trade data is enabled
                _trade_capacity = chunk_args["chunksize"] * reader.trade_capacity
                trade_dtype = np.dtype(
                    [
                        ("timestamp", "i8"),
                        ("price", "f8"),
                        ("size", "f8"),
                        ("side", "i1"),
                        ("array_id", "i8"),
                    ]
                )
                reader._create_buffer_if_needed("trades", instrument, (_trade_capacity,), trade_dtype)

                # Create quote buffer if quote data is enabled
                _quote_interval = recognize_timeframe(reader.quote_interval)
                _orderbook_interval = recognize_timeframe(reader.orderbook_interval)
                orderbook_period = int(np.round(_orderbook_interval / _quote_interval))
                quote_chunksize = chunk_args["chunksize"] * orderbook_period
                quote_dtype = np.dtype(
                    [
                        ("timestamp", "i8"),
                        ("bid", "f8"),
                        ("ask", "f8"),
                        ("bid_size", "f8"),
                        ("ask_size", "f8"),
                    ]
                )
                reader._create_buffer_if_needed("quotes", instrument, (quote_chunksize,), quote_dtype)
                start_time_ns = start.value
                stop_time_ns = stop.value

                while not stop_event.is_set():
                    stop_reached = reader._next_batch(
                        ctx=ctx,
                        instrument=instrument,
                        chunksize=chunk_args["chunksize"],
                        data_type=chunk_args["data_type"],
                        interval=_quote_interval,
                        orderbook_period=orderbook_period,
                        tick_size_pct=chunk_args["tick_size_pct"],
                        depth=chunk_args["depth"],
                        start_time=start_time_ns,
                        stop_time=stop_time_ns,
                    )

                    # Get records for enabled data types
                    records_map = {}
                    if "quote" in queues:
                        records_map["quote"] = reader._get_records("quote", instrument)
                    if "trade" in queues:
                        records_map["trade"] = reader._get_records("trade", instrument)
                    if "orderbook" in queues:
                        records_map["orderbook"] = reader._get_records("orderbook", instrument)

                    # If all records are None, we're done
                    if all(records is None for records in records_map.values()):
                        for queue in queues.values():
                            queue.put(None)  # Signal end of data
                        break

                    # Process and put records for each enabled type
                    for data_type, records in records_map.items():
                        if records is not None:
                            # Convert records to a serializable format if needed
                            if isinstance(records, zip):
                                # Create a deep copy of the zipped orderbook data
                                records = [
                                    (
                                        ts,
                                        bid_price,
                                        ask_price,
                                        tick_size,
                                        bid_size.copy(),
                                        ask_size.copy(),
                                    )
                                    for ts, bid_price, ask_price, tick_size, bid_size, ask_size in records
                                ]
                            elif isinstance(records, np.ndarray):
                                records = records.copy()  # Ensure we have a clean copy for IPC

                            # Keep trying to put records until success or stop requested
                            while not stop_event.is_set():
                                try:
                                    queues[data_type].put(records, timeout=1.0)
                                    break  # Successfully put records, exit retry loop
                                except Exception:  # Handle any queue errors including Full
                                    continue  # Try again if queue is full

                            if stop_event.is_set():
                                break  # Exit the outer loop if stop was requested

                    if stop_event.is_set():
                        break

                    for data_type in queues:
                        reader._mark_processed(data_type, instrument)

                    if stop_reached:
                        break

            except Exception as e:
                error_queue.put(e)
                for queue in queues.values():
                    queue.put(None)  # Signal error
            finally:
                if ctx is not None:
                    reader._close_context(instrument)

        self.worker = self._mp_factory_factory(
            target=_prefetch_worker,
            args=(
                path,
                data_id,
                start,
                stop,
                chunk_args,
                self.queues,
                self.error_queue,
                self.stop_event,
            ),
            daemon=True,
        )
        self.worker.start()

    def get_next(self, data_type: str, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Get next available chunk for the specified data type.

        Args:
            data_type: Type of data to get ("quote", "trade", or "orderbook")
            timeout: How long to wait for the next chunk (in seconds)

        Returns:
            The next chunk of data or None if no more data

        Raises:
            Exception: If an error occurred in the worker process
            queue.Empty: If timeout is reached before data is available
        """
        if data_type not in self.queues:
            raise ValueError(f"Invalid data type: {data_type}")

        # Check for errors first
        try:
            error = self.error_queue.get_nowait()
            raise error
        except queue.Empty:
            pass

        # Get next chunk
        try:
            chunk = self.queues[data_type].get(timeout=timeout)
            return chunk
        except queue.Empty:
            if self.worker is not None and not self.worker.is_alive():
                # Check for errors one last time
                try:
                    error = self.error_queue.get_nowait()
                    raise error
                except queue.Empty:
                    pass
                return None
            raise

    def stop(self) -> None:
        """Stop the prefetching process"""
        if self.worker is not None and self.worker.is_alive():
            # Set stop event first
            self.stop_event.set()
            logger.debug("Stop event set")

            # Give the worker a chance to exit gracefully
            self.worker.join(timeout=1)

            if self.worker.is_alive():
                logger.debug("Worker still alive after graceful stop, terminating...")
                if isinstance(self.worker, Process):
                    # If it's still alive, terminate it forcefully
                    try:
                        self.worker.terminate()
                        # Give it a very short time to terminate
                        self.worker.join(timeout=0.1)
                    except Exception as e:
                        logger.warning(f"Error during worker termination: {e}")

                    if self.worker.is_alive():
                        logger.warning("Worker still alive after terminate, attempting kill")
                        try:
                            if hasattr(self.worker, "kill"):
                                self.worker.kill()
                        except Exception as e:
                            logger.warning(f"Error during worker kill: {e}")


@reader("hft")
class HftDataReader(DataReader):
    """
    Reads npz files containing orderbooks and trades in the same format
    as https://github.com/nkaz001/hftbacktest.
    """

    def __init__(
        self,
        path: str | Path = "/hft-data",
        quote_interval: str = "1s",
        orderbook_interval: str = "1s",
        trade_capacity: int = 30_000,
        max_prefetch: int = 3,
        enable_orderbook: bool = True,
        enable_quote: bool = False,
        enable_trade: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initialize HftDataReader.

        Args:
            path: Path to the directory containing the HFT data.
                  Expected structure: path/exchange/symbol/{date}.npz
            quote_interval: Interval for quote data
            orderbook_interval: Interval for orderbook data
            trade_capacity: Maximum number of trades to store per chunk
            max_prefetch: Maximum number of chunks to prefetch
            enable_orderbook: Whether to process orderbook data (default: True)
            enable_quote: Whether to process quote data (default: False)
            enable_trade: Whether to process trade data (default: False)
        """
        path = Path(path)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Path {path} does not exist or is not a directory")
        self.path = path
        self.quote_interval = quote_interval
        self.orderbook_interval = orderbook_interval
        self.trade_capacity = trade_capacity
        self.max_prefetch = max_prefetch
        self.enable_orderbook = enable_orderbook
        self.enable_quote = enable_quote
        self.enable_trade = enable_trade
        self.instrument_to_context = {}
        self._data_id_to_instrument = {}
        self._instrument_to_name_to_buffer = defaultdict(dict)
        self._instrument_to_quote_index = defaultdict(int)
        self._instrument_to_trade_index = defaultdict(int)
        self._instrument_to_orderbook_index = defaultdict(int)
        self._prefetchers: dict[str, HftChunkPrefetcher] = {}
        self._prefetcher_ranges = {}  # Store time ranges for prefetchers
        self._debug = debug

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize: int = 5000,
        data_type: str | None = None,
        tick_size_pct: float = 0.0,
        depth: int = 100,
        timeframe: str | None = None,
    ) -> Iterable:
        """
        Supported data types: ["quote", "trade", "orderbook"]

        If tick_size_pct is set to 0, min tick size will be used.
        """
        if data_type is None:
            raise ValueError("data_type is required")
        if data_type not in ["quote", "trade", "orderbook"]:
            raise ValueError(f"Invalid data type: {data_type}")
        if not chunksize:
            raise ValueError("chunksize must be greater than 0")

        # Check if the requested data type is enabled
        if (
            (data_type == "quote" and not self.enable_quote)
            or (data_type == "trade" and not self.enable_trade)
            or (data_type == "orderbook" and not self.enable_orderbook)
        ):
            raise ValueError(f"Data type {data_type} is not enabled")

        # - handle start and stop
        _start_raw, _stop = handle_start_stop(start, stop, lambda x: pd.Timestamp(x))
        _start = _start_raw.floor("d")  # we must to start from day's start
        assert isinstance(_start, pd.Timestamp) and isinstance(_stop, pd.Timestamp)

        # Check if we need to recreate the prefetcher
        should_create_prefetcher = True
        if data_id in self._prefetchers:
            existing_range = self._prefetcher_ranges.get(data_id)
            if existing_range is not None:
                existing_start, existing_stop = existing_range
                if existing_start == _start and existing_stop == _stop:
                    should_create_prefetcher = False

        # Stop existing prefetcher if we need to recreate it
        if should_create_prefetcher and data_id in self._prefetchers:
            logger.debug(f"Stopping prefetcher for {data_id}")
            self._prefetchers[data_id].stop()
            del self._prefetchers[data_id]
            if data_id in self._prefetcher_ranges:
                del self._prefetcher_ranges[data_id]

        # Create and start prefetcher if needed
        if should_create_prefetcher:
            logger.debug(f"Creating prefetcher for {data_id}")
            prefetcher = HftChunkPrefetcher(
                max_prefetch=self.max_prefetch,
                enable_quote=self.enable_quote,
                enable_trade=self.enable_trade,
                enable_orderbook=self.enable_orderbook,
                debug=self._debug,
            )
            chunk_args = {
                "chunksize": chunksize,
                "data_type": data_type,
                "tick_size_pct": tick_size_pct,
                "depth": depth,
                "quote_interval": self.quote_interval,
                "orderbook_interval": self.orderbook_interval,
                "trade_capacity": self.trade_capacity,
            }
            prefetcher.start(self.path, data_id, _start_raw, _stop, chunk_args)
            logger.debug(f"Started prefetcher for {data_id}")
            self._prefetchers[data_id] = prefetcher
            self._prefetcher_ranges[data_id] = (_start, _stop)

        def _iter_chunks():
            try:
                prefetcher = self._prefetchers[data_id]
                while True:
                    try:
                        records = prefetcher.get_next(data_type, timeout=1)
                        if records is None:
                            break

                        transform.start_transform(data_id, self._get_field_names(data_type), start=start, stop=stop)
                        transform.process_data(records)
                        yield transform.collect()

                    except queue.Empty:
                        continue  # Try again if timeout

            finally:
                # Cleanup prefetcher when iteration is done
                if data_id in self._prefetchers:
                    self._prefetchers[data_id].stop()
                    del self._prefetchers[data_id]
                    if data_id in self._prefetcher_ranges:
                        del self._prefetcher_ranges[data_id]

        return _iter_chunks()

    def get_time_ranges(self, data_id: str, dtype: str) -> tuple[np.datetime64, np.datetime64]:
        """
        Returns first and last time for the specified symbol and data type in the reader's storage
        """
        exchange, symbol = data_id.split(":")
        instrument = lookup.find_symbol(exchange, symbol)
        if instrument is None:
            raise ValueError(f"Instrument {data_id} not found")

        _exchange = exchange.lower()
        _exchange = HFT_EXCHANGE_MAPPERS.get(_exchange, _exchange)

        _path = self.path / _exchange / symbol.upper()
        if not _path.exists():
            raise ValueError(f"Data for {data_id} not found at {_path}")

        _files = sorted(_path.glob("*.npz"))
        if not _files:
            raise ValueError(f"No data for {data_id} found at {_path}")

        _files = sorted([str(f.stem) for f in _files])
        return np.datetime64(_files[0]), np.datetime64(_files[-1])

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        _exchange = exchange.lower()
        _exchange = HFT_EXCHANGE_MAPPERS.get(_exchange, _exchange)
        _path = self.path / _exchange
        if not _path.exists():
            raise ValueError(f"Data for {exchange} not found at {_path}")
        symbols = [f.name for f in _path.iterdir() if f.is_dir()]
        return symbols

    def _get_or_create_context(self, data_id: str, start: pd.Timestamp, stop: pd.Timestamp):
        exchange, symbol = data_id.split(":")
        _exchange = exchange.lower()
        _exchange = HFT_EXCHANGE_MAPPERS.get(_exchange, _exchange)

        _instrument = lookup.find_symbol(exchange, symbol)
        if _instrument is None:
            raise ValueError(f"Instrument {data_id} not found")

        if _instrument in self.instrument_to_context:
            return self.instrument_to_context[_instrument]

        _path = self.path / _exchange / symbol.upper()
        if not _path.exists():
            raise ValueError(f"Data for {data_id} not found at {_path}")

        _files = sorted(_path.glob("*.npz"))
        if not _files:
            raise ValueError(f"No data for {data_id} found at {_path}")

        _start, _stop = str(start - pd.Timedelta("1sec")), str(stop - pd.Timedelta("1sec"))
        _files = sorted([str(f) for f in _files if _start <= f.stem <= _stop])

        ctx = ROIVectorMarketDepthBacktest(self._create_backtest_assets(_files, _instrument))
        self.instrument_to_context[_instrument] = ctx
        self._data_id_to_instrument[data_id] = _instrument
        self._align_context_to_second(ctx)
        return ctx

    def _align_context_to_second(self, ctx: IStrategyContext):
        # - initialize ctx by elapsing 100ms
        ctx.elapse(100_000_000)
        # - get to next (second - 1ms)
        _dt = pd.Timestamp(ctx.current_timestamp, "ns")
        _delta_to_sec = _dt.ceil("s") - _dt - pd.Timedelta("1ms")
        ctx.elapse(_delta_to_sec.total_seconds() * 1_000_000_000)

    def close(self) -> None:
        """Clean up all resources"""
        # Stop and remove all prefetchers
        for prefetcher in self._prefetchers.values():
            prefetcher.stop()
        self._prefetchers.clear()

        # Close all contexts
        for instrument in list(self.instrument_to_context.keys()):
            self._close_context(instrument)

    def __del__(self):
        """Ensure cleanup when object is deleted"""
        self.close()

    def _close_context(self, instrument: Instrument):
        if instrument not in self.instrument_to_context:
            return
        ctx = self.instrument_to_context.pop(instrument)
        ctx.close()

    def _next_batch(
        self,
        ctx: IStrategyContext,
        instrument: Instrument,
        chunksize: int,
        data_type: str,
        interval: int,
        orderbook_period: int,
        tick_size_pct: float,
        depth: int,
        start_time: int,
        stop_time: int,
    ) -> bool:
        match data_type:
            case "quote":
                if self._instrument_to_quote_index[instrument] > 0 or not self.enable_quote:
                    return
            case "trade":
                if self._instrument_to_trade_index[instrument] > 0 or not self.enable_trade:
                    return
            case "orderbook":
                if self._instrument_to_orderbook_index[instrument] > 0 or not self.enable_orderbook:
                    return

        (
            quote_index,
            trade_index,
            orderbook_index,
            stop_reached,
        ) = _simulate_hft(
            ctx=ctx,
            start_time=start_time,
            stop_time=stop_time,
            ob_timestamp=self._instrument_to_name_to_buffer["ob_timestamp"][instrument],
            bid_price_buffer=self._instrument_to_name_to_buffer["bid_price"][instrument],
            ask_price_buffer=self._instrument_to_name_to_buffer["ask_price"][instrument],
            bid_size_buffer=self._instrument_to_name_to_buffer["bid_size"][instrument],
            ask_size_buffer=self._instrument_to_name_to_buffer["ask_size"][instrument],
            tick_size_buffer=self._instrument_to_name_to_buffer["tick_size"][instrument],
            trade_buffer=self._instrument_to_name_to_buffer["trades"][instrument],
            quote_buffer=self._instrument_to_name_to_buffer["quotes"][instrument],
            batch_size=chunksize,
            interval=interval,
            orderbook_period=orderbook_period,
            tick_size_pct=tick_size_pct,
            max_levels=depth,
            enable_quote=self.enable_quote,
            enable_trade=self.enable_trade,
            enable_orderbook=self.enable_orderbook,
        )

        self._instrument_to_quote_index[instrument] = quote_index if self.enable_quote else 0
        self._instrument_to_trade_index[instrument] = trade_index if self.enable_trade else 0
        self._instrument_to_orderbook_index[instrument] = orderbook_index if self.enable_orderbook else 0

        return stop_reached

    def _create_backtest_assets(self, files: list[str], instrument: Instrument) -> list[BacktestAsset]:
        mid_price = _get_initial_mid_price(files)
        roi_lb, roi_ub = mid_price / 4, mid_price * 4
        return [
            BacktestAsset()
            .data(files)
            .tick_size(instrument.tick_size)
            .lot_size(instrument.lot_size)
            .last_trades_capacity(self.trade_capacity)
            .roi_lb(roi_lb)
            .roi_ub(roi_ub)
        ]

    @staticmethod
    def _get_field_names(data_type: str) -> list[str]:
        # TODO: pass correct field names
        if data_type == "quote":
            return ["timestamp", "bid", "ask", "bid_size", "ask_size"]
        elif data_type == "trade":
            return ["timestamp", "price", "size", "side", "array_id"]
        elif data_type == "orderbook":
            return ["timestamp", "top_bid", "top_ask", "tick_size", "bids", "asks"]
        else:
            raise ValueError(f"Invalid data type: {data_type}")

    def _create_buffer_if_needed(
        self, name: str, instrument: Instrument, shape: tuple[int, ...], dtype: np.dtype
    ) -> None:
        chunksize = shape[0]
        if (
            instrument not in self._instrument_to_name_to_buffer[name]
            or self._instrument_to_name_to_buffer[name][instrument].shape[0] != chunksize
        ):
            self._instrument_to_name_to_buffer[name][instrument] = np.zeros(shape, dtype=dtype)

    def _get_buffer(self, name: str, instrument: Instrument, max_index: int) -> np.ndarray:
        return self._instrument_to_name_to_buffer[name][instrument][:max_index]

    def _get_records(self, data_type: str, instrument: Instrument) -> Iterable | None:
        index = 0
        match data_type:
            case "quote":
                index = self._instrument_to_quote_index[instrument]
            case "trade":
                index = self._instrument_to_trade_index[instrument]
            case "orderbook":
                index = self._instrument_to_orderbook_index[instrument]

        if not index:
            return None

        match data_type:
            case "quote":
                return self._get_buffer("quotes", instrument, index)
            case "trade":
                return self._get_buffer("trades", instrument, index)
            case "orderbook":
                return zip(
                    self._get_buffer("ob_timestamp", instrument, index),
                    self._get_buffer("bid_price", instrument, index),
                    self._get_buffer("ask_price", instrument, index),
                    self._get_buffer("tick_size", instrument, index),
                    self._get_buffer("bid_size", instrument, index),
                    self._get_buffer("ask_size", instrument, index),
                )
            case _:
                raise ValueError(f"Invalid data type: {data_type}")

    def _mark_processed(self, data_type: str, instrument: Instrument):
        match data_type:
            case "quote":
                self._instrument_to_quote_index[instrument] = 0
            case "trade":
                self._instrument_to_trade_index[instrument] = 0
            case "orderbook":
                self._instrument_to_orderbook_index[instrument] = 0
            case _:
                raise ValueError(f"Invalid data type: {data_type}")


def _get_initial_mid_price(instr_files: list[str]) -> float:
    snapshot = np.load(instr_files[0])["data"]

    best_bid = max(snapshot[snapshot["ev"] & (BUY_EVENT | TRADE_EVENT) == (BUY_EVENT | TRADE_EVENT)]["px"])
    best_ask = min(snapshot[snapshot["ev"] & (SELL_EVENT | TRADE_EVENT) == (SELL_EVENT | TRADE_EVENT)]["px"])
    return (best_bid + best_ask) / 2.0


@njit
def _simulate_hft(
    ctx: IStrategyContext,
    ob_timestamp: np.ndarray,
    bid_price_buffer: np.ndarray,
    ask_price_buffer: np.ndarray,
    bid_size_buffer: np.ndarray,
    ask_size_buffer: np.ndarray,
    tick_size_buffer: np.ndarray,
    trade_buffer: np.ndarray,
    quote_buffer: np.ndarray,
    batch_size: int,
    start_time: int,
    stop_time: int,
    interval: int = 1_000_000_000,
    orderbook_period: int = 1,
    tick_size_pct: float = 0.0,
    max_levels: int = 100,
    enable_quote: bool = True,
    enable_trade: bool = True,
    enable_orderbook: bool = True,
) -> tuple[int, int, int, bool]:
    orderbook_index = 0
    quote_index = 0
    trade_index = 0
    stop_reached = False

    while ctx.elapse(interval) == 0 and orderbook_index < batch_size:
        # - skip if we are before the start time
        if ctx.current_timestamp < start_time:
            continue

        depth = ctx.depth(0)

        # record quote
        if enable_quote:
            quote_buffer[quote_index]["timestamp"] = ctx.current_timestamp
            quote_buffer[quote_index]["bid"] = depth.best_bid
            quote_buffer[quote_index]["ask"] = depth.best_ask
            quote_buffer[quote_index]["bid_size"] = depth.bid_qty_at_tick(depth.best_bid_tick)
            quote_buffer[quote_index]["ask_size"] = depth.ask_qty_at_tick(depth.best_ask_tick)

        # record trades
        if enable_trade:
            trades = ctx.last_trades(0)
            for trade in trades:
                trade_buffer[trade_index]["timestamp"] = trade.local_ts
                trade_buffer[trade_index]["price"] = trade.px
                trade_buffer[trade_index]["size"] = trade.qty
                trade_buffer[trade_index]["side"] = (trade.ev & BUY_EVENT == BUY_EVENT) * 2 - 1
                trade_buffer[trade_index]["array_id"] = quote_index
                trade_index += 1

        if enable_orderbook and quote_index % orderbook_period == 0:
            # record orderbook
            ob_timestamp[orderbook_index] = ctx.current_timestamp

            bid_price_buffer[orderbook_index] = depth.best_bid
            ask_price_buffer[orderbook_index] = depth.best_ask

            level_ticks = 1
            if tick_size_pct > 0.0:
                mid_price = (depth.best_bid + depth.best_ask) / 2.0
                level_ticks = max(int(np.round((mid_price * tick_size_pct / 100.0) / depth.tick_size)), 1)

            tick_size_buffer[orderbook_index] = depth.tick_size * level_ticks

            for i in range(max_levels):
                bid_size_buffer[orderbook_index, i] = 0
                ask_size_buffer[orderbook_index, i] = 0
                for j in range(level_ticks):
                    bid_size_buffer[orderbook_index, i] += depth.bid_qty_at_tick(
                        depth.best_bid_tick - i * level_ticks - j
                    )
                    ask_size_buffer[orderbook_index, i] += depth.ask_qty_at_tick(
                        depth.best_ask_tick + i * level_ticks + j
                    )

            orderbook_index += 1

        ctx.clear_last_trades(0)
        quote_index += 1

        # - stop if we reached the stop time
        if ctx.current_timestamp >= stop_time:
            stop_reached = True
            break

    return quote_index, trade_index, orderbook_index, stop_reached
