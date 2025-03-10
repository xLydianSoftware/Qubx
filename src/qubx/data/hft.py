from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, TypeAlias

import numpy as np
import pandas as pd
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, ROIVectorMarketDepthBacktest
from hftbacktest.binding import ROIVectorMarketDepthBacktest as IStrategyContext
from hftbacktest.types import BUY_EVENT, SELL_EVENT, TRADE_EVENT
from numba import njit

from qubx.core.basics import Instrument
from qubx.core.lookups import lookup
from qubx.core.utils import recognize_timeframe
from qubx.utils.time import handle_start_stop

from .readers import DataReader, DataTransformer

HFT_EXCHANGE_MAPPERS = {
    "bitfinex.f": "bitfinex-derivatives",
    "binance.um": "binance-futures",
}


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
        enable_quotes: bool = True,
        enable_orderbooks: bool = True,
        enable_trades: bool = True,
    ) -> None:
        """
        Initialize HftDataReader.

        Args:
            path: Path to the directory containing the HFT data.
                  Expected structure: path/exchange/symbol/{date}.npz
        """
        path = Path(path)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Path {path} does not exist or is not a directory")
        self.path = path
        self.quote_interval = quote_interval
        self.orderbook_interval = orderbook_interval
        self.enable_quotes = enable_quotes
        self.enable_orderbooks = enable_orderbooks
        self.enable_trades = enable_trades
        self.instrument_to_context = {}
        self._data_id_to_instrument = {}
        self._instrument_to_name_to_buffer = defaultdict(dict)

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

        _start, _stop = handle_start_stop(start, stop, pd.Timestamp)
        assert isinstance(_start, pd.Timestamp) and isinstance(_stop, pd.Timestamp)

        ctx = self._get_or_create_context(data_id, _start, _stop)
        instrument = self._data_id_to_instrument[data_id]
        names = self._get_field_names(data_type)

        if self._should_create_buffer("ob_timestamp", instrument, chunksize):
            self._instrument_to_name_to_buffer["ob_timestamp"][instrument] = np.zeros(chunksize, dtype=np.int64)
        if self._should_create_buffer("bid_price", instrument, chunksize):
            self._instrument_to_name_to_buffer["bid_price"][instrument] = np.zeros(chunksize, dtype=np.float64)
        if self._should_create_buffer("ask_price", instrument, chunksize):
            self._instrument_to_name_to_buffer["ask_price"][instrument] = np.zeros(chunksize, dtype=np.float64)
        if self._should_create_buffer("bid_size", instrument, chunksize):
            self._instrument_to_name_to_buffer["bid_size"][instrument] = np.zeros((chunksize, depth), dtype=np.float64)
        if self._should_create_buffer("ask_size", instrument, chunksize):
            self._instrument_to_name_to_buffer["ask_size"][instrument] = np.zeros((chunksize, depth), dtype=np.float64)
        if self._should_create_buffer("tick_size", instrument, chunksize):
            self._instrument_to_name_to_buffer["tick_size"][instrument] = np.zeros(chunksize, dtype=np.float64)

        _quote_interval = recognize_timeframe(self.quote_interval)
        _orderbook_interval = recognize_timeframe(self.orderbook_interval)

        def _iter_chunks():
            try:
                while True:
                    records = self._next_batch(
                        ctx=ctx,
                        instrument=instrument,
                        data_type=data_type,
                        chunksize=chunksize,
                        quote_interval=_quote_interval,
                        orderbook_interval=_orderbook_interval,
                        tick_size_pct=tick_size_pct,
                        depth=depth,
                    )
                    if not records:
                        break
                    transform.start_transform(data_id, names, start=start, stop=stop)
                    transform.process_data(records)
                    yield transform.collect()
            finally:
                self._close_context(instrument)

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

    def _close_context(self, instrument: Instrument):
        if instrument not in self.instrument_to_context:
            return
        ctx = self.instrument_to_context.pop(instrument)
        ctx.close()

    def _next_batch(
        self,
        ctx: IStrategyContext,
        instrument: Instrument,
        data_type: str,
        chunksize: int,
        quote_interval: int,
        orderbook_interval: int,
        tick_size_pct: float,
        depth: int,
    ) -> Iterable | None:
        # - if buffer is empty, read data from ctx
        orderbook_period = int(np.round(orderbook_interval / quote_interval))

        ob_timestamp, bid_price_buffer, ask_price_buffer, tick_size_buffer, bid_size_buffer, ask_size_buffer = (
            _simulate_hft(
                ctx=ctx,
                ob_timestamp=self._instrument_to_name_to_buffer["ob_timestamp"][instrument],
                bid_price_buffer=self._instrument_to_name_to_buffer["bid_price"][instrument],
                ask_price_buffer=self._instrument_to_name_to_buffer["ask_price"][instrument],
                bid_size_buffer=self._instrument_to_name_to_buffer["bid_size"][instrument],
                ask_size_buffer=self._instrument_to_name_to_buffer["ask_size"][instrument],
                tick_size_buffer=self._instrument_to_name_to_buffer["tick_size"][instrument],
                batch_size=chunksize,
                interval=quote_interval,
                orderbook_period=orderbook_period,
                tick_size_pct=tick_size_pct,
                max_levels=depth,
            )
        )

        if len(ob_timestamp) == 0:
            return None

        return zip(ob_timestamp, bid_price_buffer, ask_price_buffer, tick_size_buffer, bid_size_buffer, ask_size_buffer)

    @staticmethod
    def _create_backtest_assets(files: list[str], instrument: Instrument) -> list[BacktestAsset]:
        mid_price = _get_initial_mid_price(files)
        roi_lb, roi_ub = mid_price / 4, mid_price * 4
        return [
            BacktestAsset()
            .data(files)
            .tick_size(instrument.tick_size)
            .lot_size(instrument.lot_size)
            .last_trades_capacity(30000)
            .roi_lb(roi_lb)
            .roi_ub(roi_ub)
        ]

    @staticmethod
    def _get_field_names(data_type: str) -> list[str]:
        # TODO: pass correct field names
        if data_type == "quote":
            return ["timestamp", "bid", "ask", "bid_size", "ask_size"]
        elif data_type == "trade":
            return ["timestamp", "price", "size", "side"]
        elif data_type == "orderbook":
            return ["timestamp", "top_bid", "top_ask", "tick_size", "bids", "asks"]
        else:
            raise ValueError(f"Invalid data type: {data_type}")

    def _should_create_buffer(self, name: str, instrument: Instrument, chunksize: int) -> bool:
        return (
            name not in self._instrument_to_name_to_buffer
            or self._instrument_to_name_to_buffer[name][instrument].shape[0] != chunksize
        )


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
    batch_size: int,
    interval: int = 1_000_000_000,
    orderbook_period: int = 1,
    tick_size_pct: float = 0.0,
    max_levels: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = 0
    ts = 0

    while ctx.elapse(interval) == 0 and index < batch_size:
        if ts % orderbook_period == 0:
            depth = ctx.depth(0)

            ob_timestamp[index] = ctx.current_timestamp

            bid_price_buffer[index] = depth.best_bid
            ask_price_buffer[index] = depth.best_ask

            level_ticks = 1
            if tick_size_pct > 0.0:
                mid_price = (depth.best_bid + depth.best_ask) / 2.0
                level_ticks = max(int(np.round((mid_price * tick_size_pct / 100.0) / depth.tick_size)), 1)

            tick_size_buffer[index] = depth.tick_size * level_ticks

            for i in range(max_levels):
                bid_size_buffer[index, i] = 0
                ask_size_buffer[index, i] = 0
                for j in range(level_ticks):
                    bid_size_buffer[index, i] += depth.bid_qty_at_tick(depth.best_bid_tick - i * level_ticks - j)
                    ask_size_buffer[index, i] += depth.ask_qty_at_tick(depth.best_ask_tick + i * level_ticks + j)

            index += 1

        ts += 1

    return (
        ob_timestamp[:index],
        bid_price_buffer[:index],
        ask_price_buffer[:index],
        tick_size_buffer[:index],
        bid_size_buffer[:index],
        ask_size_buffer[:index],
    )
