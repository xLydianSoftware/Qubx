from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from ccxt.pro import Exchange
from qubx import logger
from qubx.core.basics import DataType, Instrument
from qubx.data.readers import DataReader, DataTransformer
from qubx.data.registry import reader
from qubx.utils.misc import AsyncThreadLoop
from qubx.utils.time import handle_start_stop

from .factory import get_ccxt_exchange
from .utils import ccxt_find_instrument, instrument_to_ccxt_symbol


@reader("ccxt")
class CcxtDataReader(DataReader):
    SUPPORTED_DATA_TYPES = {"ohlc"}

    _exchanges: dict[str, Exchange]
    _loop: AsyncThreadLoop
    _max_bars: int = 10_000
    _max_history: pd.Timedelta

    def __init__(
        self,
        exchanges: list[str],
        max_bars: int = 10_000,
        max_history: str = "30d",
    ):
        _exchanges = [e.upper() for e in exchanges]
        _loop = None
        self._exchanges = {}
        for exchange in _exchanges:
            self._exchanges[exchange] = get_ccxt_exchange(exchange, loop=_loop)
            if _loop is None and hasattr(self._exchanges[exchange], "asyncio_loop"):
                _loop = getattr(self._exchanges[exchange], "asyncio_loop")

        assert _loop is not None, "Loop is not set"
        self._loop = AsyncThreadLoop(_loop)

        self._max_bars = max_bars
        self._max_history = pd.Timedelta(max_history)
        self._exchange_to_symbol_to_instrument = defaultdict(dict)

        _tasks = []
        for exchange in self._exchanges:
            _tasks.append(self._loop.submit(self._exchanges[exchange].load_markets()))
        for task in _tasks:
            task.result()

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        timeframe: str = "1m",
        data_type: str = DataType.OHLC,
        **kwargs,
    ) -> Iterable | list:
        if data_type not in self.SUPPORTED_DATA_TYPES:
            return []

        instrument = self._get_instrument(data_id)
        if instrument is None:
            return []

        _timeframe = pd.Timedelta(timeframe or "1m")
        _start, _stop = self._get_start_stop(start, stop, _timeframe)

        if _start > _stop:
            return []

        ccxt_symbol = instrument_to_ccxt_symbol(instrument)
        exchange = self._exchanges[instrument.exchange]
        data = self._fetch_ohlcv(ccxt_symbol, timeframe, _start, _stop, exchange)
        column_names = self._get_column_names(data_type)

        if not chunksize:
            transform.start_transform(data_id, column_names, start=start, stop=stop)
            transform.process_data(data)
            return transform.collect()

        def _iter_chunks():
            for i in range(0, len(data), chunksize):
                chunk = data[i : i + chunksize]
                transform.start_transform(data_id, column_names, start=start, stop=stop)
                transform.process_data(chunk)
                yield transform.collect()

        return _iter_chunks()

    def get_names(self, **kwargs) -> list[str]:
        return list(self._exchanges.keys())

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        if dtype not in self.SUPPORTED_DATA_TYPES:
            return []

        exchange = exchange.upper()
        if exchange not in self._exchanges:
            return []

        _exchange = self._exchanges[exchange]
        markets = _exchange.markets
        if markets is None:
            return []

        symbols = list(markets.keys())

        instruments = [
            ccxt_find_instrument(symbol, _exchange, self._exchange_to_symbol_to_instrument[exchange])
            for symbol in symbols
        ]
        return [f"{exchange}:{instrument.symbol}" for instrument in instruments]

    def get_time_ranges(self, symbol: str, dtype: str) -> tuple[np.datetime64 | None, np.datetime64 | None]:
        if dtype != "ohlc":
            return None, None

        end_time = pd.Timestamp.now()
        start_time = end_time - self._max_history
        return start_time.to_datetime64(), end_time.to_datetime64()

    def close(self):
        tasks = []
        for exchange in self._exchanges:
            if hasattr(self._exchanges[exchange], "close") and callable(self._exchanges[exchange].close):
                tasks.append(self._loop.submit(self._exchanges[exchange].close()))

        for task in tasks:
            try:
                task.result(timeout=5)  # Wait up to 5 seconds for the connection to close
            except Exception as e:
                logger.error(f"Error closing exchange connection: {e}")

    def _get_instrument(self, data_id: str) -> Instrument | None:
        parts = data_id.split(":")
        if len(parts) != 2:
            return None
        exchange, symbol = parts
        exchange = exchange.upper()
        if exchange not in self._exchanges:
            return None
        return ccxt_find_instrument(symbol, self._exchanges[exchange], self._exchange_to_symbol_to_instrument[exchange])

    def _get_start_stop(
        self, start: str | None, stop: str | None, timeframe: pd.Timedelta
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        if not stop:
            stop = pd.Timestamp.now().isoformat()
        _start, _stop = handle_start_stop(start, stop, convert=lambda x: pd.Timestamp(x))
        assert isinstance(_stop, pd.Timestamp)
        if not _start:
            _start = _stop - timeframe * self._max_bars
        assert isinstance(_start, pd.Timestamp)

        if _start < (_max_time := pd.Timestamp.now() - self._max_history):
            _start = _max_time

        return _start, _stop

    def _fetch_ohlcv(
        self, symbol: str, timeframe: str, start: pd.Timestamp, stop: pd.Timestamp, exchange: Exchange
    ) -> list:
        since = int(start.timestamp() * 1000)
        until = int(stop.timestamp() * 1000)

        future = self._loop.submit(self._async_fetch_ohlcv(symbol, timeframe, since, until, exchange))

        try:
            ohlcv = future.result()
            return [
                [
                    pd.Timestamp(candle[0], unit="ms").to_pydatetime(),
                    float(candle[1]),  # open
                    float(candle[2]),  # high
                    float(candle[3]),  # low
                    float(candle[4]),  # close
                    float(candle[5]),  # volume
                ]
                for candle in ohlcv
            ]
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return []

    async def _async_fetch_ohlcv(self, symbol: str, timeframe: str, since: int, until: int, exchange: Exchange) -> list:
        all_candles = []

        # CCXT has a limit on how many candles can be fetched at once
        # We need to fetch in batches
        limit = 1000
        current_since = since

        while True:
            try:
                candles = await exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)

                if not candles:
                    break

                all_candles.extend(candles)

                # Update the since parameter for the next batch
                last_timestamp = candles[-1][0]

                # If we've reached the until timestamp, stop fetching
                if until and last_timestamp >= until:
                    # Filter out candles after the until timestamp
                    all_candles = [c for c in all_candles if c[0] <= until]
                    break

                # If we got fewer candles than the limit, we've reached the end
                if len(candles) < limit:
                    break

                # Set the since parameter to the timestamp of the last candle + 1
                current_since = last_timestamp + 1

            except Exception as e:
                logger.error(f"Error fetching OHLCV data: {e}")
                break

        return all_candles

    def _get_column_names(self, data_type: str) -> list[str]:
        match data_type:
            case DataType.OHLC:
                return ["timestamp", "open", "high", "low", "close", "volume"]
            case _:
                return []
