"""XLighter data reader for historical data fetching"""

import asyncio
from typing import Iterable, cast

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import DataType, Instrument, MarketType
from qubx.core.lookups import lookup
from qubx.data.readers import DataReader, DataTransformer
from qubx.data.registry import reader
from qubx.utils.misc import AsyncThreadLoop
from qubx.utils.time import handle_start_stop, now_utc

from .client import LighterClient
from .utils import get_market_id


@reader("xlighter")
class LighterReader(DataReader):
    """
    Data reader for Lighter exchange.

    Fetches historical OHLC data and funding payments via Lighter REST API.
    Uses AsyncThreadLoop pattern for async API calls.

    Supported data types:
        - ohlc: Historical candlestick data
        - funding_payment: Historical funding payment data

    Note:
        This reader requires a pre-configured LighterClient to be passed in.
        It will share the client's event loop for efficient resource usage.
    """

    SUPPORTED_DATA_TYPES = {"ohlc", "funding_payment"}

    def __init__(
        self,
        client: LighterClient,
        max_bars: int = 10_000,
        max_history: str = "30d",
    ):
        """
        Initialize XLighter data reader.

        Args:
            client: Pre-configured LighterClient instance
            max_bars: Maximum bars to fetch per request (default: 10,000)
            max_history: Maximum historical data to fetch (default: 30d)
        """
        self.client = client
        self._loop = client._loop
        self._async_loop = AsyncThreadLoop(self._loop)
        self._max_bars = max_bars
        self._max_history = pd.Timedelta(max_history)
        self._info("Initialized data reader")

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
            self._warning(f"Instrument not found: {data_id}")
            return []

        timeframe = timeframe or "1m"
        _timeframe = cast(pd.Timedelta, pd.Timedelta(timeframe))
        _start, _stop = self._get_start_stop(start, stop, _timeframe)

        if _start > _stop:
            return []

        data = self._fetch_data(instrument, data_type, timeframe, _start, _stop)
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

    def get_candles(
        self,
        exchange: str,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        stop: str | pd.Timestamp | None = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        if exchange not in self.get_names():
            return pd.DataFrame()

        start_ts, stop_ts = self._get_start_stop(start, stop, cast(pd.Timedelta, pd.Timedelta(timeframe)))

        instruments_to_fetch = self._get_instruments_for_symbols(symbols)
        if not instruments_to_fetch:
            self._warning("No instruments found for the specified symbols")
            return pd.DataFrame()

        self._info(f"Fetching candle data for {len(instruments_to_fetch)} symbols from {start_ts} to {stop_ts}")

        future = self._async_loop.submit(
            self._async_fetch_candles_for_all_instruments(instruments_to_fetch, timeframe, start_ts, stop_ts)
        )

        all_candle_data = future.result()

        if not all_candle_data:
            self._info("No candle data found")
            return pd.DataFrame()

        df = pd.DataFrame(all_candle_data)
        df = df.sort_values("timestamp")
        df = df.set_index(["timestamp", "symbol"])

        self._info(f"Fetched {len(df)} candle records for {len(instruments_to_fetch)} symbols")
        return df

    def get_funding_payment(
        self,
        exchange: str,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        stop: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if exchange not in self.get_names():
            return pd.DataFrame()

        start_ts, stop_ts = self._get_start_stop(start, stop, cast(pd.Timedelta, pd.Timedelta("1h")))
        self._debug(f"Fetching funding data for {symbols} from {start_ts} to {stop_ts}")

        start_ts = start_ts.floor("1h")

        instruments_to_fetch = self._get_instruments_for_symbols(symbols)
        if not instruments_to_fetch:
            self._warning("No instruments found for the specified symbols")
            return pd.DataFrame()

        future = self._async_loop.submit(
            self._async_fetch_funding_for_all_instruments(instruments_to_fetch, start_ts, stop_ts)
        )

        all_funding_data = future.result()

        if not all_funding_data:
            self._info("No funding payment data found")
            return pd.DataFrame()

        df = pd.DataFrame(all_funding_data)
        df = df.sort_values("timestamp")
        df = df.set_index(["timestamp", "symbol"])

        if len(instruments_to_fetch) > 5:
            self._info(f"Fetched {len(df)} funding payment records for {len(instruments_to_fetch)} symbols")
        else:
            self._info(f"Fetched {len(df)} funding payment records for {', '.join(symbols or [])}")

        return df

    def get_names(self, **kwargs) -> list[str]:
        return ["LIGHTER"]

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        if dtype not in self.SUPPORTED_DATA_TYPES:
            return []
        if exchange.upper() != "LIGHTER":
            return []
        instruments = lookup.find_instruments(exchange="LIGHTER", market_type=MarketType.SWAP)
        return [instrument.symbol for instrument in instruments]

    def get_time_ranges(self, symbol: str, dtype: str) -> tuple[np.datetime64 | None, np.datetime64 | None]:
        if dtype not in self.SUPPORTED_DATA_TYPES:
            return None, None
        end_time = now_utc()
        start_time = end_time - self._max_history
        return start_time.to_datetime64(), end_time.to_datetime64()

    def close(self):
        logger.debug("<yellow>[Lighter]</yellow> closed")

    def _fetch_data(
        self, instrument: Instrument, data_type: str, timeframe: str, start: pd.Timestamp, stop: pd.Timestamp
    ) -> list[tuple]:
        match data_type:
            case DataType.OHLC:
                return self._fetch_ohlcv(instrument, timeframe, start, stop)
            case DataType.FUNDING_PAYMENT:
                return self._fetch_fundings(instrument, start, stop)
            case _:
                raise ValueError(f"Unsupported data type: {data_type}")

    async def _async_fetch_funding_for_all_instruments(
        self, instruments: list[Instrument], start: pd.Timestamp, stop: pd.Timestamp
    ) -> list[dict]:
        coroutines = [self._async_fetch_fundings_single(instrument, start, stop) for instrument in instruments]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        all_funding_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._warning(f"Failed to fetch funding data for {instruments[i].symbol}: {result}")
                continue

            assert isinstance(result, list)
            for timestamp, funding_rate, funding_interval_hours in result:
                all_funding_data.append(
                    {
                        "timestamp": timestamp,
                        "symbol": instruments[i].symbol,
                        "funding_rate": funding_rate,
                        "funding_interval_hours": funding_interval_hours,
                    }
                )

        return all_funding_data

    async def _async_fetch_candles_for_all_instruments(
        self,
        instruments: list[Instrument],
        timeframe: str,
        start: pd.Timestamp,
        stop: pd.Timestamp,
    ) -> list[dict]:
        coroutines = [self._async_fetch_ohlcv(instrument, timeframe, start, stop) for instrument in instruments]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        all_candle_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._warning(f"Failed to fetch candle data for {instruments[i].symbol}: {result}")
                continue

            assert isinstance(result, list)
            for timestamp, open_price, high, low, close, volume, quote_volume in result:
                all_candle_data.append(
                    {
                        "timestamp": timestamp,
                        "symbol": instruments[i].symbol,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "quote_volume": quote_volume,
                    }
                )

        return all_candle_data

    def _get_instrument(self, data_id: str) -> Instrument | None:
        parts = data_id.split(":")
        if len(parts) < 2:
            return None

        exchange, symbol = parts[0], parts[-1]
        if exchange.upper() != "LIGHTER":
            return None

        return lookup.find_symbol(exchange.upper(), symbol.upper())

    def _get_instruments_for_symbols(self, symbols: list[str] | None) -> list[Instrument]:
        if symbols is None:
            # Return all instruments
            return lookup.find_instruments(exchange="LIGHTER", market_type=MarketType.SWAP)

        instruments = []
        for symbol in symbols:
            instrument = lookup.find_symbol(exchange="LIGHTER", symbol=symbol.upper())
            if instrument:
                instruments.append(instrument)

        return instruments

    def _get_start_stop(
        self, start: str | pd.Timestamp | None, stop: str | pd.Timestamp | None, timeframe: pd.Timedelta
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        if not stop:
            stop = now_utc().isoformat()
        _start, _stop = handle_start_stop(start, stop, convert=lambda x: pd.Timestamp(x))
        assert isinstance(_stop, pd.Timestamp)

        if not _start:
            _start = _stop - timeframe * self._max_bars
        assert isinstance(_start, pd.Timestamp)

        if _start < (_max_time := now_utc() - self._max_history):
            _start = _max_time

        _start = cast(pd.Timestamp, _start)
        _stop = cast(pd.Timestamp, _stop)
        return _start, _stop

    async def _async_fetch_ohlcv(
        self,
        instrument: Instrument,
        timeframe: str,
        start: pd.Timestamp,
        stop: pd.Timestamp,
    ) -> list[tuple]:
        since = int(start.timestamp() * 1000)
        until = int(stop.timestamp() * 1000)

        # Get market ID
        try:
            market_id = get_market_id(instrument)
        except ValueError:
            self._error(f"Market ID not found for {instrument.symbol}")
            return []

        try:
            # Fetch candlesticks via async API
            candlesticks = await self.client.get_candlesticks(
                market_id=market_id,
                resolution=timeframe,
                start_timestamp=since,
                end_timestamp=until,
                count_back=self._max_bars,
            )

            # Convert to Qubx format
            # Lighter format: timestamp, open, high, low, close, volume0, volume1
            # Qubx format: timestamp, open, high, low, close, volume
            ohlcv_data = []
            for candle in candlesticks:
                timestamp = pd.Timestamp(candle["timestamp"], unit="ms").to_pydatetime()
                ohlcv_data.append(
                    [
                        timestamp,
                        float(candle["open"]),
                        float(candle["high"]),
                        float(candle["low"]),
                        float(candle["close"]),
                        float(candle.get("volume0", 0.0)),  # base volume
                        float(candle.get("volume1", 0.0)),  # quote volume
                    ]
                )

            return ohlcv_data

        except Exception as e:
            self._error(f"Error fetching OHLCV data for {instrument.symbol}: {e}")
            return []

    def _fetch_ohlcv(
        self,
        instrument: Instrument,
        timeframe: str,
        start: pd.Timestamp,
        stop: pd.Timestamp,
    ) -> list[tuple]:
        future = self._async_loop.submit(self._async_fetch_ohlcv(instrument, timeframe, start, stop))
        return future.result()

    async def _async_fetch_fundings_single(
        self, instrument: Instrument, start: pd.Timestamp, stop: pd.Timestamp
    ) -> list[tuple]:
        since = int(start.timestamp() * 1000)
        until = int(stop.timestamp() * 1000)

        try:
            market_id = get_market_id(instrument)
        except ValueError:
            self._error(f"Market ID not found for {instrument.symbol}")
            return []

        try:
            # Fetch funding data via async API
            fundings = await self.client.get_fundings(
                market_id=market_id,
                resolution="1h",  # Lighter uses 1-hour funding
                start_timestamp=since,
                end_timestamp=until,
                count_back=self._max_bars,
            )

            # Convert to our format
            funding_data = []
            for funding_item in fundings:
                # Lighter returns timestamps in seconds, not milliseconds
                timestamp = pd.Timestamp(funding_item["timestamp"], unit="s").to_pydatetime()

                # Filter by stop time
                if timestamp > stop:
                    continue

                side = funding_item.get("direction", None)
                if side is None:
                    continue

                # Extract funding rate - Lighter uses 'rate' field
                # The rate is in percentage, so we need to divide by 100.0 to get the actual rate
                funding_rate = float(funding_item.get("rate", 0.0)) / 100.0 * (1 if side == "long" else -1)

                funding_data.append(
                    (
                        timestamp,
                        float(funding_rate),
                        1.0,  # Lighter uses 1-hour funding
                    )
                )

            return funding_data

        except Exception as e:
            self._error(f"Error fetching funding data for {instrument.symbol}: {e}")
            return []

    def _fetch_fundings(self, instrument: Instrument, start: pd.Timestamp, stop: pd.Timestamp) -> list[tuple]:
        future = self._async_loop.submit(self._async_fetch_fundings_single(instrument, start, stop))
        return future.result()

    def _get_column_names(self, data_type: str) -> list[str]:
        match data_type:
            case DataType.OHLC:
                return ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
            case DataType.FUNDING_PAYMENT:
                return ["timestamp", "funding_rate", "funding_interval_hours"]
            case _:
                return []

    def _info(self, message: str) -> None:
        logger.info(f"<yellow>[Lighter]</yellow> {message}")

    def _debug(self, message: str) -> None:
        logger.debug(f"<yellow>[Lighter]</yellow> {message}")

    def _warning(self, message: str) -> None:
        logger.warning(f"<yellow>[Lighter]</yellow> {message}")

    def _error(self, message: str) -> None:
        logger.error(f"<yellow>[Lighter]</yellow> {message}")
