import asyncio
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd

from ccxt.pro import Exchange
from qubx import logger
from qubx.core.basics import DataType
from qubx.data.readers import DataReader, DataTransformer
from qubx.data.registry import reader
from qubx.utils.misc import AsyncThreadLoop

from .factory import get_ccxt_exchange
from .utils import ccxt_find_instrument, instrument_to_ccxt_symbol


@reader("ccxt")
class CcxtDataReader(DataReader):
    """
    Data reader for CCXT exchanges.

    This reader can be used to fetch OHLC bars from CCXT exchanges.

    Example:
    ```
    # Using the URI format
    reader = ReaderRegistry.get("ccxt::binance.um")

    # Or directly
    reader = CcxtDataReader("binance.um", max_bars=5000)

    # Then fetch data
    bars = reader.read(
        "BTCUSDT",
        start="2023-01-01 00:00",
        stop="2023-01-02 00:00",
        timeframe="1h"
    )

    # Process data in chunks to reduce memory usage
    chunks = reader.read(
        "BTCUSDT",
        start="2023-01-01 00:00",
        stop="2023-01-02 00:00",
        timeframe="1h",
        chunksize=100  # Process 100 candles at a time
    )

    for chunk in chunks:
        # Process each chunk
        print(f"Got chunk with {len(chunk)} candles")

    # Check available timeframes
    timeframes = reader.get_available_timeframes()
    print(f"Available timeframes: {timeframes}")

    # Get the latest data
    latest_data = reader.get_latest_data("BTCUSDT", timeframe="1h", limit=100)
    print(f"Latest data: {latest_data}")

    # Get default time range (1 month back to now)
    start_time, end_time = reader.get_time_ranges("BTCUSDT", "ohlc")
    print(f"Default time range: {start_time} to {end_time}")
    ```

    The reader supports the following features:
    - Fetching OHLC bars for any symbol supported by the exchange
    - Processing data in chunks to reduce memory usage
    - Transforming data using DataTransformer classes
    - Getting available timeframes for the exchange
    - Getting the latest data for a symbol
    - Providing default time ranges for historical data

    Note:
    - The reader has a limit on how far back in time it can fetch data, controlled by the `max_bars` parameter
      (default: 10,000 bars). Requests for data beyond this limit will return empty data or be adjusted.
    """

    _exchange: Exchange
    _exchange_name: str
    _loop: Optional[AsyncThreadLoop] = None
    _max_bars: int = 10_000

    def __init__(self, exchange_name: str, **kwargs):
        """
        Initialize the CCXT data reader.

        Args:
            exchange_name: The name of the exchange (e.g., "binance.um")
            **kwargs: Additional arguments to pass to the CCXT exchange constructor
        """
        self._exchange_name = exchange_name.upper()
        self._exchange = get_ccxt_exchange(self._exchange_name, **kwargs)
        self._loop = AsyncThreadLoop(self._exchange.asyncio_loop)
        self._max_bars = kwargs.get("max_bars", 10_000)
        self._symbol_to_instrument = {}
        self._loop.submit(self._exchange.load_markets()).result()

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        timeframe: str = "1m",
        **kwargs,
    ) -> Iterable | List:
        """
        Read OHLC bars from the exchange.

        Args:
            data_id: The symbol to fetch data for (e.g., "BTCUSDT")
            start: Start time for the data (e.g., "2023-01-01 00:00")
            stop: End time for the data (e.g., "2023-01-02 00:00")
            transform: Data transformer to apply to the data
            chunksize: Chunk size for the data (if > 0, returns an iterator over chunks)
            timeframe: Timeframe for the OHLC bars (e.g., "1m", "1h", "1d")
            **kwargs: Additional arguments

        Returns:
            List of OHLC bars or transformed data, or an iterator if chunksize > 0
        """
        _data_id = data_id.split(":")[1] if ":" in data_id else data_id
        instrument = ccxt_find_instrument(_data_id, self._exchange, self._symbol_to_instrument)
        data_id = instrument_to_ccxt_symbol(instrument)

        timeframe = timeframe or "1m"

        # Convert start and stop to timestamps if provided
        since = None
        if start:
            since = int(pd.Timestamp(start).timestamp() * 1000)

        until = None
        if stop:
            until = int(pd.Timestamp(stop).timestamp() * 1000)

        # Check if the requested time range exceeds the max_bars limit
        if since is not None:
            # Calculate the time period for each candle based on the timeframe
            timeframe_ms = self._timeframe_to_milliseconds(timeframe)

            # Calculate the maximum time we can go back from current time
            if until:
                max_lookback_time = until - (self._max_bars * timeframe_ms)
            else:
                current_time = int(pd.Timestamp.now().timestamp() * 1000)
                max_lookback_time = current_time - (self._max_bars * timeframe_ms)

            # If the requested start time is earlier than the max lookback time,
            # log a warning and return empty data
            if since < max_lookback_time:
                logger.debug(
                    f"Requested start time {pd.Timestamp(since, unit='ms')} is earlier than "
                    f"the maximum lookback time {pd.Timestamp(max_lookback_time, unit='ms')} "
                    f"(limited by max_bars={self._max_bars}). Returning empty data."
                )
                # Apply the transformer with empty data
                column_names = ["timestamp", "open", "high", "low", "close", "volume"]
                transform.start_transform(data_id, column_names, start, stop)
                return transform.collect()

        # Apply the transformer
        column_names = ["timestamp", "open", "high", "low", "close", "volume"]
        transform.start_transform(data_id, column_names, start, stop)

        # If chunksize is specified, process data in chunks
        if chunksize > 0:
            return self._read_in_chunks(data_id, timeframe, since, until, transform, chunksize)

        # Otherwise fetch and process all data at once
        ohlcv = self._fetch_ohlcv(data_id, timeframe, since, until)

        # Convert to a list of lists for the transformer
        data = [
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

        if data:
            transform.process_data(data)

        return transform.collect()

    def _read_in_chunks(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int],
        until: Optional[int],
        transform: DataTransformer,
        chunksize: int,
    ) -> Iterable:
        """
        Read data in chunks directly from the exchange when possible.

        Args:
            symbol: The symbol to fetch data for
            timeframe: Timeframe for the OHLC bars
            since: Start time in milliseconds
            until: End time in milliseconds
            transform: The transformer to apply
            chunksize: The size of each chunk

        Returns:
            An iterator over the transformed chunks
        """
        # For CCXT, we need to determine the time period for each chunk
        # based on the timeframe and chunksize

        # Calculate the time period for each candle based on the timeframe
        timeframe_ms = self._timeframe_to_milliseconds(timeframe)

        # If we don't have a start time, we can't chunk by time
        if since is None:
            logger.info(f"No start time provided for {symbol}, fetching all data and chunking in memory")
            # Fetch all data and process in memory chunks
            ohlcv = self._fetch_ohlcv(symbol, timeframe, since, until)
            logger.info(f"Fetched {len(ohlcv)} candles for {symbol}")

            data = [
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
            return self._process_in_chunks(data, transform, chunksize)

        # Calculate the time period for each chunk
        chunk_period_ms = timeframe_ms * chunksize

        # Calculate the maximum time we can go back from current time
        if until:
            max_lookback_time = until - (self._max_bars * timeframe_ms)
        else:
            current_time = int(pd.Timestamp.now().timestamp() * 1000)
            max_lookback_time = current_time - (self._max_bars * timeframe_ms)

        # If the requested start time is earlier than the max lookback time,
        # adjust it to the max lookback time
        if since < max_lookback_time:
            logger.debug(
                f"Requested start time {pd.Timestamp(since, unit='ms')} is earlier than "
                f"the maximum lookback time {pd.Timestamp(max_lookback_time, unit='ms')} "
                f"(limited by max_bars={self._max_bars}). Adjusting start time."
            )
            since = max_lookback_time

        # Log the chunking parameters
        start_time = pd.Timestamp(since, unit="ms")
        end_time = pd.Timestamp(until, unit="ms") if until else "now"
        chunk_period = pd.Timedelta(milliseconds=chunk_period_ms)
        logger.debug(
            f"Fetching {symbol} data from {start_time} to {end_time} in chunks of {chunksize} {timeframe} candles ({chunk_period})"
        )

        # Create a generator that yields chunks of data
        def chunk_generator():
            current_since = since
            chunk_num = 1

            while True:
                # Calculate the end time for this chunk
                chunk_until = current_since + chunk_period_ms

                # If we've reached the until time, stop
                if until is not None and current_since >= until:
                    logger.info(f"Reached end time for {symbol}, stopping")
                    break

                # If the chunk end time is beyond the until time, cap it
                if until is not None and chunk_until > until:
                    chunk_until = until

                # Log the chunk we're fetching
                chunk_start = pd.Timestamp(current_since, unit="ms")
                chunk_end = pd.Timestamp(chunk_until, unit="ms")
                logger.debug(f"Fetching chunk {chunk_num} for {symbol}: {chunk_start} to {chunk_end}")

                # Fetch the chunk
                chunk_ohlcv = self._fetch_ohlcv(symbol, timeframe, current_since, chunk_until)

                # If no data was returned, we're done
                if not chunk_ohlcv:
                    logger.debug(f"No data returned for chunk {chunk_num} for {symbol}, stopping")
                    break

                logger.info(f"Fetched {len(chunk_ohlcv)} candles for chunk {chunk_num} for {symbol}")

                # Convert to the format expected by the transformer
                chunk_data = [
                    [
                        pd.Timestamp(candle[0], unit="ms").to_pydatetime(),
                        float(candle[1]),  # open
                        float(candle[2]),  # high
                        float(candle[3]),  # low
                        float(candle[4]),  # close
                        float(candle[5]),  # volume
                    ]
                    for candle in chunk_ohlcv
                ]

                # If no data was returned, we're done
                if not chunk_data:
                    logger.info(f"No data after conversion for chunk {chunk_num} for {symbol}, stopping")
                    break

                # Create a copy of the transformer for this chunk
                chunk_transform = type(transform)()
                chunk_transform.__dict__.update(transform.__dict__)

                # Process the chunk
                chunk_transform.process_data(chunk_data)

                # Yield the transformed chunk
                result = chunk_transform.collect()
                logger.info(f"Processed chunk {chunk_num} for {symbol}")
                yield result

                # Update the since time for the next chunk
                # Use the timestamp of the last candle + 1 to avoid duplicates
                current_since = chunk_ohlcv[-1][0] + 1

                # If we've reached the until time, stop
                if until is not None and current_since >= until:
                    logger.info(f"Reached end time after chunk {chunk_num} for {symbol}, stopping")
                    break

                chunk_num += 1

        return chunk_generator()

    def _timeframe_to_milliseconds(self, timeframe: str) -> int:
        """
        Convert a timeframe string to milliseconds.

        Args:
            timeframe: The timeframe string (e.g., "1m", "1h", "1d")

        Returns:
            The timeframe in milliseconds
        """
        # Parse the timeframe string
        if timeframe.endswith("m"):
            return int(timeframe[:-1]) * 60 * 1000
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 60 * 60 * 1000
        elif timeframe.endswith("d"):
            return int(timeframe[:-1]) * 24 * 60 * 60 * 1000
        elif timeframe.endswith("w"):
            return int(timeframe[:-1]) * 7 * 24 * 60 * 60 * 1000
        elif timeframe.endswith("M"):
            return int(timeframe[:-1]) * 30 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

    def _process_in_chunks(self, data: List, transform: DataTransformer, chunksize: int) -> Iterable:
        """
        Process data in chunks.

        Args:
            data: The data to process
            transform: The transformer to apply
            chunksize: The size of each chunk

        Returns:
            An iterator over the transformed chunks
        """
        from qubx.data.readers import _list_to_chunked_iterator

        # Create a generator that yields chunks of data
        def chunk_generator():
            for chunk in _list_to_chunked_iterator(data, chunksize):
                # Create a copy of the transformer for each chunk
                chunk_transform = type(transform)()
                chunk_transform.__dict__.update(transform.__dict__)

                # Process the chunk
                chunk_transform.process_data(chunk)

                # Yield the transformed chunk
                yield chunk_transform.collect()

        return chunk_generator()

    def _fetch_ohlcv(
        self, symbol: str, timeframe: str, since: Optional[int] = None, until: Optional[int] = None
    ) -> List:
        """
        Fetch OHLC bars from the exchange.

        Args:
            symbol: The symbol to fetch data for
            timeframe: Timeframe for the OHLC bars
            since: Start time in milliseconds
            until: End time in milliseconds

        Returns:
            List of OHLC bars
        """
        if self._loop is None:
            return []

        future = asyncio.run_coroutine_threadsafe(
            self._async_fetch_ohlcv(symbol, timeframe, since, until), self._loop.loop
        )

        try:
            return future.result()
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return []

    async def _async_fetch_ohlcv(
        self, symbol: str, timeframe: str, since: Optional[int] = None, until: Optional[int] = None
    ) -> List:
        """
        Asynchronously fetch OHLC bars from the exchange.

        Args:
            symbol: The symbol to fetch data for
            timeframe: Timeframe for the OHLC bars
            since: Start time in milliseconds
            until: End time in milliseconds

        Returns:
            List of OHLC bars
        """
        all_candles = []

        # CCXT has a limit on how many candles can be fetched at once
        # We need to fetch in batches
        limit = 1000
        current_since = since

        while True:
            try:
                candles = await self._exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)

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

    def get_names(self, **kwargs) -> List[str]:
        """
        Get the list of available symbols.

        Returns:
            List of available symbols
        """
        if self._loop is None:
            return []

        future = asyncio.run_coroutine_threadsafe(self._exchange.load_markets(), self._loop.loop)

        try:
            markets = future.result()
            return list(markets.keys())
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            return []

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        """
        Get the list of available symbols for the given exchange and data type.

        Args:
            exchange: The exchange name
            dtype: The data type

        Returns:
            List of available symbols
        """
        if exchange.lower() != self._exchange_name.lower():
            return []

        ccxt_symbols = self.get_names()
        instruments = [
            ccxt_find_instrument(symbol, self._exchange, self._symbol_to_instrument) for symbol in ccxt_symbols
        ]
        return [f"{self._exchange_name}:{instrument.symbol}" for instrument in instruments]

    def get_time_ranges(self, symbol: str, dtype: str) -> tuple[np.datetime64 | None, np.datetime64 | None]:
        """
        Get the time range for the given symbol and data type.

        Args:
            symbol: The symbol
            dtype: The data type

        Returns:
            Tuple of (start_time, end_time)

        Note:
            Since CCXT doesn't provide a way to get the time range for a symbol,
            this method returns a default range of 1 month back to the current time.
        """
        if dtype != DataType.OHLC:
            return None, None

        # Get current time
        end_time = pd.Timestamp.now()

        # Get start time (1 month back)
        start_time = end_time - pd.Timedelta(days=30)

        # Convert to numpy datetime64
        return start_time.to_datetime64(), end_time.to_datetime64()

    def get_available_timeframes(self) -> list[str]:
        """
        Get the list of available timeframes for the exchange.

        Returns:
            List of available timeframes
        """
        if self._loop is None or not hasattr(self._exchange, "timeframes"):
            return []

        # Some exchanges provide a list of available timeframes
        if hasattr(self._exchange, "timeframes") and self._exchange.timeframes:
            return list(self._exchange.timeframes)

        # Default timeframes that most exchanges support
        return ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M"]

    def get_latest_data(
        self, symbol: str, timeframe: str = "1m", limit: int = 100, transform: DataTransformer = DataTransformer()
    ) -> Any:
        """
        Get the most recent data for a symbol.

        Args:
            symbol: The symbol to fetch data for
            timeframe: Timeframe for the OHLC bars
            limit: Maximum number of candles to fetch
            transform: Data transformer to apply to the data

        Returns:
            The most recent data for the symbol
        """
        # Ensure limit doesn't exceed max_bars
        if limit > self._max_bars:
            logger.warning(
                f"Requested limit {limit} exceeds max_bars {self._max_bars}. Limiting to {self._max_bars} candles."
            )
            limit = self._max_bars

        logger.info(f"Fetching latest {limit} candles for {symbol} with timeframe {timeframe}")

        # Check if the loop is available
        if self._loop is None:
            logger.error("AsyncThreadLoop is not initialized")
            return []

        # Fetch the latest candles
        future = asyncio.run_coroutine_threadsafe(
            self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit), self._loop.loop
        )

        try:
            ohlcv = future.result()
            logger.info(f"Fetched {len(ohlcv)} candles for {symbol}")

            # Convert to a list of lists for the transformer
            data = [
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

            # Apply the transformer
            column_names = ["timestamp", "open", "high", "low", "close", "volume"]
            transform.start_transform(symbol, column_names, None, None)

            if data:
                transform.process_data(data)

            return transform.collect()

        except Exception as e:
            logger.error(f"Error fetching latest data for {symbol}: {e}")
            return []

    def close(self):
        """
        Close the CCXT exchange connection.
        """
        # AsyncThreadLoop doesn't have a close method
        # We need to close the exchange connection
        if hasattr(self, "_exchange") and self._exchange and hasattr(self, "_loop") and self._loop:
            # Close the exchange connection if it has a close method
            if hasattr(self._exchange, "close") and callable(self._exchange.close):
                try:
                    future = self._loop.submit(self._exchange.close())
                    future.result(timeout=5)  # Wait up to 5 seconds for the connection to close
                except Exception as e:
                    logger.error(f"Error closing exchange connection: {e}")
