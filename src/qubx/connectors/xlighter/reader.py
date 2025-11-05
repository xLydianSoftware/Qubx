"""XLighter data reader for historical data fetching"""

import asyncio
from typing import Iterable, cast

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import DataType, Instrument
from qubx.core.lookups import lookup
from qubx.data.readers import DataReader, DataTransformer
from qubx.data.registry import reader
from qubx.utils.misc import AsyncThreadLoop
from qubx.utils.time import handle_start_stop, now_utc

from .client import LighterClient
from .instruments import LighterInstrumentLoader


@reader("xlighter")
class XLighterDataReader(DataReader):
    """
    Data reader for XLighter (Lighter) exchange.

    Fetches historical OHLC data and funding payments via Lighter REST API.
    Uses AsyncThreadLoop pattern for async API calls.

    Supported data types:
        - ohlc: Historical candlestick data

    Note:
        This reader requires a pre-configured LighterClient to be passed in.
        It will share the client's event loop for efficient resource usage.

    Example:
        ```python
        # Create client first (from factory)
        client = get_xlighter_client(
            api_key="0xYourAddress",
            secret="0xYourPrivateKey",
            account_index=225671,
            api_key_index=2
        )

        # Create reader with client
        reader = XLighterDataReader(
            client=client,
            max_history="30d"
        )
        ```
    """

    SUPPORTED_DATA_TYPES = {"ohlc"}

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
        # Store client
        self.client = client

        # Use client's event loop (shared resource)
        self._loop = client._loop
        self._async_loop = AsyncThreadLoop(self._loop)

        # Create and load instrument loader as a proper task (required by aiohttp)
        import concurrent.futures

        init_future = concurrent.futures.Future()

        def create_init_task():
            """Create init task in the event loop"""
            task = asyncio.create_task(self._async_init())
            task.add_done_callback(
                lambda t: init_future.set_result(t.result())
                if not t.exception()
                else init_future.set_exception(t.exception())
            )

        self._loop.call_soon_threadsafe(create_init_task)
        self.instrument_loader = init_future.result()

        self._max_bars = max_bars
        self._max_history = pd.Timedelta(max_history)

        logger.info(
            f"XLighterDataReader initialized: {len(self.instrument_loader.instruments)} instruments loaded, "
            f"max_history={max_history}"
        )

    async def _async_init(self):
        """
        Initialize instrument loader in async context.

        Returns:
            LighterInstrumentLoader instance
        """
        # Create and load instruments
        instrument_loader = LighterInstrumentLoader(self.client)
        await instrument_loader.load_instruments()

        return instrument_loader

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
        """
        Read historical data for an instrument.

        Args:
            data_id: Data identifier in format "LIGHTER:SWAP:BTCUSDC"
            start: Start time (ISO format or timestamp)
            stop: Stop time (ISO format or timestamp)
            transform: Data transformer
            chunksize: Chunk size for iteration (0 = no chunking)
            timeframe: Timeframe (e.g., "1m", "5m", "1h", "1d")
            data_type: Data type (only "ohlc" supported)
            **kwargs: Additional parameters

        Returns:
            Iterable or list of data
        """
        if data_type not in self.SUPPORTED_DATA_TYPES:
            return []

        instrument = self._get_instrument(data_id)
        if instrument is None:
            logger.warning(f"Instrument not found: {data_id}")
            return []

        timeframe = timeframe or "1m"
        _timeframe = pd.Timedelta(timeframe)
        _start, _stop = self._get_start_stop(start, stop, _timeframe)

        if _start > _stop:
            return []

        # Fetch OHLC data
        data = self._fetch_ohlcv(instrument, timeframe, _start, _stop)
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
        """
        Get list of exchange names.

        Returns:
            List containing ["LIGHTER"]
        """
        return ["LIGHTER"]

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        """
        Get list of available symbols for exchange.

        Args:
            exchange: Exchange name (should be "LIGHTER")
            dtype: Data type (should be "ohlc")

        Returns:
            List of symbol identifiers in format "LIGHTER:SWAP:BTCUSDC"
        """
        if dtype not in self.SUPPORTED_DATA_TYPES:
            return []

        if exchange.upper() != "LIGHTER":
            return []

        # Return all instrument IDs
        return list(self.instrument_loader.instruments.keys())

    def get_time_ranges(self, symbol: str, dtype: str) -> tuple[np.datetime64 | None, np.datetime64 | None]:
        """
        Get available time range for a symbol.

        Args:
            symbol: Symbol identifier
            dtype: Data type

        Returns:
            Tuple of (start_time, end_time) as numpy datetime64
        """
        if dtype != "ohlc":
            return None, None

        end_time = now_utc()
        start_time = end_time - self._max_history
        return start_time.to_datetime64(), end_time.to_datetime64()

    def close(self):
        """
        Close the reader and release resources.

        Note: Does not close the client or stop the event loop since they are shared.
        The client should be managed by the component that created it.
        """
        logger.debug("XLighterDataReader closed")

    def get_funding_payment(
        self,
        exchange: str,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        stop: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Get funding payment data for symbols.

        Args:
            exchange: Exchange name (should be "LIGHTER")
            symbols: List of symbols in Qubx format (e.g., ["BTCUSDC", "ETHUSDC"])
            start: Start time
            stop: Stop time

        Returns:
            DataFrame with MultiIndex [timestamp, symbol] and columns:
                - funding_rate: Funding rate value
                - funding_interval_hours: Funding interval (1.0 for Lighter)
        """
        if exchange.upper() != "LIGHTER":
            return pd.DataFrame(columns=["funding_rate", "funding_interval_hours"])  # type: ignore

        # Handle time range
        start_ts = pd.Timestamp(start) if start else pd.Timestamp.now() - pd.Timedelta(days=7)
        stop_ts = pd.Timestamp(stop) if stop else pd.Timestamp.now()

        start_ts = cast(pd.Timestamp, start_ts)
        stop_ts = cast(pd.Timestamp, stop_ts)

        # Apply max_history limitation
        if self._max_history:
            max_history_start = stop_ts - self._max_history
            if start_ts < max_history_start:
                logger.debug(
                    f"Adjusting start time from {start_ts} to {max_history_start} "
                    f"due to max_history={self._max_history}"
                )
                start_ts = max_history_start

        # Convert to milliseconds
        since = int(start_ts.timestamp() * 1000)
        until = int(stop_ts.timestamp() * 1000)

        # Get instruments to fetch
        instruments_to_fetch = self._get_instruments_for_symbols(symbols)
        if not instruments_to_fetch:
            logger.warning("No instruments found for the specified symbols")
            return pd.DataFrame(columns=["funding_rate", "funding_interval_hours"])

        # Fetch funding data for each instrument
        all_funding_data = []

        for instrument in instruments_to_fetch:
            try:
                market_id = self.instrument_loader.get_market_id(instrument.symbol)
                if market_id is None:
                    logger.warning(f"Market ID not found for {instrument.symbol}")
                    continue

                # Fetch funding data via async API
                future = self._async_loop.submit(
                    self.client.get_fundings(
                        market_id=market_id,
                        resolution="1h",  # Lighter uses 1-hour funding
                        start_timestamp=since,
                        end_timestamp=until,
                        count_back=self._max_bars,
                    )
                )

                fundings = future.result()

                # Convert to our format
                for funding_item in fundings:
                    # Lighter returns timestamps in seconds, not milliseconds
                    timestamp = pd.Timestamp(funding_item["timestamp"], unit="s")

                    # Filter by stop time
                    if timestamp > stop_ts:
                        continue

                    # Extract funding rate - Lighter uses 'rate' field
                    # The rate is in percentage, so we need to divide by 100.0 to get the actual rate
                    funding_rate = funding_item.get("rate", 0.0) / 100.0

                    all_funding_data.append(
                        {
                            "timestamp": timestamp,
                            "symbol": instrument.symbol,
                            "funding_rate": float(funding_rate),
                            "funding_interval_hours": 1.0,  # Lighter uses 1-hour funding
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to fetch funding data for {instrument.symbol}: {e}")
                continue

        if not all_funding_data:
            logger.info("No funding payment data found")
            return pd.DataFrame(columns=["funding_rate", "funding_interval_hours"])

        # Create DataFrame
        df = pd.DataFrame(all_funding_data)
        df = df.sort_values("timestamp")
        df = df.set_index(["timestamp", "symbol"])

        logger.info(f"Fetched {len(df)} funding payment records for {len(instruments_to_fetch)} symbols")
        return df

    def get_candles(
        self,
        exchange: str,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        stop: str | pd.Timestamp | None = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        Get candlestick data for symbols within specified time range.

        Args:
            exchange: Exchange name (should be "LIGHTER")
            symbols: List of symbols in Qubx format (e.g., ["BTCUSDC", "ETHUSDC"]). If None, fetches all symbols.
            start: Start time (ISO format or timestamp)
            stop: Stop time (ISO format or timestamp)
            timeframe: Timeframe for candles (e.g., "1m", "5m", "1h", "1d")

        Returns:
            DataFrame with MultiIndex [timestamp, symbol] and columns:
                - open: Opening price
                - high: Highest price
                - low: Lowest price
                - close: Closing price
                - volume: Trading volume
        """
        if exchange.upper() != "LIGHTER":
            logger.warning(f"Exchange {exchange} not supported by XLighterDataReader")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Handle time range
        start_ts = pd.Timestamp(start) if start else pd.Timestamp.now() - pd.Timedelta(days=7)
        stop_ts = pd.Timestamp(stop) if stop else pd.Timestamp.now()

        # Apply max_history limitation
        if self._max_history:
            max_history_start = stop_ts - self._max_history
            if start_ts < max_history_start:
                logger.debug(
                    f"Adjusting start time from {start_ts} to {max_history_start} "
                    f"due to max_history={self._max_history}"
                )
                start_ts = max_history_start

        # Get instruments to fetch
        instruments_to_fetch = self._get_instruments_for_symbols(symbols)
        if not instruments_to_fetch:
            logger.warning("No instruments found for the specified symbols")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Fetch candle data for each instrument
        all_candle_data = []

        for instrument in instruments_to_fetch:
            try:
                # Fetch OHLCV data using existing method
                ohlcv_list = self._fetch_ohlcv(instrument, timeframe, start_ts, stop_ts)

                if not ohlcv_list:
                    logger.debug(f"No candle data found for {instrument.symbol}")
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(ohlcv_list, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["symbol"] = instrument.symbol
                all_candle_data.append(df)

            except Exception as e:
                logger.error(f"Failed to fetch candle data for {instrument.symbol}: {e}")
                continue

        if not all_candle_data:
            logger.info("No candle data found")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Combine all DataFrames
        combined_df = pd.concat(all_candle_data, ignore_index=True)
        combined_df = combined_df.sort_values("timestamp")
        combined_df = combined_df.set_index(["timestamp", "symbol"])

        logger.info(f"Fetched {len(combined_df)} candle records for {len(instruments_to_fetch)} symbols")
        return combined_df

    def _get_instrument(self, data_id: str) -> Instrument | None:
        """
        Get instrument from data ID.

        Args:
            data_id: Data ID in format "LIGHTER:SWAP:BTCUSDC" or "LIGHTER:BTCUSDC

        Returns:
            Instrument object or None
        """
        parts = data_id.split(":")
        if len(parts) < 2:
            return None

        exchange, symbol = parts[0], parts[-1]
        if exchange.upper() != "LIGHTER":
            return None

        return lookup.find_symbol(exchange.upper(), symbol.upper())

    def _get_instruments_for_symbols(self, symbols: list[str] | None) -> list[Instrument]:
        """
        Get instruments for symbols.

        Args:
            symbols: List of symbols in Qubx format or None for all

        Returns:
            List of Instrument objects
        """
        if symbols is None:
            # Return all instruments
            return list(self.instrument_loader.instruments.values())

        instruments = []
        for symbol in symbols:
            instrument = self.instrument_loader.get_instrument_by_symbol(symbol)
            if instrument:
                instruments.append(instrument)

        return instruments

    def _get_start_stop(
        self, start: str | None, stop: str | None, timeframe: pd.Timedelta
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Parse and validate start/stop times.

        Args:
            start: Start time string
            stop: Stop time string
            timeframe: Timeframe as Timedelta

        Returns:
            Tuple of (start_timestamp, stop_timestamp)
        """
        if not stop:
            stop = now_utc().isoformat()
        _start, _stop = handle_start_stop(start, stop, convert=lambda x: pd.Timestamp(x))
        assert isinstance(_stop, pd.Timestamp)

        if not _start:
            _start = _stop - timeframe * self._max_bars
        assert isinstance(_start, pd.Timestamp)

        # Apply max_history limit
        if _start < (_max_time := now_utc() - self._max_history):
            _start = _max_time

        return _start, _stop

    def _fetch_ohlcv(
        self,
        instrument: Instrument,
        timeframe: str,
        start: pd.Timestamp,
        stop: pd.Timestamp,
    ) -> list:
        """
        Fetch OHLCV data for instrument.

        Args:
            instrument: Instrument to fetch
            timeframe: Timeframe string (e.g., "1h")
            start: Start timestamp
            stop: Stop timestamp

        Returns:
            List of OHLCV records
        """
        since = int(start.timestamp() * 1000)
        until = int(stop.timestamp() * 1000)

        # Get market ID
        market_id = self.instrument_loader.get_market_id(instrument.symbol)
        if market_id is None:
            logger.error(f"Market ID not found for {instrument.symbol}")
            return []

        # Fetch candlesticks via async API
        future = self._async_loop.submit(
            self.client.get_candlesticks(
                market_id=market_id,
                resolution=timeframe,
                start_timestamp=since,
                end_timestamp=until,
                count_back=self._max_bars,
            )
        )

        try:
            candlesticks = future.result()

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
                        float(candle.get("volume0", 0.0)),  # Use volume0 as primary volume
                    ]
                )

            return ohlcv_data

        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {instrument.symbol}: {e}")
            return []

    def _get_column_names(self, data_type: str) -> list[str]:
        """
        Get column names for data type.

        Args:
            data_type: Data type

        Returns:
            List of column names
        """
        match data_type:
            case DataType.OHLC:
                return ["timestamp", "open", "high", "low", "close", "volume"]
            case _:
                return []
