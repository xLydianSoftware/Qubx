import asyncio
from collections import defaultdict
from typing import Iterable, cast

import numpy as np
import pandas as pd

from ccxt.pro import Exchange
from qubx import logger
from qubx.core.basics import DataType, Instrument
from qubx.core.lookups import lookup
from qubx.data.readers import DataReader, DataTransformer
from qubx.data.registry import reader
from qubx.utils.misc import AsyncThreadLoop
from qubx.utils.time import handle_start_stop, now_utc, timedelta_to_str

from .exchanges import READER_CAPABILITIES, ReaderCapabilities
from .factory import get_ccxt_exchange
from .utils import ccxt_find_instrument, instrument_to_ccxt_symbol


@reader("ccxt")
class CcxtDataReader(DataReader):
    SUPPORTED_DATA_TYPES = {"ohlc", "funding_payment"}

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
        self._max_history = cast(pd.Timedelta, pd.Timedelta(max_history))
        self._capabilities = READER_CAPABILITIES.copy()
        self._exchange_to_symbol_to_instrument = defaultdict(dict)
        self._funding_intervals_cache = {}  # Cache funding interval dictionaries per exchange

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

        timeframe = timeframe or "1m"
        _timeframe = cast(pd.Timedelta, pd.Timedelta(timeframe))
        _start, _stop = self._get_start_stop(start, stop, _timeframe)

        if _start > _stop:
            return []

        exchange = self._exchanges[instrument.exchange]
        data = self._fetch_data(instrument, data_type, timeframe, _start, _stop, exchange)
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

        end_time = now_utc()
        start_time = end_time - self._max_history
        return start_time.to_datetime64(), end_time.to_datetime64()

    def get_candles(
        self,
        exchange: str,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        stop: str | pd.Timestamp | None = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        Returns pandas DataFrame of candle data for given exchange and symbols within specified time range.

        Args:
            exchange: Exchange identifier (e.g., "BINANCE.UM") - must match configured exchange
            symbols: List of symbols in Qubx format (e.g., ["BTCUSDT", "ETHUSDT"]). If None, fetches all symbols.
            start: Start time for filtering. If None, fetches recent history.
            stop: Stop time for filtering. If None, fetches up to current time.
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h", "1d")

        Returns:
            DataFrame with MultiIndex [timestamp, symbol] and columns [open, high, low, close, volume, quote_volume]
            Symbols in the DataFrame are in Qubx format (e.g., "BTCUSDT")
        """
        exchange_upper = exchange.upper()
        if exchange_upper not in self._exchanges:
            self._warning(f"Exchange {exchange} not found in configured exchanges: {list(self._exchanges.keys())}")
            return pd.DataFrame()

        # Parse time range
        timeframe_td = cast(pd.Timedelta, pd.Timedelta(timeframe))
        start_ts, stop_ts = self._get_start_stop(start, stop, timeframe_td)

        if start_ts >= stop_ts:
            return pd.DataFrame()

        # Find instruments to fetch
        instruments_to_fetch = self._find_instruments(exchange_upper, symbols) if symbols else None

        if instruments_to_fetch is not None and not instruments_to_fetch:
            self._warning("No instruments found for the specified symbols")
            return pd.DataFrame()

        # If no symbols specified, get all instruments for the exchange
        if instruments_to_fetch is None:
            instruments_to_fetch = lookup.find_instruments(exchange_upper, as_of=stop_ts)

        self._info(f"Fetching candle data for {len(instruments_to_fetch)} symbols from {start_ts} to {stop_ts}")

        # Fetch candles for all instruments
        future = self._loop.submit(
            self._async_fetch_candles_for_all_instruments(
                instruments_to_fetch, timeframe, start_ts, stop_ts, exchange_upper
            )
        )

        all_candle_data = future.result()

        if not all_candle_data:
            self._info("No candle data found")
            return pd.DataFrame()

        # Create DataFrame
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
        """
        Returns pandas DataFrame of funding payments for given exchange and symbols within specified time range.

        This method fetches funding rate history from CCXT exchanges and formats it to match
        the MultiQdbConnector interface for compatibility with existing strategies.

        Args:
            exchange: Exchange identifier (e.g., "BINANCE.UM") - must match configured exchange
            symbols: List of symbols in Qubx format (e.g., ["BTCUSDT", "ETHUSDT"]). If None, fetches all symbols.
            start: Start time for filtering. If None, fetches recent history.
            stop: Stop time for filtering. If None, fetches up to current time.

        Returns:
            DataFrame with MultiIndex [timestamp, symbol] and columns [funding_rate, funding_interval_hours]
            Symbols in the DataFrame are in Qubx format (e.g., "BTCUSDT")
        """
        # Validate exchange matches configured exchanges
        exchange_upper = exchange.upper()
        if exchange_upper not in self._exchanges:
            self._warning(f"Exchange {exchange} not found in configured exchanges: {list(self._exchanges.keys())}")
            return pd.DataFrame()

        instruments = self._find_instruments(exchange_upper, symbols) if symbols else None

        # Handle time range
        start_ts = None
        stop_ts = None
        if start:
            start_ts = pd.Timestamp(start)
        if stop:
            stop_ts = pd.Timestamp(stop)

        # Default to last 7 days if no start time specified
        if not start_ts:
            start_ts = pd.Timestamp.now() - pd.Timedelta(days=7)

        start_ts = cast(pd.Timestamp, start_ts)
        stop_ts = cast(pd.Timestamp, stop_ts)

        # Apply max_history limitation if both start and stop are specified
        if start_ts and stop_ts and self._max_history:
            max_history_start = stop_ts - self._max_history
            if start_ts < max_history_start:
                start_ts = max_history_start

        if start_ts >= stop_ts:
            return pd.DataFrame()

        # Convert to milliseconds for CCXT
        since = int(start_ts.timestamp() * 1000)
        until = int(stop_ts.timestamp() * 1000) if stop_ts else None

        # Get the exchange instance
        ccxt_exchange = self._exchanges[exchange_upper]

        # Determine fetching strategy based on exchange capabilities
        exchange_caps = self._capabilities.get(exchange_upper.lower(), ReaderCapabilities())
        should_use_bulk = (instruments is None or len(instruments) > 10) and exchange_caps.supports_bulk_funding
        instruments = cast(list[Instrument], instruments)

        if should_use_bulk:
            # Use batch fetching for all symbols or large symbol lists
            fundings = self._batch_fetch_funding_with_pagination(
                ccxt_exchange, exchange, instruments, since, until, start_ts, stop_ts
            )
            self._debug(
                f"Batch fetched {len(fundings)} funding records for symbols: {[instrument.symbol for instrument in instruments]}"
            )
            return fundings
        else:
            # Check if we're trying to fetch all symbols but exchange doesn't support bulk
            if instruments is None and not exchange_caps.supports_bulk_funding:
                raise ValueError(
                    f"Exchange {exchange} does not support bulk funding rate history. "
                    f"Please specify a list of symbols instead of fetching all symbols."
                )

            # Use individual fetching for small symbol lists
            fundings = self._individual_fetch_funding(
                ccxt_exchange, exchange, instruments, since, until, start_ts, stop_ts
            )
            self._debug(
                f"Fetched {len(fundings)} funding records for symbols: {[instrument.symbol for instrument in instruments]}"
            )
            return fundings

    def _batch_fetch_funding_with_pagination(
        self,
        ccxt_exchange,
        exchange: str,
        instruments: list[Instrument] | None,
        since: int,
        until: int | None,
        start_ts: pd.Timestamp,
        stop_ts: pd.Timestamp | None,
    ) -> pd.DataFrame:
        """
        Proper batch funding fetching with pagination following Binance API documentation.

        Uses limit=1000 and paginates from start_time to end_time, removing duplicates.
        Converts symbols from CCXT format to Qubx format.
        """
        try:
            symbols = [instrument.symbol for instrument in instruments] if instruments else None
            all_funding_data = []
            current_since = since
            page = 1
            limit = 1000  # Binance API max limit

            self._debug(f"Starting batch pagination from {pd.Timestamp(since, unit='ms')}")
            self._loop.submit(self._get_funding_intervals_for_exchange(exchange)).result()

            while current_since < (until or int(pd.Timestamp.now().timestamp() * 1000)):
                self._debug(f"Page {page}: Fetching from {pd.Timestamp(current_since, unit='ms')}")

                # Fetch batch of funding rate history
                batch_data = self._loop.submit(
                    ccxt_exchange.fetch_funding_rate_history(
                        None,  # All symbols
                        since=current_since,
                        limit=limit,
                    )
                ).result()

                if not batch_data:
                    self._debug(f"No more data returned on page {page}")
                    break

                # Filter by end time and collect
                filtered_batch = []
                latest_timestamp = current_since

                for item in batch_data:
                    timestamp = item["timestamp"]
                    if until is None or timestamp <= until:
                        filtered_batch.append(item)
                        latest_timestamp = max(latest_timestamp, timestamp)

                self._debug(f"Page {page}: {len(batch_data)} -> {len(filtered_batch)} records after time filtering")

                if not filtered_batch:
                    self._debug(f"No records within time range on page {page}")
                    break

                all_funding_data.extend(filtered_batch)

                # Update current_since for next iteration
                # Don't add 1ms because there might be multiple records at the same timestamp
                if latest_timestamp == current_since:
                    # Timestamp not advancing, add 1ms to avoid infinite loop
                    current_since = latest_timestamp + 1
                else:
                    current_since = latest_timestamp

                # If we got less than the limit, we've reached the end
                if len(batch_data) < limit:
                    self._debug(f"Page {page}: Got {len(batch_data)} < {limit}, reached end")
                    break

                page += 1

                # Safety check to avoid infinite loops
                if page > 100:
                    self._warning(f"Safety break: too many pages ({page})")
                    break

            self._debug(f"Pagination complete: {len(all_funding_data)} raw records from {page} pages")

            if not all_funding_data:
                return pd.DataFrame()

            # Remove duplicates (since times are inclusive)
            seen = set()
            unique_data = []
            duplicates_removed = 0

            for item in all_funding_data:
                key = (item["timestamp"], item["symbol"])
                if key not in seen:
                    seen.add(key)
                    unique_data.append(item)
                else:
                    duplicates_removed += 1

            self._debug(f"Removed {duplicates_removed} duplicates, {len(unique_data)} unique records")

            # Convert to Qubx format and create DataFrame
            processed_data = []

            for item in unique_data:
                timestamp = cast(pd.Timestamp, pd.Timestamp(item["timestamp"], unit="ms"))

                # Convert symbol from CCXT to Qubx format
                ccxt_symbol = item["symbol"]
                qubx_symbol = self._ccxt_symbol_to_qubx(ccxt_symbol)

                # Filter by requested symbols if specified (symbols are in Qubx format)
                if symbols is not None and qubx_symbol not in symbols:
                    continue

                funding_rate = item.get("fundingRate", 0.0)

                # Use exchange-specific funding interval (lookup from cache or default)
                exchange_caps = self._capabilities.get(exchange.lower(), ReaderCapabilities())
                funding_interval_hours = self._get_funding_interval_hours_for_symbol(
                    exchange.upper(), ccxt_symbol, exchange_caps.default_funding_interval_hours
                )
                funding_timeframe = timedelta_to_str(cast(pd.Timedelta, pd.Timedelta(hours=funding_interval_hours)))

                processed_data.append(
                    {
                        "timestamp": timestamp.floor(funding_timeframe),
                        "symbol": qubx_symbol,
                        "funding_rate": funding_rate,
                        "funding_interval_hours": funding_interval_hours,
                    }
                )

            # Create DataFrame
            if not processed_data:
                self._info("No matching symbols found after filtering")
                return pd.DataFrame()

            df = pd.DataFrame(processed_data)
            df = df.sort_values("timestamp")
            df = df.set_index(["timestamp", "symbol"])

            unique_symbols = df.index.get_level_values("symbol").unique()
            self._debug(f"Batch fetch returned {len(df)} records for {len(unique_symbols)} symbols")

            return df

        except Exception as e:
            self._warning(f"Batch fetching failed for {exchange}: {e}")
            return pd.DataFrame()

    def close(self):
        tasks = []
        for exchange in self._exchanges:
            if hasattr(self._exchanges[exchange], "close") and callable(self._exchanges[exchange].close):
                tasks.append(self._loop.submit(self._exchanges[exchange].close()))

        for task in tasks:
            try:
                task.result(timeout=5)  # Wait up to 5 seconds for the connection to close
            except Exception as e:
                self._error(f"Error closing exchange connection: {e}")

    def _get_instrument(self, data_id: str) -> Instrument | None:
        parts = data_id.split(":")
        if len(parts) != 2:
            return None
        exchange, symbol = parts
        return lookup.find_symbol(exchange, symbol)

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

        return _start, _stop

    def _fetch_data(
        self,
        instrument: Instrument,
        data_type: str,
        timeframe: str,
        start: pd.Timestamp,
        stop: pd.Timestamp,
        exchange: Exchange,
    ) -> list:
        """Route to appropriate data fetcher based on data type."""
        match data_type:
            case DataType.OHLC:
                ccxt_symbol = instrument_to_ccxt_symbol(instrument)
                return self._fetch_ohlcv(ccxt_symbol, timeframe, start, stop, exchange)
            case DataType.FUNDING_PAYMENT:
                return self._fetch_funding_for_read(instrument, start, stop, exchange)
            case _:
                raise ValueError(f"Unsupported data type: {data_type}")

    def _fetch_ohlcv(
        self, symbol: str, timeframe: str, start: pd.Timestamp, stop: pd.Timestamp, exchange: Exchange
    ) -> list:
        since = int(start.timestamp() * 1000)
        until = int(stop.timestamp() * 1000)

        # Normalize timeframe for CCXT (e.g., '1D' -> '1d')
        normalized_timeframe = timeframe.lower()

        future = self._loop.submit(self._async_fetch_ohlcv(symbol, normalized_timeframe, since, until, exchange))

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
            self._error(f"Error fetching OHLCV data: {e}")
            return []

    async def _async_fetch_ohlcv(self, symbol: str, timeframe: str, since: int, until: int, exchange: Exchange) -> list:
        all_candles = []

        # CCXT has a limit on how many candles can be fetched at once
        # We need to fetch in batches
        limit = 1000
        current_since = since

        # Normalize timeframe for CCXT (e.g., '1D' -> '1d')
        normalized_timeframe = timeframe.lower()

        self._debug(
            f"[{symbol}] Fetching OHLCV({timeframe}) data from {pd.Timestamp(current_since, unit='ms')} to {pd.Timestamp(until, unit='ms')}"
        )

        while True:
            try:
                candles = await exchange.fetch_ohlcv(symbol, normalized_timeframe, since=current_since, limit=limit)

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
                self._error(f"[{symbol}] Error fetching OHLCV data: {e}")
                break

        return all_candles

    async def _async_fetch_candles_for_all_instruments(
        self,
        instruments: list[Instrument],
        timeframe: str,
        start: pd.Timestamp,
        stop: pd.Timestamp,
        exchange: str,
    ) -> list[dict]:
        """
        Fetch candle data for all instruments concurrently using asyncio.gather().
        """
        ccxt_exchange = self._exchanges[exchange]

        # Create coroutines for all instruments
        coroutines = [
            self._async_fetch_ohlcv(
                instrument_to_ccxt_symbol(instrument),
                timeframe,
                int(start.timestamp() * 1000),
                int(stop.timestamp() * 1000),
                ccxt_exchange,
            )
            for instrument in instruments
        ]

        # Execute all coroutines concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Collect successful results and log failures
        all_candle_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._warning(f"Failed to fetch candle data for {instruments[i].symbol}: {result}")
                continue

            assert isinstance(result, list)
            for candle in result:
                timestamp = pd.Timestamp(candle[0], unit="ms")
                all_candle_data.append(
                    {
                        "timestamp": timestamp,
                        "symbol": instruments[i].symbol,
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                        "quote_volume": float(candle[5])
                        * float(candle[4]),  # Approximate quote volume as volume * close
                    }
                )

        return all_candle_data

    def _fetch_funding_for_read(
        self, instrument: Instrument, start: pd.Timestamp, stop: pd.Timestamp, exchange: Exchange
    ) -> list[tuple]:
        """
        Fetch funding data for single instrument and return as list of tuples.

        This method wraps the existing async funding fetch logic and converts
        the dict format to tuple format for compatibility with the read() method's
        DataTransformer.

        Returns:
            List of tuples: (timestamp, funding_rate, funding_interval_hours)
        """
        since = int(start.timestamp() * 1000)

        # Use the existing async method that handles per-symbol funding intervals
        future = self._loop.submit(
            self._async_fetch_funding_for_instrument(exchange, instrument, since, stop, instrument.exchange.upper())
        )

        funding_dicts = future.result()

        # Convert from dict format to tuple format for DataTransformer
        return [(item["timestamp"], item["funding_rate"], item["funding_interval_hours"]) for item in funding_dicts]

    def _find_instruments(self, exchange: str, symbols: list[str]) -> list[Instrument]:
        exchange_upper = exchange.upper()
        if exchange_upper not in self._exchanges:
            return []
        _exchange = self._exchanges[exchange]
        markets = _exchange.markets
        if markets is None:
            return []

        ccxt_symbols = list(markets.keys())

        instruments = [
            ccxt_find_instrument(symbol, _exchange, self._exchange_to_symbol_to_instrument[exchange_upper])
            for symbol in ccxt_symbols
        ]
        symbol_to_instrument = {instrument.symbol: instrument for instrument in instruments}
        return [symbol_to_instrument[symbol] for symbol in symbols if symbol in symbol_to_instrument]

    def _ccxt_symbol_to_qubx(self, ccxt_symbol: str) -> str:
        """
        Convert CCXT symbol format to Qubx format.
        CCXT: 'BTC/USDT:USDT' -> Qubx: 'BTCUSDT'
        """
        if "/" in ccxt_symbol:
            base_quote = ccxt_symbol.split("/")[0]  # Get 'BTC' from 'BTC/USDT:USDT'
            quote_part = ccxt_symbol.split("/")[1]  # Get 'USDT:USDT' from 'BTC/USDT:USDT'
            quote = quote_part.split(":")[0]  # Get 'USDT' from 'USDT:USDT'
            return f"{base_quote}{quote}"
        else:
            # Already in simple format
            return ccxt_symbol

    def _individual_fetch_funding(
        self,
        ccxt_exchange,
        exchange: str,
        instruments: list[Instrument],
        since: int,
        until: int | None,
        start_ts: pd.Timestamp,
        stop_ts: pd.Timestamp | None,
    ) -> pd.DataFrame:
        # Submit single async task that gathers all individual requests
        future = self._loop.submit(
            self._async_fetch_funding_for_all_instruments(ccxt_exchange, instruments, since, stop_ts, exchange)
        )

        # Wait for all parallel requests to complete
        all_funding_data = future.result()

        # Convert to DataFrame
        if not all_funding_data:
            self._info(f"No funding payment data found for exchange {exchange}")
            return pd.DataFrame()

        df = pd.DataFrame(all_funding_data)
        df = df.sort_values("timestamp")
        df = df.set_index(["timestamp", "symbol"])
        return df

    async def _async_fetch_funding_for_all_instruments(
        self,
        ccxt_exchange,
        instruments: list[Instrument],
        since: int,
        stop_ts: pd.Timestamp | None,
        exchange: str,
    ) -> list[dict]:
        """
        Fetch funding data for all instruments concurrently using asyncio.gather().
        """
        # Create coroutines for all instruments
        coroutines = [
            self._async_fetch_funding_for_instrument(ccxt_exchange, instrument, since, stop_ts, exchange)
            for instrument in instruments
        ]

        # Execute all coroutines concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Collect successful results and log failures
        all_funding_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._warning(f"Failed to fetch funding data for {instruments[i].symbol}: {result}")
            else:
                all_funding_data.extend(cast(list[dict], result))

        return all_funding_data

    async def _async_fetch_funding_for_instrument(
        self,
        ccxt_exchange,
        instrument: Instrument,
        since: int,
        stop_ts: pd.Timestamp | None,
        exchange: str,
    ) -> list[dict]:
        """
        Async method to fetch funding data for a single instrument.
        Returns list of funding data dictionaries.
        """
        try:
            # Convert Qubx symbol to CCXT symbol for API call
            ccxt_symbol = instrument_to_ccxt_symbol(instrument)
            qubx_symbol = instrument.symbol

            await self._get_funding_intervals_for_exchange(exchange)

            # Fetch funding rate history for this symbol
            funding_history = await ccxt_exchange.fetch_funding_rate_history(ccxt_symbol, since=since, limit=1000)

            if not funding_history:
                self._debug(f"No funding history found for {qubx_symbol} ({ccxt_symbol}) on {exchange}")
                return []

            # Convert CCXT format to our expected format
            instrument_data = []
            for item in funding_history:
                timestamp = cast(pd.Timestamp, pd.Timestamp(item["timestamp"], unit="ms"))

                # Filter by stop time if specified
                if stop_ts and timestamp > stop_ts:
                    continue

                funding_rate = item.get("fundingRate", 0.0)

                # Use exchange-specific funding interval (lookup from cache or default)
                exchange_caps = self._capabilities.get(exchange.lower(), ReaderCapabilities())
                funding_interval_hours = self._get_funding_interval_hours_for_symbol(
                    exchange.upper(), ccxt_symbol, exchange_caps.default_funding_interval_hours
                )
                funding_timeframe = timedelta_to_str(cast(pd.Timedelta, pd.Timedelta(hours=funding_interval_hours)))

                instrument_data.append(
                    {
                        "timestamp": timestamp.floor(funding_timeframe),
                        "symbol": qubx_symbol,  # Use original Qubx symbol
                        "funding_rate": funding_rate,
                        "funding_interval_hours": funding_interval_hours,
                    }
                )

            return instrument_data

        except Exception as e:
            self._warning(f"Error fetching funding data for {instrument.symbol} on {exchange}: {e}")
            return []

    def _qubx_symbol_to_ccxt(self, qubx_symbol: str) -> str:
        """
        Convert Qubx symbol format to CCXT symbol format for API calls.
        Qubx: 'BTCUSDT' -> CCXT: 'BTCUSDT' (for individual API calls, CCXT accepts simple format)
        """
        # For individual API calls, CCXT accepts the simple format
        # The complex format (BTC/USDT:USDT) is used in batch responses
        return qubx_symbol

    async def _get_funding_intervals_for_exchange(self, exchange_name: str) -> dict[str, float]:
        """
        Get funding intervals dictionary for an exchange, with caching.
        """
        if exchange_name in self._funding_intervals_cache:
            return self._funding_intervals_cache[exchange_name]

        ccxt_exchange = self._exchanges[exchange_name]
        intervals = {}

        # Check if exchange has get_funding_interval_hours method
        if hasattr(ccxt_exchange, "get_funding_interval_hours"):
            try:
                intervals = await getattr(ccxt_exchange, "get_funding_interval_hours")()
                self._debug(f"Retrieved {len(intervals)} funding intervals for {exchange_name}")
            except Exception as e:
                self._debug(f"Failed to get funding intervals for {exchange_name}: {e}")

        # Cache the result (even if empty)
        self._funding_intervals_cache[exchange_name] = intervals
        return intervals

    def _get_funding_interval_hours_for_symbol(
        self, exchange_name: str, ccxt_symbol: str, default_hours: float
    ) -> float:
        """
        Get funding interval for a specific symbol, with exchange-specific lookup and fallback.
        """
        if exchange_name in self._funding_intervals_cache:
            intervals_dict = self._funding_intervals_cache[exchange_name]
        else:
            raise ValueError(
                f"First fetch funding intervals for {exchange_name} before fetching funding interval for a symbol"
            )
        interval = intervals_dict.get(ccxt_symbol, default_hours)
        if isinstance(interval, str):
            return pd.Timedelta(interval).total_seconds() / 3600
        return float(interval)

    def _get_column_names(self, data_type: str) -> list[str]:
        match data_type:
            case DataType.OHLC:
                return ["timestamp", "open", "high", "low", "close", "volume"]
            case DataType.FUNDING_PAYMENT:
                return ["timestamp", "funding_rate", "funding_interval_hours"]
            case _:
                return []

    def _info(self, message: str) -> None:
        logger.info(f"<blue>[CCXT]</blue> {message}")

    def _debug(self, message: str) -> None:
        logger.debug(f"<blue>[CCXT]</blue> {message}")

    def _warning(self, message: str) -> None:
        logger.warning(f"<blue>[CCXT]</blue> {message}")

    def _error(self, message: str) -> None:
        logger.error(f"<blue>[CCXT]</blue> {message}")
