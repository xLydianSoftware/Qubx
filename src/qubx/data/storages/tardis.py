import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from tardis_dev import datasets

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.containers import RawData, RawMultiData
from qubx.data.registry import storage
from qubx.data.storage import IReader, IStorage, Transformable
from qubx.utils.time import handle_start_stop

# API endpoint for Tardis
TARDIS_API_URL = "https://api.tardis.dev/v1"

# Data directory for cached Tardis data
TARDIS_DATA_DIR = Path(os.getenv("TARDIS_DATA_DIR", "/data/tardis"))

# API key for Tardis (optional, but required for full data access)
TARDIS_API_KEY = os.getenv("TARDIS_API_KEY", "")

# Exchange name mappings (Qubx convention -> Tardis ID)
EXCHANGE_MAPPINGS = {
    "binance.um": "binance-futures",
    "binance.cm": "binance-delivery",
    "binance.pm": "binance-futures",
    "bitfinex.f": "bitfinex-derivatives",
}

# Reverse mappings (Tardis ID -> Qubx convention)
EXCHANGE_REVERSE_MAPPINGS = {v: k for k, v in EXCHANGE_MAPPINGS.items()}

# Market type mappings (Qubx convention -> Tardis symbol type)
MARKET_TYPE_MAPPINGS = {
    "swap": "perpetual",
    "perp": "perpetual",
    "perpetual": "perpetual",
    "future": "future",
    "futures": "future",
}

# Reverse market type mappings (Tardis -> Qubx)
MARKET_TYPE_REVERSE = {
    "perpetual": "SWAP",
    "future": "FUTURE",
}

# DataType to Tardis channel mapping (for read() method)
DTYPE_TO_CHANNEL = {
    DataType.TRADE: "trade",
    DataType.ORDERBOOK: "depth",
    DataType.QUOTE: "bookTicker",
    DataType.FUNDING_RATE: "markPrice",
    DataType.LIQUIDATION: "forceOrder",
}

# Channel to download data_type mapping (Tardis API channel -> file data_type)
CHANNEL_TO_DATA_TYPE = {
    "trade": "trades",
    "aggTrade": "trades",
    "depth": "incremental_book_L2",
    "depthSnapshot": "book_snapshot_25",
    "bookTicker": "quotes",
    "forceOrder": "liquidations",
    "markPrice": "derivative_ticker",
    "funding": "funding",
    "openInterest": "open_interest",
    "ticker": "ticker",
}

# Channel to Qubx DataType mapping
CHANNEL_TO_DTYPE = {
    "trade": DataType.TRADE,
    "aggTrade": DataType.TRADE,
    "depth": DataType.ORDERBOOK,
    "depthSnapshot": DataType.ORDERBOOK,
    "bookTicker": DataType.QUOTE,
    "markPrice": DataType.FUNDING_RATE,
    "forceOrder": DataType.LIQUIDATION,
    "funding": DataType.FUNDING_RATE,
    "openInterest": DataType.NONE,  # Custom type
    "ticker": DataType.NONE,  # Custom type
}


def _normalize_exchange(exchange: str) -> str:
    """Convert Qubx exchange name to Tardis exchange ID."""
    return EXCHANGE_MAPPINGS.get(exchange.lower(), exchange.lower())


def _denormalize_exchange(tardis_id: str) -> str:
    """Convert Tardis exchange ID back to Qubx convention."""
    return EXCHANGE_REVERSE_MAPPINGS.get(tardis_id, tardis_id)


def _normalize_market_type(market: str) -> str:
    """Convert Qubx market type to Tardis symbol type."""
    return MARKET_TYPE_MAPPINGS.get(market.lower(), market.lower())


def _denormalize_market_type(tardis_type: str) -> str:
    """Convert Tardis symbol type to Qubx market type."""
    return MARKET_TYPE_REVERSE.get(tardis_type, tardis_type.upper())


def _file_name_nested(exchange: str, data_type: str, date: datetime, symbol: str, format: str) -> str:
    """
    Generate file path matching existing structure in /data/tardis/.

    Structure: {exchange}/{data_type}/{date}_{symbol}.{format}.gz
    Example: binance-futures/trades/2024-11-01_BTCUSDT.csv.gz
    """
    return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


@storage("tardis")
class TardisStorage(IStorage):
    """
    Storage implementation for Tardis.dev historical cryptocurrency market data.

    Tardis provides tick-level historical data for 50+ cryptocurrency exchanges
    including trades, order books, funding rates, liquidations, and more.

    Data is automatically downloaded and cached to TARDIS_DATA_DIR (default: /data/tardis/).
    Existing files are skipped during download.

    Usage:
        storage = TardisStorage()

        # List available exchanges
        exchanges = storage.get_exchanges()

        # List available market types for an exchange (SWAP, FUTURE)
        markets = storage.get_market_types("binance-futures")

        # Get a reader for specific exchange and market type
        reader = storage.get_reader("BINANCE.UM", "SWAP")

        # Read trades data
        data = reader.read("BTCUSDT", DataType.TRADE, "2024-01-01", "2024-01-02")

        # Read orderbook data
        data = reader.read("BTCUSDT", DataType.ORDERBOOK, "2024-01-01", "2024-01-02")

        # Transform to pandas DataFrame (note: Tardis uses microseconds)
        from qubx.data import PandasFrame
        df = data.transform(PandasFrame(timestamp_units="us"))

    See: https://docs.tardis.dev/api/python
    """

    _exchanges_cache: list[dict] | None
    _exchange_details_cache: dict[str, dict]

    def __init__(self, api_url: str = TARDIS_API_URL, only_enabled: bool = True):
        """
        Initialize TardisStorage.

        Args:
            api_url: Base URL for Tardis API (default: https://api.tardis.dev/v1)
            only_enabled: If True, only return enabled (active) exchanges
        """
        self.api_url = api_url.rstrip("/")
        self.only_enabled = only_enabled
        self._exchanges_cache = None
        self._exchange_details_cache = {}

    def _fetch_exchanges(self) -> list[dict]:
        """Fetch list of all exchanges from Tardis API."""
        if self._exchanges_cache is not None:
            return self._exchanges_cache

        try:
            resp = httpx.get(f"{self.api_url}/exchanges", timeout=30)
            resp.raise_for_status()
            self._exchanges_cache = resp.json()
            return self._exchanges_cache
        except Exception as e:
            logger.error(f"Failed to fetch exchanges from Tardis: {e}")
            return []

    def _get_exchange_details(self, exchange: str) -> dict | None:
        """
        Get detailed information about an exchange.

        Uses httpx instead of tardis_dev.get_exchange_details() to avoid
        event loop conflicts in Jupyter notebooks.

        Args:
            exchange: Exchange ID (e.g., 'binance-futures')

        Returns:
            Exchange details dict or None if not found
        """
        tardis_id = _normalize_exchange(exchange)

        if tardis_id in self._exchange_details_cache:
            return self._exchange_details_cache[tardis_id]

        try:
            resp = httpx.get(f"{self.api_url}/exchanges/{tardis_id}", timeout=30)
            resp.raise_for_status()
            details = resp.json()
            self._exchange_details_cache[tardis_id] = details
            return details
        except Exception as e:
            logger.error(f"Failed to get exchange details for {tardis_id}: {e}")
            return None

    def get_exchanges(self) -> list[str]:
        """
        Get list of available exchanges.

        Returns:
            List of exchange IDs (e.g., ['binance-futures', 'deribit', 'bitmex', ...])
        """
        exchanges = self._fetch_exchanges()

        if self.only_enabled:
            return [ex["id"] for ex in exchanges if ex.get("enabled", False)]

        return [ex["id"] for ex in exchanges]

    def get_market_types(self, exchange: str) -> list[str]:
        """
        Get available market types for an exchange.

        Market types represent the contract type (SWAP for perpetual, FUTURE for dated).

        Args:
            exchange: Exchange ID (e.g., 'binance-futures' or 'binance.um')

        Returns:
            List of available market types (e.g., ['SWAP', 'FUTURE'])
        """
        details = self._get_exchange_details(exchange)
        if details is None:
            return []

        # Extract unique symbol types and convert to Qubx convention
        symbols = details.get("availableSymbols", [])
        types = set(s.get("type", "perpetual") for s in symbols)
        return sorted([_denormalize_market_type(t) for t in types])

    def get_data_types(self, exchange: str) -> list[str]:
        """
        Get available data channels for an exchange.

        Data channels represent the type of market data (trade, depth, quotes, etc.).

        Args:
            exchange: Exchange ID (e.g., 'binance-futures' or 'binance.um')

        Returns:
            List of available channels (e.g., ['trade', 'aggTrade', 'depth', ...])
        """
        details = self._get_exchange_details(exchange)
        if details is None:
            return []

        return details.get("availableChannels", [])

    def get_symbols(self, exchange: str) -> list[str]:
        """
        Get available symbols for an exchange.

        Args:
            exchange: Exchange ID

        Returns:
            List of symbol IDs (e.g., ['btcusdt', 'ethusdt', ...])
        """
        details = self._get_exchange_details(exchange)
        if details is None:
            return []

        symbols = details.get("availableSymbols", [])
        return [s["id"] for s in symbols]

    def get_reader(self, exchange: str, market: str) -> "TardisReader":
        """
        Get a reader for specific exchange and market type.

        Args:
            exchange: Exchange ID (e.g., 'binance-futures', 'BINANCE.UM')
            market: Market type (e.g., 'SWAP', 'FUTURE', 'perpetual')

        Returns:
            TardisReader instance for reading data
        """
        tardis_id = _normalize_exchange(exchange)
        market_type = _normalize_market_type(market)

        # Validate exchange exists
        details = self._get_exchange_details(tardis_id)
        if details is None:
            raise ValueError(f"Exchange '{exchange}' not found in Tardis")

        # Filter symbols by market type
        all_symbols = details.get("availableSymbols", [])
        filtered_symbols = [s for s in all_symbols if s.get("type", "perpetual") == market_type]

        if not filtered_symbols:
            available_types = sorted(set(s.get("type", "perpetual") for s in all_symbols))
            raise ValueError(
                f"Market type '{market}' not available for exchange '{tardis_id}'. "
                f"Available types: {[_denormalize_market_type(t) for t in available_types]}"
            )

        # Get available channels
        available_channels = details.get("availableChannels", [])

        return TardisReader(tardis_id, market_type, filtered_symbols, available_channels)


class TardisReader(IReader):
    """
    Reader for Tardis.dev market data.

    Reads historical tick-level data for a specific exchange and market type.
    Automatically downloads missing data from Tardis.dev and caches locally.

    Supports multiple data types (trade, orderbook, quotes, etc.) through the
    dtype parameter in read().
    """

    def __init__(self, exchange: str, market_type: str, symbols: list[dict], channels: list[str]):
        """
        Initialize TardisReader.

        Args:
            exchange: Tardis exchange ID
            market_type: Tardis symbol type (e.g., 'perpetual', 'future')
            symbols: List of symbol dicts filtered by market type
            channels: List of available channels for this exchange
        """
        self.exchange = exchange
        self.market_type = market_type
        self._channels = channels
        self._symbols = {s["id"].upper(): s for s in symbols}

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        """
        Get available symbols for this reader.

        Returns:
            List of symbol IDs (uppercase, e.g., ['BTCUSDT', 'ETHUSDT', ...])
        """
        return list(self._symbols.keys())

    def get_data_types(self, data_id: str) -> list[DataType]:
        """
        Get available data types for a symbol.

        Returns data types based on available channels for this exchange.
        """
        # Map available channels to DataTypes
        dtypes = []
        for channel in self._channels:
            dtype = CHANNEL_TO_DTYPE.get(channel)
            if dtype and dtype != DataType.NONE and dtype not in dtypes:
                dtypes.append(dtype)
        return dtypes if dtypes else [DataType.NONE]

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple:
        """
        Get available time range for a symbol.

        Returns:
            Tuple of (start_time, end_time) as numpy datetime64
        """
        symbol_info = self._symbols.get(data_id.upper())
        if symbol_info is None:
            return (np.datetime64("NaT"), np.datetime64("NaT"))

        available_since = symbol_info.get("availableSince")
        available_to = symbol_info.get("availableTo")

        start = np.datetime64(available_since) if available_since else np.datetime64("NaT")
        end = np.datetime64(available_to) if available_to else np.datetime64(pd.Timestamp.now().floor("D"))

        return (start, end)

    def _resolve_channel(self, dtype: DataType | str) -> str:
        """
        Resolve DataType to Tardis channel.

        Args:
            dtype: DataType enum or string

        Returns:
            Tardis channel name (e.g., 'trade', 'depth')

        Raises:
            ValueError: If dtype is not supported
        """
        # Normalize dtype
        if isinstance(dtype, str):
            try:
                dtype = DataType[dtype.upper()]
            except KeyError:
                raise ValueError(f"Unknown data type: {dtype}")

        # Map DataType to channel
        channel = DTYPE_TO_CHANNEL.get(dtype)
        if channel is None:
            supported = list(DTYPE_TO_CHANNEL.keys())
            raise ValueError(f"Data type {dtype} not supported. Supported types: {supported}")

        # Check if channel is available for this exchange
        if channel not in self._channels:
            # Try alternative channels (e.g., aggTrade instead of trade)
            if channel == "trade" and "aggTrade" in self._channels:
                return "aggTrade"
            raise ValueError(
                f"Channel '{channel}' for {dtype} not available for exchange '{self.exchange}'. "
                f"Available channels: {self._channels}"
            )

        return channel

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
        chunksize: int = 0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        """
        Read data from Tardis.

        Automatically downloads missing data from Tardis.dev (skips existing files).

        Args:
            data_id: Symbol or list of symbols (e.g., 'BTCUSDT')
            dtype: Data type to read (e.g., DataType.TRADE, DataType.ORDERBOOK)
            start: Start time (ISO format or pandas-parseable string)
            stop: Stop time
            chunksize: If > 0, return iterator yielding daily chunks (one day per iteration).
                       Tardis data is stored in daily files, so chunksize is always treated as 1.
            **kwargs: Additional arguments

        Returns:
            If chunksize=0: Transformable data container (RawData or RawMultiData)
            If chunksize>0: Iterator yielding daily RawData/RawMultiData chunks
        """
        # Normalize inputs
        symbols = [data_id.upper()] if isinstance(data_id, str) else [s.upper() for s in data_id]

        # Map dtype to channel and then to Tardis data_type
        channel = self._resolve_channel(dtype)
        data_type = CHANNEL_TO_DATA_TYPE.get(channel, channel)

        # Parse dates
        start_dt, stop_dt = handle_start_stop(start, stop)
        if start_dt is None:
            raise ValueError("Start date is required for TardisReader.read()")

        start_date = str(pd.Timestamp(start_dt).date())
        stop_date = str(pd.Timestamp(stop_dt).date()) if stop_dt else start_date

        # Chunked reading - yield day by day
        if chunksize > 0:
            if chunksize != 1:
                logger.warning(
                    f"TardisReader chunksize={chunksize} is not supported. "
                    "Tardis data is stored in daily files, so chunksize is reset to 1 (one day per chunk)."
                )
            return self._read_chunked_by_day(symbols, data_type, start_date, stop_date, dtype)

        # Non-chunked reading - download all and concatenate
        self._download_data(symbols, data_type, start_date, stop_date)
        return self._read_csv_files(symbols, data_type, start_date, stop_date, dtype)

    def _download_data(self, symbols: list[str], data_type: str, start_date: str, stop_date: str) -> None:
        """
        Download data from Tardis.dev.

        Automatically skips files that already exist locally.

        Args:
            symbols: List of symbols to download
            data_type: Tardis data type (e.g., 'trades', 'incremental_book_L2')
            start_date: Start date (YYYY-MM-DD)
            stop_date: Stop date (YYYY-MM-DD)
        """
        # Check if all files already exist locally
        missing_files = self._find_missing_files(symbols, data_type, start_date, stop_date)
        if not missing_files:
            logger.debug(f"All files already cached for {symbols} from {start_date} to {stop_date}")
            return

        if not TARDIS_API_KEY:
            logger.warning(
                "TARDIS_API_KEY not set. Only first day of each month is accessible. "
                "Set the environment variable for full data access."
            )

        logger.info(
            f"Downloading {self.exchange}/{data_type} for {symbols} "
            f"from {start_date} to {stop_date} ({len(missing_files)} files to download)"
        )

        try:
            # Use nest_asyncio to allow running in Jupyter notebooks
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            pass  # nest_asyncio not installed, hope we're not in Jupyter

        try:
            datasets.download(
                exchange=self.exchange,
                data_types=[data_type],
                symbols=symbols,
                from_date=start_date,
                to_date=stop_date,
                api_key=TARDIS_API_KEY,
                download_dir=str(TARDIS_DATA_DIR),
                get_filename=_file_name_nested,
            )
        except Exception as e:
            logger.error(f"Failed to download data from Tardis: {e}")
            raise

    def _find_missing_files(
        self, symbols: list[str], data_type: str, start_date: str, stop_date: str
    ) -> list[tuple[str, str]]:
        """
        Find which files are missing locally.

        Returns:
            List of (symbol, date) tuples for missing files
        """
        missing = []
        data_dir = TARDIS_DATA_DIR / self.exchange / data_type

        # Generate all expected dates
        current = pd.Timestamp(start_date)
        end = pd.Timestamp(stop_date)

        while current < end:
            date_str = str(current.date())
            for symbol in symbols:
                expected_file = data_dir / f"{date_str}_{symbol}.csv.gz"
                if not expected_file.exists():
                    missing.append((symbol, date_str))
            current += pd.Timedelta(days=1)

        return missing

    def _read_csv_files(
        self,
        symbols: list[str],
        data_type: str,
        start_date: str,
        stop_date: str,
        dtype: DataType | str,
    ) -> Transformable:
        """
        Read downloaded CSV files and return as RawData/RawMultiData.

        Args:
            symbols: List of symbols to read
            data_type: Tardis data type (e.g., 'trades')
            start_date: Start date (YYYY-MM-DD)
            stop_date: Stop date (YYYY-MM-DD)
            dtype: DataType for the result

        Returns:
            RawData for single symbol, RawMultiData for multiple symbols
        """
        data_dir = TARDIS_DATA_DIR / self.exchange / data_type

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        results = []

        for symbol in symbols:
            # Find matching files
            pattern = f"*_{symbol}.csv.gz"
            all_files = sorted(data_dir.glob(pattern))

            # Filter by date range
            files = []
            for f in all_files:
                file_date = f.stem.split("_")[0]
                if start_date <= file_date <= stop_date:
                    files.append(f)

            if not files:
                logger.warning(f"No data files found for {symbol} in {data_dir}")
                continue

            # Read and concatenate files
            dfs = []
            for f in files:
                try:
                    df = pd.read_csv(f)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to read {f}: {e}")

            if not dfs:
                continue

            df = pd.concat(dfs, ignore_index=True)

            # Get column names and convert to list of lists
            columns = list(df.columns)
            raw_data = df.values.tolist()

            # Resolve dtype if it's a string
            if isinstance(dtype, str):
                try:
                    resolved_dtype = DataType[dtype.upper()]
                except KeyError:
                    resolved_dtype = DataType.NONE
            else:
                resolved_dtype = dtype

            results.append(RawData(symbol, columns, resolved_dtype, raw_data))

        if not results:
            raise ValueError(f"No data found for symbols {symbols} in date range {start_date} to {stop_date}")

        if len(results) == 1:
            return results[0]

        return RawMultiData(results)

    def _read_chunked_by_day(
        self,
        symbols: list[str],
        data_type: str,
        start_date: str,
        stop_date: str,
        dtype: DataType | str,
    ) -> Iterator[Transformable]:
        """
        Yield data day-by-day for memory-efficient processing.

        Each iteration yields one day's data:
        - RawData if single symbol
        - RawMultiData if multiple symbols

        Downloads each day lazily (only when needed).

        Args:
            symbols: List of symbols to read
            data_type: Tardis data type (e.g., 'trades')
            start_date: Start date (YYYY-MM-DD)
            stop_date: Stop date (YYYY-MM-DD)
            dtype: DataType for the result

        Yields:
            RawData or RawMultiData for each day
        """
        data_dir = TARDIS_DATA_DIR / self.exchange / data_type

        # Resolve dtype once
        if isinstance(dtype, str):
            try:
                resolved_dtype = DataType[dtype.upper()]
            except KeyError:
                resolved_dtype = DataType.NONE
        else:
            resolved_dtype = dtype

        current = pd.Timestamp(start_date)
        end = pd.Timestamp(stop_date)

        while current < end:
            date_str = str(current.date())
            next_date_str = str((current + pd.Timedelta(days=1)).date())

            # Download only this day if missing (lazy download)
            self._download_data(symbols, data_type, date_str, next_date_str)

            # Read this day's files
            day_results = []
            for symbol in symbols:
                file_path = data_dir / f"{date_str}_{symbol}.csv.gz"
                if not file_path.exists():
                    logger.debug(f"No data file for {symbol} on {date_str}")
                    continue

                try:
                    df = pd.read_csv(file_path)
                    columns = list(df.columns)
                    raw_data = df.values.tolist()
                    day_results.append(RawData(symbol, columns, resolved_dtype, raw_data))
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")

            # Yield this day's data
            if day_results:
                if len(day_results) == 1:
                    yield day_results[0]
                else:
                    yield RawMultiData(day_results)

            current += pd.Timedelta(days=1)
