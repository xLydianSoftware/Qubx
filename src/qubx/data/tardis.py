import json
import threading
import time
from contextlib import contextmanager
from os.path import exists, expanduser
from pathlib import Path
from queue import Queue
from typing import Any, Iterable

import numpy as np
import orjson
import pandas as pd
import requests
from pyarrow import csv

from qubx import logger
from qubx.data.registry import reader
from qubx.utils.ntp import time_now
from qubx.utils.orderbook import accumulate_orderbook_levels
from qubx.utils.time import handle_start_stop

from .readers import DataReader, DataTransformer

TARDIS_EXCHANGE_MAPPERS = {
    "bitfinex.f": "bitfinex-derivatives",
    "binance.um": "binance-futures",
}


class TardisCsvDataReader(DataReader):
    def __init__(self, path: str | Path) -> None:
        _path = expanduser(path)
        if not exists(_path):
            raise ValueError(f"Folder is not found at {path}")
        self.path = Path(_path)

    def get_names(self, exchange: str | None = None, data_type: str | None = None) -> list[str]:
        symbols = []
        exchanges = [exchange] if exchange else self.get_exchanges()
        for exchange in exchanges:
            exchange_path = Path(self.path) / exchange
            if not exists(exchange_path):
                raise ValueError(f"Exchange is not found at {exchange_path}")
            data_types = [data_type] if data_type else self.get_data_types(exchange)
            for data_type in data_types:
                data_type_path = exchange_path / data_type
                if not exists(data_type_path):
                    return []
                symbols += self._get_symbols(data_type_path)
        return symbols

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        timeframe=None,
        data_type="trades",
    ) -> Iterable | Any:
        if chunksize > 0:
            raise NotImplementedError("Chunksize is not supported for TardisCsvDataReader")
        exchange, symbol = data_id.split(":")
        _exchange = exchange.lower()
        _exchange = TARDIS_EXCHANGE_MAPPERS.get(_exchange, _exchange)
        t_0, t_1 = handle_start_stop(start, stop, lambda x: pd.Timestamp(x).date().isoformat())
        _path = self.path / _exchange / data_type
        if not _path.exists():
            raise ValueError(f"Data type is not found at {_path}")
        _files = sorted(_path.glob(f"*_{symbol}.csv.gz"))
        if not _files:
            return None
        _dates = [file.stem.split("_")[0] for file in _files]
        if t_0 is None:
            t_0 = _dates[0]
        if t_1 is None:
            t_1 = _dates[-1]
        _filt_files = [file for file in _files if t_0 <= file.stem.split("_")[0] <= t_1]

        tables = []
        fieldnames = None
        for f_path in _filt_files:
            table = csv.read_csv(
                f_path,
                parse_options=csv.ParseOptions(ignore_empty_lines=True),
            )
            if not fieldnames:
                fieldnames = table.column_names
            tables.append(table.to_pandas())

        transform.start_transform(data_id, fieldnames or [], start=start, stop=stop)
        raw_data = pd.concat(tables).to_numpy()
        transform.process_data(raw_data)

        return transform.collect()

    def get_exchanges(self) -> list[str]:
        return [exchange.name for exchange in self.path.iterdir() if exchange.is_dir()]

    def get_data_types(self, exchange: str) -> list[str]:
        exchange_path = Path(self.path) / exchange
        return [data_type.name for data_type in exchange_path.iterdir() if data_type.is_dir()]

    def _get_symbols(self, data_type_path: Path) -> list[str]:
        symbols = set()
        for file in data_type_path.glob("*.gz"):
            parts = file.stem.replace(".csv", "").split("_")
            if len(parts) == 2:
                symbols.add(parts[1])
        return list(symbols)


@reader("tardis")
class TardisMachineReader(DataReader):
    """
    Data reader for Tardis.dev API which provides access to historical cryptocurrency market data.

    This reader connects to both the Tardis.dev HTTP API for exchange information and
    the Tardis Machine server for data retrieval using the normalized API endpoint.

    Parameters:
    -----------
    api_key : str | None
        Optional API key for authentication with Tardis.dev API. If not provided,
        only the first day of each month of data is accessible (free access).
    tardis_api_url : str
        URL for the Tardis.dev API (default: "https://api.tardis.dev/v1")
    tardis_machine_url : str
        URL for the Tardis Machine server (default: "http://quantlab:8010")
    """

    # Define the standard column names for each normalized data type
    NORMALIZED_DATA_COLUMNS = {
        "trade": ["timestamp", "type", "symbol", "exchange", "side", "price", "amount", "localTimestamp"],
        "book_change": ["timestamp", "type", "symbol", "exchange", "side", "price", "amount", "localTimestamp"],
        # "book_snapshot": [
        #     "timestamp",
        #     "type",
        #     "symbol",
        #     "exchange",
        #     "name",
        #     "depth",
        #     "interval",
        #     "bids",
        #     "asks",
        #     "localTimestamp",
        # ],
        # We replace the book snapshot with our custom format
        "book_snapshot": [
            "timestamp",
            "top_bid",
            "top_ask",
            "tick_size",
            "bids",
            "asks",
        ],
        "derivative_ticker": [
            "timestamp",
            "type",
            "symbol",
            "exchange",
            "fundingRate",
            "fundingTimestamp",
            "openInterest",
            "localTimestamp",
        ],
        "trade_bar": [
            "timestamp",
            "type",
            "symbol",
            "exchange",
            "name",
            "interval",
            "kind",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "buyVolume",
            "sellVolume",
            "trades",
            "vwap",
            "openTimestamp",
            "closeTimestamp",
            "localTimestamp",
        ],
        "disconnect": ["type", "exchange", "localTimestamp"],
    }

    def __init__(
        self,
        tardis_api_url: str = "https://api.tardis.dev/v1",
        tardis_machine_url: str = "http://quantlab:8010",
    ) -> None:
        self.api_url = tardis_api_url.rstrip("/")
        self.machine_url = tardis_machine_url.rstrip("/")
        self._exchanges_cache = None
        self._exchange_info_cache = {}

    def _get_headers(self) -> dict:
        """Get headers for API requests including authentication if available"""
        headers = {"Content-Type": "application/json"}
        return headers

    def _safe_log_response(self, response):
        """Safely log response content without HTML tags that could cause issues with logger"""
        try:
            # Try to parse as JSON first
            return str(response.json())
        except:
            # If not JSON, return status code only to avoid HTML parsing issues
            return f"Status code: {response.status_code}"

    def _get_normalized_data_type(self, data_type: str, timeframe: str | None = None) -> list[str]:
        """
        Map input data type to normalized data types used by Tardis Machine API

        Parameters:
        -----------
        data_type : str
            Input data type (e.g., 'trade', 'orderBookL2', 'book')
        timeframe : str | None
            Optional timeframe for aggregation (e.g., '1m', '5m')

        Returns:
        --------
        list[str]
            List of normalized data types to request
        """
        if timeframe:
            return [f"trade_bar_{timeframe}"]

        # Map common data types to normalized data types
        if data_type in ["orderBookL2", "book", "orderbook"]:
            # For now, we'll only support book_snapshot and not book_change
            # as book_change can have isSnapshot=True which requires special handling
            # TODO: maybe change to 1000 levels and 1000ms interval
            return ["book_snapshot_500_1000ms"]
        elif data_type == "book_change":
            raise NotImplementedError(
                "book_change data type is not implemented yet due to complexity with isSnapshot handling"
            )
        elif data_type in ["trade", "trades"]:
            return ["trade"]
        elif data_type == "derivative_ticker":
            return ["derivative_ticker"]
        elif data_type.startswith("book_snapshot_"):
            # Allow direct specification of book_snapshot with levels and interval
            return [data_type]
        else:
            # Default to the provided data type
            return [data_type]

    def _get_column_names(self, data_type: str) -> list[str]:
        """
        Get standard column names for a normalized data type

        Parameters:
        -----------
        data_type : str
            Normalized data type (e.g., 'trade', 'book_change')

        Returns:
        --------
        list[str]
            List of column names in the standard order
        """
        # Extract the base type from data types like 'trade_bar_1m'
        base_type = data_type
        if data_type.startswith("trade_bar_"):
            base_type = "trade_bar"
        elif data_type.startswith("book_snapshot_"):
            base_type = "book_snapshot"

        return self.NORMALIZED_DATA_COLUMNS.get(base_type, [])

    def _parse_record(self, record: dict, data_type: str, tick_size_pct: float = 0.01, depth: int = 100) -> list | None:
        """
        Parse a record from the normalized API response into a list of values

        Parameters:
        -----------
        record : dict
            Record from the normalized API response
        data_type : str
            Normalized data type of the record

        Returns:
        --------
        list
            List of values in the standard order for the data type
        """
        if data_type == "book_snapshot":
            return self._parse_book_snapshot(record, tick_size_pct=tick_size_pct, depth=depth)

        if data_type == "trade":
            record["side"] = 1 if record["side"] == "buy" else -1

        column_names = self._get_column_names(data_type)
        values = [record.get(col) for col in column_names]
        return values

    def _parse_book_snapshot(self, record: dict, tick_size_pct: float = 0.01, depth: int = 100) -> list | None:
        """
        Parse a book snapshot record from the normalized API response into a list of values
        """
        if tick_size_pct == 0:
            raise ValueError("tick_size_pct=0 is not supported yet")

        bids = record.get("bids", [])
        asks = record.get("asks", [])
        if not bids or not asks:
            return None

        # bids are of type [{"price": float, "size": float}, ...]
        # turn them to bids array of shape (depth, 2)
        bids = np.array([[bid["price"], bid["amount"]] for bid in bids if "price" in bid and "amount" in bid])
        asks = np.array([[ask["price"], ask["amount"]] for ask in asks if "price" in ask and "amount" in ask])

        top_bid, top_ask = bids[0][0], asks[0][0]
        mid_price = (top_bid + top_ask) / 2
        tick_size = mid_price * tick_size_pct / 100

        raw_bids = np.zeros(depth, dtype=np.float64)
        raw_asks = np.zeros(depth, dtype=np.float64)

        top_bid, _bids = accumulate_orderbook_levels(bids, raw_bids, tick_size, True, depth, False)
        top_ask, _asks = accumulate_orderbook_levels(asks, raw_asks, tick_size, False, depth, False)

        return [record["localTimestamp"], top_bid, top_ask, tick_size, _bids, _asks]

    def get_names(self, exchange: str | None = None, data_type: str | None = None) -> list[str]:
        """
        Get available data names (symbols) for the specified exchange and data type

        Parameters:
        -----------
        exchange : str | None
            Optional exchange name to filter symbols
        data_type : str | None
            Optional data type to filter symbols (e.g., 'trades', 'book')

        Returns:
        --------
        list[str]
            List of available data names in the format 'exchange:symbol'
        """
        if not exchange:
            exchanges = self.get_exchanges()
            symbols = []
            for exch in exchanges:
                symbols.extend(self.get_names(exch, data_type))
            return symbols

        # Get exchange info to get available symbols
        exchange_info = self.get_exchange_info(exchange)
        if not exchange_info or "availableSymbols" not in exchange_info:
            return []

        symbols = []
        for symbol_info in exchange_info["availableSymbols"]:
            symbol_id = symbol_info["id"]

            # Filter by data type if specified
            if data_type and "availableChannels" in exchange_info:
                if data_type not in exchange_info["availableChannels"]:
                    continue

            symbols.append(f"{exchange}:{symbol_id}")

        return symbols

    @contextmanager
    def _stream_context(self, response):
        """Context manager to handle streaming response cleanup"""
        line_queue = Queue(maxsize=1000)  # Buffer for lines from the stream
        stop_event = threading.Event()

        def _read_lines():
            """Thread function to read lines from the response stream"""
            try:
                for line in response.iter_lines():
                    if stop_event.is_set():
                        break
                    if not line:
                        continue
                    line_queue.put(line)
            except Exception as e:
                logger.error(f"Error reading lines from stream: {str(e)}")
            finally:
                line_queue.put(None)  # Signal end of stream

        # Start the reader thread
        reader_thread = threading.Thread(target=_read_lines)
        reader_thread.daemon = True
        reader_thread.start()

        try:
            yield line_queue
        finally:
            stop_event.set()
            reader_thread.join()

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        timeframe=None,
        data_type="trade",
        tick_size_pct: float = 0.01,
        depth: int = 100,
        **kwargs,
    ) -> Iterable | Any:
        """
        Read data from Tardis Machine server using the normalized API endpoint

        Parameters:
        -----------
        data_id : str
            Data identifier in the format 'exchange:symbol'
        start : str | None
            Start date/time for data retrieval
        stop : str | None
            End date/time for data retrieval
        transform : DataTransformer
            Transformer to process the retrieved data
        chunksize : int
            Size of chunks for processing data. If > 0, data will be processed in chunks.
        timeframe : str | None
            Optional timeframe for data aggregation (e.g., '1m', '5m')
            If provided, will use trade_bar_{timeframe} data type
        data_type : str
            Type of data to retrieve (e.g., 'trade', 'book_change', 'book_snapshot')
        tick_size_pct : float
            Tick size percentage for book_snapshot data type
        depth : int
            Depth for book_snapshot data type
        **kwargs : dict
            Additional keyword arguments

        Returns:
        --------
        Iterable | Any
            Processed data according to the specified transformer
        """
        try:
            exchange, symbol = data_id.split(":", 1)
        except ValueError:
            raise ValueError(f"Invalid data_id format: {data_id}. Expected format: 'exchange:symbol'")

        exchange = TARDIS_EXCHANGE_MAPPERS.get(exchange.lower(), exchange)

        start_date, end_date = handle_start_stop(start, stop)
        if not start_date:
            raise ValueError("Start date must be provided for TardisMachineReader")

        # Handle start and stop dates
        actual_start_date = pd.Timestamp(start_date)
        start_date = (
            actual_start_date.floor("d").isoformat()
            if data_type.startswith(("book_snapshot", "orderbook"))
            else actual_start_date.isoformat()
        )
        end_date = pd.Timestamp(stop).isoformat() if stop else (actual_start_date + pd.Timedelta(days=1)).isoformat()

        # latest 15min of data is not available
        _now = pd.Timestamp(time_now())
        _now_offset = pd.Timedelta("15min")
        if _now - pd.Timestamp(stop) < _now_offset:
            actual_start_date -= _now_offset
            end_date = (_now - _now_offset).isoformat()

        # Determine the data types to request
        data_types = self._get_normalized_data_type(data_type, timeframe)

        # Format the request for the normalized API
        options = {
            "exchange": exchange,
            "symbols": [symbol],
            "from": start_date,
            "to": end_date,
            "dataTypes": data_types,
            "withDisconnectMessages": False,
        }

        # Construct the URL with options as a query parameter
        url = f"{self.machine_url}/replay-normalized?options={json.dumps(options)}"

        logger.debug(f"Requesting data from Tardis Machine normalized API: {url}")
        response = requests.get(url, headers=self._get_headers(), stream=True)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch data from Tardis Machine normalized API: {self._safe_log_response(response)}"
            )

        try:
            record_type = None
            column_names = []  # Initialize with empty list instead of None
            current_chunk = []
            prev_record_time = None

            def _iter_chunks():
                nonlocal record_type, column_names, current_chunk, prev_record_time
                with self._stream_context(response) as line_queue:
                    while True:
                        line = line_queue.get()
                        if line is None:  # End of stream
                            break

                        # Parse the JSON object
                        record = self._parse_line(line)
                        if not record:
                            continue

                        # Skip disconnect messages unless specifically requested
                        if record.get("type") == "disconnect" and "disconnect" not in data_types:
                            continue

                        # Get the record type
                        current_type = record.get("type", "")

                        # For the first valid record, set the record type and column names
                        if not record_type and current_type:
                            record_type = current_type
                            column_names = self._get_column_names(record_type)

                        # Only process records of the same type
                        if current_type == record_type:
                            # Skip records before actual start time for book snapshot data
                            if current_type == "book_snapshot":
                                if "localTimestamp" not in record:
                                    continue
                                record_time = pd.Timestamp(record["localTimestamp"])
                                if prev_record_time is None or record_time.floor("h") != prev_record_time.floor("h"):
                                    prev_record_time = record_time
                                    logger.debug(f"[{data_id}] :: New hour: {record_time}")
                                if record_time < actual_start_date:
                                    continue

                            # Parse the record into a list of values
                            values = self._parse_record(record, current_type, tick_size_pct=tick_size_pct, depth=depth)
                            if values:
                                current_chunk.append(values)

                                # If we have enough records for a chunk, process and yield it
                                if len(current_chunk) >= chunksize:
                                    transform.start_transform(data_id, column_names, start=start, stop=stop)
                                    transform.process_data(current_chunk)
                                    yield transform.collect()
                                    current_chunk = []

                    # Process any remaining records
                    if current_chunk:
                        transform.start_transform(data_id, column_names, start=start, stop=stop)
                        transform.process_data(current_chunk)
                        yield transform.collect()

            if chunksize > 0:
                return _iter_chunks()

            # For non-chunked processing, collect all records first
            all_records = []
            with self._stream_context(response) as line_queue:
                while True:
                    line = line_queue.get()
                    if line is None:  # End of stream
                        break

                    record = self._parse_line(line)
                    if not record:
                        continue

                    if record.get("type") == "disconnect" and "disconnect" not in data_types:
                        continue

                    current_type = record.get("type", "")

                    if not record_type and current_type:
                        record_type = current_type
                        column_names = self._get_column_names(record_type)

                    if current_type == record_type:
                        if current_type == "book_snapshot":
                            if "localTimestamp" not in record:
                                continue
                            record_time = pd.Timestamp(record["localTimestamp"])
                            if record_time < actual_start_date:
                                continue

                        values = self._parse_record(record, current_type, tick_size_pct=tick_size_pct, depth=depth)
                        if values:
                            all_records.append(values)

            # Process all records at once
            if all_records and record_type:
                transform.start_transform(data_id, column_names, start=start, stop=stop)
                transform.process_data(all_records)
                return transform.collect()
            else:
                logger.warning("No valid records found in Tardis Machine normalized response")
                return None

        except Exception as e:
            logger.warning(f"Error processing Tardis Machine normalized response: {str(e)}")
            return None

    def get_exchanges(self) -> list[str]:
        """
        Get list of available exchanges from Tardis.dev API

        Returns:
        --------
        list[str]
            List of exchange IDs
        """
        if self._exchanges_cache is not None:
            return self._exchanges_cache

        try:
            url = f"{self.api_url}/exchanges"
            response = requests.get(url, headers=self._get_headers())

            if response.status_code != 200:
                logger.warning(f"Failed to get exchanges: {self._safe_log_response(response)}")
                return []

            exchanges_data = response.json()
            self._exchanges_cache = [exchange["id"] for exchange in exchanges_data]
            return self._exchanges_cache

        except Exception as e:
            logger.error(f"Error getting exchanges: {str(e)}")
            return []

    def get_exchange_info(self, exchange: str) -> dict | None:
        """
        Get detailed information about an exchange from Tardis.dev API

        Parameters:
        -----------
        exchange : str
            Exchange ID

        Returns:
        --------
        dict | None
            Dictionary containing exchange information or None if not found
        """
        exchange = TARDIS_EXCHANGE_MAPPERS.get(exchange.lower(), exchange)
        if exchange in self._exchange_info_cache:
            return self._exchange_info_cache[exchange]

        try:
            url = f"{self.api_url}/exchanges/{exchange}"
            response = requests.get(url, headers=self._get_headers())

            if response.status_code != 200:
                logger.warning(f"Failed to get exchange info for {exchange}: {self._safe_log_response(response)}")
                return None

            exchange_info = response.json()
            self._exchange_info_cache[exchange] = exchange_info
            return exchange_info

        except Exception as e:
            logger.error(f"Error getting exchange info for {exchange}: {str(e)}")
            return None

    def get_data_types(self, exchange: str) -> list[str]:
        """
        Get available data types (channels) for an exchange

        Parameters:
        -----------
        exchange : str
            Exchange ID

        Returns:
        --------
        list[str]
            List of available data types
        """
        exchange_info = self.get_exchange_info(exchange)
        if not exchange_info or "availableChannels" not in exchange_info:
            return []

        return exchange_info["availableChannels"]

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        """
        Get symbols for a specific exchange and data type

        Parameters:
        -----------
        exchange : str
            Exchange ID
        dtype : str
            Data type

        Returns:
        --------
        list[str]
            List of symbols in the format 'exchange:symbol'
        """
        exchange_info = self.get_exchange_info(exchange)
        if not exchange_info or "availableSymbols" not in exchange_info:
            return []

        # Check if the data type is available for this exchange
        if dtype and "availableChannels" in exchange_info:
            if dtype not in exchange_info["availableChannels"]:
                return []

        return [f"{exchange}:{symbol['id']}" for symbol in exchange_info["availableSymbols"]]

    def get_time_ranges(self, symbol: str, dtype: str) -> tuple[np.datetime64, np.datetime64]:
        """
        Get available time range for a symbol and data type

        Parameters:
        -----------
        symbol : str
            Symbol in the format 'exchange:symbol'
        dtype : str
            Data type

        Returns:
        --------
        tuple[np.datetime64, np.datetime64]
            Tuple of (start_time, end_time)
        """
        try:
            exchange, symbol_id = symbol.split(":")
            exchange_info = self.get_exchange_info(exchange)
            if not exchange_info or "availableSymbols" not in exchange_info:
                return np.datetime64("NaT"), np.datetime64("NaT")

            # Find the symbol in the available symbols
            for symbol_info in exchange_info["availableSymbols"]:
                if symbol_info["id"].lower() == symbol_id.lower():
                    # Get the availability dates
                    available_since = symbol_info.get("availableSince")
                    if available_since:
                        start_time = np.datetime64(available_since)
                        # End time is typically the current date
                        end_time = np.datetime64(pd.Timestamp.now().date())
                        return start_time, end_time

            return np.datetime64("NaT"), np.datetime64("NaT")

        except Exception as e:
            logger.error(f"Error getting time ranges for {symbol}: {str(e)}")
            return np.datetime64("NaT"), np.datetime64("NaT")

    def _parse_line(self, line: str) -> dict | None:
        try:
            record = orjson.loads(line)
            for col in ["timestamp", "localTimestamp"]:
                record[col] = record[col].rstrip("Z")
            return record
        except Exception as _:
            logger.warning(f"Failed to parse line as JSON: {line[:100]}...")
            return None
