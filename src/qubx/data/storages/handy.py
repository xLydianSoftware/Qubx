"""
In-memory storage for ad-hoc research data.

Supports multiple input formats for quick research and backtesting
without requiring external data sources.
"""

from collections import defaultdict
from collections.abc import Iterator

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import DataType
from qubx.core.utils import time_delta_to_str
from qubx.data.containers import IteratorsMaster, RawData, RawMultiData
from qubx.data.registry import storage
from qubx.data.storage import IReader, IStorage, Transformable
from qubx.utils.time import handle_start_stop, infer_series_frequency

# - known market type defaults for common exchanges
_MARKET_DEFAULTS = {
    "BINANCE.UM": "SWAP",
    "BINANCE.PM": "SWAP",
    "BINANCE": "SPOT",
    "BYBIT": "SWAP",
    "OKX": "SWAP",
}

# - column sets for data type detection
_OHLC_REQUIRED = {"open", "high", "low", "close"}
_QUOTE_INDICATORS = {"bid", "ask", "bid_price", "ask_price"}
_TRADE_PRICE = {"price"}
_TRADE_SIZE = {"size", "amount", "quantity"}


def _infer_dtype(df: pd.DataFrame) -> str:
    """
    Infer DataType string from DataFrame columns.
    Returns string like 'ohlc(1h)', 'quote', 'trade', etc.
    """
    cols = {c.lower() for c in df.columns}

    if _OHLC_REQUIRED.issubset(cols):
        try:
            tf = time_delta_to_str(infer_series_frequency(df).item())
        except (ValueError, TypeError):
            tf = "1h"
        # - normalize to lowercase: DataPump always requests "ohlc(1d)", "ohlc(1min)", etc.
        # - time_delta_to_str may return mixed-case e.g. "1D", "1Min", "15Min"
        return DataType.OHLC[tf.lower()]

    if cols & _QUOTE_INDICATORS:
        return DataType.QUOTE

    if _TRADE_PRICE & cols and _TRADE_SIZE & cols:
        return DataType.TRADE

    if "funding_rate" in cols:
        return DataType.FUNDING_RATE

    if "open_interest" in cols:
        return DataType.OPEN_INTEREST

    if cols & {"market_cap", "total_volume"}:
        return DataType.FUNDAMENTAL

    return DataType.RECORD


def _parse_exchange_market(exchange: str | None) -> tuple[str, str]:
    """
    Parse exchange string into (exchange, market_type) tuple.

    'BINANCE.UM'      -> ('BINANCE.UM', 'SWAP')
    'BINANCE.UM:SWAP' -> ('BINANCE.UM', 'SWAP')
    None              -> ('HANDY', 'DATA')
    """
    if exchange is None:
        return ("HANDY", "DATA")

    parts = exchange.split(":")
    if len(parts) >= 2:
        return (parts[0].upper(), parts[1].upper())

    ex = parts[0].upper()
    return (ex, _MARKET_DEFAULTS.get(ex, "SWAP"))


def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has a named DatetimeIndex so RawData.from_pandas
    correctly resets it into a time column.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.name:
            df = df.copy()
            df.index.name = "timestamp"
        return df

    # - check if there's a time-like column already in the data
    for col in ("time", "timestamp", "datetime", "date", "open_time", "ts"):
        if col in df.columns:
            return df

    return df


class HandyReader(IReader):
    """
    In-memory reader serving DataFrames stored in HandyStorage.
    """

    def __init__(self):
        # - {symbol_upper -> {dtype_str -> pd.DataFrame}}
        self._data: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)

    def add(self, symbol: str, dtype: str, df: pd.DataFrame):
        """
        Add a DataFrame for given symbol and data type.
        """
        self._data[symbol.upper()][dtype] = df

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        if str(dtype) == str(DataType.ALL):
            return list(self._data.keys())
        ds = str(dtype)
        return [sym for sym, dtypes in self._data.items() if ds in dtypes]

    def get_data_types(self, data_id: str) -> list[str]:
        return list(self._data.get(data_id.upper(), {}).keys())

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[np.datetime64, np.datetime64]:
        df = self._data.get(data_id.upper(), {}).get(str(dtype))
        if df is None or df.empty:
            return (np.datetime64("NaT"), np.datetime64("NaT"))
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)
        return (np.datetime64(idx[0]), np.datetime64(idx[-1]))

    def _filter_time(self, df: pd.DataFrame, start: str | None, stop: str | None) -> pd.DataFrame:
        """
        Filter DataFrame by time range using handle_start_stop.
        """
        if start is None and stop is None:
            return df
        t0, t1 = handle_start_stop(start, stop)
        if t0 is not None:
            df = df[df.index >= pd.Timestamp(t0)]
        if t1 is not None:
            # - exclusive stop boundary (start <= timestamp < stop) to match QuestDB/CSV semantics
            df = df[df.index < pd.Timestamp(t1)]
        return df

    def _read_single(
        self,
        data_id: str,
        dtype: DataType | str,
        start: str | None = None,
        stop: str | None = None,
        chunksize: int = 0,
    ) -> RawData | Iterator[RawData]:
        dtype_str = str(dtype)
        df = self._data.get(data_id.upper(), {}).get(dtype_str)
        if df is None:
            available = self.get_data_types(data_id)
            raise ValueError(f"No data for '{data_id}' of type '{dtype_str}'. Available: {available}")

        df = self._filter_time(df, start, stop)

        if chunksize > 0:

            def _chunks():
                for i in range(0, len(df), chunksize):
                    chunk = df.iloc[i : i + chunksize]
                    yield RawData.from_pandas(data_id, dtype_str, chunk)

            return _chunks()

        return RawData.from_pandas(data_id, dtype_str, df)

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None = None,
        stop: str | None = None,
        chunksize: int = 0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        if isinstance(data_id, (list, tuple, set)):
            # - empty collection → all available symbols for this dtype
            ids = self.get_data_id(dtype) if not data_id else list(data_id)
            multi = [self._read_single(d, dtype, start, stop, chunksize) for d in ids]
            return IteratorsMaster(multi) if chunksize > 0 else RawMultiData(multi)
        return self._read_single(data_id, dtype, start, stop, chunksize)


@storage("handy")
class HandyStorage(IStorage):
    """
    In-memory storage for ad-hoc research data.

    Supports multiple input formats:

        # - Full instrument spec in keys
        s = HandyStorage({"BINANCE.UM:SWAP:BTCUSDT": df1, "BINANCE.UM:SWAP:ETHUSDT": df2})

        # - Symbol keys + exchange parameter
        s = HandyStorage({"BTCUSDT": df1, "ETHUSDT": df2}, exchange="BINANCE.UM")

        # - Multiple data types per symbol (list of DataFrames)
        s = HandyStorage({"BTCUSDT": [df_ohlc, df_quotes]}, exchange="BINANCE.UM:SWAP")

        # - MultiIndex DataFrame with (timestamp, symbol) levels
        s = HandyStorage(multi_df, exchange="BINANCE.UM:SWAP")

    Usage with reader:
        reader = s["BINANCE.UM", "SWAP"]
        raw = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-06-01")
        df = raw.to_pd()
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame | list[pd.DataFrame]] | pd.DataFrame,
        exchange: str | None = None,
    ):
        # - {exchange_upper -> {market_upper -> HandyReader}}
        self._readers: dict[str, dict[str, HandyReader]] = defaultdict(dict)
        self._ingest(data, exchange)

    def _get_or_create_reader(self, exchange: str, market: str) -> HandyReader:
        ex, mt = exchange.upper(), market.upper()
        if mt not in self._readers[ex]:
            self._readers[ex][mt] = HandyReader()
        return self._readers[ex][mt]

    def _ingest_single(self, symbol: str, df: pd.DataFrame, exchange: str, market: str):
        df = _ensure_time_index(df)
        reader = self._get_or_create_reader(exchange, market)
        dtype_str = _infer_dtype(df)
        reader.add(symbol, dtype_str, df)
        logger.debug(f"HandyStorage: ingested {symbol} as {dtype_str} into {exchange}:{market}")

    def _ingest(
        self,
        data: dict[str, pd.DataFrame | list[pd.DataFrame]] | pd.DataFrame,
        exchange: str | None,
    ):
        # - Form 4: MultiIndex DataFrame
        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.MultiIndex):
                raise ValueError("DataFrame input must have MultiIndex (timestamp, symbol)")

            ex, mt = _parse_exchange_market(exchange)
            idx_names = [str(n).lower() if n else "" for n in data.index.names]
            time_names = {"timestamp", "time", "date", "datetime", "open_time", "ts"}

            # - detect which level is time vs symbol
            if idx_names[0] in time_names:
                symbol_level = 1
            elif idx_names[1] in time_names:
                symbol_level = 0
            else:
                # - assume first level is time (most common: timestamp, symbol)
                symbol_level = 1

            for symbol in data.index.get_level_values(symbol_level).unique():
                df_sym = data.xs(symbol, level=symbol_level)
                if not isinstance(df_sym.index, pd.DatetimeIndex):
                    df_sym.index = pd.DatetimeIndex(df_sym.index)
                self._ingest_single(str(symbol), df_sym, ex, mt)
            return

        # - Forms 1-3: dict input
        for key, value in data.items():
            parts = key.split(":")
            if len(parts) >= 3 and exchange is None:
                # - Form 1: "BINANCE.UM:SWAP:BTCUSDT"
                ex, mt, symbol = parts[0], parts[1], ":".join(parts[2:])
            else:
                # - Forms 2 & 3: symbol-only keys
                symbol = key
                ex, mt = _parse_exchange_market(exchange)

            if isinstance(value, list):
                # - Form 3: list of DataFrames per symbol
                for df in value:
                    self._ingest_single(symbol, df, ex, mt)
            else:
                self._ingest_single(symbol, value, ex, mt)

    def get_exchanges(self) -> list[str]:
        return list(self._readers.keys())

    def get_market_types(self, exchange: str) -> list[str]:
        return list(self._readers.get(exchange.upper(), {}).keys())

    def get_reader(self, exchange: str, market: str) -> IReader:
        ex, mt = exchange.upper(), market.upper()
        if ex not in self._readers or mt not in self._readers[ex]:
            available = {e: list(ms.keys()) for e, ms in self._readers.items()}
            raise ValueError(f"No data for {ex}:{mt}. Available: {available}")
        return self._readers[ex][mt]
