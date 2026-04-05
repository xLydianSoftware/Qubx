"""
CCXT-backed IStorage / IReader for live and recent market data.

Architecture
------------
- ``CcxtStorage`` — top-level IStorage, registered as ``@storage("ccxt")``.
  Exchange-agnostic: any exchange is connected **lazily** on the first
  ``get_reader(exchange, market)`` call.  Multiple exchanges share one
  ``AsyncThreadLoop``.

- ``CcxtReader`` — thin per-(exchange, market) IReader that delegates to
  the parent ``CcxtStorage``.

Data pipeline
-------------
CCXT returns raw Python lists/dicts.  These are converted **directly** to
PyArrow RecordBatches (via ``RawData.from_record_batch``) — no intermediate
pandas DataFrames are created.

Registration
------------
    StorageRegistry.get("ccxt")          # or just CcxtStorage()
    StorageRegistry.get("ccxt::")        # same

    # multiple exchanges via MultiStorage
    MultiStorage([CcxtStorage(), CcxtStorage()])
    # (or a single instance handles all via repeated get_reader calls)

Supported data types
--------------------
- ``ohlc(<timeframe>)``  — OHLCV candles  (e.g. ``ohlc(1h)``)
- ``funding_payment``    — funding payment history
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from ccxt.pro import Exchange

from qubx import logger
from qubx.core.basics import DataType, Instrument
from qubx.data.containers import IteratorsMaster, RawData, RawMultiData
from qubx.data.registry import storage
from qubx.data.storage import IReader, IStorage, Transformable
from qubx.utils.misc import AsyncThreadLoop
from qubx.utils.time import handle_start_stop, now_utc, to_timedelta

# - default market type per well-known exchange name
_EXCHANGE_MARKET_DEFAULTS: dict[str, str] = {
    "BINANCE": "SPOT",
    "BINANCE.UM": "SWAP",
    "BINANCE.CM": "SWAP",
    "BINANCE.PM": "SWAP",
    "HYPERLIQUID": "SWAP",
    "HYPERLIQUID.F": "SWAP",
    "BYBIT": "SWAP",
    "OKX": "SWAP",
    "OKX.F": "SWAP",
    "KRAKEN.F": "SWAP",
    "BITFINEX": "SPOT",
    "BITFINEX.F": "SWAP",
}

# - PyArrow schemas for each supported data type
_OHLCV_SCHEMA = pa.schema(
    [
        ("timestamp", pa.timestamp("ms")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("quote_volume", pa.float64()),
    ]
)

_FUNDING_SCHEMA = pa.schema(
    [
        ("timestamp", pa.timestamp("ms")),
        ("funding_rate", pa.float64()),
        ("funding_interval_hours", pa.float64()),
    ]
)


# ------------------------------------------------------------------
# Arrow helpers — build RawData without any pandas DataFrames
# ------------------------------------------------------------------


def _candles_to_raw(symbol: str, dtype_str: str, candles: list) -> RawData:
    """
    Build ``RawData`` directly from CCXT raw candle list.

    Each candle is ``[ts_ms, open, high, low, close, volume]``.
    """
    if candles:
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([c[0] for c in candles], type=pa.timestamp("ms")),
                pa.array([float(c[1]) for c in candles]),
                pa.array([float(c[2]) for c in candles]),
                pa.array([float(c[3]) for c in candles]),
                pa.array([float(c[4]) for c in candles]),
                pa.array([float(c[5]) for c in candles]),
                pa.array([float(c[4]) * float(c[5]) for c in candles]),
            ],
            schema=_OHLCV_SCHEMA,
        )
    else:
        batch = pa.RecordBatch.from_pydict(
            {f.name: pa.array([], type=f.type) for f in _OHLCV_SCHEMA},
            schema=_OHLCV_SCHEMA,
        )
    return RawData.from_record_batch(symbol, dtype_str, batch)


def _funding_rows_to_raw(
    symbol: str,
    rows: list[tuple[int, float, float]],
) -> RawData:
    """
    Build ``RawData`` from funding rows ``[(ts_ms_floored, rate, hours), ...]``.
    """
    if rows:
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([r[0] for r in rows], type=pa.timestamp("ms")),
                pa.array([r[1] for r in rows]),
                pa.array([r[2] for r in rows]),
            ],
            schema=_FUNDING_SCHEMA,
        )
    else:
        batch = pa.RecordBatch.from_pydict(
            {f.name: pa.array([], type=f.type) for f in _FUNDING_SCHEMA},
            schema=_FUNDING_SCHEMA,
        )
    return RawData.from_record_batch(symbol, "funding_payment", batch)


def _floor_ms(ts_ms: int, interval_hours: float) -> int:
    """
    Floor a millisecond timestamp to the nearest funding interval boundary.
    """
    interval_ms = int(interval_hours * 3_600_000)
    return (ts_ms // interval_ms) * interval_ms


def _infer_funding_interval_hours(timestamps_ms: list[int], default_hours: float = 8.0) -> list[float]:
    """
    Infer per-record funding interval from consecutive timestamp differences.

    Uses the time gap to the *next* record (or previous for the last one).
    Falls back to *default_hours* for single records or when the inferred
    gap is outside a reasonable range (0.5 h – 24 h).
    """
    n = len(timestamps_ms)
    if n == 0:
        return []
    if n == 1:
        return [default_hours]

    intervals: list[float] = []
    for i in range(n):
        if i < n - 1:
            diff_h = (timestamps_ms[i + 1] - timestamps_ms[i]) / 3_600_000
        else:
            diff_h = (timestamps_ms[i] - timestamps_ms[i - 1]) / 3_600_000
        if diff_h < 0.5 or diff_h > 24:
            diff_h = default_hours
        intervals.append(diff_h)
    return intervals


class CcxtReader(IReader):
    """
    Per-(exchange, market) IReader backed by ``CcxtStorage``.

    Delegates all fetching to the parent storage's async methods.
    Results are returned as ``RawData`` / ``RawMultiData`` built
    directly from PyArrow RecordBatches.

    Lifecycle is owned by the parent ``CcxtStorage`` — do **not** call
    ``close()`` on a reader; call ``CcxtStorage.close()`` instead.
    """

    def __init__(self, exchange: str, market: str, storage_ref: "CcxtStorage") -> None:
        self._exchange = exchange.upper()
        self._market = market.upper()
        self._storage = storage_ref

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        return self._storage._get_symbols(self._exchange, self._market)

    def get_data_types(self, data_id: str) -> list[DataType]:
        return [DataType.OHLC, DataType.FUNDING_PAYMENT]

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[np.datetime64, np.datetime64]:
        """
        Approximate time range: (now − max_history, now).
        """
        end_time = now_utc()
        start_time = end_time - self._storage._max_history
        return start_time.to_datetime64(), end_time.to_datetime64()

    def _read_single(
        self,
        symbol: str,
        dtype_str: str,
        dt: DataType,
        params: dict[str, Any],
        start: str | None,
        stop: str | None,
        chunksize: int,
    ) -> RawData | Iterator[RawData]:
        raw = self._storage._fetch_single(self._exchange, self._market, symbol, dt, params, dtype_str, start, stop)
        if chunksize > 0:

            def _chunks(r: RawData = raw) -> Iterator[RawData]:
                total = len(r)
                for i in range(0, max(total, 1), chunksize):
                    batch_slice = r.data.slice(i, min(chunksize, total - i))
                    yield RawData.from_record_batch(symbol, dtype_str, batch_slice)

            return _chunks()
        return raw

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None = None,
        stop: str | None = None,
        chunksize: int = 0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        dtype_str = str(dtype) if isinstance(dtype, DataType) else dtype
        dt, params = DataType.from_str(dtype_str)

        if isinstance(data_id, (list, tuple, set)):
            symbols: list[str] = self.get_data_id(dt) if not data_id else list(data_id)
            if len(symbols) > 20:
                logger.warning(
                    f"[CCXT] Skipping bulk read of {len(symbols)} symbols for {dtype_str} on {self._exchange} — "
                    f"ccxt is too slow for bulk queries. Use a database storage (e.g., qdb) instead."
                )
                return RawMultiData([])
            raw_list: list[RawData] = self._storage._fetch_multi(
                self._exchange, self._market, symbols, dt, params, dtype_str, start, stop
            )

            if chunksize > 0:

                def _to_iter(r: RawData) -> Iterator[RawData]:
                    sym = r.data_id
                    total = len(r)
                    for i in range(0, max(total, 1), chunksize):
                        batch_slice = r.data.slice(i, min(chunksize, total - i))
                        yield RawData.from_record_batch(sym, dtype_str, batch_slice)

                return IteratorsMaster([_to_iter(r) for r in raw_list])

            return RawMultiData(raw_list)

        return self._read_single(data_id, dtype_str, dt, params, start, stop, chunksize)

    def close(self) -> None:
        # - lifecycle owned by CcxtStorage
        pass


@storage("ccxt")
class CcxtStorage(IStorage):
    """
    IStorage implementation backed by CCXT Pro for live and recent market data.

    Exchange-agnostic: any exchange is connected **lazily** on the first
    ``get_reader(exchange, market)`` call.  All exchanges share one
    ``AsyncThreadLoop``.

    Registration name: ``ccxt`` (via ``@storage`` decorator).

    Usage::

        from qubx.data.storages.ccxt import CcxtStorage

        s = CcxtStorage()
        r = s.get_reader("BINANCE.UM", "SWAP")
        df = r.read("BTCUSDT", "ohlc(1h)", "2025-01-01", "2025-02-01").to_pd()

        # - second exchange reuses the same loop (no extra thread)
        r2 = s.get_reader("HYPERLIQUID", "SWAP")

    Args:
        max_bars:    Maximum candles per CCXT fetch request (default 10 000).
        max_history: Maximum look-back window (default ``"3650d"``).

    Note:
        An optional first positional string argument is accepted but ignored.
        This allows the URI form ``ccxt::BINANCE.UM`` to work via ``StorageRegistry.get`` — the exchange name in the URI is not needed
        since exchanges are connected lazily via ``get_reader()``.
    """

    _max_bars: int
    _max_history: pd.Timedelta
    _loop: AsyncThreadLoop | None
    _capabilities: Any | None

    def __init__(
        self,
        _uri_hint: str = "",  # - ignored; absorbed from URI "ccxt::EXCHANGE" via StorageRegistry
        max_bars: int = 10_000,
        max_history: str = "3650d",
    ) -> None:
        self._max_bars = max_bars
        self._max_history = to_timedelta(max_history)
        # - shared async loop (created from first exchange connection)
        self._loop = None
        self._capabilities = None
        # - per-exchange state (all lazily populated)
        self._exchanges: dict[str, Exchange] = {}
        self._symbol_maps: dict[tuple[str, str], dict[str, tuple[str, Instrument]]] = {}
        self._instrument_caches: dict[str, dict[str, Instrument]] = {}

    def _ensure_exchange(self, exchange: str) -> Exchange:
        """
        Create and cache the CCXT Pro exchange object for *exchange* on first use.
        All exchanges share the same ``AsyncThreadLoop``.
        """
        if exchange in self._exchanges:
            return self._exchanges[exchange]

        from qubx.connectors.ccxt.exchanges import READER_CAPABILITIES
        from qubx.connectors.ccxt.factory import get_ccxt_exchange

        # - reuse existing loop so all exchanges run on one thread
        existing_loop = self._loop.loop if self._loop is not None else None
        ccxt_ex = get_ccxt_exchange(exchange, loop=existing_loop)

        if self._loop is None:
            _loop = getattr(ccxt_ex, "asyncio_loop", None)
            assert _loop is not None, f"CcxtStorage: asyncio_loop not found on {exchange}"
            self._loop = AsyncThreadLoop(_loop)

        if self._capabilities is None:
            self._capabilities = READER_CAPABILITIES.copy()

        self._loop.submit(ccxt_ex.load_markets()).result()
        self._exchanges[exchange] = ccxt_ex
        self._instrument_caches[exchange] = {}
        return ccxt_ex

    def get_exchanges(self) -> list[str]:
        # - return only exchanges that have been connected so far
        return list(self._exchanges.keys())

    def get_market_types(self, exchange: str) -> list[str]:
        return [_EXCHANGE_MARKET_DEFAULTS.get(exchange.upper(), "SWAP")]

    def get_reader(self, exchange: str, market: str) -> IReader:
        _ex = exchange.upper()
        self._ensure_exchange(_ex)
        return CcxtReader(_ex, market.upper(), self)

    def close(self) -> None:
        """
        Close all CCXT exchange connections.
        """
        if not self._exchanges:
            return
        for ex_name, ccxt_ex in list(self._exchanges.items()):
            if self._loop is not None and callable(getattr(ccxt_ex, "close", None)):
                try:
                    self._loop.submit(ccxt_ex.close()).result(timeout=5)
                except Exception as e:
                    logger.warning(f"[CCXT] Error closing {ex_name}: {e}")
        self._exchanges.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _get_or_build_symbol_map(self, exchange: str, market: str) -> dict[str, tuple[str, Instrument]]:
        """
        Lazy-build ``{qubx_symbol: (ccxt_symbol, Instrument)}`` for *(exchange, market)*.

        Filters by CCXT market type so that e.g. options on Gate.io never
        collide with perpetual swaps that share the same base/quote pair.
        """
        key = (exchange, market)
        if key in self._symbol_maps:
            return self._symbol_maps[key]

        from qubx.connectors.ccxt.utils import ccxt_find_instrument

        ccxt_ex = self._exchanges[exchange]
        instr_cache = self._instrument_caches[exchange]
        target_type = market.lower()  # Qubx uses "SWAP"; CCXT uses "swap"

        sym_map: dict[str, tuple[str, Instrument]] = {}
        for ccxt_sym in list(ccxt_ex.markets.keys()):
            try:
                if ccxt_ex.markets[ccxt_sym].get("type", "").lower() != target_type:
                    continue
                instr = ccxt_find_instrument(ccxt_sym, ccxt_ex, instr_cache)
                sym_map[instr.symbol] = (ccxt_sym, instr)
            except Exception:
                pass
        self._symbol_maps[key] = sym_map
        return sym_map

    def _get_symbols(self, exchange: str, market: str) -> list[str]:
        return list(self._get_or_build_symbol_map(exchange, market).keys())

    def _find_instrument(self, exchange: str, market: str, qubx_symbol: str) -> tuple[str, Instrument] | None:
        return self._get_or_build_symbol_map(exchange, market).get(qubx_symbol)

    def _find_instruments(self, exchange: str, market: str, qubx_symbols: list[str]) -> list[tuple[str, Instrument]]:
        sym_map = self._get_or_build_symbol_map(exchange, market)
        result = []
        for sym in qubx_symbols:
            entry = sym_map.get(sym)
            if entry is not None:
                result.append(entry)
            else:
                logger.warning(f"[CCXT] Symbol '{sym}' not found on {exchange} — skipped")
        return result

    def _get_start_stop(
        self,
        start: str | pd.Timestamp | None,
        stop: str | pd.Timestamp | None,
        timeframe: pd.Timedelta,
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

    async def _async_paginated_fetch(
        self,
        ccxt_ex: Exchange,
        method: Callable,
        since: int,
        until: int,
        limit: int = 1000,
        get_ts: Callable = lambda item: item["timestamp"],
        max_pages: int = 200,
        **method_kwargs: Any,
    ) -> list:
        """
        Generic paginated async fetch for any CCXT method returning timestamped records.

        Args:
            method:        Bound async method, e.g. ``ccxt_ex.fetch_ohlcv``.
            since/until:   Millisecond timestamp range (inclusive).
            get_ts:        Extracts the ms timestamp from a single record.
            max_pages:     Safety cap on the number of pagination rounds.
            method_kwargs: Extra keyword arguments forwarded to *method* on every call.
        """
        all_items: list = []
        current_since = since

        for _ in range(max_pages):
            batch = await method(since=current_since, limit=limit, **method_kwargs)
            if not batch:
                break
            for item in batch:
                if get_ts(item) <= until:
                    all_items.append(item)
            last_ts = get_ts(batch[-1])
            if last_ts >= until:
                break
            # no progress — exchange returned same or older data
            if last_ts < current_since:
                break
            current_since = last_ts + 1

        return all_items

    async def _async_fetch_ohlcv(
        self,
        ccxt_ex: Exchange,
        ccxt_symbol: str,
        exc_tf: str,
        since: int,
        until: int,
    ) -> list:
        """
        Paginated async OHLCV fetch.
        Returns raw candle list ``[[ts_ms, o, h, l, c, v], ...]``.
        """
        _ex_id = ccxt_ex.id.upper()
        logger.debug(
            f"[{_ex_id}] [{ccxt_symbol}] fetching OHLCV({exc_tf}) "
            f"from {pd.Timestamp(since, unit='ms')} to {pd.Timestamp(until, unit='ms')}"
        )
        return await self._async_paginated_fetch(
            ccxt_ex,
            ccxt_ex.fetch_ohlcv,
            since,
            until,
            get_ts=lambda c: c[0],
            symbol=ccxt_symbol,
            timeframe=exc_tf,
        )

    async def _async_fetch_ohlcv_multi(
        self,
        ccxt_ex: Exchange,
        instruments_info: list[tuple[str, str]],  # (ccxt_sym, qubx_sym)
        exc_tf: str,
        since: int,
        until: int,
    ) -> dict[str, list]:
        """
        Concurrent OHLCV fetch for multiple instruments via asyncio.gather.
        Returns ``{qubx_sym: raw_candles}``.
        """
        coros = [self._async_fetch_ohlcv(ccxt_ex, ccxt_sym, exc_tf, since, until) for ccxt_sym, _ in instruments_info]
        results = await asyncio.gather(*coros, return_exceptions=True)
        out: dict[str, list] = {}
        for i, result in enumerate(results):
            _, qubx_sym = instruments_info[i]
            if isinstance(result, Exception):
                logger.warning(f"[CCXT] Failed to fetch OHLCV for {qubx_sym}: {result}")
                out[qubx_sym] = []
            else:
                out[qubx_sym] = cast(list, result)
        return out

    def _ccxt_sym_to_qubx(self, ccxt_symbol: str) -> str:
        if "/" in ccxt_symbol:
            base = ccxt_symbol.split("/")[0]
            quote = ccxt_symbol.split("/")[1].split(":")[0]
            return f"{base}{quote}"
        return ccxt_symbol

    @staticmethod
    def _build_funding_rows(
        history: list[dict],
        default_hours: float,
    ) -> list[tuple[int, float, float]]:
        """
        Convert raw CCXT funding dicts to ``[(ts_ms_floored, rate, hours), ...]``
        with per-record interval inferred from timestamps.
        """
        if not history:
            return []
        timestamps = [item["timestamp"] for item in history]
        intervals = _infer_funding_interval_hours(timestamps, default_hours)
        return [
            (_floor_ms(item["timestamp"], hours), item.get("fundingRate", 0.0), hours)
            for item, hours in zip(history, intervals)
        ]

    async def _async_fetch_funding_one(
        self,
        ccxt_ex: Exchange,
        instrument: Instrument,
        ccxt_symbol: str,
        since: int,
        until: int,
        default_hours: float,
    ) -> list[tuple[int, float, float]]:
        """
        Paginated async per-instrument funding fetch.
        Returns ``[(ts_ms_floored, funding_rate, interval_hours), ...]``.
        """
        try:
            history = await self._async_paginated_fetch(
                ccxt_ex,
                ccxt_ex.fetch_funding_rate_history,
                since,
                until,
                symbol=ccxt_symbol,
            )
            return self._build_funding_rows(history, default_hours)
        except Exception as e:
            logger.warning(f"[CCXT] Error fetching funding for {instrument.symbol}: {e}")
            return []

    async def _async_fetch_funding_multi(
        self,
        ccxt_ex: Exchange,
        instruments_info: list[tuple[str, Instrument]],  # (ccxt_sym, instr)
        since: int,
        until: int,
        default_hours: float,
    ) -> dict[str, list[tuple[int, float, float]]]:
        """
        Concurrent per-instrument funding fetch via asyncio.gather.
        Returns ``{qubx_sym: [(ts_ms, rate, hours), ...]}``.
        """
        coros = [
            self._async_fetch_funding_one(ccxt_ex, instr, ccxt_sym, since, until, default_hours)
            for ccxt_sym, instr in instruments_info
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        out: dict[str, list[tuple[int, float, float]]] = {}
        for i, result in enumerate(results):
            qubx_sym = instruments_info[i][1].symbol
            if isinstance(result, Exception):
                logger.warning(f"[CCXT] Failed funding fetch for {qubx_sym}: {result}")
                out[qubx_sym] = []
            else:
                out[qubx_sym] = cast(list[tuple[int, float, float]], result)
        return out

    def _bulk_fetch_funding_with_pagination(
        self,
        exchange: str,
        ccxt_ex: Exchange,
        since: int,
        until: int,
        symbols: list[str] | None,
        default_hours: float,
    ) -> dict[str, list[tuple[int, float, float]]]:
        """
        Paginated bulk funding fetch (all symbols in one API call).
        Returns ``{qubx_sym: [(ts_ms_floored, rate, hours), ...]}``.
        """
        assert self._loop is not None
        try:
            all_items: list[dict] = self._loop.submit(
                self._async_paginated_fetch(
                    ccxt_ex,
                    ccxt_ex.fetch_funding_rate_history,
                    since,
                    until,
                    symbol=None,
                )
            ).result()

            # - deduplicate and group by ccxt symbol
            seen: set[tuple] = set()
            by_ccxt_sym: dict[str, list[dict]] = {}
            for item in all_items:
                key = (item["timestamp"], item["symbol"])
                if key in seen:
                    continue
                seen.add(key)
                qubx_sym = self._ccxt_sym_to_qubx(item["symbol"])
                if symbols is not None and qubx_sym not in symbols:
                    continue
                by_ccxt_sym.setdefault(qubx_sym, []).append(item)

            # - infer intervals per symbol from actual timestamps
            out: dict[str, list[tuple[int, float, float]]] = {}
            for qubx_sym, items in by_ccxt_sym.items():
                items.sort(key=lambda x: x["timestamp"])
                out[qubx_sym] = self._build_funding_rows(items, default_hours)

            return out
        except Exception as e:
            logger.warning(f"[CCXT] Bulk funding fetch failed for {exchange}: {e}")
            return {}

    def _fetch_funding_by_symbol(
        self,
        exchange: str,
        instruments_info: list[tuple[str, Instrument]],
        qubx_symbols: list[str] | None,
        start: str | None,
        stop: str | None,
    ) -> dict[str, list[tuple[int, float, float]]]:
        """
        Route to bulk or per-instrument funding fetch based on exchange capabilities.
        Returns ``{qubx_sym: [(ts_ms_floored, rate, hours), ...]}``.
        """
        assert self._loop is not None

        from qubx.connectors.ccxt.exchanges import ReaderCapabilities

        caps = (self._capabilities.get(exchange.lower()) if self._capabilities else None) or ReaderCapabilities()

        ccxt_ex = self._exchanges[exchange]
        # - resolve start/stop through the same handle_start_stop path used by OHLC
        # - default timeframe = 8h (typical funding interval) for the fallback window
        _start, _stop = self._get_start_stop(start, stop, to_timedelta("8h"))
        since = int(_start.timestamp() * 1000)
        stop_ts_ms = int(_stop.timestamp() * 1000)

        should_bulk = (not instruments_info or len(instruments_info) > 10) and caps.supports_bulk_funding
        if should_bulk:
            return self._bulk_fetch_funding_with_pagination(
                exchange,
                ccxt_ex,
                since,
                stop_ts_ms,
                qubx_symbols,
                caps.default_funding_interval_hours,
            )
        else:
            if not caps.supports_bulk_funding and not instruments_info:
                raise ValueError(
                    f"Exchange '{exchange}' does not support bulk funding fetch. Specify a list of symbols."
                )
            return self._loop.submit(
                self._async_fetch_funding_multi(
                    ccxt_ex, instruments_info, since, stop_ts_ms, caps.default_funding_interval_hours
                )
            ).result()

    def _fetch_single(
        self,
        exchange: str,
        market: str,
        qubx_symbol: str,
        dt: DataType,
        params: dict[str, Any],
        dtype_str: str,
        start: str | None,
        stop: str | None,
    ) -> RawData:
        """
        Fetch data for a **single** symbol and return a ``RawData``.
        """
        assert self._loop is not None
        ccxt_ex = self._exchanges[exchange]

        entry = self._find_instrument(exchange, market, qubx_symbol)
        if entry is None:
            logger.warning(f"[CCXT] Symbol '{qubx_symbol}' not found on {exchange}")
            return _candles_to_raw(qubx_symbol, dtype_str, [])

        ccxt_sym, instr = entry

        match dt:
            case DataType.OHLC:
                from qubx.connectors.ccxt.utils import ccxt_convert_timeframe_to_exchange_format

                timeframe = params.get("timeframe", "1h")
                _tf = to_timedelta(timeframe)
                _start, _stop = self._get_start_stop(start, stop, _tf)
                exc_tf = ccxt_convert_timeframe_to_exchange_format(timeframe)
                if exc_tf is None or exc_tf not in ccxt_ex.timeframes:
                    raise ValueError(f"Timeframe '{timeframe}' not supported by {exchange}")
                since = int(_start.timestamp() * 1000)
                until = int(_stop.timestamp() * 1000)
                candles = self._loop.submit(self._async_fetch_ohlcv(ccxt_ex, ccxt_sym, exc_tf, since, until)).result()
                return _candles_to_raw(qubx_symbol, dtype_str, candles)

            case DataType.FUNDING_PAYMENT:
                by_sym = self._fetch_funding_by_symbol(exchange, [(ccxt_sym, instr)], [qubx_symbol], start, stop)
                return _funding_rows_to_raw(qubx_symbol, by_sym.get(qubx_symbol, []))

            case _:
                raise ValueError(f"CcxtStorage: unsupported dtype '{dt}'. Supported: ohlc(*), funding_payment")

    def _fetch_multi(
        self,
        exchange: str,
        market: str,
        qubx_symbols: list[str],
        dt: DataType,
        params: dict[str, Any],
        dtype_str: str,
        start: str | None,
        stop: str | None,
    ) -> list[RawData]:
        """
        Batch-fetch data for multiple symbols using asyncio.gather where possible.
        Returns a ``list[RawData]`` in the same order as *qubx_symbols*.
        """
        assert self._loop is not None
        ccxt_ex = self._exchanges[exchange]
        instruments_info = self._find_instruments(exchange, market, qubx_symbols)

        match dt:
            case DataType.OHLC:
                from qubx.connectors.ccxt.utils import ccxt_convert_timeframe_to_exchange_format

                timeframe = params.get("timeframe", "1h")
                _tf = to_timedelta(timeframe)
                _start, _stop = self._get_start_stop(start, stop, _tf)
                exc_tf = ccxt_convert_timeframe_to_exchange_format(timeframe)
                if exc_tf is None or exc_tf not in ccxt_ex.timeframes:
                    raise ValueError(f"Timeframe '{timeframe}' not supported by {exchange}")
                since = int(_start.timestamp() * 1000)
                until = int(_stop.timestamp() * 1000)

                by_sym: dict[str, list] = self._loop.submit(
                    self._async_fetch_ohlcv_multi(
                        ccxt_ex,
                        [(ccxt_sym, instr.symbol) for ccxt_sym, instr in instruments_info],
                        exc_tf,
                        since,
                        until,
                    )
                ).result()

                return [_candles_to_raw(sym, dtype_str, by_sym.get(sym, [])) for sym in qubx_symbols]

            case DataType.FUNDING_PAYMENT:
                by_sym_f = self._fetch_funding_by_symbol(exchange, instruments_info, qubx_symbols, start, stop)
                return [_funding_rows_to_raw(sym, by_sym_f.get(sym, [])) for sym in qubx_symbols]

            case _:
                raise ValueError(f"CcxtStorage: unsupported dtype '{dt}'. Supported: ohlc(*), funding_payment")
