"""
Tests for OHLC historical data pagination in get_historical_ohlc.

Verifies that the pagination loop correctly fetches all requested bars
even when the exchange returns fewer bars per request than the internal
MAX_BARS_PER_REQUEST_FOR_PROVIDER limit (e.g., OKX returns ~100 bars).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qubx.connectors.ccxt.handlers.ohlc import OhlcDataHandler
from qubx.core.basics import Instrument, MarketType


def _make_instrument() -> Instrument:
    return Instrument(
        symbol="BTCUSDT",
        market_type=MarketType.SWAP,
        exchange="OKX.F",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTC-USDT-SWAP",
        tick_size=0.1,
        lot_size=0.01,
        min_size=0.01,
        contract_size=0.01,
    )


def _make_ohlcv_page(start_ts_ms: int, count: int, tf_ms: int) -> list[list]:
    """Generate a page of OHLCV data starting at start_ts_ms."""
    return [
        [start_ts_ms + i * tf_ms, 50000.0, 50100.0, 49900.0, 50050.0, 100.0]
        for i in range(count)
    ]


def _build_handler() -> tuple[OhlcDataHandler, MagicMock]:
    """Build an OhlcDataHandler with mocked dependencies."""
    data_provider = MagicMock()
    data_provider._get_exch_timeframe.return_value = "1h"
    # start_since = now - nbarsback * tf, using a fixed timestamp
    data_provider._time_msec_nbars_back.return_value = 1_000_000_000

    exchange_manager = MagicMock()
    handler = OhlcDataHandler(
        data_provider=data_provider,
        exchange_manager=exchange_manager,
        exchange_id="OKX.F",
    )
    return handler, exchange_manager


class TestOhlcPagination:
    """Test get_historical_ohlc pagination logic."""

    def test_fetches_all_bars_when_exchange_returns_small_pages(self):
        """When exchange returns 100 bars per request (like OKX), loop should continue until we have enough."""
        handler, exchange_manager = _build_handler()
        instrument = _make_instrument()
        tf_ms = 3_600_000  # 1h in ms
        requested_bars = 500
        page_size = 100

        # Build pages of 100 bars each
        call_count = 0
        start_ts = 1_000_000_000

        async def mock_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
            nonlocal call_count
            page_start = start_ts + call_count * page_size * tf_ms
            call_count += 1
            return _make_ohlcv_page(page_start, page_size, tf_ms)

        exchange_manager.exchange.fetch_ohlcv = AsyncMock(side_effect=mock_fetch_ohlcv)

        bars = asyncio.get_event_loop().run_until_complete(
            handler.get_historical_ohlc(instrument, "1h", requested_bars)
        )

        assert len(bars) >= requested_bars
        # Should have made ~5 calls (500 / 100)
        assert call_count >= requested_bars // page_size

    def test_fetches_all_bars_when_exchange_returns_large_pages(self):
        """When exchange returns 1000 bars per request (like Binance), fewer calls needed."""
        handler, exchange_manager = _build_handler()
        instrument = _make_instrument()
        tf_ms = 3_600_000
        requested_bars = 2000
        page_size = 1000

        call_count = 0
        start_ts = 1_000_000_000

        async def mock_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
            nonlocal call_count
            page_start = start_ts + call_count * page_size * tf_ms
            call_count += 1
            return _make_ohlcv_page(page_start, page_size, tf_ms)

        exchange_manager.exchange.fetch_ohlcv = AsyncMock(side_effect=mock_fetch_ohlcv)

        bars = asyncio.get_event_loop().run_until_complete(
            handler.get_historical_ohlc(instrument, "1h", requested_bars)
        )

        assert len(bars) >= requested_bars
        assert call_count >= 2  # 2000 / 1000

    def test_stops_when_exchange_returns_empty(self):
        """Loop should stop when exchange returns no data."""
        handler, exchange_manager = _build_handler()
        instrument = _make_instrument()
        tf_ms = 3_600_000
        page_size = 50

        call_count = 0
        start_ts = 1_000_000_000

        async def mock_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                page_start = start_ts + (call_count - 1) * page_size * tf_ms
                return _make_ohlcv_page(page_start, page_size, tf_ms)
            return []  # No more data

        exchange_manager.exchange.fetch_ohlcv = AsyncMock(side_effect=mock_fetch_ohlcv)

        bars = asyncio.get_event_loop().run_until_complete(
            handler.get_historical_ohlc(instrument, "1h", 1000)
        )

        assert len(bars) == 100  # 2 pages * 50
        assert call_count == 3  # 2 successful + 1 empty

    def test_stops_when_no_new_bars_added(self):
        """Loop should stop when exchange keeps returning the same data (no progress)."""
        handler, exchange_manager = _build_handler()
        instrument = _make_instrument()

        # Always return the same single bar
        same_bar = [[1_000_000_000, 50000.0, 50100.0, 49900.0, 50050.0, 100.0]]
        exchange_manager.exchange.fetch_ohlcv = AsyncMock(return_value=same_bar)

        bars = asyncio.get_event_loop().run_until_complete(
            handler.get_historical_ohlc(instrument, "1h", 100)
        )

        assert len(bars) == 1  # Only 1 unique bar

    def test_deduplicates_overlapping_bars(self):
        """Bars with the same timestamp should be deduplicated."""
        handler, exchange_manager = _build_handler()
        instrument = _make_instrument()
        tf_ms = 3_600_000

        call_count = 0
        start_ts = 1_000_000_000

        async def mock_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
            nonlocal call_count
            call_count += 1
            # Each page overlaps by 1 bar with the previous page
            page_start = start_ts + (call_count - 1) * 9 * tf_ms  # 10 bars, overlap 1
            return _make_ohlcv_page(page_start, 10, tf_ms)

        exchange_manager.exchange.fetch_ohlcv = AsyncMock(side_effect=mock_fetch_ohlcv)

        bars = asyncio.get_event_loop().run_until_complete(
            handler.get_historical_ohlc(instrument, "1h", 30)
        )

        # With 10 bars per page and 1 overlap, need ~4 pages for 30 unique bars
        assert len(bars) >= 30
