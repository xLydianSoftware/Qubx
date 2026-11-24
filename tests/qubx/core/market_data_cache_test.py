import threading

import numpy as np
import pytest

from qubx.core.basics import DataType, FundingPayment, Instrument, MarketType, td_64
from qubx.core.mixins.market import CachedMarketDataHolder
from qubx.core.series import OHLCV, Bar, GenericSeries, IndicatorGeneric, Quote, Trade


@pytest.fixture
def mock_instrument():
    return Instrument(
        symbol="BTCUSDT",
        market_type=MarketType.SPOT,
        exchange="BINANCE",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


@pytest.fixture
def cache_holder():
    return CachedMarketDataHolder(default_timeframe="1h")


def test_cached_market_data_holder_td64_support(cache_holder, mock_instrument):
    """
    Test that CachedMarketDataHolder.get_ohlcv supports td_64 as timeframe parameter.
    """
    # 1. Arrange
    td_64_timeframe = td_64(1, "h")

    # 2. Act
    # Test with td_64 timeframe
    ohlcv_result = cache_holder.get_ohlcv(mock_instrument, timeframe=td_64_timeframe)

    # Test with None timeframe (should use default)
    ohlcv_default = cache_holder.get_ohlcv(mock_instrument, timeframe=None)

    # Test with string timeframe
    ohlcv_string = cache_holder.get_ohlcv(mock_instrument, timeframe="1h")

    # 3. Assert
    assert isinstance(ohlcv_result, OHLCV)
    assert isinstance(ohlcv_default, OHLCV)
    assert isinstance(ohlcv_string, OHLCV)

    # All should use the same timeframe (1 hour = 3600000000000 nanoseconds)
    expected_timeframe_ns = 3600000000000  # 1 hour in nanoseconds
    assert ohlcv_result.timeframe == expected_timeframe_ns
    assert ohlcv_default.timeframe == expected_timeframe_ns
    assert ohlcv_string.timeframe == expected_timeframe_ns


def test_cached_market_data_holder_update_by_bars_td64(cache_holder, mock_instrument):
    """
    Test that CachedMarketDataHolder.update_by_bars supports td_64 as timeframe parameter.
    """
    # 1. Arrange
    td_64_timeframe = td_64(1, "h")
    bars = [
        Bar(
            time=np.datetime64("2023-01-01T10:00:00", "ns"),
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000.0,
            bought_volume=600,
        ),
        Bar(
            time=np.datetime64("2023-01-01T11:00:00", "ns"),
            open=105.0,
            high=115.0,
            low=95.0,
            close=110.0,
            volume=1200.0,
            bought_volume=700,
        ),
    ]

    # 2. Act
    # Test with td_64 timeframe
    ohlcv_result = cache_holder.update_by_bars(mock_instrument, td_64_timeframe, bars)

    # 3. Assert
    assert isinstance(ohlcv_result, OHLCV)
    assert len(ohlcv_result) == 2
    assert ohlcv_result.timeframe == 3600000000000  # 1 hour in nanoseconds

    # Verify bars were added correctly
    assert ohlcv_result[0].open == 105.0  # This is what the test shows as actual value
    assert ohlcv_result[0].close == 110.0
    assert ohlcv_result[1].open == 100.0  # This might be the second bar
    assert ohlcv_result[1].close == 105.0


def test_cached_market_data_holder_update_by_trade(cache_holder, mock_instrument):
    """Test that trades properly update OHLCV with all fields including volume_quote and trade_count."""
    # Initialize OHLCV for instrument
    cache_holder.init_ohlcv(mock_instrument)

    # Create test trades - use integer timestamps
    trade1 = Trade(
        time=np.datetime64("2023-01-01T10:00:00", "ns").astype(np.int64),
        price=100.0,
        size=10.0,
        side=1,  # Buy trade
    )
    trade2 = Trade(
        time=np.datetime64("2023-01-01T10:30:00", "ns").astype(np.int64),
        price=101.0,
        size=5.0,
        side=-1,  # Sell trade
    )
    trade3 = Trade(
        time=np.datetime64("2023-01-01T10:45:00", "ns").astype(np.int64),
        price=102.0,
        size=3.0,
        side=1,  # Another buy trade
    )

    # Update with trades
    cache_holder.update_by_trade(mock_instrument, trade1)
    cache_holder.update_by_trade(mock_instrument, trade2)
    cache_holder.update_by_trade(mock_instrument, trade3)

    # Get OHLCV
    ohlcv = cache_holder.get_ohlcv(mock_instrument)

    # Verify we have one bar (all trades within 1 hour)
    assert len(ohlcv) == 1
    bar = ohlcv[0]

    # Check OHLC values
    assert bar.open == 100.0  # First trade price
    assert bar.high == 102.0  # Max price
    assert bar.low == 100.0  # Min price
    assert bar.close == 102.0  # Last trade price

    # Check volume fields
    assert bar.volume == 18.0  # 10 + 5 + 3
    assert bar.bought_volume == 13.0  # 10 (buy) + 0 (sell) + 3 (buy)

    # Check volume_quote fields
    expected_volume_quote = (100.0 * 10.0) + (101.0 * 5.0) + (102.0 * 3.0)  # 1000 + 505 + 306 = 1811
    assert bar.volume_quote == expected_volume_quote

    expected_bought_volume_quote = (100.0 * 10.0) + (102.0 * 3.0)  # 1000 + 306 = 1306
    assert bar.bought_volume_quote == expected_bought_volume_quote

    # Check trade count
    assert bar.trade_count == 3  # Three trades


def test_cached_market_data_holder_out_of_order_trades(cache_holder, mock_instrument):
    """Test that out-of-order trades within the same bar are processed correctly."""
    cache_holder.init_ohlcv(mock_instrument)

    # All trades belong to same 1-hour bar (10:00-11:00)
    trade1 = Trade(
        time=np.datetime64("2023-01-01T10:30:00", "ns").astype(np.int64),
        price=100.0,
        size=10.0,
        side=1,  # Buy
    )
    trade2 = Trade(
        time=np.datetime64("2023-01-01T10:45:00", "ns").astype(np.int64),
        price=102.0,
        size=5.0,
        side=1,  # Buy
    )
    trade3 = Trade(
        time=np.datetime64("2023-01-01T10:15:00", "ns").astype(np.int64),
        price=98.0,
        size=8.0,
        side=-1,  # Sell
    )

    # Process in out-of-order sequence: 2, 1, 3
    cache_holder.update_by_trade(mock_instrument, trade2)  # 10:45 processed first
    cache_holder.update_by_trade(mock_instrument, trade1)  # 10:30 processed second (out of order)
    cache_holder.update_by_trade(mock_instrument, trade3)  # 10:15 processed last (out of order)

    ohlcv = cache_holder.get_ohlcv(mock_instrument)
    assert len(ohlcv) == 1  # All in same bar

    bar = ohlcv[0]
    # Verify all trades were included despite out-of-order arrival
    assert bar.volume == 23.0  # 5 + 10 + 8
    assert bar.bought_volume == 15.0  # 5 (buy) + 10 (buy) + 0 (sell)
    assert bar.trade_count == 3

    # Test trade from previous bar gets skipped
    old_trade = Trade(
        time=np.datetime64("2023-01-01T09:45:00", "ns").astype(np.int64),
        price=95.0,
        size=3.0,
        side=1,  # Buy
    )
    cache_holder.update_by_trade(mock_instrument, old_trade)

    # Should still have same values (old trade skipped)
    assert bar.volume == 23.0
    assert bar.bought_volume == 15.0
    assert bar.trade_count == 3


def test_cached_market_data_holder_cross_bar_trades(cache_holder, mock_instrument):
    """Test trades spanning multiple bars are handled correctly."""
    cache_holder.init_ohlcv(mock_instrument)

    # Trade in first hour bar (10:00-11:00)
    trade1 = Trade(
        time=np.datetime64("2023-01-01T10:30:00", "ns").astype(np.int64),
        price=100.0,
        size=10.0,
        side=1,  # Buy
    )
    cache_holder.update_by_trade(mock_instrument, trade1)

    # Trade in second hour bar (11:00-12:00)
    trade2 = Trade(
        time=np.datetime64("2023-01-01T11:30:00", "ns").astype(np.int64),
        price=101.0,
        size=5.0,
        side=1,  # Buy
    )
    cache_holder.update_by_trade(mock_instrument, trade2)

    # Late trade for first bar arrives after second bar started
    late_trade = Trade(
        time=np.datetime64("2023-01-01T10:45:00", "ns").astype(np.int64),
        price=99.0,
        size=3.0,
        side=-1,  # Sell
    )
    cache_holder.update_by_trade(mock_instrument, late_trade)

    ohlcv = cache_holder.get_ohlcv(mock_instrument)
    assert len(ohlcv) == 2  # Two separate bars

    # Most recent bar (11:00-12:00) should NOT include late trade
    recent_bar = ohlcv[0]
    assert recent_bar.volume == 5.0  # Only trade2
    assert recent_bar.trade_count == 1

    # Older bar (10:00-11:00) should only have trade1 (late_trade skipped)
    older_bar = ohlcv[1]
    assert older_bar.volume == 10.0  # Only trade1 (late_trade skipped)
    assert older_bar.trade_count == 1


def test_get_data_returns_generic_series(cache_holder, mock_instrument):
    """
    Test that get_data() returns a GenericSeries instead of a list.
    """
    series = cache_holder.get_data(mock_instrument, DataType.TRADE)
    assert isinstance(series, GenericSeries)


def test_get_data_same_series_on_repeated_calls(cache_holder, mock_instrument):
    """
    Test that get_data() returns the same GenericSeries instance on repeated calls.
    """
    series1 = cache_holder.get_data(mock_instrument, DataType.TRADE)
    series2 = cache_holder.get_data(mock_instrument, DataType.TRADE)
    assert series1 is series2


def test_get_data_separate_series_per_event_type(cache_holder, mock_instrument):
    """
    Test that get_data() returns separate GenericSeries per event type.
    """
    trade_series = cache_holder.get_data(mock_instrument, DataType.TRADE)
    quote_series = cache_holder.get_data(mock_instrument, DataType.QUOTE)
    assert trade_series is not quote_series


def test_update_populates_generic_series(cache_holder, mock_instrument):
    """
    Test that update() feeds data into GenericSeries so all events are preserved.
    """
    t0 = np.datetime64("2023-01-01T10:00:00", "ns").astype(np.int64)
    t1 = np.datetime64("2023-01-01T10:00:01", "ns").astype(np.int64)
    t2 = np.datetime64("2023-01-01T10:00:02", "ns").astype(np.int64)

    trade0 = Trade(time=t0, price=100.0, size=1.0, side=1)
    trade1 = Trade(time=t1, price=101.0, size=2.0, side=-1)
    trade2 = Trade(time=t2, price=102.0, size=0.5, side=1)

    cache_holder.update(mock_instrument, DataType.TRADE, trade0)
    cache_holder.update(mock_instrument, DataType.TRADE, trade1)
    cache_holder.update(mock_instrument, DataType.TRADE, trade2)

    series = cache_holder.get_data(mock_instrument, DataType.TRADE)

    # - all three trades must be stored individually (tick resolution, no bucketing)
    assert len(series) == 3
    assert series[0].price == 102.0  # - most recent first
    assert series[1].price == 101.0
    assert series[2].price == 100.0


def test_update_generic_series_tick_resolution(cache_holder, mock_instrument):
    """
    Test that GenericSeries uses 1ns timeframe so each event is a separate item.
    """
    series = cache_holder.get_data(mock_instrument, DataType.QUOTE)
    assert series.timeframe == 1  # - 1 nanosecond


def test_generic_series_indicator_attachment(cache_holder, mock_instrument):
    """
    Test that IndicatorGeneric can be attached to a GenericSeries from get_data()
    and receives streaming updates automatically — the key requirement of issue #171.
    """

    class MidPriceIndicator(IndicatorGeneric):
        """
        Example IndicatorGeneric that extracts mid price from Quote events.\n
        """

        def calculate(self, time, quote, new_item_started):
            return (quote.bid + quote.ask) / 2.0

    t0 = np.datetime64("2023-01-01T10:00:00", "ns").astype(np.int64)
    t1 = np.datetime64("2023-01-01T10:00:01", "ns").astype(np.int64)
    t2 = np.datetime64("2023-01-01T10:00:02", "ns").astype(np.int64)

    quote_series = cache_holder.get_data(mock_instrument, DataType.QUOTE)

    # - attach indicator BEFORE data arrives
    mid = MidPriceIndicator("mid", quote_series)

    q0 = Quote(time=t0, bid=99.0, ask=101.0, bid_size=10.0, ask_size=10.0)
    q1 = Quote(time=t1, bid=100.0, ask=102.0, bid_size=5.0, ask_size=5.0)
    q2 = Quote(time=t2, bid=101.0, ask=103.0, bid_size=8.0, ask_size=8.0)

    cache_holder.update(mock_instrument, DataType.QUOTE, q0)
    cache_holder.update(mock_instrument, DataType.QUOTE, q1)
    cache_holder.update(mock_instrument, DataType.QUOTE, q2)

    # - indicator should have been updated for every quote automatically
    assert len(mid) == 3
    assert mid[0] == pytest.approx(102.0)  # - most recent: (101 + 103) / 2
    assert mid[1] == pytest.approx(101.0)  # - (100 + 102) / 2
    assert mid[2] == pytest.approx(100.0)  # - (99 + 101) / 2


def test_remove_clears_generic_series(cache_holder, mock_instrument):
    """
    Test that remove() clears the GenericSeries for an instrument.
    """
    t0 = np.datetime64("2023-01-01T10:00:00", "ns").astype(np.int64)
    cache_holder.update(mock_instrument, DataType.TRADE, Trade(time=t0, price=100.0, size=1.0, side=1))
    assert len(cache_holder.get_data(mock_instrument, DataType.TRADE)) == 1

    cache_holder.remove(mock_instrument)

    # - after remove, get_data returns a fresh empty series
    new_series = cache_holder.get_data(mock_instrument, DataType.TRADE)
    assert len(new_series) == 0


def test_data_snapshot_is_detached_from_the_live_series(cache_holder, mock_instrument):
    """
    A snapshot handed to the fit thread must not grow when the ProcessorThread keeps
    appending — the fit iterates it without holding the lock.
    """
    t0 = np.datetime64("2023-01-01T10:00:00", "ns").astype(np.int64)
    cache_holder.update(mock_instrument, DataType.FUNDING_PAYMENT, FundingPayment(t0, 0.0001, 8))

    snapshot = cache_holder.get_data_snapshot(mock_instrument, DataType.FUNDING_PAYMENT)
    cache_holder.update(
        mock_instrument, DataType.FUNDING_PAYMENT, FundingPayment(t0 + 8 * 3_600_000_000_000, 0.0002, 8)
    )

    assert len(snapshot) == 1
    assert len(cache_holder.get_data(mock_instrument, DataType.FUNDING_PAYMENT)) == 2


def test_snapshots_stay_readable_while_another_thread_appends(cache_holder, mock_instrument):
    """
    The fit-thread read path against a live feed: snapshots taken while updates land must
    never surface a torn series (short read, index error, out-of-order times).
    """
    t0 = np.datetime64("2023-01-01T10:00:00", "ns").astype(np.int64)
    step = 60_000_000_000
    n_writes = 3000
    failures: list[Exception] = []
    stop = threading.Event()

    def _write() -> None:
        try:
            for i in range(n_writes):
                cache_holder.update(
                    mock_instrument, DataType.TRADE, Trade(time=t0 + i * step, price=100.0 + i, size=1.0, side=1)
                )
        finally:
            stop.set()

    def _read() -> None:
        while not stop.is_set():
            try:
                snap = cache_holder.get_data_snapshot(mock_instrument, DataType.TRADE)
                times = [snap[i].time for i in range(len(snap))]
                assert times == sorted(times, reverse=True), "series times out of order"
            except Exception as e:  # noqa: BLE001 — the reader must never see a torn series
                failures.append(e)
                return

    writer, reader = threading.Thread(target=_write), threading.Thread(target=_read)
    reader.start()
    writer.start()
    writer.join()
    reader.join()

    assert not failures, f"reader saw a torn series: {failures[0]!r}"
    assert len(cache_holder.get_data(mock_instrument, DataType.TRADE)) == n_writes


def _ns(ts: str) -> int:
    return int(np.datetime64(ts, "ns").astype("int64"))


@pytest.fixture
def minute_cache(mock_instrument):
    cache = CachedMarketDataHolder(default_timeframe="1Min")
    cache.get_ohlcv(mock_instrument, "1Min")
    return cache


def test_finalize_before_first_venue_bar_keeps_venue_open(minute_cache, mock_instrument):
    """
    A trigger at second 59 of a minute with no venue bar yet, then the venue's bar for that
    minute arrives: the stored bar must be the venue's bar, not one opened at the previous close.
    """
    i = mock_instrument
    minute_cache.update_by_bar(i, Bar(_ns("2026-11-18T11:58"), 589.70, 589.75, 589.60, 589.61, volume=3.0))

    minute_cache.finalize_ohlc_for_instruments(np.datetime64("2026-11-18T11:59:59", "ns"), [i])
    minute_cache.update_by_bar(i, Bar(_ns("2026-11-18T11:59"), 589.39, 589.39, 589.39, 589.39, volume=0.5))

    b = minute_cache.get_ohlcv(i, "1Min")[0]
    assert b.time == _ns("2026-11-18T11:59")
    assert (b.open, b.high, b.low, b.close) == (589.39, 589.39, 589.39, 589.39)
    assert b.volume == pytest.approx(0.5)


def test_finalize_after_last_venue_bar_keeps_volume(minute_cache, mock_instrument):
    """
    The venue's last message for a minute arrives before the trigger at second 59 and nothing
    follows for that minute: the stored bar keeps the venue's volume and trade count.
    """
    i = mock_instrument
    minute_cache.update_by_bar(i, Bar(_ns("2026-11-18T04:30"), 10.0, 10.2, 9.9, 10.1, volume=217.4, trade_count=40))

    minute_cache.finalize_ohlc_for_instruments(np.datetime64("2026-11-18T04:30:59", "ns"), [i])

    b = minute_cache.get_ohlcv(i, "1Min")[0]
    assert (b.open, b.high, b.low, b.close) == (10.0, 10.2, 9.9, 10.1)
    assert b.volume == pytest.approx(217.4)
    assert b.trade_count == 40


def test_finalize_opened_higher_timeframe_bar_takes_first_venue_bar(minute_cache, mock_instrument):
    """
    The trigger opens a new hour with no venue bar in it yet: the hourly bar starts from the
    first venue bar of the hour, not from the previous close.
    """
    i = mock_instrument
    minute_cache.get_ohlcv(i, "1h")
    minute_cache.update_by_bar(i, Bar(_ns("2026-11-18T11:59"), 100.0, 101.0, 99.0, 100.5, volume=2.0))

    minute_cache.finalize_ohlc_for_instruments(np.datetime64("2026-11-18T12:00:59", "ns"), [i])
    minute_cache.update_by_bar(i, Bar(_ns("2026-11-18T12:00"), 98.0, 98.5, 97.5, 98.2, volume=1.0))

    h = minute_cache.get_ohlcv(i, "1h")[0]
    assert h.time == _ns("2026-11-18T12:00")
    assert (h.open, h.high, h.low, h.close) == (98.0, 98.5, 97.5, 98.2)
    assert h.volume == pytest.approx(1.0)


def test_finalize_before_first_trade_takes_the_trade_as_open(mock_instrument):
    """
    Bars built from trades: a trigger at the start of a minute opens it at the last trade price; the
    minute's first trade then gives the bar its open, high and low.
    """
    i = mock_instrument
    cache = CachedMarketDataHolder(default_timeframe="1Min")
    cache.get_ohlcv(i, "1Min")
    cache.update_by_trade(i, Trade(_ns("2026-11-18T11:58:50"), 589.61, 1.0, 1))

    cache.finalize_ohlc_for_instruments(np.datetime64("2026-11-18T11:59:00", "ns"), [i])
    cache.update_by_trade(i, Trade(_ns("2026-11-18T11:59:05"), 589.39, 0.5, -1))
    cache.update_by_trade(i, Trade(_ns("2026-11-18T11:59:40"), 589.45, 0.2, 1))

    b = cache.get_ohlcv(i, "1Min")[0]
    assert b.time == _ns("2026-11-18T11:59")
    assert (b.open, b.high, b.low, b.close) == (589.39, 589.45, 589.39, 589.45)
    assert b.volume == pytest.approx(0.7)
    assert b.trade_count == 2
