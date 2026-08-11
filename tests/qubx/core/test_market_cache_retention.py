import numpy as np
import pytest

from qubx.core.basics import Instrument, MarketType
from qubx.core.mixins.market import CachedMarketDataHolder
from qubx.core.series import Bar, OHLCV


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


def test_per_type_cap_applies_to_generic_series():
    h = CachedMarketDataHolder(max_buffer_size=10_000, per_type_lengths={"orderbook": 4, "funding_rate": 64})
    assert h._resolve_series_length("orderbook") == 4
    assert h._resolve_series_length("orderbook(0,1)") == 4  # parameterized form matches base name
    assert h._resolve_series_length("funding_rate") == 64
    assert h._resolve_series_length("quote") == 10_000  # unmatched -> default
    # _resolve_series_length is a pure lookup; production OHLC creation goes through
    # the separately-guarded _resolve_ohlc_length and stays unbounded here.
    assert h._resolve_series_length("ohlc(1h)") == 10_000  # ohlc key unset -> default


def test_ohlc_key_honored():
    h = CachedMarketDataHolder(max_buffer_size=10_000, per_type_lengths={"ohlc": 500})
    assert h._resolve_series_length("ohlc(1h)") == 500


def test_resolve_ohlc_length_unconfigured_stays_unbounded():
    h = CachedMarketDataHolder(max_buffer_size=10_000, per_type_lengths={})
    assert h._resolve_ohlc_length(np.inf) == np.inf


def test_resolve_ohlc_length_configured_caps_default_sentinel():
    h = CachedMarketDataHolder(max_buffer_size=10_000, per_type_lengths={"ohlc": 500})
    assert h._resolve_ohlc_length(np.inf) == 500


def test_resolve_ohlc_length_never_overrides_callers_explicit_cap():
    # Caller-supplied max_size (not the np.inf default sentinel) always passes
    # through untouched, even when an "ohlc" cap is configured.
    h = CachedMarketDataHolder(max_buffer_size=10_000, per_type_lengths={"ohlc": 500})
    assert h._resolve_ohlc_length(50) == 50


def test_update_by_bars_creates_capped_series_when_ohlc_configured(mock_instrument):
    # Subscription warmup for a non-default timeframe reaches OHLC creation via
    # update_by_bars before init_ohlcv/get_ohlcv ever touches that (instrument,
    # timeframe) pair -- must honor a configured "ohlc" cap too.
    h = CachedMarketDataHolder(default_timeframe="1h", per_type_lengths={"ohlc": 500})
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
    ]
    ohlcv = h.update_by_bars(mock_instrument, "5m", bars)
    assert isinstance(ohlcv, OHLCV)
    assert ohlcv.max_series_length == 500


def test_update_by_bars_creates_unbounded_series_when_ohlc_unconfigured(mock_instrument):
    h = CachedMarketDataHolder(default_timeframe="1h", per_type_lengths={})
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
    ]
    ohlcv = h.update_by_bars(mock_instrument, "5m", bars)
    assert isinstance(ohlcv, OHLCV)
    assert ohlcv.max_series_length == np.inf


def test_funding_rate_has_slots():
    import numpy as np
    from qubx.core.basics import FundingRate

    fr = FundingRate(time=np.datetime64(0, "ns"), rate=0.0001, interval="1h", next_funding_time=np.datetime64(3600_000_000_000, "ns"))
    assert not hasattr(fr, "__dict__")
    with pytest.raises(AttributeError):
        fr.extra = 1  # type: ignore[attr-defined]
