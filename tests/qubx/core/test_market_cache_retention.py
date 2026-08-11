from qubx.core.mixins.market import CachedMarketDataHolder


def test_per_type_cap_applies_to_generic_series():
    h = CachedMarketDataHolder(max_buffer_size=10_000, per_type_lengths={"orderbook": 4, "funding_rate": 64})
    assert h._resolve_series_length("orderbook") == 4
    assert h._resolve_series_length("orderbook(0,1)") == 4  # parameterized form matches base name
    assert h._resolve_series_length("funding_rate") == 64
    assert h._resolve_series_length("quote") == 10_000  # unmatched -> default
    assert h._resolve_series_length("ohlc(1h)") == 10_000  # ohlc key unset -> default


def test_ohlc_key_honored():
    h = CachedMarketDataHolder(max_buffer_size=10_000, per_type_lengths={"ohlc": 500})
    assert h._resolve_series_length("ohlc(1h)") == 500
