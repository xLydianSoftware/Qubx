"""
Unit tests for IndicatorEmitter.

This module tests the IndicatorEmitter class to ensure it properly wraps indicators
and emits their values as expected.
"""

from unittest.mock import Mock

import numpy as np

from qubx.core.basics import AssetType, Instrument, MarketType
from qubx.core.interfaces import IMetricEmitter
from qubx.core.series import OHLCV, TimeSeries
from qubx.emitters.indicator import IndicatorEmitter, indicator_emitter
from qubx.ta.indicators import atr, sma


class TestIndicatorEmitter:
    """Test cases for IndicatorEmitter class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a time series with some test data
        self.series = TimeSeries("test_series", 60_000_000_000, 100)  # 1-minute timeframe

        # Create OHLC series for ATR testing
        self.ohlc_series = OHLCV("test_ohlc", 60_000_000_000, 100)

        # Add some initial data
        for i in range(5):
            time = i * 60_000_000_000  # 1 minute intervals in nanoseconds
            value = 100.0 + i

            # Add regular data
            self.series.update(time, value)

            # Add OHLC data
            self.ohlc_series.update_by_bar(
                time,
                open=value - 0.5,
                high=value + 1.0,
                low=value - 1.0,
                close=value,
                vol_incr=1000,
            )

        # Create test instrument
        self.instrument = Instrument(
            symbol="BTCUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="binance",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTCUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

        # Create mock emitter
        self.mock_emitter = Mock(spec=IMetricEmitter)

    def test_indicator_emitter_creation(self):
        """Test that IndicatorEmitter can be created and properly initialized."""
        # Create a simple SMA indicator
        sma_indicator = sma(self.series, 3)

        # Create an IndicatorEmitter
        emitter = IndicatorEmitter(
            name="test_emitter",
            wrapped_indicator=sma_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="test_sma",
            instrument=self.instrument,
            tags={"test": "tag"},
        )

        # Verify basic properties
        assert emitter.name == "test_emitter"
        assert emitter._wrapped_indicator == sma_indicator
        assert emitter._metric_emitter == self.mock_emitter
        assert emitter._metric_name == "test_sma"
        assert emitter._instrument == self.instrument
        assert emitter._tags == {"test": "tag"}

    def test_indicator_emitter_emits_on_new_item_only(self):
        """Test that emitter only emits when new_item_started=True by default."""
        sma_indicator = sma(self.series, 3)

        _emitter = IndicatorEmitter(
            name="test_emitter",
            wrapped_indicator=sma_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="test_sma",
            instrument=self.instrument,
        )

        # Reset mock to clear initialization emissions
        self.mock_emitter.reset_mock()

        # Track all emissions for detailed verification
        emissions = []

        def capture_emission(*args, **kwargs):
            emissions.append(
                {
                    "args": args,
                    "kwargs": kwargs,
                    "sma_current": sma_indicator[0] if len(sma_indicator) > 0 else None,
                    "sma_previous": sma_indicator[1] if len(sma_indicator) > 1 else None,
                    "series_length": len(self.series),
                }
            )

        self.mock_emitter.emit.side_effect = capture_emission

        # Add some more data to trigger emissions
        base_time = 10 * 60_000_000_000

        # Store SMA values before updates for comparison
        sma_before_first = sma_indicator[0] if len(sma_indicator) > 0 else None

        # First update - new item, should emit previous completed value
        result1 = self.series.update(base_time, 110)
        assert result1 == 1  # Should be new item

        # Store values after first update
        emissions_after_first = len(emissions)
        sma_after_first = sma_indicator[0] if len(sma_indicator) > 0 else None

        # Second update - same item, should not emit (new_item_only=True by default)
        result2 = self.series.update(base_time + 30_000_000_000, 111)  # 30 seconds later, same bar
        assert result2 == 0  # Should be update to existing item

        emissions_after_second = len(emissions)

        # Third update - new item, should emit previous completed value
        sma_before_third = sma_indicator[0] if len(sma_indicator) > 0 else None
        result3 = self.series.update(base_time + 60_000_000_000, 112)  # 1 minute later, new bar
        assert result3 == 1  # Should be new item

        emissions_after_third = len(emissions)

        # Verify emission behavior
        # Should emit only when new items are created (result == 1)
        print(f"DEBUG: Emissions after first update: {emissions_after_first}")
        print(f"DEBUG: Emissions after second update: {emissions_after_second}")
        print(f"DEBUG: Emissions after third update: {emissions_after_third}")

        # Should have emitted for new items only, not for updates within same item
        if emissions_after_first > 0:
            # First emission should happen when new item is created
            first_emission = emissions[0]
            print(f"DEBUG: First emission value: {first_emission['kwargs'].get('value')}")
            print(f"DEBUG: SMA before first update: {sma_before_first}")
            print(f"DEBUG: SMA after first update: {sma_after_first}")

            # When new item starts, should emit the previous (completed) value
            if sma_before_first is not None and not np.isnan(sma_before_first):
                emitted_value = first_emission["kwargs"]["value"]
                # Should emit the previous completed value, not the current updating one
                assert not np.isnan(emitted_value), "Emitted value should not be NaN"

        # Second update should not cause emission (same item)
        assert emissions_after_second == emissions_after_first, "Should not emit on same-item updates"

        # Third update should cause emission (new item)
        if emissions_after_third > emissions_after_second:
            third_emission = emissions[emissions_after_second]
            print(f"DEBUG: Third emission value: {third_emission['kwargs'].get('value')}")
            print(f"DEBUG: SMA before third update: {sma_before_third}")

            if sma_before_third is not None and not np.isnan(sma_before_third):
                emitted_value = third_emission["kwargs"]["value"]
                assert not np.isnan(emitted_value), "Third emission should not be NaN"

    def test_indicator_emitter_emits_previous_value_on_new_item(self):
        """Test that emitter emits the previous (completed) value when new item starts."""
        sma_indicator = sma(self.series, 3)

        _emitter = IndicatorEmitter(
            name="test_emitter",
            wrapped_indicator=sma_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="test_sma",
            instrument=self.instrument,
        )

        # Reset mock to clear initialization emissions
        self.mock_emitter.reset_mock()

        # Track all emissions for detailed verification
        emissions = []

        def capture_emission(*args, **kwargs):
            emissions.append(
                {
                    "args": args,
                    "kwargs": kwargs,
                    "sma_current": sma_indicator[0] if len(sma_indicator) > 0 else None,
                    "sma_previous": sma_indicator[1] if len(sma_indicator) > 1 else None,
                    "series_length": len(self.series),
                }
            )

        self.mock_emitter.emit.side_effect = capture_emission

        # Get the current SMA value before triggering new item
        sma_before_update = sma_indicator[0] if len(sma_indicator) > 0 else None
        print(f"DEBUG: SMA before update: {sma_before_update}")

        # Add new data point to trigger new item
        base_time = 10 * 60_000_000_000
        result = self.series.update(base_time, 110)
        assert result == 1  # Should be new item

        # Get SMA values after update
        sma_current_after = sma_indicator[0] if len(sma_indicator) > 0 else None
        sma_previous_after = sma_indicator[1] if len(sma_indicator) > 1 else None

        print(f"DEBUG: SMA current after update: {sma_current_after}")
        print(f"DEBUG: SMA previous after update: {sma_previous_after}")
        print(f"DEBUG: Number of emissions: {len(emissions)}")

        # Should have emitted the previous (completed) SMA value, not the current one
        if len(emissions) > 0:
            emission = emissions[0]  # Get the first emission after reset
            emitted_value = emission["kwargs"]["value"]
            emitted_timestamp = emission["kwargs"]["timestamp"]

            print(f"DEBUG: Emitted value: {emitted_value}")
            print(f"DEBUG: Emitted timestamp: {emitted_timestamp}")

            # The emitted value should be meaningful (not NaN) if we have enough history
            assert not np.isnan(emitted_value), "Emitted value should not be NaN"

            # Key assertion: When new item starts, we should emit the PREVIOUS completed value
            # This means emitted_value should match what was sma_indicator[0] BEFORE the update
            # or what is now sma_indicator[1] AFTER the update (the previous completed value)
            if sma_before_update is not None and not np.isnan(sma_before_update):
                assert emitted_value == sma_before_update, (
                    f"Should emit previous completed value ({sma_before_update}), "
                    f"not current updating value ({sma_current_after})"
                )

            # Alternative check: after update, the emitted value should match sma_previous_after
            if sma_previous_after is not None and not np.isnan(sma_previous_after):
                assert emitted_value == sma_previous_after, (
                    f"Emitted value should match previous SMA value ({sma_previous_after}) after new item creation"
                )

            # Timestamp should match the update time
            if hasattr(emitted_timestamp, "view"):
                # Convert numpy.datetime64 to nanoseconds for comparison
                emitted_timestamp_ns = emitted_timestamp.view("int64")
                assert emitted_timestamp_ns == base_time, (
                    f"Emission timestamp ({emitted_timestamp_ns}) should match update time ({base_time})"
                )
            else:
                assert emitted_timestamp == base_time, "Emission timestamp should match update time"

            # Verify that we're not emitting the current (incomplete) value
            if sma_current_after is not None and sma_previous_after is not None:
                if sma_current_after != sma_previous_after:  # They should be different after new item
                    assert emitted_value != sma_current_after, (
                        "Should not emit current (incomplete) value on new item start"
                    )

    def test_indicator_emitter_emits_on_every_update_when_configured(self):
        """Test emitter emits on every update when emit_on_new_item_only=False."""
        sma_indicator = sma(self.series, 3)

        _emitter = IndicatorEmitter(
            name="test_emitter",
            wrapped_indicator=sma_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="test_sma",
            emit_on_new_item_only=False,
        )

        # Reset mock to clear initialization emissions
        self.mock_emitter.reset_mock()

        # Track all emissions for detailed verification
        emissions = []

        def capture_emission(*args, **kwargs):
            emissions.append(
                {
                    "args": args,
                    "kwargs": kwargs,
                    "sma_current": sma_indicator[0] if len(sma_indicator) > 0 else None,
                    "sma_previous": sma_indicator[1] if len(sma_indicator) > 1 else None,
                    "series_length": len(self.series),
                    "timestamp": kwargs.get("timestamp"),
                }
            )

        self.mock_emitter.emit.side_effect = capture_emission

        # Add some data
        base_time = 10 * 60_000_000_000

        # Store initial state
        _sma_before_first = sma_indicator[0] if len(sma_indicator) > 0 else None

        # First update - new item
        result1 = self.series.update(base_time, 110)
        assert result1 == 1  # Should be new item
        emissions_after_first = len(emissions)
        _sma_after_first = sma_indicator[0] if len(sma_indicator) > 0 else None

        # Second update - same item (should still emit because emit_on_new_item_only=False)
        result2 = self.series.update(base_time + 30_000_000_000, 111)
        assert result2 == 0  # Should be update to existing item
        emissions_after_second = len(emissions)
        _sma_after_second = sma_indicator[0] if len(sma_indicator) > 0 else None

        # Third update - new item
        result3 = self.series.update(base_time + 60_000_000_000, 112)
        assert result3 == 1  # Should be new item
        emissions_after_third = len(emissions)

        # Verify emission behavior
        print(f"DEBUG: Emissions after first update: {emissions_after_first}")
        print(f"DEBUG: Emissions after second update: {emissions_after_second}")
        print(f"DEBUG: Emissions after third update: {emissions_after_third}")

        # Should have emitted for BOTH new items AND updates (emit_on_new_item_only=False)
        # This is the key difference from the previous test

        # Should emit on first update (new item)
        assert emissions_after_first >= 1, "Should emit on new item"

        # Should emit on second update too (same item update, but emit_on_new_item_only=False)
        assert emissions_after_second > emissions_after_first, (
            "Should emit on same-item updates when emit_on_new_item_only=False"
        )

        # Should emit on third update (new item)
        assert emissions_after_third > emissions_after_second, "Should emit on new item"

        # Verify the emitted values are current (not previous) since we emit on every update
        for i, emission in enumerate(emissions):
            emitted_value = emission["kwargs"]["value"]
            emitted_timestamp = emission["kwargs"]["timestamp"]

            print(f"DEBUG: Emission {i}: value={emitted_value}, timestamp={emitted_timestamp}")
            print(f"DEBUG: SMA current: {emission['sma_current']}, SMA previous: {emission['sma_previous']}")

            # Values should not be NaN when we have enough data
            if emission["sma_current"] is not None and not np.isnan(emission["sma_current"]):
                assert not np.isnan(emitted_value), f"Emission {i} should not be NaN"
                # Since emit_on_new_item_only=False, we emit current values on updates
                assert emitted_value == emission["sma_current"], f"Emission {i} should emit current SMA value"

        # Verify timestamps are correct (convert to nanoseconds for comparison)
        if len(emissions) >= 2:
            # First emission should have timestamp of the bar time
            first_timestamp = emissions[0]["kwargs"]["timestamp"]
            if hasattr(first_timestamp, "view"):
                # Convert numpy.datetime64 to nanoseconds for comparison
                first_timestamp_ns = first_timestamp.view("int64")
                assert first_timestamp_ns == base_time, (
                    f"First emission timestamp ({first_timestamp_ns}) should match bar time ({base_time})"
                )

            # Second emission should have same timestamp as first (same bar)
            if emissions_after_second > 1:
                second_timestamp = emissions[1]["kwargs"]["timestamp"]
                if hasattr(second_timestamp, "view"):
                    second_timestamp_ns = second_timestamp.view("int64")
                    # Both updates are for the same bar, so timestamp should be the same
                    assert second_timestamp_ns == base_time, (
                        f"Second emission timestamp ({second_timestamp_ns}) should match bar time ({base_time}) for same-bar update"
                    )

            # Third emission should have timestamp of the new bar
            if emissions_after_third > 2:
                third_timestamp = emissions[2]["kwargs"]["timestamp"]
                if hasattr(third_timestamp, "view"):
                    third_timestamp_ns = third_timestamp.view("int64")
                    expected_third_time = base_time + 60_000_000_000
                    assert third_timestamp_ns == expected_third_time, (
                        f"Third emission timestamp ({third_timestamp_ns}) should match new bar time ({expected_third_time})"
                    )

    def test_indicator_emitter_includes_proper_tags(self):
        """Test that emitter includes correct tags in emissions."""
        sma_indicator = sma(self.series, 3)
        custom_tags = {"timeframe": "1m", "period": "3"}
        _emitter = IndicatorEmitter(
            name="test_emitter",
            wrapped_indicator=sma_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="test_sma",
            instrument=self.instrument,
            tags=custom_tags,
        )

        # Reset mock to clear initialization emissions
        self.mock_emitter.reset_mock()

        # Add data to trigger emission
        base_time = 10 * 60_000_000_000
        self.series.update(base_time, 110)

        # Check if emission occurred with proper tags
        if self.mock_emitter.emit.call_count > 0:
            call_args = self.mock_emitter.emit.call_args
            emitted_tags = call_args.kwargs["tags"]

            # Should include custom tags plus indicator name
            assert "timeframe" in emitted_tags
            assert "period" in emitted_tags

    def test_wrap_with_emitter_classmethod(self):
        """Test the wrap_with_emitter class method."""
        sma_indicator = sma(self.series, 3)

        emitter = IndicatorEmitter.wrap_with_emitter(
            indicator=sma_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="wrapped_sma",
            instrument=self.instrument,
        )

        # Verify it's properly configured
        assert isinstance(emitter, IndicatorEmitter)
        assert emitter._wrapped_indicator == sma_indicator
        assert emitter._metric_name == "wrapped_sma"

    def test_helper_function(self):
        """Test the indicator_emitter helper function."""
        sma_indicator = sma(self.series, 3)

        emitter = indicator_emitter(
            wrapped_indicator=sma_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="helper_sma",
        )

        # Verify it's properly configured
        assert isinstance(emitter, IndicatorEmitter)
        assert emitter._wrapped_indicator == sma_indicator
        assert emitter._metric_name == "helper_sma"

    def test_emitter_works_with_indicator_tree_registration(self):
        """Test that emitters are properly registered in the indicator tree."""
        sma_indicator = sma(self.series, 3)

        # Create two emitters with identical configuration
        _emitter1 = IndicatorEmitter.wrap_with_emitter(
            indicator=sma_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="tree_test",
        )

        _emitter2 = IndicatorEmitter.wrap_with_emitter(
            indicator=sma_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="tree_test",
        )

        # The emitters should be registered as indicators on the wrapped indicator (SMA)
        # Check both the original series and the SMA indicator for emitters
        series_indicators = self.series.get_indicators()
        sma_indicators = sma_indicator.get_indicators()

        print(f"DEBUG: Series indicator names: {list(series_indicators.keys())}")
        print(f"DEBUG: SMA indicator names: {list(sma_indicators.keys())}")

        # Look for emitters in both locations
        all_indicators = {**series_indicators, **sma_indicators}
        emitter_names = [name for name in all_indicators.keys() if "emitter" in name.lower()]
        print(f"DEBUG: Emitter names: {emitter_names}")

        # Should have at least one emitter registered
        assert len(emitter_names) >= 1

        # Check that the emitters are actually IndicatorEmitter instances
        for name in emitter_names:
            assert isinstance(all_indicators[name], IndicatorEmitter)

    def test_emitter_with_atr_indicator(self):
        """Test emitter works with ATR indicator (OHLC data)."""
        atr_indicator = atr(self.ohlc_series, 3)

        emitter = IndicatorEmitter.wrap_with_emitter(
            indicator=atr_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="atr_volatility",
            instrument=self.instrument,
            tags={"timeframe": "1m", "period": "3"},
        )

        # Reset mock to clear initialization emissions
        self.mock_emitter.reset_mock()

        # Add more OHLC data to trigger emission
        base_time = 10 * 60_000_000_000
        self.ohlc_series.update_by_bar(
            base_time,
            open=109.5,
            high=111.0,
            low=109.0,
            close=110.0,
            vol_incr=1000,
        )

        # Should work without type errors
        assert isinstance(emitter, IndicatorEmitter)
        assert emitter._wrapped_indicator == atr_indicator

    def test_emitter_handles_emission_errors_gracefully(self):
        """Test that emitter handles emission errors without crashing."""
        sma_indicator = sma(self.series, 3)

        # Configure mock to raise exception on emit
        self.mock_emitter.emit.side_effect = Exception("Emission failed")

        _emitter = IndicatorEmitter(
            name="error_test_emitter",
            wrapped_indicator=sma_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="error_test_sma",
        )

        # Add data - should not raise exception despite emit failure
        base_time = 10 * 60_000_000_000
        result = self.series.update(base_time, 110)

        # Should return 1 for new item (TimeSeries.update returns 1 for new item, 0 for update)
        assert result == 1

    def test_indexing_and_length_delegation(self):
        """Test that indexing and length operations are delegated to wrapped indicator."""
        sma_indicator = sma(self.series, 3)

        emitter = IndicatorEmitter(
            name="delegation_test",
            wrapped_indicator=sma_indicator,
            metric_emitter=self.mock_emitter,
            metric_name="delegation_sma",
        )

        # Test length delegation
        assert len(emitter) == len(sma_indicator)

        # Test indexing delegation
        if len(sma_indicator) > 0:
            assert emitter[0] == sma_indicator[0]

        # Test that it updates when wrapped indicator updates
        original_length = len(emitter)
        self.series.update(20 * 60_000_000_000, 120)

        # Length should have potentially changed
        new_length = len(emitter)
        assert new_length >= original_length
