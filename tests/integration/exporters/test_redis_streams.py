"""
Integration tests for the Redis Streams Exporter.

These tests require Redis to be running.
"""

import numpy as np
import pytest
import redis

from qubx.core.basics import AssetType, Instrument, MarketType, Signal, TargetPosition
from qubx.exporters.redis_streams import RedisStreamsExporter

# MockAccountViewer is now imported from conftest.py via the fixture


@pytest.fixture
def instruments():
    """Fixture for test instruments."""
    return [
        Instrument(
            symbol="BTC-USDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTCUSDT",
            tick_size=0.01,
            lot_size=0.00001,
            min_size=0.0001,
        ),
        Instrument(
            symbol="ETH-USDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETHUSDT",
            tick_size=0.01,
            lot_size=0.00001,
            min_size=0.0001,
        ),
    ]


@pytest.fixture
def signals(instruments):
    """Fixture for test signals."""
    return [
        Signal(instruments[0], 1.0, reference_price=50000.0, group="test_group"),
        Signal(instruments[1], -0.5, reference_price=3000.0, group="test_group"),
    ]


@pytest.fixture
def target_positions(instruments, signals):
    """Fixture for test target positions."""
    time_now = np.datetime64("now")
    return [
        TargetPosition(time_now, signals[0], 0.1),
        TargetPosition(time_now, signals[1], -0.05),
    ]


@pytest.fixture
def clear_redis_streams(redis_service):
    """Fixture to clear Redis streams before each test."""
    r = redis.from_url(redis_service)

    # Clear all streams used in tests
    stream_keys = [
        "strategy:test_strategy:signals",
        "strategy:test_strategy:targets",
        "strategy:test_strategy:position_changes",
    ]

    for key in stream_keys:
        try:
            # Delete the stream if it exists
            r.delete(key)
        except Exception:
            pass

    yield

    # Clean up after tests
    for key in stream_keys:
        try:
            r.delete(key)
        except Exception:
            pass


@pytest.mark.integration
class TestRedisStreamsExporter:
    """Integration tests for the RedisStreamsExporter."""

    def test_export_signals(self, redis_service, account_viewer, signals, clear_redis_streams):
        """Test exporting signals with the default formatter."""
        # Create the exporter
        exporter = RedisStreamsExporter(
            redis_url=redis_service,
            strategy_name="test_strategy",
            export_signals=True,
            export_targets=False,
            export_position_changes=False,
        )

        # Export the signals
        current_time = np.datetime64("now")
        exporter.export_signals(current_time, signals, account_viewer)

        # Connect to Redis and check if the signals were exported
        r = redis.from_url(redis_service)

        # Get the signals from the stream
        stream_key = "strategy:test_strategy:signals"
        result = r.xread({stream_key: "0-0"}, count=10)

        # Check if we got results
        assert result, "No signals found in Redis stream"

        # Get the messages from the result
        # Note: We're ignoring type checking here to avoid linter errors
        messages = result[0][1]  # type: ignore

        # Check if we got the expected number of signals
        assert len(messages) == 2, f"Expected 2 signals, got {len(messages)}"

        # Check the content of the signals
        for i, (msg_id, msg_data) in enumerate(messages):
            # Convert byte keys/values to strings
            msg_data = {k.decode(): v.decode() for k, v in msg_data.items()}

            # Check basic fields
            assert "timestamp" in msg_data
            assert msg_data["instrument"] == signals[i].instrument.symbol
            assert msg_data["exchange"] == signals[i].instrument.exchange
            assert float(msg_data["direction"]) == signals[i].signal
            assert msg_data["group"] == "test_group"
            assert float(msg_data["reference_price"]) == signals[i].reference_price

    @pytest.mark.integration
    def test_export_target_positions(self, redis_service, account_viewer, target_positions, clear_redis_streams):
        """Test exporting target positions."""
        # Create the exporter
        exporter = RedisStreamsExporter(
            redis_url=redis_service,
            strategy_name="test_strategy",
            export_signals=False,
            export_targets=True,
            export_position_changes=False,
        )

        # Export the target positions
        current_time = np.datetime64("now")
        exporter.export_target_positions(current_time, target_positions, account_viewer)

        # Connect to Redis and check if the target positions were exported
        r = redis.from_url(redis_service)

        # Get the target positions from the stream
        stream_key = "strategy:test_strategy:targets"
        result = r.xread({stream_key: "0-0"}, count=10)

        # Check if we got results
        assert result, "No target positions found in Redis stream"

        # Get the messages from the result
        # Note: We're ignoring type checking here to avoid linter errors
        messages = result[0][1]  # type: ignore

        # Check if we got the expected number of target positions
        assert len(messages) == 2, f"Expected 2 target positions, got {len(messages)}"

        # Check the content of the target positions
        for i, (msg_id, msg_data) in enumerate(messages):
            # Convert byte keys/values to strings
            msg_data = {k.decode(): v.decode() for k, v in msg_data.items()}

            # Check basic fields
            assert "timestamp" in msg_data
            assert msg_data["instrument"] == target_positions[i].instrument.symbol
            assert msg_data["exchange"] == target_positions[i].instrument.exchange
            assert float(msg_data["target_size"]) == target_positions[i].target_position_size
            assert "price" in msg_data  # Price might be empty string if None

    @pytest.mark.integration
    def test_export_leverage_changes(self, redis_service, account_viewer, instruments, clear_redis_streams):
        """Test exporting leverage changes."""
        # Create the exporter
        exporter = RedisStreamsExporter(
            redis_url=redis_service,
            strategy_name="test_strategy",
            export_signals=False,
            export_targets=False,
            export_position_changes=True,
        )

        # Export a leverage change
        current_time = np.datetime64("now")
        instrument = instruments[0]
        price = 50000.0

        # Set the previous leverage
        exporter._instrument_to_previous_leverage[instrument] = 1.0

        # Export the leverage change
        exporter.export_position_changes(current_time, instrument, price, account_viewer)

        # Connect to Redis and check if the leverage change was exported
        r = redis.from_url(redis_service)

        # Get the leverage changes from the stream
        stream_key = "strategy:test_strategy:position_changes"
        result = r.xread({stream_key: "0-0"}, count=10)

        # Check if we got results
        assert result, "No leverage changes found in Redis stream"

        # Get the messages from the result
        # Note: We're ignoring type checking here to avoid linter errors
        messages = result[0][1]  # type: ignore

        # Check if we got any messages
        assert len(messages) > 0, "No leverage changes found in Redis stream"

        # Get the latest message
        msg_id, msg_data = messages[-1]

        # Convert byte keys/values to strings
        msg_data = {k.decode(): v.decode() for k, v in msg_data.items()}

        # Check basic fields
        assert "timestamp" in msg_data
        assert msg_data["instrument"] == instrument.symbol
        assert msg_data["exchange"] == instrument.exchange
        assert float(msg_data["price"]) == price
