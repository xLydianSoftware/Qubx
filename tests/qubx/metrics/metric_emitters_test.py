"""
Unit tests for metric emitters.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from qubx.core.interfaces import IMetricEmitter, IStrategyContext
from qubx.metrics.base import BaseMetricEmitter
from qubx.metrics.composite import CompositeMetricEmitter
from qubx.metrics.null import NullMetricEmitter


class TestBaseMetricEmitter:
    """Test the BaseMetricEmitter class."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock strategy context."""
        context = MagicMock(spec=IStrategyContext)
        context.get_total_capital.return_value = 10000.0
        context.get_net_leverage.return_value = 0.5
        context.get_gross_leverage.return_value = 1.0
        context.instruments = ["BTC-USD", "ETH-USD"]

        # Mock positions
        position1 = MagicMock()
        position1.quantity = 1.0
        position1.pnl = 100.0
        position1.unrealized_pnl.return_value = 50.0

        instrument1 = MagicMock()
        instrument1.symbol = "BTC-USD"
        instrument1.exchange = "binance"
        instrument1.min_size = 0.001

        positions = {instrument1: position1}
        context.get_positions.return_value = positions

        # Mock quote
        quote = MagicMock()
        quote.mid_price.return_value = 50000.0
        context.quote.return_value = quote

        return context

    @pytest.fixture
    def emitter(self):
        """Create a BaseMetricEmitter instance for testing."""

        class TestEmitter(BaseMetricEmitter):
            """Test implementation of BaseMetricEmitter."""

            def __init__(self, stats_to_emit=None, stats_interval="1m"):
                super().__init__(stats_to_emit, stats_interval)
                self.emitted_gauges = []
                self.emitted_counters = []
                self.emitted_summaries = []

            def emit_gauge(self, name, value, tags=None):
                self.emitted_gauges.append((name, value, tags))

            def emit_counter(self, name, value=1.0, tags=None):
                self.emitted_counters.append((name, value, tags))

            def emit_summary(self, name, value, tags=None):
                self.emitted_summaries.append((name, value, tags))

        return TestEmitter()

    def test_emit_strategy_stats(self, emitter, mock_context):
        """Test that emit_strategy_stats emits the correct metrics."""
        emitter.emit_strategy_stats(mock_context)

        # Check that the context was stored
        assert emitter._context == mock_context

        # Check that the correct gauges were emitted
        gauge_names = [g[0] for g in emitter.emitted_gauges]
        assert "total_capital" in gauge_names
        assert "net_leverage" in gauge_names
        assert "gross_leverage" in gauge_names
        assert "universe_size" in gauge_names
        assert "position_count" in gauge_names
        assert "position_pnl" in gauge_names
        assert "position_unrealized_pnl" in gauge_names
        assert "position_leverage" in gauge_names

        # Check specific values
        for name, value, tags in emitter.emitted_gauges:
            if name == "total_capital":
                assert value == 10000.0
            elif name == "net_leverage":
                assert value == 0.5
            elif name == "gross_leverage":
                assert value == 1.0
            elif name == "universe_size":
                assert value == 2
            elif name == "position_count":
                assert value == 1
            elif name == "position_pnl":
                assert value == 100.0
                assert tags == {"symbol": "BTC-USD", "exchange": "binance"}
            elif name == "position_unrealized_pnl":
                assert value == 50.0
                assert tags == {"symbol": "BTC-USD", "exchange": "binance"}
            elif name == "position_leverage":
                assert value == (1.0 * 50000.0 / 10000.0) * 100  # (quantity * price / capital) * 100
                assert tags == {"symbol": "BTC-USD", "exchange": "binance"}

    def test_notify_first_call(self, emitter, mock_context):
        """Test that the first notify call initializes the last emission time."""
        # Set the context
        emitter._context = mock_context

        # Call notify with a timestamp
        timestamp = np.datetime64("2023-01-01T12:00:00")
        emitter.notify(timestamp)

        # Check that the last emission time was set
        assert emitter._last_emission_time == pd.Timestamp(timestamp)

        # Check that no metrics were emitted
        assert len(emitter.emitted_gauges) == 0

    def test_notify_before_interval(self, emitter, mock_context):
        """Test that notify doesn't emit metrics before the interval has passed."""
        # Set the context and last emission time
        emitter._context = mock_context
        emitter._last_emission_time = pd.Timestamp("2023-01-01T12:00:00")

        # Call notify with a timestamp before the interval has passed
        timestamp = np.datetime64("2023-01-01T12:00:30")  # 30 seconds later
        emitter.notify(timestamp)

        # Check that no metrics were emitted
        assert len(emitter.emitted_gauges) == 0

    def test_notify_after_interval(self, emitter, mock_context):
        """Test that notify emits metrics after the interval has passed."""
        # Set the context and last emission time
        emitter._context = mock_context
        emitter._last_emission_time = pd.Timestamp("2023-01-01T12:00:00")

        # Call notify with a timestamp after the interval has passed
        timestamp = np.datetime64("2023-01-01T12:01:00")  # 1 minute later
        emitter.notify(timestamp)

        # Check that metrics were emitted
        assert len(emitter.emitted_gauges) > 0

        # Check that the last emission time was updated
        assert emitter._last_emission_time == pd.Timestamp(timestamp)

    def test_custom_stats_to_emit(self, mock_context):
        """Test that only the specified stats are emitted."""

        # Create an emitter with custom stats to emit
        class TestEmitter(BaseMetricEmitter):
            def __init__(self, stats_to_emit=None, stats_interval="1m"):
                super().__init__(stats_to_emit, stats_interval)
                self.emitted_gauges = []

            def emit_gauge(self, name, value, tags=None):
                self.emitted_gauges.append((name, value, tags))

        emitter = TestEmitter(stats_to_emit=["total_capital", "net_leverage"])

        # Emit stats
        emitter.emit_strategy_stats(mock_context)

        # Check that only the specified stats were emitted
        gauge_names = [g[0] for g in emitter.emitted_gauges]
        assert "total_capital" in gauge_names
        assert "net_leverage" in gauge_names
        assert "gross_leverage" not in gauge_names
        assert "universe_size" not in gauge_names

    def test_custom_stats_interval(self, mock_context):
        """Test that the custom stats interval is respected."""

        # Create an emitter with a custom stats interval
        class TestEmitter(BaseMetricEmitter):
            def __init__(self, stats_to_emit=None, stats_interval="1m"):
                super().__init__(stats_to_emit, stats_interval)
                self.emitted_gauges = []

            def emit_gauge(self, name, value, tags=None):
                self.emitted_gauges.append((name, value, tags))

        emitter = TestEmitter(stats_interval="2m")

        # Set the context and last emission time
        emitter._context = mock_context
        emitter._last_emission_time = pd.Timestamp("2023-01-01T12:00:00")

        # Call notify with a timestamp before the interval has passed
        timestamp = np.datetime64("2023-01-01T12:01:00")  # 1 minute later
        emitter.notify(timestamp)

        # Check that no metrics were emitted
        assert len(emitter.emitted_gauges) == 0

        # Call notify with a timestamp after the interval has passed
        timestamp = np.datetime64("2023-01-01T12:02:00")  # 2 minutes later
        emitter.notify(timestamp)

        # Check that metrics were emitted
        assert len(emitter.emitted_gauges) > 0


class TestNullMetricEmitter:
    """Test the NullMetricEmitter class."""

    @pytest.fixture
    def emitter(self):
        """Create a NullMetricEmitter instance."""
        return NullMetricEmitter()

    def test_emit_gauge(self, emitter):
        """Test that emit_gauge does nothing."""
        # This should not raise any exceptions
        emitter.emit_gauge("test_gauge", 1.0, {"tag": "value"})

    def test_emit_counter(self, emitter):
        """Test that emit_counter does nothing."""
        # This should not raise any exceptions
        emitter.emit_counter("test_counter", 1.0, {"tag": "value"})

    def test_emit_summary(self, emitter):
        """Test that emit_summary does nothing."""
        # This should not raise any exceptions
        emitter.emit_summary("test_summary", 1.0, {"tag": "value"})

    def test_notify(self, emitter):
        """Test that notify method works with NullMetricEmitter."""
        # This should not raise any exceptions
        timestamp = np.datetime64("2023-01-01T12:00:00")
        emitter.notify(timestamp)


class TestCompositeMetricEmitter:
    """Test the CompositeMetricEmitter class."""

    @pytest.fixture
    def emitters(self):
        """Create mock emitters."""
        emitter1 = MagicMock(spec=IMetricEmitter)
        emitter2 = MagicMock(spec=IMetricEmitter)
        return emitter1, emitter2

    @pytest.fixture
    def composite(self, emitters):
        """Create a CompositeMetricEmitter instance."""
        return CompositeMetricEmitter(list(emitters))

    def test_emit_gauge(self, composite, emitters):
        """Test that emit_gauge delegates to all emitters."""
        emitter1, emitter2 = emitters
        composite.emit_gauge("test_gauge", 1.0, {"tag": "value"})
        emitter1.emit_gauge.assert_called_once_with("test_gauge", 1.0, {"tag": "value"})
        emitter2.emit_gauge.assert_called_once_with("test_gauge", 1.0, {"tag": "value"})

    def test_emit_counter(self, composite, emitters):
        """Test that emit_counter delegates to all emitters."""
        emitter1, emitter2 = emitters
        composite.emit_counter("test_counter", 1.0, {"tag": "value"})
        emitter1.emit_counter.assert_called_once_with("test_counter", 1.0, {"tag": "value"})
        emitter2.emit_counter.assert_called_once_with("test_counter", 1.0, {"tag": "value"})

    def test_emit_summary(self, composite, emitters):
        """Test that emit_summary delegates to all emitters."""
        emitter1, emitter2 = emitters
        composite.emit_summary("test_summary", 1.0, {"tag": "value"})
        emitter1.emit_summary.assert_called_once_with("test_summary", 1.0, {"tag": "value"})
        emitter2.emit_summary.assert_called_once_with("test_summary", 1.0, {"tag": "value"})

    def test_emit_gauge_with_exception(self, composite, emitters):
        """Test that emit_gauge continues even if one emitter raises an exception."""
        emitter1, emitter2 = emitters
        emitter1.emit_gauge.side_effect = Exception("Test exception")
        # This should not raise an exception
        composite.emit_gauge("test_gauge", 1.0, {"tag": "value"})
        emitter1.emit_gauge.assert_called_once_with("test_gauge", 1.0, {"tag": "value"})
        emitter2.emit_gauge.assert_called_once_with("test_gauge", 1.0, {"tag": "value"})

    def test_emit_counter_with_exception(self, composite, emitters):
        """Test that emit_counter continues even if one emitter raises an exception."""
        emitter1, emitter2 = emitters
        emitter1.emit_counter.side_effect = Exception("Test exception")
        # This should not raise an exception
        composite.emit_counter("test_counter", 1.0, {"tag": "value"})
        emitter1.emit_counter.assert_called_once_with("test_counter", 1.0, {"tag": "value"})
        emitter2.emit_counter.assert_called_once_with("test_counter", 1.0, {"tag": "value"})

    def test_emit_summary_with_exception(self, composite, emitters):
        """Test that emit_summary continues even if one emitter raises an exception."""
        emitter1, emitter2 = emitters
        emitter1.emit_summary.side_effect = Exception("Test exception")
        # This should not raise an exception
        composite.emit_summary("test_summary", 1.0, {"tag": "value"})
        emitter1.emit_summary.assert_called_once_with("test_summary", 1.0, {"tag": "value"})
        emitter2.emit_summary.assert_called_once_with("test_summary", 1.0, {"tag": "value"})

    def test_notify(self, composite, emitters):
        """Test that notify delegates to all emitters."""
        emitter1, emitter2 = emitters
        timestamp = np.datetime64("2023-01-01T12:00:00")
        composite.notify(timestamp)
        emitter1.notify.assert_called_once_with(timestamp)
        emitter2.notify.assert_called_once_with(timestamp)

    def test_notify_with_exception(self, composite, emitters):
        """Test that notify continues even if one emitter raises an exception."""
        emitter1, emitter2 = emitters
        emitter1.notify.side_effect = Exception("Test exception")
        timestamp = np.datetime64("2023-01-01T12:00:00")
        # This should not raise an exception
        composite.notify(timestamp)
        emitter1.notify.assert_called_once_with(timestamp)
        emitter2.notify.assert_called_once_with(timestamp)


class TestPrometheusMetricEmitter:
    """Test the PrometheusMetricEmitter class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry."""
        return MagicMock()

    @pytest.fixture
    def mock_gauge(self):
        """Create a mock gauge."""
        gauge = MagicMock()
        gauge_instance = MagicMock()
        gauge.return_value = gauge_instance
        gauge_instance.labels.return_value = gauge_instance
        return gauge

    @pytest.fixture
    def mock_counter(self):
        """Create a mock counter."""
        counter = MagicMock()
        counter_instance = MagicMock()
        counter.return_value = counter_instance
        counter_instance.labels.return_value = counter_instance
        return counter

    @pytest.fixture
    def mock_summary(self):
        """Create a mock summary."""
        summary = MagicMock()
        summary_instance = MagicMock()
        summary.return_value = summary_instance
        summary_instance.labels.return_value = summary_instance
        return summary

    @pytest.fixture
    def mock_push_gateway(self):
        """Create a mock push_to_gateway function."""
        return MagicMock()

    @pytest.fixture
    def emitter(self, mock_registry, mock_push_gateway):
        """Create a PrometheusMetricEmitter instance with a mock registry."""
        with patch("qubx.metrics.prometheus.REGISTRY", mock_registry):
            with patch("qubx.metrics.prometheus.push_to_gateway", mock_push_gateway):
                from qubx.metrics.prometheus import PrometheusMetricEmitter

                return PrometheusMetricEmitter(
                    strategy_name="test_strategy", pushgateway_url="http://localhost:9091", expose_http=False
                )

    def test_emit_gauge(self, emitter, mock_registry, mock_gauge, mock_push_gateway):
        """Test that emit_gauge creates and updates a gauge metric."""
        with patch("qubx.metrics.prometheus.Gauge", mock_gauge):
            with patch("qubx.metrics.prometheus.push_to_gateway", mock_push_gateway):
                # Call the method
                emitter.emit_gauge("test_gauge", 1.0, {"tag": "value"})

                # Verify the calls
                mock_gauge.assert_called_once()
                mock_gauge.return_value.labels.assert_called_once_with(strategy="test_strategy", tag="value")
                mock_gauge.return_value.labels.return_value.set.assert_called_once_with(1.0)
                mock_push_gateway.assert_called_once()

    def test_emit_counter(self, emitter, mock_registry, mock_counter, mock_push_gateway):
        """Test that emit_counter creates and updates a counter metric."""
        with patch("qubx.metrics.prometheus.Counter", mock_counter):
            with patch("qubx.metrics.prometheus.push_to_gateway", mock_push_gateway):
                # Call the method
                emitter.emit_counter("test_counter", 1.0, {"tag": "value"})

                # Verify the calls
                mock_counter.assert_called_once()
                mock_counter.return_value.labels.assert_called_once_with(strategy="test_strategy", tag="value")
                mock_counter.return_value.labels.return_value.inc.assert_called_once_with(1.0)
                mock_push_gateway.assert_called_once()

    def test_emit_summary(self, emitter, mock_registry, mock_summary, mock_push_gateway):
        """Test that emit_summary creates and updates a summary metric."""
        with patch("qubx.metrics.prometheus.Summary", mock_summary):
            with patch("qubx.metrics.prometheus.push_to_gateway", mock_push_gateway):
                # Call the method
                emitter.emit_summary("test_summary", 1.0, {"tag": "value"})

                # Verify the calls
                mock_summary.assert_called_once()
                mock_summary.return_value.labels.assert_called_once_with(strategy="test_strategy", tag="value")
                mock_summary.return_value.labels.return_value.observe.assert_called_once_with(1.0)
                mock_push_gateway.assert_called_once()

    def test_emit_gauge_without_pushgateway(self, mock_registry, mock_gauge, mock_push_gateway):
        """Test that emit_gauge works without a pushgateway."""
        with patch("qubx.metrics.prometheus.REGISTRY", mock_registry):
            with patch("qubx.metrics.prometheus.Gauge", mock_gauge):
                with patch("qubx.metrics.prometheus.push_to_gateway", mock_push_gateway):
                    # Create a new emitter without a pushgateway
                    from qubx.metrics.prometheus import PrometheusMetricEmitter

                    emitter = PrometheusMetricEmitter(
                        strategy_name="test_strategy", pushgateway_url=None, expose_http=False
                    )

                    # Call the method
                    emitter.emit_gauge("test_gauge", 1.0, {"tag": "value"})

                    # Verify the calls
                    mock_gauge.assert_called_once()
                    mock_gauge.return_value.labels.assert_called_once_with(strategy="test_strategy", tag="value")
                    mock_gauge.return_value.labels.return_value.set.assert_called_once_with(1.0)
                    mock_push_gateway.assert_not_called()
