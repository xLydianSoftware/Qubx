"""
Unit tests for metric emitters.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qubx.core.interfaces import IMetricEmitter, IStrategyContext
from qubx.emitters.base import BaseMetricEmitter
from qubx.emitters.composite import CompositeMetricEmitter


class TestBaseMetricEmitter:
    """Test the BaseMetricEmitter class."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock strategy context."""
        mock = MagicMock(spec=IStrategyContext)
        mock.get_total_capital.return_value = 10000.0
        mock.get_net_leverage.return_value = 0.5
        mock.get_gross_leverage.return_value = 0.7
        mock.instruments = ["BTC-USD", "ETH-USD"]

        # Mock positions
        position1 = MagicMock()
        position1.quantity = 1.0
        position1.entry_price = 50000.0
        position1.current_price = 55000.0
        position1.unrealized_pnl = 5000.0
        position1.realized_pnl = 1000.0

        instrument1 = MagicMock()
        instrument1.min_size = 0.001
        instrument1.__str__.return_value = "BTC-USD"

        position2 = MagicMock()
        position2.quantity = 0.0  # Inactive position
        position2.entry_price = 2000.0
        position2.current_price = 2100.0
        position2.unrealized_pnl = 0.0
        position2.realized_pnl = 500.0

        instrument2 = MagicMock()
        instrument2.min_size = 0.01
        instrument2.__str__.return_value = "ETH-USD"

        mock.get_positions.return_value = {instrument1: position1, instrument2: position2}
        mock.get_leverages.return_value = {instrument1: 1.0, instrument2: 0.0}

        return mock

    @pytest.fixture
    def emitter(self):
        """Create a test emitter."""

        class TestEmitter(BaseMetricEmitter):
            """Test emitter implementation."""

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
        """Test that emit_strategy_stats emits the expected metrics."""
        emitter.emit_strategy_stats(mock_context)

        # Check that the context was stored
        assert emitter._context == mock_context

        # Check that the expected gauges were emitted
        expected_gauges = [
            ("total_capital", 10000.0, None),
            ("net_leverage", 0.5, None),
            ("gross_leverage", 0.7, None),
            ("universe_size", 2, None),
            ("position_count", 1, None),  # Only one active position
        ]

        # Check position-level metrics
        position_gauges = [g for g in emitter.emitted_gauges if g[0].startswith("position_")]
        assert len(position_gauges) >= 2  # At least 2 position metrics (position_count and position_pnl)

        # Check that all expected gauges were emitted
        for expected in expected_gauges:
            assert expected in emitter.emitted_gauges, f"Expected gauge {expected} not emitted"

    def test_notify_first_call(self, emitter, mock_context):
        """Test that the first call to notify initializes the last emission time."""
        # First call should just initialize the last emission time
        timestamp = pd.Timestamp("2023-01-01 12:00:00").to_numpy()
        emitter._context = mock_context
        emitter.notify(timestamp)

        assert emitter._last_emission_time == timestamp
        assert len(emitter.emitted_gauges) == 0  # No metrics should be emitted

    def test_notify_before_interval(self, emitter, mock_context):
        """Test that notify doesn't emit metrics before the interval has passed."""
        # Set up initial state
        emitter._context = mock_context
        emitter._last_emission_time = pd.Timestamp("2023-01-01 12:00:00")

        # Call notify before the interval has passed
        timestamp = pd.Timestamp("2023-01-01 12:00:30").to_numpy()  # 30 seconds later
        emitter.notify(timestamp)

        assert emitter._last_emission_time == pd.Timestamp("2023-01-01 12:00:00")
        assert len(emitter.emitted_gauges) == 0  # No metrics should be emitted

    def test_notify_after_interval(self, emitter, mock_context):
        """Test that notify emits metrics after the interval has passed."""
        # Set up initial state
        emitter._context = mock_context
        emitter._last_emission_time = pd.Timestamp("2023-01-01 12:00:00")

        # Call notify after the interval has passed
        timestamp = pd.Timestamp("2023-01-01 12:01:30").to_numpy()  # 1 minute 30 seconds later
        emitter.notify(timestamp)

        assert emitter._last_emission_time == timestamp
        assert len(emitter.emitted_gauges) > 0  # Metrics should be emitted

    def test_custom_stats_to_emit(self, mock_context):
        """Test that custom stats_to_emit works correctly."""

        class TestEmitter(BaseMetricEmitter):
            def __init__(self, stats_to_emit=None, stats_interval="1m"):
                super().__init__(stats_to_emit, stats_interval)
                self.emitted_gauges = []

            def emit_gauge(self, name, value, tags=None):
                self.emitted_gauges.append((name, value, tags))

        # Create emitter with custom stats to emit
        emitter = TestEmitter(stats_to_emit=["total_capital", "net_leverage"])
        emitter.emit_strategy_stats(mock_context)

        # Check that only the specified metrics were emitted
        assert len(emitter.emitted_gauges) == 2
        assert ("total_capital", 10000.0, None) in emitter.emitted_gauges
        assert ("net_leverage", 0.5, None) in emitter.emitted_gauges
        assert not any(g[0] == "gross_leverage" for g in emitter.emitted_gauges)

    def test_custom_stats_interval(self, mock_context):
        """Test that custom stats_interval works correctly."""

        class TestEmitter(BaseMetricEmitter):
            def __init__(self, stats_to_emit=None, stats_interval="1m"):
                super().__init__(stats_to_emit, stats_interval)
                self.emitted_gauges = []

            def emit_gauge(self, name, value, tags=None):
                self.emitted_gauges.append((name, value, tags))

        # Create emitter with custom stats interval
        emitter = TestEmitter(stats_interval="5m")
        emitter._context = mock_context
        emitter._last_emission_time = pd.Timestamp("2023-01-01 12:00:00")

        # Call notify before the interval has passed
        timestamp = pd.Timestamp("2023-01-01 12:04:00").to_numpy()  # 4 minutes later
        emitter.notify(timestamp)
        assert len(emitter.emitted_gauges) == 0  # No metrics should be emitted

        # Call notify after the interval has passed
        timestamp = pd.Timestamp("2023-01-01 12:05:30").to_numpy()  # 5 minutes 30 seconds later
        emitter.notify(timestamp)
        assert len(emitter.emitted_gauges) > 0  # Metrics should be emitted


class TestCompositeMetricEmitter:
    """Test the CompositeMetricEmitter class."""

    @pytest.fixture
    def emitters(self):
        """Create mock emitters."""
        emitter1 = MagicMock(spec=IMetricEmitter)
        emitter2 = MagicMock(spec=IMetricEmitter)
        return [emitter1, emitter2]

    @pytest.fixture
    def composite(self, emitters):
        """Create a composite emitter with mock emitters."""
        return CompositeMetricEmitter(emitters=emitters)

    def test_emit_gauge(self, composite, emitters):
        """Test that emit_gauge calls emit_gauge on all emitters."""
        composite.emit_gauge("test", 1.0, {"tag": "value"})
        for emitter in emitters:
            emitter.emit_gauge.assert_called_once_with("test", 1.0, {"tag": "value"})

    def test_emit_counter(self, composite, emitters):
        """Test that emit_counter calls emit_counter on all emitters."""
        composite.emit_counter("test", 1.0, {"tag": "value"})
        for emitter in emitters:
            emitter.emit_counter.assert_called_once_with("test", 1.0, {"tag": "value"})

    def test_emit_summary(self, composite, emitters):
        """Test that emit_summary calls emit_summary on all emitters."""
        composite.emit_summary("test", 1.0, {"tag": "value"})
        for emitter in emitters:
            emitter.emit_summary.assert_called_once_with("test", 1.0, {"tag": "value"})

    def test_emit_gauge_with_exception(self, composite, emitters):
        """Test that emit_gauge handles exceptions from emitters."""
        emitters[0].emit_gauge.side_effect = Exception("Test exception")
        # This should not raise an exception
        composite.emit_gauge("test", 1.0)
        # The second emitter should still be called
        emitters[1].emit_gauge.assert_called_once()

    def test_emit_counter_with_exception(self, composite, emitters):
        """Test that emit_counter handles exceptions from emitters."""
        emitters[0].emit_counter.side_effect = Exception("Test exception")
        # This should not raise an exception
        composite.emit_counter("test", 1.0)
        # The second emitter should still be called
        emitters[1].emit_counter.assert_called_once()

    def test_emit_summary_with_exception(self, composite, emitters):
        """Test that emit_summary handles exceptions from emitters."""
        emitters[0].emit_summary.side_effect = Exception("Test exception")
        # This should not raise an exception
        composite.emit_summary("test", 1.0)
        # The second emitter should still be called
        emitters[1].emit_summary.assert_called_once()

    def test_notify(self, composite, emitters):
        """Test that notify calls notify on all emitters."""
        timestamp = pd.Timestamp("2023-01-01 12:00:00").to_numpy()
        composite.notify(timestamp)
        for emitter in emitters:
            emitter.notify.assert_called_once_with(timestamp)

    def test_notify_with_exception(self, composite, emitters):
        """Test that notify handles exceptions from emitters."""
        timestamp = pd.Timestamp("2023-01-01 12:00:00").to_numpy()
        emitters[0].notify.side_effect = Exception("Test exception")
        # This should not raise an exception
        composite.notify(timestamp)
        # The second emitter should still be called
        emitters[1].notify.assert_called_once_with(timestamp)


class TestPrometheusMetricEmitter:
    """Test the PrometheusMetricEmitter class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry."""
        registry = MagicMock()
        registry.get_sample_value.return_value = 1.0
        return registry

    @pytest.fixture
    def mock_gauge(self):
        """Create a mock gauge."""
        gauge = MagicMock()
        gauge.return_value = gauge
        gauge.labels.return_value = gauge
        return gauge

    @pytest.fixture
    def mock_counter(self):
        """Create a mock counter."""
        counter = MagicMock()
        counter.return_value = counter
        counter.labels.return_value = counter
        return counter

    @pytest.fixture
    def mock_summary(self):
        """Create a mock summary."""
        summary = MagicMock()
        summary.return_value = summary
        summary.labels.return_value = summary
        return summary

    @pytest.fixture
    def mock_push_gateway(self):
        """Create a mock push_to_gateway function."""
        return MagicMock()

    @pytest.fixture
    def emitter(self, mock_registry, mock_push_gateway):
        """Create a PrometheusMetricEmitter with mocks."""
        with patch("qubx.emitters.prometheus.REGISTRY", mock_registry):
            with patch("qubx.emitters.prometheus.push_to_gateway", mock_push_gateway):
                from qubx.emitters.prometheus import PrometheusMetricEmitter

                return PrometheusMetricEmitter(pushgateway_url="localhost:9091", strategy_name="test")

    def test_emit_gauge(self, emitter, mock_registry, mock_gauge, mock_push_gateway):
        """Test that emit_gauge creates and updates a gauge."""
        with patch("qubx.emitters.prometheus.Gauge", mock_gauge):
            with patch("qubx.emitters.prometheus.push_to_gateway", mock_push_gateway):
                emitter.emit_gauge("test", 1.0, {"tag": "value"})
                mock_gauge.assert_called_once()
                mock_gauge.labels.assert_called_once_with(strategy="test", tag="value")
                mock_gauge.labels().set.assert_called_once_with(1.0)
                mock_push_gateway.assert_called_once()

    def test_emit_counter(self, emitter, mock_registry, mock_counter, mock_push_gateway):
        """Test that emit_counter creates and updates a counter."""
        with patch("qubx.emitters.prometheus.Counter", mock_counter):
            with patch("qubx.emitters.prometheus.push_to_gateway", mock_push_gateway):
                emitter.emit_counter("test", 1.0, {"tag": "value"})
                mock_counter.assert_called_once()
                mock_counter.labels.assert_called_once_with(strategy="test", tag="value")
                mock_counter.labels().inc.assert_called_once_with(1.0)
                mock_push_gateway.assert_called_once()

    def test_emit_summary(self, emitter, mock_registry, mock_summary, mock_push_gateway):
        """Test that emit_summary creates and updates a summary."""
        with patch("qubx.emitters.prometheus.Summary", mock_summary):
            with patch("qubx.emitters.prometheus.push_to_gateway", mock_push_gateway):
                emitter.emit_summary("test", 1.0, {"tag": "value"})
                mock_summary.assert_called_once()
                mock_summary.labels.assert_called_once_with(strategy="test", tag="value")
                mock_summary.labels().observe.assert_called_once_with(1.0)
                mock_push_gateway.assert_called_once()

    def test_emit_gauge_without_pushgateway(self, mock_registry, mock_gauge, mock_push_gateway):
        """Test that emit_gauge works without a pushgateway."""
        with patch("qubx.emitters.prometheus.REGISTRY", mock_registry):
            with patch("qubx.emitters.prometheus.Gauge", mock_gauge):
                with patch("qubx.emitters.prometheus.push_to_gateway", mock_push_gateway):
                    from qubx.emitters.prometheus import PrometheusMetricEmitter

                    emitter = PrometheusMetricEmitter(strategy_name="test")  # No pushgateway URL
                    emitter.emit_gauge("test", 1.0)
                    mock_gauge.assert_called_once()
                    mock_gauge().set.assert_called_once_with(1.0)
                    mock_push_gateway.assert_not_called()
