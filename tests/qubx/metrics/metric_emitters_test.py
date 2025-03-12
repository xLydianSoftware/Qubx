"""
Unit tests for metric emitters.
"""

from unittest.mock import MagicMock, patch

import pytest

from qubx.core.interfaces import IMetricEmitter
from qubx.metrics.composite import CompositeMetricEmitter
from qubx.metrics.null import NullMetricEmitter


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
