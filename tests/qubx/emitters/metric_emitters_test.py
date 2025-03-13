"""
Unit tests for metric emitters.
"""

import datetime
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
                self.emitted_metrics = []

            def _emit_impl(self, name, value, tags, timestamp=None):
                self.emitted_metrics.append((name, value, tags, timestamp))

        return TestEmitter()

    def test_emit_strategy_stats(self, emitter, mock_context):
        """Test that emit_strategy_stats emits the expected metrics."""
        emitter.emit_strategy_stats(mock_context)

        # Check that the context was stored
        assert emitter._context == mock_context

        # Check that the expected metrics were emitted
        expected_metrics = [
            ("total_capital", 10000.0, {}, None),
            ("net_leverage", 0.5, {}, None),
            ("gross_leverage", 0.7, {}, None),
            ("universe_size", 2, {}, None),
            ("position_count", 1, {}, None),  # Only one active position
        ]

        # Check position-level metrics
        position_metrics = [m for m in emitter.emitted_metrics if m[0].startswith("position_")]
        assert len(position_metrics) >= 2  # At least 2 position metrics (position_count and position_pnl)

        # Check that all expected metrics were emitted
        for expected in expected_metrics:
            name, value, _, _ = expected
            matching = [m for m in emitter.emitted_metrics if m[0] == name and m[1] == value]
            assert matching, f"Expected metric {name} with value {value} not found"

    def test_notify_first_call(self, emitter, mock_context):
        """Test that the first call to notify initializes the last emission time."""
        # Set the context
        emitter._context = mock_context

        # Mock the time
        timestamp = pd.Timestamp("2023-01-01 00:00:00").to_numpy()
        mock_context.time.return_value = timestamp

        # Call notify
        emitter.notify(timestamp)

        # Check that the last emission time was set
        # The BaseMetricEmitter converts the numpy datetime64 to a pandas Timestamp
        assert emitter._last_emission_time == pd.Timestamp(timestamp)

        # Check that no metrics were emitted
        assert len(emitter.emitted_metrics) == 0

    def test_notify_before_interval(self, emitter, mock_context):
        """Test that notify doesn't emit metrics before the interval has passed."""
        # Set up the emitter with a context
        emitter._context = mock_context

        # Set the last emission time
        emitter._last_emission_time = pd.Timestamp("2023-01-01 00:00:00")

        # Call notify with a time before the interval has passed
        emitter.notify(pd.Timestamp("2023-01-01 00:00:30"))  # 30 seconds later

        # Check that no metrics were emitted
        assert len(emitter.emitted_metrics) == 0

    def test_notify_after_interval(self, emitter, mock_context):
        """Test that notify emits metrics after the interval has passed."""
        # Set up the emitter with a context
        emitter._context = mock_context

        # Set the last emission time
        emitter._last_emission_time = pd.Timestamp("2023-01-01 00:00:00")

        # Call notify with a time after the interval has passed
        emitter.notify(pd.Timestamp("2023-01-01 00:01:30"))  # 1 minute and 30 seconds later

        # Check that metrics were emitted
        assert len(emitter.emitted_metrics) > 0

    def test_custom_stats_to_emit(self, mock_context):
        """Test that custom stats_to_emit works correctly."""

        class TestEmitter(BaseMetricEmitter):
            """Test emitter implementation."""

            def __init__(self, stats_to_emit=None, stats_interval="1m"):
                super().__init__(stats_to_emit, stats_interval)
                self.emitted_metrics = []

            def _emit_impl(self, name, value, tags, timestamp=None):
                self.emitted_metrics.append((name, value, tags, timestamp))

        # Create an emitter with custom stats to emit
        emitter = TestEmitter(stats_to_emit=["total_capital", "net_leverage"])

        # Emit strategy stats
        emitter.emit_strategy_stats(mock_context)

        # Check that only the specified metrics were emitted
        assert len(emitter.emitted_metrics) == 2
        assert emitter.emitted_metrics[0][0] == "total_capital"
        assert emitter.emitted_metrics[1][0] == "net_leverage"

    def test_custom_stats_interval(self, mock_context):
        """Test that custom stats_interval works correctly."""

        class TestEmitter(BaseMetricEmitter):
            """Test emitter implementation."""

            def __init__(self, stats_to_emit=None, stats_interval="1m"):
                super().__init__(stats_to_emit, stats_interval)
                self.emitted_metrics = []

            def _emit_impl(self, name, value, tags, timestamp=None):
                self.emitted_metrics.append((name, value, tags, timestamp))

        # Create an emitter with a custom stats interval
        emitter = TestEmitter(stats_interval="2m")
        emitter._context = mock_context

        # Set the last emission time
        emitter._last_emission_time = pd.Timestamp("2023-01-01 00:00:00")

        # Call notify with a time before the interval has passed
        emitter.notify(pd.Timestamp("2023-01-01 00:01:30").to_numpy())  # 1 minute and 30 seconds later

        # Check that no metrics were emitted
        assert len(emitter.emitted_metrics) == 0

        # Call notify with a time after the interval has passed
        emitter.notify(pd.Timestamp("2023-01-01 00:02:30").to_numpy())  # 2 minutes and 30 seconds later

        # Check that metrics were emitted
        assert len(emitter.emitted_metrics) > 0


class TestCompositeMetricEmitter:
    """Test the CompositeMetricEmitter class."""

    @pytest.fixture
    def emitters(self):
        """Create a list of mock emitters."""
        emitter1 = MagicMock(spec=IMetricEmitter)
        emitter2 = MagicMock(spec=IMetricEmitter)
        return [emitter1, emitter2]

    @pytest.fixture
    def composite(self, emitters):
        """Create a CompositeMetricEmitter with mock emitters."""
        return CompositeMetricEmitter(emitters)

    def test_emit(self, composite, emitters):
        """Test that emit delegates to all emitters."""
        composite.emit("test", 1.0, {"tag": "value"}, pd.Timestamp("2023-01-01").to_numpy())
        for emitter in emitters:
            emitter.emit.assert_called_once_with("test", 1.0, {"tag": "value"}, pd.Timestamp("2023-01-01").to_numpy())

    def test_emit_with_exception(self, composite, emitters):
        """Test that emit handles exceptions from emitters."""
        emitters[0].emit.side_effect = Exception("Test exception")
        composite.emit("test", 1.0, {"tag": "value"})
        emitters[0].emit.assert_called_once()
        emitters[1].emit.assert_called_once()

    def test_emit_strategy_stats(self, composite, emitters):
        """Test that emit_strategy_stats delegates to all emitters."""
        context = MagicMock(spec=IStrategyContext)
        composite.emit_strategy_stats(context)
        for emitter in emitters:
            emitter.emit_strategy_stats.assert_called_once_with(context)

    def test_emit_strategy_stats_with_exception(self, composite, emitters):
        """Test that emit_strategy_stats handles exceptions from emitters."""
        emitters[0].emit_strategy_stats.side_effect = Exception("Test exception")
        context = MagicMock(spec=IStrategyContext)
        composite.emit_strategy_stats(context)
        emitters[0].emit_strategy_stats.assert_called_once()
        emitters[1].emit_strategy_stats.assert_called_once()

    def test_notify(self, composite, emitters):
        """Test that notify delegates to all emitters."""
        timestamp = pd.Timestamp("2023-01-01").to_numpy()
        composite.notify(timestamp)
        for emitter in emitters:
            emitter.notify.assert_called_once_with(timestamp)

    def test_notify_with_exception(self, composite, emitters):
        """Test that notify handles exceptions from emitters."""
        emitters[0].notify.side_effect = Exception("Test exception")
        timestamp = pd.Timestamp("2023-01-01").to_numpy()
        composite.notify(timestamp)
        emitters[0].notify.assert_called_once()
        emitters[1].notify.assert_called_once()


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

    def test_emit_gauge_metric(self, emitter, mock_registry, mock_gauge, mock_push_gateway):
        """Test that emit with gauge metric type creates and updates a gauge."""
        with patch("qubx.emitters.prometheus.Gauge", mock_gauge):
            with patch("qubx.emitters.prometheus.push_to_gateway", mock_push_gateway):
                emitter.emit("test", 1.0, {"tag": "value"}, metric_type="gauge")
                mock_gauge.assert_called_once()
                mock_gauge.labels.assert_called_once_with(strategy="test", tag="value")
                mock_gauge.labels().set.assert_called_once_with(1.0)
                mock_push_gateway.assert_called_once()

    def test_emit_counter_metric(self, emitter, mock_registry, mock_counter, mock_push_gateway):
        """Test that emit with counter metric type creates and updates a counter."""
        with patch("qubx.emitters.prometheus.Counter", mock_counter):
            with patch("qubx.emitters.prometheus.push_to_gateway", mock_push_gateway):
                emitter.emit("test", 1.0, {"tag": "value"}, metric_type="counter")
                mock_counter.assert_called_once()
                mock_counter.labels.assert_called_once_with(strategy="test", tag="value")
                mock_counter.labels().inc.assert_called_once_with(1.0)
                mock_push_gateway.assert_called_once()

    def test_emit_summary_metric(self, emitter, mock_registry, mock_summary, mock_push_gateway):
        """Test that emit with summary metric type creates and updates a summary."""
        with patch("qubx.emitters.prometheus.Summary", mock_summary):
            with patch("qubx.emitters.prometheus.push_to_gateway", mock_push_gateway):
                emitter.emit("test", 1.0, {"tag": "value"}, metric_type="summary")
                mock_summary.assert_called_once()
                mock_summary.labels.assert_called_once_with(strategy="test", tag="value")
                mock_summary.labels().observe.assert_called_once_with(1.0)
                mock_push_gateway.assert_called_once()

    def test_emit_without_pushgateway(self, mock_registry, mock_gauge, mock_push_gateway):
        """Test that emit works without a pushgateway."""
        with patch("qubx.emitters.prometheus.REGISTRY", mock_registry):
            with patch("qubx.emitters.prometheus.Gauge", mock_gauge):
                with patch("qubx.emitters.prometheus.push_to_gateway", mock_push_gateway):
                    from qubx.emitters.prometheus import PrometheusMetricEmitter

                    emitter = PrometheusMetricEmitter(strategy_name="test")  # No pushgateway URL
                    emitter.emit("test", 1.0)
                    mock_gauge.assert_called_once()
                    mock_gauge().set.assert_called_once_with(1.0)
                    mock_push_gateway.assert_not_called()


class TestQuestDBMetricEmitter:
    """Test the QuestDBMetricEmitter class."""

    @pytest.fixture
    def mock_sender(self):
        """Create a mock QuestDB sender."""
        sender = MagicMock()
        sender.from_conf.return_value = sender
        sender.table.return_value = sender
        sender.symbol.return_value = sender
        sender.double_column.return_value = sender
        return sender

    @pytest.fixture
    def emitter(self, mock_sender):
        """Create a QuestDBMetricEmitter with mocks."""
        with patch("qubx.emitters.questdb.Sender", mock_sender):
            from qubx.emitters.questdb import QuestDBMetricEmitter

            return QuestDBMetricEmitter(strategy_name="test")

    def test_init(self, mock_sender):
        """Test that the emitter initializes correctly."""
        with patch("qubx.emitters.questdb.Sender", mock_sender):
            from qubx.emitters.questdb import QuestDBMetricEmitter

            emitter = QuestDBMetricEmitter(strategy_name="test", host="testhost", port=9999, table_name="test_table")
            mock_sender.from_conf.assert_called_once_with("http::addr=testhost:9999;")
            assert emitter._table_name == "test_table"
            assert emitter._default_tags["strategy"] == "test"

    def test_emit(self, emitter, mock_sender):
        """Test that emit sends data to QuestDB."""
        timestamp = pd.Timestamp("2023-01-01").to_numpy()

        # Mock the _convert_timestamp method to return a datetime
        with patch.object(emitter, "_convert_timestamp", return_value=datetime.datetime(2023, 1, 1)):
            emitter.emit("test_metric", 42.0, {"tag1": "value1"}, timestamp)

            # Check that the table was created
            mock_sender.table.assert_called_once_with("qubx_metrics")

            # Check that the metric name was added
            mock_sender.symbol.assert_any_call("metric_name", "test_metric")

            # Check that the tags were added
            mock_sender.symbol.assert_any_call("tag1", "value1")
            mock_sender.symbol.assert_any_call("strategy", "test")

            # Check that the value was added
            mock_sender.double_column.assert_called_once_with("value", 42.0)

            # Check that the timestamp was used
            mock_sender.at.assert_called_once_with(datetime.datetime(2023, 1, 1))

    def test_emit_without_timestamp(self, emitter, mock_sender):
        """Test that emit works without a timestamp."""
        # Mock datetime.now to return a fixed datetime
        mock_now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now

            emitter.emit("test_metric", 42.0, {"tag1": "value1"})

            # Check that the table was created
            mock_sender.table.assert_called_once_with("qubx_metrics")

            # Check that the metric name was added
            mock_sender.symbol.assert_any_call("metric_name", "test_metric")

            # Check that the tags were added
            mock_sender.symbol.assert_any_call("tag1", "value1")
            mock_sender.symbol.assert_any_call("strategy", "test")

            # Check that the value was added
            mock_sender.double_column.assert_called_once_with("value", 42.0)

            # Check that a timestamp was used (current time)
            mock_sender.at.assert_called_once_with(mock_now)

    def test_emit_with_connection_error(self, mock_sender):
        """Test that emit handles connection errors gracefully."""
        # Make the sender raise an exception
        mock_sender.from_conf.side_effect = Exception("Connection error")

        with patch("qubx.emitters.questdb.Sender", mock_sender):
            from qubx.emitters.questdb import QuestDBMetricEmitter

            emitter = QuestDBMetricEmitter(strategy_name="test")

            # This should not raise an exception
            emitter.emit("test_metric", 42.0)

            # The sender should be None
            assert emitter._sender is None

    def test_emit_with_send_error(self, emitter, mock_sender):
        """Test that emit handles send errors gracefully."""
        # Make the table method raise an exception
        mock_sender.table.side_effect = Exception("Send error")

        # This should not raise an exception
        emitter.emit("test_metric", 42.0)

        # The table method should have been called
        mock_sender.table.assert_called_once()

    def test_close_connection(self, emitter, mock_sender):
        """Test that the connection is closed when the emitter is destroyed."""
        # Call __del__ manually
        emitter.__del__()

        # Check that close was called
        mock_sender.close.assert_called_once()
