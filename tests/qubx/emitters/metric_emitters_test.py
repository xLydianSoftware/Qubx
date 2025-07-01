"""
Unit tests for metric emitters.
"""

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qubx.core.basics import Signal, dt_64
from qubx.core.interfaces import IAccountViewer, IMetricEmitter, IStrategyContext
from qubx.emitters.base import BaseMetricEmitter
from qubx.emitters.composite import CompositeMetricEmitter


@pytest.fixture
def mock_account():
    """Create a mock account viewer."""
    mock = MagicMock(spec=IAccountViewer)
    mock.get_total_capital.return_value = 10000.0
    mock.get_net_leverage.return_value = 0.5
    mock.get_gross_leverage.return_value = 0.7
    return mock


@pytest.fixture
def mock_signals():
    """Create mock signals for testing."""
    # Use mock instruments instead of full Instrument objects
    instrument1 = MagicMock()
    instrument1.symbol = "BTCUSDT"
    instrument1.exchange = "binance"
    instrument1.__str__ = lambda: "binance:SPOT:BTCUSDT"

    instrument2 = MagicMock()
    instrument2.symbol = "ETHUSDT"
    instrument2.exchange = "binance"
    instrument2.__str__ = lambda: "binance:SPOT:ETHUSDT"

    signal1 = Signal(
        time=dt_64(pd.Timestamp("2023-01-01 12:00:00")),
        instrument=instrument1,
        signal=1.0,
        price=50000.0,
        stop=49000.0,
        take=52000.0,
        reference_price=50000.0,
        group="test_group",
        comment="Test signal 1",
        options={"test": "option"},
        is_service=False,
    )

    signal2 = Signal(
        time=dt_64(pd.Timestamp("2023-01-01 12:00:01")),
        instrument=instrument2,
        signal=-1.0,
        price=2000.0,
        stop=2100.0,
        take=1900.0,
        reference_price=2000.0,
        group="test_group",
        comment="Test signal 2",
        options={"test": "option2"},
        is_service=True,
    )

    return [signal1, signal2]


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
        mock.is_simulation = False
        mock.time.return_value = pd.Timestamp("2023-01-01 00:00:00").to_numpy()

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

        # Set up positions
        mock.get_positions.return_value = {
            instrument1: position1,
            instrument2: position2,
        }

        # Set up quotes
        quote1 = MagicMock()
        quote1.mid_price.return_value = 55000.0

        quote2 = MagicMock()
        quote2.mid_price.return_value = 2100.0

        mock.quote.side_effect = lambda instr: quote1 if str(instr) == "BTC-USD" else quote2

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

        # Check that metrics were emitted
        assert len(emitter.emitted_metrics) > 0

        # Check for specific metrics
        metric_names = [m[0] for m in emitter.emitted_metrics]
        assert "total_capital" in metric_names
        assert "net_leverage" in metric_names
        assert "gross_leverage" in metric_names

    def test_notify_first_call(self, emitter, mock_context):
        """Test that the first call to notify initializes the last emission time."""
        # Call notify
        emitter.notify(mock_context)

        # Check that the last emission time was set
        assert emitter._last_emission_time is not None
        assert emitter._last_emission_time == pd.Timestamp(mock_context.time())

        # No metrics should have been emitted
        assert len(emitter.emitted_metrics) == 0

    def test_notify_before_interval(self, emitter, mock_context):
        """Test that notify doesn't emit metrics before the interval has passed."""
        # Set the last emission time
        emitter._last_emission_time = pd.Timestamp("2023-01-01 00:00:00")

        # Set the context time to be before the interval has passed
        mock_context.time.return_value = pd.Timestamp("2023-01-01 00:00:30").to_numpy()  # 30 seconds later

        # Call notify
        emitter.notify(mock_context)

        # No metrics should have been emitted
        assert len(emitter.emitted_metrics) == 0

    def test_notify_after_interval(self, emitter, mock_context):
        """Test that notify emits metrics after the interval has passed."""
        # Set the last emission time
        emitter._last_emission_time = pd.Timestamp("2023-01-01 00:00:00")

        # Set the context time to be after the interval has passed
        mock_context.time.return_value = pd.Timestamp("2023-01-01 00:01:30").to_numpy()  # 1 minute and 30 seconds later

        # Call notify
        emitter.notify(mock_context)

        # Metrics should have been emitted
        assert len(emitter.emitted_metrics) > 0

    def test_emit_signals(self, emitter, mock_signals, mock_account):
        """Test that emit_signals works correctly."""
        time = dt_64(pd.Timestamp("2023-01-01 12:00:00"))

        # Should not raise any exception and should do nothing (base implementation)
        emitter.emit_signals(time, mock_signals, mock_account)

    def test_emit_signals_empty_list(self, emitter, mock_account):
        """Test that emit_signals handles empty signal list."""
        time = dt_64(pd.Timestamp("2023-01-01 12:00:00"))

        # Should not raise any exception
        emitter.emit_signals(time, [], mock_account)

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
        metric_names = [m[0] for m in emitter.emitted_metrics]
        assert "total_capital" in metric_names
        assert "net_leverage" in metric_names
        assert "gross_leverage" not in metric_names

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

        # Set the last emission time
        emitter._last_emission_time = pd.Timestamp("2023-01-01 00:00:00")

        # Set the context time to be before the custom interval has passed
        mock_context.time.return_value = pd.Timestamp("2023-01-01 00:01:30").to_numpy()  # 1 minute and 30 seconds later

        # Call notify
        emitter.notify(mock_context)

        # No metrics should have been emitted (interval is 2m)
        assert len(emitter.emitted_metrics) == 0

        # Set the context time to be after the custom interval has passed
        mock_context.time.return_value = pd.Timestamp(
            "2023-01-01 00:02:30"
        ).to_numpy()  # 2 minutes and 30 seconds later

        # Call notify
        emitter.notify(mock_context)

        # Metrics should have been emitted
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

    def test_emit_signals(self, composite, emitters, mock_signals, mock_account):
        """Test that emit_signals calls all child emitters."""
        time = dt_64(pd.Timestamp("2023-01-01 12:00:00"))

        composite.emit_signals(time, mock_signals, mock_account)

        # Check that all emitters were called
        for emitter in emitters:
            emitter.emit_signals.assert_called_once_with(time, mock_signals, mock_account)

    def test_emit_signals_with_exception(self, composite, emitters, mock_signals, mock_account):
        """Test that emit_signals handles exceptions from child emitters."""
        time = dt_64(pd.Timestamp("2023-01-01 12:00:00"))

        # Make one emitter raise an exception
        emitters[0].emit_signals.side_effect = Exception("Test exception")

        # Should not raise an exception
        composite.emit_signals(time, mock_signals, mock_account)

        # Check that all emitters were still called
        for emitter in emitters:
            emitter.emit_signals.assert_called_once_with(time, mock_signals, mock_account)


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

    def test_emit_signals(self, emitter, mock_signals, mock_account, mock_gauge, mock_push_gateway):
        """Test that emit_signals creates Prometheus metrics for signals."""
        time = dt_64(pd.Timestamp("2023-01-01 12:00:00"))

        with patch("qubx.emitters.prometheus.Gauge", mock_gauge):
            with patch("qubx.emitters.prometheus.push_to_gateway", mock_push_gateway):
                emitter.emit_signals(time, mock_signals, mock_account)

                # Should create gauges for each signal
                assert mock_gauge.call_count >= len(mock_signals)

                # Should push to gateway if configured
                if emitter._pushgateway_url:
                    mock_push_gateway.assert_called()

    def test_emit_signals_with_error(self, emitter, mock_signals, mock_account, mock_gauge):
        """Test that emit_signals handles errors gracefully."""
        time = dt_64(pd.Timestamp("2023-01-01 12:00:00"))

        # Make gauge creation fail
        mock_gauge.side_effect = Exception("Prometheus error")

        with patch("qubx.emitters.prometheus.Gauge", mock_gauge):
            # Should not raise an exception
            emitter.emit_signals(time, mock_signals, mock_account)


class TestQuestDBMetricEmitter:
    """Test the QuestDBMetricEmitter class."""

    @pytest.fixture
    def mock_sender(self):
        """Create a mock QuestDB sender."""
        sender = MagicMock()
        sender.from_conf.return_value = sender
        sender.establish.return_value = None
        sender.row.return_value = None
        sender.flush.return_value = None
        sender.close.return_value = None
        return sender

    @pytest.fixture
    def emitter(self, mock_sender):
        """Create a QuestDBMetricEmitter with mocks."""
        with patch("qubx.emitters.questdb.Sender", mock_sender):
            from qubx.emitters.questdb import QuestDBMetricEmitter

            return QuestDBMetricEmitter(tags={"strategy": "test"})

    def test_init(self, mock_sender):
        """Test that the emitter initializes correctly."""
        with patch("qubx.emitters.questdb.Sender", mock_sender):
            from qubx.emitters.questdb import QuestDBMetricEmitter

            emitter = QuestDBMetricEmitter(
                host="testhost", port=9999, table_name="test_table", tags={"strategy": "test"}
            )
            mock_sender.from_conf.assert_called_once_with("http::addr=testhost:9999;")
            mock_sender.establish.assert_called_once()
            assert emitter._table_name == "test_table"
            assert emitter._default_tags["strategy"] == "test"

    def test_emit(self, emitter, mock_sender):
        """Test that emit sends data to QuestDB."""
        timestamp = pd.Timestamp("2023-01-01").to_numpy()

        # Mock the _convert_timestamp method to return a datetime
        dt_timestamp = datetime.datetime(2023, 1, 1)
        with patch.object(emitter, "_convert_timestamp", return_value=dt_timestamp):
            emitter.emit("test_metric", 42.0, {"tag1": "value1"}, timestamp)

            # Check that row was called with the correct arguments
            expected_symbols = {"metric_name": "test_metric", "tag1": "value1", "strategy": "test"}
            expected_columns = {"value": 42.0}
            mock_sender.row.assert_called_once_with(
                "qubx_metrics", symbols=expected_symbols, columns=expected_columns, at=dt_timestamp
            )

    def test_emit_without_timestamp(self, emitter, mock_sender):
        """Test that emit works without a timestamp."""
        # Mock datetime.now to return a fixed datetime
        mock_now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now

            emitter.emit("test_metric", 42.0, {"tag1": "value1"})

            # Check that row was called with the correct arguments
            expected_symbols = {"metric_name": "test_metric", "tag1": "value1", "strategy": "test"}
            expected_columns = {"value": 42.0}
            mock_sender.row.assert_called_once_with(
                "qubx_metrics", symbols=expected_symbols, columns=expected_columns, at=mock_now
            )

    def test_emit_with_connection_error(self, mock_sender):
        """Test that emit handles connection errors gracefully."""
        # Make the sender raise an exception
        mock_sender.from_conf.side_effect = Exception("Connection error")

        with patch("qubx.emitters.questdb.Sender", mock_sender):
            from qubx.emitters.questdb import QuestDBMetricEmitter

            emitter = QuestDBMetricEmitter(tags={"strategy": "test"})

            # This should not raise an exception
            emitter.emit("test_metric", 42.0)

            # The sender should be None
            assert emitter._sender is None

    def test_emit_with_send_error(self, emitter, mock_sender):
        """Test that emit handles send errors gracefully."""
        # Make the row method raise an exception
        mock_sender.row.side_effect = Exception("Send error")

        # This should not raise an exception
        emitter.emit("test_metric", 42.0)

        # The row method should have been called
        mock_sender.row.assert_called_once()

    def test_close_connection(self, emitter, mock_sender):
        """Test that the connection is closed when the emitter is destroyed."""
        # Call __del__ manually
        emitter.__del__()

        # Check that close was called
        mock_sender.close.assert_called_once()

    def test_notify_flush(self, emitter, mock_sender):
        """Test that notify flushes the sender after the flush interval."""
        # Create a mock context
        mock_context = MagicMock(spec=IStrategyContext)
        mock_context.is_simulation = False
        mock_context.time.return_value = pd.Timestamp("2023-01-01 12:00:00").to_numpy()

        # First call to notify should set _last_flush
        emitter.notify(mock_context)
        mock_sender.flush.assert_not_called()

        # Set _last_flush manually to a known value
        emitter._last_flush = pd.Timestamp("2023-01-01 12:00:00")

        # Mock the current time to be after the flush interval
        current_time = pd.Timestamp("2023-01-01 12:00:10")  # 10 seconds later

        # Create a timedelta that's greater than the flush interval
        # The default flush interval is "5s" (5 seconds)
        with patch("pandas.Timestamp.now", return_value=current_time):
            # Second call to notify should flush because current_time - _last_flush >= flush_interval
            emitter.notify(mock_context)
            mock_sender.flush.assert_called_once()

    def test_emit_signals(self, emitter, mock_sender, mock_signals, mock_account):
        """Test that emit_signals sends signals to QuestDB."""
        time = dt_64(pd.Timestamp("2023-01-01 12:00:00"))

        # Mock the necessary methods
        with patch.object(emitter, "_convert_timestamp", return_value=datetime.datetime(2023, 1, 1, 12, 0, 0)):
            with patch.object(emitter._executor, "submit") as mock_submit:
                emitter.emit_signals(time, mock_signals, mock_account)

                # Should submit a single background task that handles all signals
                assert mock_submit.call_count == 1
                # Verify the task was called with the correct arguments
                mock_submit.assert_called_once_with(emitter._emit_signals_to_questdb, time, mock_signals)

    def test_emit_signals_with_connection_error(self, mock_sender, mock_signals, mock_account):
        """Test that emit_signals handles connection errors gracefully."""
        time = dt_64(pd.Timestamp("2023-01-01 12:00:00"))

        # Create an emitter with no connection
        mock_sender.from_conf.side_effect = Exception("Connection error")

        with patch("qubx.emitters.questdb.Sender", mock_sender):
            from qubx.emitters.questdb import QuestDBMetricEmitter

            emitter = QuestDBMetricEmitter(tags={"strategy": "test"})

            # This should not raise an exception
            emitter.emit_signals(time, mock_signals, mock_account)

    def test_emit_signals_empty_list(self, emitter, mock_sender, mock_account):
        """Test that emit_signals handles empty signal list."""
        time = dt_64(pd.Timestamp("2023-01-01 12:00:00"))

        # Should not raise an exception
        emitter.emit_signals(time, [], mock_account)

        # Should not submit any tasks
        with patch.object(emitter._executor, "submit") as mock_submit:
            emitter.emit_signals(time, [], mock_account)
            mock_submit.assert_not_called()
