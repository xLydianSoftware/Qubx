from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from qubx.backtester.simulated_data import DataFetcher, IterableSimulationData
from qubx.core.basics import AssetType, DataType, FundingPayment, Instrument, MarketEvent, MarketType, dt_64
from qubx.core.mixins.processing import ProcessingManager
from qubx.data.readers import AsFundingPayments, DataReader


class TestFundingPaymentSubscription:
    """Test suite for funding payment subscription and event processing."""

    @pytest.fixture
    def sample_funding_payment(self):
        """Sample funding payment object for testing."""
        timestamp = pd.Timestamp("2025-01-08 00:00:00").asm8
        return FundingPayment(time=timestamp, symbol="BTCUSDT", funding_rate=0.0001, funding_interval_hours=8)

    @pytest.fixture
    def mock_instrument(self):
        """Mock instrument for testing."""
        return Instrument(
            symbol="BTCUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            exchange="binance",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTCUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

    @pytest.fixture
    def sample_raw_funding_data(self):
        """Sample raw funding payment data as it would come from QuestDB."""
        return [
            ["2025-01-08 00:00:00", "BTCUSDT", 0.0001, 8],
            ["2025-01-08 08:00:00", "BTCUSDT", -0.0002, 8],
            ["2025-01-08 16:00:00", "BTCUSDT", 0.0003, 8],
        ]

    def test_funding_payment_transformer_basic(self, sample_raw_funding_data):
        """Test AsFundingPayments transformer basic functionality."""
        transformer = AsFundingPayments()

        # Start transformation
        column_names = ["timestamp", "symbol", "funding_rate", "funding_interval_hours"]
        transformer.start_transform("test", column_names)

        # Process data
        transformer.process_data(sample_raw_funding_data)

        # Collect results
        funding_payments = transformer.collect()

        assert len(funding_payments) == 3
        assert all(isinstance(fp, FundingPayment) for fp in funding_payments)

        # Check first funding payment
        fp = funding_payments[0]
        assert fp.symbol == "BTCUSDT"
        assert fp.funding_rate == 0.0001
        assert fp.funding_interval_hours == 8

    def test_funding_payment_transformer_column_mapping(self):
        """Test transformer handles different column arrangements."""
        transformer = AsFundingPayments()

        # Test with different column order
        column_names = ["funding_rate", "timestamp", "funding_interval_hours", "symbol"]
        raw_data = [[0.0001, "2025-01-08 00:00:00", 8, "BTCUSDT"]]

        transformer.start_transform("test", column_names)
        transformer.process_data(raw_data)

        funding_payments = transformer.collect()
        assert len(funding_payments) == 1

        fp = funding_payments[0]
        assert fp.symbol == "BTCUSDT"
        assert fp.funding_rate == 0.0001
        assert fp.funding_interval_hours == 8

    def test_data_fetcher_funding_payment_case(self):
        """Test DataFetcher correctly handles FUNDING_PAYMENT subscription type."""
        mock_reader = Mock(spec=DataReader)

        fetcher = DataFetcher(
            fetcher_id="test_funding", reader=mock_reader, subtype=DataType.FUNDING_PAYMENT, params={}
        )

        # Verify fetcher configuration
        assert fetcher._requested_data_type == "funding_payment"
        assert fetcher._producing_data_type == "funding_payment"
        assert isinstance(fetcher._transformer, AsFundingPayments)

    def test_data_fetcher_funding_payment_with_params(self):
        """Test DataFetcher with funding payment parameters."""
        mock_reader = Mock(spec=DataReader)
        params = {"exchange": "binance", "interval": "8h"}

        fetcher = DataFetcher(
            fetcher_id="test_funding_params", reader=mock_reader, subtype=DataType.FUNDING_PAYMENT, params=params
        )

        assert fetcher._params == params
        assert fetcher._requested_data_type == "funding_payment"

    def test_processing_manager_handle_funding_payment(self, mock_instrument, sample_funding_payment):
        """Test ProcessingManager _handle_funding_payment method."""
        # Create a mock processing manager instance
        processor = Mock()
        processor._time_provider = Mock()
        processor._time_provider.time.return_value = pd.Timestamp("2025-01-08 00:00:00").asm8

        # Mock the private method
        processor._ProcessingManager__update_base_data = Mock(return_value=True)

        # Manually call the method since we can't easily instantiate ProcessingManager
        result = ProcessingManager._handle_funding_payment(
            processor, instrument=mock_instrument, event_type="funding_payment", funding_payment=sample_funding_payment
        )

        # Verify result
        assert isinstance(result, MarketEvent)
        assert result.type == "funding_payment"
        assert result.instrument == mock_instrument
        assert result.data == sample_funding_payment
        assert result.is_trigger is True  # Based on mock return value

        # Verify base data update was called
        processor._ProcessingManager__update_base_data.assert_called_once_with(
            mock_instrument, "funding_payment", sample_funding_payment
        )

    def test_funding_payment_handler_registration(self):
        """Test that _handle_funding_payment method exists and follows naming convention."""
        # Test that the handler method exists
        assert hasattr(ProcessingManager, "_handle_funding_payment")

        # Test the naming convention for handler discovery
        method_name = "_handle_funding_payment"
        handler_key = method_name.split("_handle_")[1]
        assert handler_key == "funding_payment"

        # Verify method signature
        import inspect

        sig = inspect.signature(ProcessingManager._handle_funding_payment)
        param_names = list(sig.parameters.keys())
        assert "self" in param_names
        assert "instrument" in param_names
        assert "event_type" in param_names
        assert "funding_payment" in param_names

    def test_simulated_data_funding_payment_subscription(self):
        """Test IterableSimulationData supports funding payment subscriptions."""
        mock_readers = {"funding_payment": Mock(spec=DataReader)}

        sim_data = IterableSimulationData(readers=mock_readers)

        # Test subscription parsing
        access_key, data_type, params = sim_data._parse_subscription_spec("funding_payment")

        assert access_key == "funding_payment"
        assert data_type == "funding_payment"
        assert params == {}

    def test_simulated_data_funding_payment_fetcher_creation(self):
        """Test IterableSimulationData creates funding payment fetcher correctly."""
        mock_reader = Mock(spec=DataReader)
        mock_readers = {"funding_payment": mock_reader}

        sim_data = IterableSimulationData(readers=mock_readers)
        mock_instrument = Instrument(
            symbol="BTCUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            exchange="binance",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTCUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

        # Add instruments for funding payment subscription
        sim_data.add_instruments_for_subscription("funding_payment", [mock_instrument])

        # Verify fetcher was created
        assert "funding_payment" in sim_data._subtyped_fetchers
        fetcher = sim_data._subtyped_fetchers["funding_payment"]

        assert isinstance(fetcher, DataFetcher)
        assert fetcher._requested_data_type == "funding_payment"
        assert fetcher._producing_data_type == "funding_payment"
        assert isinstance(fetcher._transformer, AsFundingPayments)

    def test_funding_payment_multiple_instruments(self):
        """Test funding payment subscription works with multiple instruments."""
        mock_reader = Mock(spec=DataReader)
        mock_readers = {"funding_payment": mock_reader}

        sim_data = IterableSimulationData(readers=mock_readers)
        instruments = [
            Instrument(
                "BTCUSDT",
                AssetType.CRYPTO,
                MarketType.SWAP,
                "binance",
                "BTC",
                "USDT",
                "USDT",
                "BTCUSDT",
                0.01,
                0.001,
                0.001,
            ),
            Instrument(
                "ETHUSDT",
                AssetType.CRYPTO,
                MarketType.SWAP,
                "binance",
                "ETH",
                "USDT",
                "USDT",
                "ETHUSDT",
                0.01,
                0.001,
                0.001,
            ),
            Instrument(
                "ADAUSDT",
                AssetType.CRYPTO,
                MarketType.SWAP,
                "binance",
                "ADA",
                "USDT",
                "USDT",
                "ADAUSDT",
                0.01,
                0.001,
                0.001,
            ),
        ]

        # Add multiple instruments
        sim_data.add_instruments_for_subscription("funding_payment", instruments)

        # Verify all instruments are subscribed
        subscribed_instruments = sim_data.get_instruments_for_subscription("funding_payment")
        assert len(subscribed_instruments) == 3
        assert set(i.symbol for i in subscribed_instruments) == {"BTCUSDT", "ETHUSDT", "ADAUSDT"}

    def test_funding_payment_subscription_removal(self):
        """Test funding payment subscription removal."""
        mock_reader = Mock(spec=DataReader)
        mock_readers = {"funding_payment": mock_reader}

        sim_data = IterableSimulationData(readers=mock_readers)
        instrument = Instrument(
            "BTCUSDT",
            AssetType.CRYPTO,
            MarketType.SWAP,
            "binance",
            "BTC",
            "USDT",
            "USDT",
            "BTCUSDT",
            0.01,
            0.001,
            0.001,
        )

        # Add and then remove subscription
        sim_data.add_instruments_for_subscription("funding_payment", [instrument])
        assert sim_data.has_subscription(instrument, "funding_payment")

        sim_data.remove_instruments_from_subscription("funding_payment", [instrument])
        assert not sim_data.has_subscription(instrument, "funding_payment")

    def test_funding_payment_error_handling(self):
        """Test funding payment subscription error handling."""
        # Test with missing reader
        sim_data = IterableSimulationData(readers={})
        instrument = Instrument(
            "BTCUSDT",
            AssetType.CRYPTO,
            MarketType.SWAP,
            "binance",
            "BTC",
            "USDT",
            "USDT",
            "BTCUSDT",
            0.01,
            0.001,
            0.001,
        )

        with pytest.raises(Exception):  # Should raise SimulationError due to missing reader
            sim_data.add_instruments_for_subscription("funding_payment", [instrument])

    def test_funding_payment_transformer_empty_data(self):
        """Test AsFundingPayments transformer handles empty data correctly."""
        transformer = AsFundingPayments()
        column_names = ["timestamp", "symbol", "funding_rate", "funding_interval_hours"]

        transformer.start_transform("test", column_names)
        transformer.process_data(None)  # No data
        transformer.process_data([])  # Empty data

        result = transformer.collect()
        assert result == []

    def test_funding_payment_transformer_buffer_accumulation(self, sample_raw_funding_data):
        """Test transformer correctly accumulates data across multiple process_data calls."""
        transformer = AsFundingPayments()
        column_names = ["timestamp", "symbol", "funding_rate", "funding_interval_hours"]

        transformer.start_transform("test", column_names)

        # Process data in chunks
        transformer.process_data(sample_raw_funding_data[:1])
        transformer.process_data(sample_raw_funding_data[1:])

        result = transformer.collect()
        assert len(result) == 3
        assert all(isinstance(fp, FundingPayment) for fp in result)
