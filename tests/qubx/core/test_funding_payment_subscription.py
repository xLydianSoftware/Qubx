import inspect
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from qubx.backtester.simulated_data import DataPump, SimulatedDataIterator
from qubx.core.basics import DataType, FundingPayment, Instrument, MarketEvent, MarketType
from qubx.core.mixins.processing import ProcessingManager
from qubx.data.containers import RawData
from qubx.data.storage import IReader, IStorage
from qubx.data.transformers import TypedRecords

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_funding_batch(
    timestamps: list[str],
    funding_rates: list[float],
    intervals: list[float],
    column_order: list[str] | None = None,
) -> pa.RecordBatch:
    """
    Build a PyArrow RecordBatch containing funding payment data.
    column_order lets tests exercise arbitrary column ordering.
    """
    columns = {
        "time": pa.array(pd.to_datetime(timestamps).asi8, type=pa.int64()),
        "funding_rate": pa.array(funding_rates, type=pa.float64()),
        "funding_interval_hours": pa.array(intervals, type=pa.float64()),
    }
    if column_order:
        columns = {k: columns[k] for k in column_order}
    return pa.RecordBatch.from_pydict(columns)


def _make_storage(reader: IReader | None = None) -> IStorage:
    """
    Create a mock IStorage that returns given reader from get_reader().
    If reader is None a fresh Mock IReader is used.
    """
    mock_storage = Mock(spec=IStorage)
    mock_storage.get_reader.return_value = reader or Mock(spec=IReader)
    return mock_storage


def _make_swap_instrument(symbol: str = "BTCUSDT", base: str = "BTC") -> Instrument:
    """Create a SWAP Instrument with minimal required fields."""
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange="binance",
        base=base,
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


class TestFundingPaymentSubscription:
    """Test suite for funding payment subscription and event processing."""

    @pytest.fixture
    def sample_funding_payment(self):
        """Sample funding payment object for testing."""
        timestamp = pd.Timestamp("2025-01-08 00:00:00").asm8
        return FundingPayment(time=timestamp, funding_rate=0.0001, funding_interval_hours=8)

    @pytest.fixture
    def mock_instrument(self):
        """Mock instrument for testing."""
        return _make_swap_instrument("BTCUSDT", "BTC")

    # -----------------------------------------------------------------------
    # TypedRecords transformer tests (FUNDING_PAYMENT path)
    # -----------------------------------------------------------------------

    def test_funding_payment_transformer_basic(self):
        """TypedRecords converts a batch of funding payment rows to FundingPayment objects."""
        timestamps = ["2025-01-08 00:00:00", "2025-01-08 08:00:00", "2025-01-08 16:00:00"]
        batch = _make_funding_batch(timestamps, [0.0001, -0.0002, 0.0003], [8.0, 8.0, 8.0])
        raw = RawData.from_record_batch("BTCUSDT", DataType.FUNDING_PAYMENT, batch)

        result = TypedRecords().process_data(raw)

        assert len(result) == 3
        assert all(isinstance(fp, FundingPayment) for fp in result)
        assert result[0].funding_rate == pytest.approx(0.0001)
        assert result[1].funding_rate == pytest.approx(-0.0002)
        assert result[0].funding_interval_hours == pytest.approx(8.0)

    def test_funding_payment_transformer_column_ordering(self):
        """TypedRecords uses column-name lookup so column order in the batch does not matter."""
        # - columns in non-standard order: funding_rate first, then time, then interval
        batch = _make_funding_batch(
            ["2025-01-08 00:00:00"],
            [0.0001],
            [8.0],
            column_order=["funding_rate", "time", "funding_interval_hours"],
        )
        raw = RawData.from_record_batch("BTCUSDT", DataType.FUNDING_PAYMENT, batch)

        result = TypedRecords().process_data(raw)

        assert len(result) == 1
        assert isinstance(result[0], FundingPayment)
        assert result[0].funding_rate == pytest.approx(0.0001)
        assert result[0].funding_interval_hours == pytest.approx(8.0)

    def test_funding_payment_transformer_empty_batch(self):
        """TypedRecords returns an empty list for a zero-row batch."""
        empty_batch = pa.RecordBatch.from_pydict(
            {
                "time": pa.array([], type=pa.int64()),
                "funding_rate": pa.array([], type=pa.float64()),
                "funding_interval_hours": pa.array([], type=pa.float64()),
            }
        )
        raw = RawData.from_record_batch("BTCUSDT", DataType.FUNDING_PAYMENT, empty_batch)

        result = TypedRecords().process_data(raw)

        assert result == []

    def test_funding_payment_transformer_field_values(self):
        """TypedRecords correctly extracts all FundingPayment fields including timestamps."""
        ts = pd.Timestamp("2025-06-15 08:00:00")
        batch = _make_funding_batch([ts.isoformat()], [-0.00015], [4.0])
        raw = RawData.from_record_batch("ETHUSDT", DataType.FUNDING_PAYMENT, batch)

        result = TypedRecords().process_data(raw)

        assert len(result) == 1
        fp = result[0]
        assert isinstance(fp, FundingPayment)
        assert fp.funding_rate == pytest.approx(-0.00015)
        assert fp.funding_interval_hours == pytest.approx(4.0)
        # - time must round-trip to the expected nanosecond value
        assert fp.time == ts.value

    # -----------------------------------------------------------------------
    # DataPump tests
    # -----------------------------------------------------------------------

    def test_data_pump_funding_payment_case(self):
        """DataPump correctly handles FUNDING_PAYMENT subscription type."""
        mock_reader = Mock(spec=IReader)

        pump = DataPump(mock_reader, "funding_payment", "binance", "SWAP")

        # - funding_payment falls into the generic 'case _' branch → TypedRecords transformer
        assert pump._requested_data_type == "funding_payment"
        assert pump._producing_data_type == "funding_payment"
        assert isinstance(pump._transformer, TypedRecords)

    def test_data_pump_funding_payment_different_exchanges(self):
        """DataPump created for different exchanges keeps correct data types independently."""
        mock_reader = Mock(spec=IReader)

        pump_binance = DataPump(mock_reader, "funding_payment", "binance", "SWAP")
        pump_okx = DataPump(mock_reader, "funding_payment", "okx", "SWAP")

        assert pump_binance._requested_data_type == "funding_payment"
        assert pump_okx._requested_data_type == "funding_payment"
        assert isinstance(pump_binance._transformer, TypedRecords)
        assert isinstance(pump_okx._transformer, TypedRecords)

    # -----------------------------------------------------------------------
    # ProcessingManager tests
    # -----------------------------------------------------------------------

    def test_processing_manager_handle_funding_payment(self, mock_instrument, sample_funding_payment):
        """ProcessingManager._handle_funding_payment builds correct MarketEvent."""
        processor = Mock()
        processor._time_provider = Mock()
        processor._time_provider.time.return_value = pd.Timestamp("2025-01-08 00:00:00").asm8
        processor._account = Mock()

        # - mock the mangled private helper that determines trigger status
        processor._ProcessingManager__update_base_data = Mock(return_value=True)

        result = ProcessingManager._handle_funding_payment(
            processor, instrument=mock_instrument, event_type="funding_payment", funding_payment=sample_funding_payment
        )

        # - verify result shape
        assert isinstance(result, MarketEvent)
        assert result.type == "funding_payment"
        assert result.instrument == mock_instrument
        assert result.data == sample_funding_payment
        assert result.is_trigger is True

        # - account must receive the funding payment before the market event is built
        processor._account.process_funding_payment.assert_called_once_with(mock_instrument, sample_funding_payment)

        # - base-data update must be called with the right args
        processor._ProcessingManager__update_base_data.assert_called_once_with(
            mock_instrument, "funding_payment", sample_funding_payment
        )

    def test_funding_payment_handler_registration(self):
        """_handle_funding_payment exists on ProcessingManager and has the expected signature."""
        # - direct access raises AttributeError if missing (no hasattr)
        handler = ProcessingManager._handle_funding_payment
        assert callable(handler)

        # - naming convention: _handle_<event_type>
        method_name = "_handle_funding_payment"
        handler_key = method_name.split("_handle_")[1]
        assert handler_key == "funding_payment"

        # - verify method signature
        sig = inspect.signature(ProcessingManager._handle_funding_payment)
        param_names = list(sig.parameters.keys())
        assert "self" in param_names
        assert "instrument" in param_names
        assert "event_type" in param_names
        assert "funding_payment" in param_names

    # -----------------------------------------------------------------------
    # SimulatedDataIterator tests
    # -----------------------------------------------------------------------

    def test_simulated_data_funding_payment_subscription(self):
        """SimulatedDataIterator._parse_subscription_spec correctly parses funding_payment."""
        sim_data = SimulatedDataIterator(storage=_make_storage())

        access_key, data_type, params = sim_data._parse_subscription_spec("funding_payment")

        assert access_key == "funding_payment"
        assert data_type == "funding_payment"
        assert params == {}

    def test_simulated_data_funding_payment_fetcher_creation(self):
        """SimulatedDataIterator creates a DataPump with TypedRecords for funding_payment."""
        mock_reader = Mock(spec=IReader)
        sim_data = SimulatedDataIterator(storage=_make_storage(mock_reader))
        mock_instrument = _make_swap_instrument("BTCUSDT")

        sim_data.add_instruments_for_subscription("funding_payment", [mock_instrument])

        # - pump is stored under key "{access_key}.{exchange}:{market_type}"
        pump_key = "funding_payment.binance:SWAP"
        assert pump_key in sim_data._pumps, f"Expected key '{pump_key}' in _pumps, got {list(sim_data._pumps)}"
        pump = sim_data._pumps[pump_key]

        assert isinstance(pump, DataPump)
        assert pump._requested_data_type == "funding_payment"
        assert pump._producing_data_type == "funding_payment"
        assert isinstance(pump._transformer, TypedRecords)

    def test_funding_payment_multiple_instruments(self):
        """Funding payment subscription handles multiple SWAP instruments."""
        sim_data = SimulatedDataIterator(storage=_make_storage())
        instruments = [
            Instrument("BTCUSDT", MarketType.SWAP, "binance", "BTC", "USDT", "USDT", "BTCUSDT", 0.01, 0.001, 0.001),
            Instrument("ETHUSDT", MarketType.SWAP, "binance", "ETH", "USDT", "USDT", "ETHUSDT", 0.01, 0.001, 0.001),
            Instrument("ADAUSDT", MarketType.SWAP, "binance", "ADA", "USDT", "USDT", "ADAUSDT", 0.01, 0.001, 0.001),
        ]

        sim_data.add_instruments_for_subscription("funding_payment", instruments)

        subscribed_instruments = sim_data.get_instruments_for_subscription("funding_payment")
        assert len(subscribed_instruments) == 3
        assert set(i.symbol for i in subscribed_instruments) == {"BTCUSDT", "ETHUSDT", "ADAUSDT"}

    def test_funding_payment_subscription_removal(self):
        """Funding payment subscription can be removed instrument-by-instrument."""
        sim_data = SimulatedDataIterator(storage=_make_storage())
        instrument = Instrument(
            "BTCUSDT", MarketType.SWAP, "binance", "BTC", "USDT", "USDT", "BTCUSDT", 0.01, 0.001, 0.001
        )

        sim_data.add_instruments_for_subscription("funding_payment", [instrument])
        assert sim_data.has_subscription(instrument, "funding_payment")

        sim_data.remove_instruments_from_subscription("funding_payment", [instrument])
        assert not sim_data.has_subscription(instrument, "funding_payment")

    def test_funding_payment_error_handling(self):
        """add_instruments_for_subscription raises when the storage cannot provide a reader."""
        mock_storage = Mock(spec=IStorage)
        mock_storage.get_reader.side_effect = Exception("No reader configured for this market")
        sim_data = SimulatedDataIterator(storage=mock_storage)
        instrument = Instrument(
            "BTCUSDT", MarketType.SWAP, "binance", "BTC", "USDT", "USDT", "BTCUSDT", 0.01, 0.001, 0.001
        )

        with pytest.raises(Exception):
            sim_data.add_instruments_for_subscription("funding_payment", [instrument])

    # -----------------------------------------------------------------------
    # SWAP filter tests
    # -----------------------------------------------------------------------

    def test_funding_payment_non_swap_instruments_filtered(self):
        """Non-SWAP instruments are silently dropped from funding payment subscriptions."""
        sim_data = SimulatedDataIterator(storage=_make_storage())

        spot_instrument = Instrument(
            "BTCUSDT", MarketType.SPOT, "binance", "BTC", "USDT", "USDT", "BTCUSDT", 0.01, 0.001, 0.001
        )
        future_instrument = Instrument(
            "BTCUSDT", MarketType.FUTURE, "binance", "BTC", "USDT", "USDT", "BTCUSDT", 0.01, 0.001, 0.001
        )

        # - no SWAP instruments → filter empties the list → no pump created
        sim_data.add_instruments_for_subscription("funding_payment", [spot_instrument, future_instrument])
        assert len(sim_data.get_instruments_for_subscription("funding_payment")) == 0

        # - SWAP instrument succeeds
        swap_instrument = Instrument(
            "BTCUSDT", MarketType.SWAP, "binance", "BTC", "USDT", "USDT", "BTCUSDT", 0.01, 0.001, 0.001
        )
        sim_data.add_instruments_for_subscription("funding_payment", [swap_instrument])
        subscribed = sim_data.get_instruments_for_subscription("funding_payment")
        assert len(subscribed) == 1
        assert subscribed[0].market_type == MarketType.SWAP

    def test_funding_payment_mixed_instrument_types(self):
        """Only SWAP instruments are kept when a mixed list is subscribed to funding_payment."""
        sim_data = SimulatedDataIterator(storage=_make_storage())

        all_instruments = [
            Instrument("BTCUSDT", MarketType.SWAP, "binance", "BTC", "USDT", "USDT", "BTCUSDT", 0.01, 0.001, 0.001),
            Instrument("ETHUSDT", MarketType.SWAP, "binance", "ETH", "USDT", "USDT", "ETHUSDT", 0.01, 0.001, 0.001),
            Instrument("ADAUSDT", MarketType.SPOT, "binance", "ADA", "USDT", "USDT", "ADAUSDT", 0.01, 0.001, 0.001),
            Instrument("LTCUSDT", MarketType.FUTURE, "binance", "LTC", "USDT", "USDT", "LTCUSDT", 0.01, 0.001, 0.001),
        ]

        sim_data.add_instruments_for_subscription("funding_payment", all_instruments)

        subscribed = sim_data.get_instruments_for_subscription("funding_payment")
        assert len(subscribed) == 2
        assert all(i.market_type == MarketType.SWAP for i in subscribed)
        assert set(i.symbol for i in subscribed) == {"BTCUSDT", "ETHUSDT"}

    def test_funding_payment_only_swap_instruments_unchanged(self):
        """All SWAP instruments pass through funding payment subscription unchanged."""
        sim_data = SimulatedDataIterator(storage=_make_storage())

        swap_instruments = [
            Instrument("BTCUSDT", MarketType.SWAP, "binance", "BTC", "USDT", "USDT", "BTCUSDT", 0.01, 0.001, 0.001),
            Instrument("ETHUSDT", MarketType.SWAP, "binance", "ETH", "USDT", "USDT", "ETHUSDT", 0.01, 0.001, 0.001),
        ]

        sim_data.add_instruments_for_subscription("funding_payment", swap_instruments)

        subscribed = sim_data.get_instruments_for_subscription("funding_payment")
        assert len(subscribed) == 2
        assert all(i.market_type == MarketType.SWAP for i in subscribed)
        assert set(i.symbol for i in subscribed) == {"BTCUSDT", "ETHUSDT"}

    def test_funding_payment_other_subscription_types_unaffected(self):
        """SWAP filtering applies only to funding_payment; other subscriptions are unrestricted."""
        sim_data = SimulatedDataIterator(storage=_make_storage())

        # - SPOT instrument subscribing to 'quote' must NOT be filtered
        spot_instrument = Instrument(
            "BTCUSDT", MarketType.SPOT, "binance", "BTC", "USDT", "USDT", "BTCUSDT", 0.01, 0.001, 0.001
        )

        sim_data.add_instruments_for_subscription("quote", [spot_instrument])

        subscribed = sim_data.get_instruments_for_subscription("quote")
        assert len(subscribed) == 1
        assert subscribed[0].market_type == MarketType.SPOT
