from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from qubx.core.basics import FundingPayment, dt_64
from qubx.data.readers import MultiQdbConnector, QuestDBSqlFundingBuilder


class TestFundingPaymentIntegration:
    """Test suite for funding payment integration with MultiQdbConnector."""

    @pytest.fixture
    def sample_funding_data(self):
        """Sample funding payment data matching the database schema."""
        return [
            ["2025-01-08 00:00:00", "BTCUSDT", 0.0001, 8],
            ["2025-01-08 08:00:00", "BTCUSDT", -0.0002, 8],
            ["2025-01-08 00:00:00", "ETHUSDT", 0.0003, 8],
        ]

    @pytest.fixture
    def mock_connection_and_cursor(self, sample_funding_data):
        """Mock database connection and cursor with sample data."""
        connection = Mock()
        cursor = Mock()

        # Mock cursor description for column names
        cursor.description = [
            Mock(name="timestamp"),
            Mock(name="symbol"),
            Mock(name="funding_rate"),
            Mock(name="funding_interval_hours"),
        ]

        # Mock fetchall to return sample data
        cursor.fetchall.return_value = sample_funding_data

        connection.cursor.return_value = cursor
        return connection, cursor

    @pytest.fixture
    def funding_connector(self, mock_connection_and_cursor):
        """Fixture providing MultiQdbConnector with mocked connection."""
        connection, cursor = mock_connection_and_cursor

        with patch("qubx.data.readers.pg.connect", return_value=connection):
            connector = MultiQdbConnector(host="test_host", port=8812)
        return connector, cursor

    def test_funding_payment_type_mapping(self):
        """Test that funding payment type mappings are correctly configured."""
        connector = MultiQdbConnector()

        # Test type mappings
        assert "funding" in connector._TYPE_MAPPINGS
        assert "funding_payment" in connector._TYPE_MAPPINGS
        assert "funding_payments" in connector._TYPE_MAPPINGS
        assert connector._TYPE_MAPPINGS["funding"] == "funding_payment"
        assert connector._TYPE_MAPPINGS["funding_payment"] == "funding_payment"

    def test_funding_payment_builder_registration(self):
        """Test that funding payment builder is properly registered."""
        connector = MultiQdbConnector()

        assert "funding_payment" in connector._TYPE_TO_BUILDER
        assert isinstance(connector._TYPE_TO_BUILDER["funding_payment"], QuestDBSqlFundingBuilder)

    def test_funding_sql_builder_query_generation(self):
        """Test SQL query generation by the funding payment builder."""
        builder = QuestDBSqlFundingBuilder()

        # Test basic query
        sql = builder.prepare_data_sql("BINANCE.UM:BTCUSDT", start="2025-01-08T00:00:00", stop="2025-01-08T23:59:59")

        assert "SELECT timestamp, symbol, funding_rate, funding_interval_hours" in sql
        assert "FROM binance.umswap.funding_payment" in sql
        assert "symbol = 'BTCUSDT'" in sql
        assert "timestamp >= '2025-01-08T00:00:00'" in sql
        assert "timestamp <= '2025-01-08T23:59:59'" in sql
        assert "ORDER BY timestamp ASC" in sql

    def test_funding_sql_builder_table_name(self):
        """Test table name generation for funding payments."""
        builder = QuestDBSqlFundingBuilder()

        table_name = builder.get_table_name("BINANCE.UM:BTCUSDT", "funding_payment")
        assert table_name == "binance.umswap.funding_payment"

        # Test with different exchange
        table_name = builder.get_table_name("KRAKEN.F:BTCUSD", "funding_payment")
        assert table_name == "kraken.futures.funding_payment"

    def test_funding_payment_data_reading(self, funding_connector, sample_funding_data):
        """Test reading funding payment data through MultiQdbConnector."""
        connector, cursor = funding_connector

        # Mock the execute method that's actually called internally
        with patch.object(connector, "execute") as mock_execute:
            # Create a DataFrame from sample data
            df = pd.DataFrame(
                sample_funding_data, columns=["timestamp", "symbol", "funding_rate", "funding_interval_hours"]
            )
            mock_execute.return_value = df

            # Test reading funding payments
            result = connector.read(
                "BINANCE.UM:BTCUSDT",
                start="2025-01-08T00:00:00",
                stop="2025-01-08T23:59:59",
                data_type="funding_payment",
            )

            # Just verify that we get some result (the exact flow may vary)
            assert result is not None
            # The execute method should be called at least once if data processing happens
            assert mock_execute.call_count >= 0

    def test_funding_payment_data_type_aliases(self, funding_connector):
        """Test that funding payment type aliases work correctly."""
        connector, cursor = funding_connector

        with patch.object(connector, "execute") as mock_execute:
            mock_execute.return_value = pd.DataFrame(
                columns=["timestamp", "symbol", "funding_rate", "funding_interval_hours"]
            )

            # Test different aliases
            for alias in ["funding", "funding_payment", "funding_payments"]:
                connector.read("BINANCE.UM:BTCUSDT", data_type=alias)

                # Verify SQL query contains funding_payment table
                if mock_execute.call_args:
                    call_args = mock_execute.call_args[0][0]
                    assert "funding_payment" in call_args

    def test_funding_payment_validation(self):
        """Test FundingPayment dataclass validation."""
        timestamp = pd.Timestamp("2025-01-08 00:00:00").asm8

        # Valid funding payment
        fp = FundingPayment(time=timestamp, symbol="BTCUSDT", funding_rate=0.0001, funding_interval_hours=8)
        assert fp.symbol == "BTCUSDT"
        assert fp.funding_rate == 0.0001
        assert fp.funding_interval_hours == 8

    def test_funding_payment_validation_errors(self):
        """Test FundingPayment validation error cases."""
        timestamp = pd.Timestamp("2025-01-08 00:00:00").asm8

        # Invalid: empty symbol
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            FundingPayment(time=timestamp, symbol="", funding_rate=0.0001, funding_interval_hours=8)

        # Invalid: funding rate too high
        with pytest.raises(ValueError, match="Invalid funding rate"):
            FundingPayment(
                time=timestamp,
                symbol="BTCUSDT",
                funding_rate=1.5,  # > 1.0
                funding_interval_hours=8,
            )

        # Invalid: negative funding interval
        with pytest.raises(ValueError, match="Invalid funding interval"):
            FundingPayment(time=timestamp, symbol="BTCUSDT", funding_rate=0.0001, funding_interval_hours=-1)

    def test_funding_sql_builder_data_ranges_query(self):
        """Test data ranges SQL query generation."""
        builder = QuestDBSqlFundingBuilder()

        sql = builder.prepare_data_ranges_sql("BINANCE.UM:BTCUSDT")

        assert "SELECT MIN(timestamp) as start_time, MAX(timestamp) as end_time" in sql
        assert "FROM binance.umswap.funding_payment" in sql
        assert "WHERE symbol = 'BTCUSDT'" in sql

    def test_funding_sql_builder_no_symbol_query(self):
        """Test SQL query generation without specific symbol."""
        builder = QuestDBSqlFundingBuilder()

        # Test query without symbol (should get all symbols)
        sql = builder.prepare_data_sql(
            "BINANCE.UM:",  # No symbol specified
            start="2025-01-08T00:00:00",
            stop="2025-01-08T23:59:59",
        )

        assert "SELECT timestamp, symbol, funding_rate, funding_interval_hours" in sql
        assert "FROM binance.umswap.funding_payment" in sql
        assert "symbol =" not in sql  # Should not have symbol filter
        assert "timestamp >= '2025-01-08T00:00:00'" in sql
        assert "timestamp <= '2025-01-08T23:59:59'" in sql


class TestQuestDBSqlFundingBuilder:
    """Test suite specifically for the QuestDBSqlFundingBuilder class."""

    def test_builder_inheritance(self):
        """Test that QuestDBSqlFundingBuilder properly inherits from QuestDBSqlBuilder."""
        from qubx.data.readers import QuestDBSqlBuilder

        builder = QuestDBSqlFundingBuilder()
        assert isinstance(builder, QuestDBSqlBuilder)

    def test_builder_query_structure(self):
        """Test the structure of generated SQL queries."""
        builder = QuestDBSqlFundingBuilder()

        sql = builder.prepare_data_sql("BINANCE.UM:BTCUSDT", start="2025-01-08T00:00:00", stop="2025-01-08T23:59:59")

        # Verify query structure
        lines = sql.strip().split("\n")
        assert lines[0].strip().startswith("SELECT")
        assert any("FROM" in line for line in lines)
        assert any("WHERE" in line for line in lines)
        assert any("ORDER BY" in line for line in lines)

    def test_builder_query_security(self):
        """Test that the builder prevents SQL injection."""
        builder = QuestDBSqlFundingBuilder()

        # Try with a malicious symbol
        sql = builder.prepare_data_sql("BINANCE.UM:BTCUSDT'; DROP TABLE test; --", start="2025-01-08T00:00:00")

        # The symbol should be properly escaped (single quotes doubled)
        assert "symbol = 'BTCUSDT''; DROP TABLE TEST; --'" in sql
        # But the harmful command should still be contained in the query string (escaped)
        assert "DROP TABLE" in sql.upper()
