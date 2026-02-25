"""
Tests for funding payment data through the new IStorage/IReader interface.

Replaces the old MultiQdbConnector/QuestDBSqlFundingBuilder-based tests.
The new architecture:
  - SQL is inline in QuestDBReader._prepare_sql_for_dtype (no builder classes)
  - Table metadata decoded via xLTableMetaInfo.decode_table_metadata (not built)
  - Data reading: storage.get_reader(exchange, market).read(symbol, dtype, start, stop)
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from qubx.core.basics import DataType, FundingPayment
from qubx.data.storages.questdb import PGConnectionHelper, QuestDBReader, xLTableMetaInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FUNDING_PAYMENT_TABLE = "binance.umswap.funding_payment"
_FUNDING_RATE_TABLE = "binance.umswap.funding_rate"
_DEFAULT_SYMBOLS = {"BTCUSDT", "ETHUSDT"}


def _make_reader(
    table_name: str = _FUNDING_PAYMENT_TABLE,
    symbols: set[str] | None = None,
) -> tuple[QuestDBReader, Mock]:
    """
    Construct a QuestDBReader backed by a mocked PGConnectionHelper.

    Returns (reader, mock_pgc) so tests can configure mock_pgc.execute.
    """
    if symbols is None:
        symbols = _DEFAULT_SYMBOLS
    xtable = xLTableMetaInfo.decode_table_metadata(table_name)
    mock_pgc = Mock(spec=PGConnectionHelper)
    reader = QuestDBReader(
        exchange="BINANCE.UM",
        market="SWAP",
        available_data=[xtable],
        pgc=mock_pgc,
        synthetic_ohlc_timeframes_types=False,
        min_symbols_for_all_data_request=50,
        symbols_by_table={table_name: symbols},
        manifest_manager=Mock(),
    )
    return reader, mock_pgc


# ---------------------------------------------------------------------------
# FundingPayment dataclass validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFundingPaymentValidation:
    """
    Validate FundingPayment dataclass constraints.

    FundingPayment(time, funding_rate, funding_interval_hours) — no symbol field;
    symbol is a DB column but is not stored in the dataclass.
    """

    def test_valid_funding_payment(self):
        """Test that a well-formed FundingPayment is accepted."""
        timestamp = pd.Timestamp("2025-01-08 00:00:00").asm8

        fp = FundingPayment(time=timestamp, funding_rate=0.0001, funding_interval_hours=8)

        assert fp.funding_rate == 0.0001
        assert fp.funding_interval_hours == 8

    def test_funding_rate_too_high_rejected(self):
        timestamp = pd.Timestamp("2025-01-08 00:00:00").asm8

        with pytest.raises(ValueError, match="Invalid funding rate"):
            FundingPayment(time=timestamp, funding_rate=1.5, funding_interval_hours=8)

    def test_negative_funding_interval_rejected(self):
        timestamp = pd.Timestamp("2025-01-08 00:00:00").asm8

        with pytest.raises(ValueError, match="Invalid funding interval"):
            FundingPayment(time=timestamp, funding_rate=0.0001, funding_interval_hours=-1)


# ---------------------------------------------------------------------------
# xLTableMetaInfo — table name decoding
# (replaces QuestDBSqlFundingBuilder.get_table_name tests)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFundingTableMetadata:
    """
    Test xLTableMetaInfo.decode_table_metadata for funding-related tables.

    The old tests used QuestDBSqlFundingBuilder.get_table_name() to build a table name
    from exchange:symbol. The new architecture is the reverse: table names are decoded
    from QuestDB's tables() system view into structured metadata.
    """

    def test_binance_funding_payment_table_decoded(self):
        meta = xLTableMetaInfo.decode_table_metadata(_FUNDING_PAYMENT_TABLE)

        assert meta is not None
        assert meta.exchange == "BINANCE.UM"
        assert meta.market_type == "SWAP"
        assert meta.dtype == DataType.FUNDING_PAYMENT
        assert meta.table_name == _FUNDING_PAYMENT_TABLE

    def test_binance_funding_rate_table_decoded(self):
        meta = xLTableMetaInfo.decode_table_metadata(_FUNDING_RATE_TABLE)

        assert meta is not None
        assert meta.exchange == "BINANCE.UM"
        assert meta.market_type == "SWAP"
        assert meta.dtype == DataType.FUNDING_RATE
        assert meta.table_name == _FUNDING_RATE_TABLE

    def test_funding_payment_dtype_round_trip(self):
        """DataType.from_str('funding_payment') must resolve correctly."""
        dt, params = DataType.from_str("funding_payment")

        assert dt == DataType.FUNDING_PAYMENT
        assert params == {}


# ---------------------------------------------------------------------------
# QuestDBReader — SQL generation and data reading for funding data
# (replaces QuestDBSqlFundingBuilder query-generation tests)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestQuestDBReaderFunding:
    """
    Test QuestDBReader._prepare_sql_for_dtype and read() for funding data.
    PGConnectionHelper is mocked — no live QuestDB required.
    """

    def test_funding_payment_sql_columns_and_table(self):
        """Generated SQL must select the correct columns from the correct table."""
        reader, _ = _make_reader(_FUNDING_PAYMENT_TABLE)
        xtable = xLTableMetaInfo.decode_table_metadata(_FUNDING_PAYMENT_TABLE)

        sql = reader._prepare_sql_for_dtype(DataType.FUNDING_PAYMENT, {"BTCUSDT"}, xtable, [], "")

        assert "SELECT timestamp, symbol, funding_rate, funding_interval_hours" in sql
        assert f'"{_FUNDING_PAYMENT_TABLE}"' in sql
        assert "ORDER BY timestamp ASC" in sql

    def test_funding_payment_sql_time_conditions(self):
        """Time-range conditions must appear in the WHERE clause."""
        reader, _ = _make_reader()
        xtable = xLTableMetaInfo.decode_table_metadata(_FUNDING_PAYMENT_TABLE)
        conditions = ["timestamp >= '2025-01-08T00:00:00'", "timestamp < '2025-01-08T23:59:59'"]

        sql = reader._prepare_sql_for_dtype(DataType.FUNDING_PAYMENT, {"BTCUSDT"}, xtable, conditions, "")

        assert "timestamp >= '2025-01-08T00:00:00'" in sql
        assert "timestamp < '2025-01-08T23:59:59'" in sql

    def test_funding_payment_sql_symbol_filter(self):
        """Requested symbol must appear in the IN clause (uppercased and quoted)."""
        reader, _ = _make_reader()
        xtable = xLTableMetaInfo.decode_table_metadata(_FUNDING_PAYMENT_TABLE)

        sql = reader._prepare_sql_for_dtype(DataType.FUNDING_PAYMENT, {"BTCUSDT"}, xtable, [], "")

        assert "'BTCUSDT'" in sql

    def test_funding_payment_sql_no_symbol_filter_when_empty(self):
        """Empty symbol set → fetch all rows — no symbol IN clause."""
        reader, _ = _make_reader()
        xtable = xLTableMetaInfo.decode_table_metadata(_FUNDING_PAYMENT_TABLE)

        sql = reader._prepare_sql_for_dtype(DataType.FUNDING_PAYMENT, set(), xtable, [], "")

        # - empty set means "all symbols" — no WHERE symbol filter
        assert "symbol in" not in sql.lower()

    def test_funding_dtype_registered_in_lookup(self):
        """After construction, funding_payment must be in _dtype_lookup."""
        reader, _ = _make_reader()

        x_type = str(DataType.FUNDING_PAYMENT).lower()
        assert x_type in reader._dtype_lookup

        symbols, xtable = reader._dtype_lookup[x_type]
        assert "BTCUSDT" in symbols
        assert xtable.dtype == DataType.FUNDING_PAYMENT

    def test_funding_payment_read_returns_data(self):
        """read() with mocked execute must return RawData containing the sample rows."""
        reader, mock_pgc = _make_reader()

        # - mock DB returning two funding payment rows for BTCUSDT
        mock_pgc.execute.return_value = (
            ["timestamp", "symbol", "funding_rate", "funding_interval_hours"],
            [
                (pd.Timestamp("2025-01-08 00:00:00"), "BTCUSDT", 0.0001, 8),
                (pd.Timestamp("2025-01-08 08:00:00"), "BTCUSDT", -0.0002, 8),
            ],
        )

        result = reader.read("BTCUSDT", "funding_payment", "2025-01-08T00:00:00", "2025-01-08T23:59:59")

        assert result is not None
        df = result.to_pd()
        assert len(df) == 2
        assert set(df.columns) >= {"symbol", "funding_rate", "funding_interval_hours"}

    def test_funding_payment_read_empty_result(self):
        """read() must handle empty DB result gracefully."""
        reader, mock_pgc = _make_reader()
        mock_pgc.execute.return_value = (
            ["timestamp", "symbol", "funding_rate", "funding_interval_hours"],
            [],
        )

        result = reader.read("BTCUSDT", "funding_payment", "2024-01-01", "2024-01-02")

        assert result is not None
        df = result.to_pd()
        assert len(df) == 0

    def test_symbol_quoting_in_generated_sql(self):
        """Symbol values must be single-quoted in the IN clause (basic injection guard)."""
        reader, _ = _make_reader(symbols={"BTCUSDT"})
        xtable = xLTableMetaInfo.decode_table_metadata(_FUNDING_PAYMENT_TABLE)

        sql = reader._prepare_sql_for_dtype(DataType.FUNDING_PAYMENT, {"BTCUSDT"}, xtable, [], "")

        # - symbol wrapped in single quotes: symbol in ('BTCUSDT')
        assert "symbol in (" in sql.lower()
        assert "'BTCUSDT'" in sql
