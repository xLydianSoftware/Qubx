"""
Tests for PostgreSQL logger and restorers.

Uses unittest.mock to simulate psycopg connections since there is no
psycopg equivalent of mongomock.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from qubx.core.basics import (
    Instrument,
    MarketType,
    RestoredState,
)
from qubx.restorers.balance import PostgresBalanceRestorer
from qubx.restorers.interfaces import (
    IBalanceRestorer,
    IPositionRestorer,
    ISignalRestorer,
    IStateRestorer,
)
from qubx.restorers.position import PostgresPositionRestorer
from qubx.restorers.signal import PostgresSignalRestorer
from qubx.restorers.state import PostgresStateRestorer

# --- Helpers ---

def create_mock_instruments():
    return {
        "BINANCE.UM:SWAP:BTCUSDT": Instrument(
            symbol="BTCUSDT",
            market_type=MarketType.SWAP,
            exchange="BINANCE.UM",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTC/USDT:USDT",
            tick_size=0.1,
            lot_size=0.001,
            min_size=0.001,
        ),
        "BINANCE.UM:SWAP:ETHUSDT": Instrument(
            symbol="ETHUSDT",
            market_type=MarketType.SWAP,
            exchange="BINANCE.UM",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETH/USDT:USDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        ),
    }


def mock_find_symbol(exchange, symbol, market_type=None):
    instruments = create_mock_instruments()
    for instrument in instruments.values():
        if instrument.exchange == exchange and instrument.symbol == symbol:
            if market_type is None or instrument.market_type == market_type:
                return instrument
    return None


@pytest.fixture
def mock_lookup():
    with (
        patch("qubx.restorers.position.lookup") as mock_pos,
        patch("qubx.restorers.signal.lookup") as mock_sig,
    ):
        mock_pos.find_symbol = mock_find_symbol
        mock_sig.find_symbol = mock_find_symbol
        yield


def _make_mock_connection():
    """Create a mock psycopg Connection with cursor context manager support."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


# --- Protocol tests ---

class TestPostgresProtocolImplementations:

    def test_position_restorer_protocol(self):
        conn, _ = _make_mock_connection()
        restorer = PostgresPositionRestorer(strategy_name="test", connection=conn)
        assert isinstance(restorer, IPositionRestorer)

    def test_signal_restorer_protocol(self):
        conn, _ = _make_mock_connection()
        restorer = PostgresSignalRestorer(strategy_name="test", connection=conn)
        assert isinstance(restorer, ISignalRestorer)

    def test_balance_restorer_protocol(self):
        conn, _ = _make_mock_connection()
        restorer = PostgresBalanceRestorer(strategy_name="test", connection=conn)
        assert isinstance(restorer, IBalanceRestorer)

    def test_state_restorer_protocol(self):
        restorer = PostgresStateRestorer(strategy_name="test")
        assert isinstance(restorer, IStateRestorer)


# --- Position restorer tests ---

class TestPostgresPositionRestorer:
    _strategy_name = "test_strategy"

    def test_with_no_data(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = None

        restorer = PostgresPositionRestorer(strategy_name=self._strategy_name, connection=conn)
        result = restorer.restore_positions()

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_with_sample_data(self, mock_lookup):
        conn, cursor = _make_mock_connection()
        ts = datetime.now(timezone.utc) - timedelta(hours=1)

        # First call: find latest run_id
        # Second call: get positions
        cursor.fetchone.return_value = ("run-123",)
        cursor.fetchall.return_value = [
            # symbol, exchange, market_type, quantity, avg_price, r_pnl, current_price, funding, commissions, ts
            ("BTCUSDT", "BINANCE.UM", "SWAP", 1.5, 90000.0, 100.0, 91000.0, 5.0, 10.0, ts),
        ]

        restorer = PostgresPositionRestorer(strategy_name=self._strategy_name, connection=conn)
        result = restorer.restore_positions()

        assert len(result) == 1
        btc_pos = next(p for i, p in result.items() if i.symbol == "BTCUSDT")
        assert btc_pos.quantity == 1.5
        assert btc_pos.position_avg_price == 90000.0


# --- Signal restorer tests ---

class TestPostgresSignalRestorer:
    _strategy_name = "test_strategy"

    def test_with_no_data(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchall.return_value = []
        cursor.description = []

        restorer = PostgresSignalRestorer(strategy_name=self._strategy_name, connection=conn)
        result = restorer.restore_signals()

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_with_sample_signals(self, mock_lookup):
        conn, cursor = _make_mock_connection()
        ts = datetime.now(timezone.utc) - timedelta(hours=1)

        # Mock cursor.description to return column names
        col_names = [
            "id", "run_id", "account_id", "strategy_name", "created_at",
            "timestamp", "symbol", "exchange", "market_type",
            "signal", "reference_price", "price", "take", "stop",
            "group_name", "comment", "service", "options", "rn",
        ]
        cursor.description = [MagicMock(name=c) for c in col_names]
        for desc, name in zip(cursor.description, col_names):
            desc.name = name

        cursor.fetchall.return_value = [
            (1, "run-123", "acc", "test_strategy", ts,
             ts, "BTCUSDT", "BINANCE.UM", "SWAP",
             "1.0", 90000.0, 90000.0, None, None,
             "", "", False, {}, 1),
        ]

        restorer = PostgresSignalRestorer(strategy_name=self._strategy_name, connection=conn)
        result = restorer.restore_signals()

        assert len(result) == 1
        btc_signals = next(s for i, s in result.items() if i.symbol == "BTCUSDT")
        assert len(btc_signals) == 1
        assert btc_signals[0].signal == 1.0

    def test_with_sample_targets(self, mock_lookup):
        conn, cursor = _make_mock_connection()
        ts = datetime.now(timezone.utc) - timedelta(hours=1)

        col_names = [
            "id", "run_id", "account_id", "strategy_name", "created_at",
            "timestamp", "symbol", "exchange", "market_type",
            "target_position", "entry_price", "take_price", "stop_price",
            "options", "rn",
        ]
        cursor.description = [MagicMock(name=c) for c in col_names]
        for desc, name in zip(cursor.description, col_names):
            desc.name = name

        cursor.fetchall.return_value = [
            (1, "run-123", "acc", "test_strategy", ts,
             ts, "BTCUSDT", "BINANCE.UM", "SWAP",
             0.5, 90000.0, None, None,
             {}, 1),
        ]

        restorer = PostgresSignalRestorer(strategy_name=self._strategy_name, connection=conn)
        result = restorer.restore_targets()

        assert len(result) == 1
        btc_targets = next(t for i, t in result.items() if i.symbol == "BTCUSDT")
        assert len(btc_targets) == 1
        assert btc_targets[0].target_position_size == 0.5


# --- Balance restorer tests ---

class TestPostgresBalanceRestorer:
    _strategy_name = "test_strategy"

    def test_with_no_data(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = None

        restorer = PostgresBalanceRestorer(strategy_name=self._strategy_name, connection=conn)
        result = restorer.restore_balances()

        assert isinstance(result, list)
        assert len(result) == 0

    def test_with_sample_data(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = ("run-123",)
        cursor.fetchall.return_value = [
            ("BINANCE.UM", "USDT", 10000.0, 1000.0),
        ]

        restorer = PostgresBalanceRestorer(strategy_name=self._strategy_name, connection=conn)
        result = restorer.restore_balances()

        assert len(result) == 1
        assert result[0].currency == "USDT"
        assert result[0].total == 10000.0
        assert result[0].locked == 1000.0
        assert result[0].free == 9000.0
        assert result[0].exchange == "BINANCE.UM"


# --- State restorer tests ---

class TestPostgresStateRestorer:
    _strategy_name = "test_strategy"
    _postgres_uri = "postgresql://localhost:5432/qubx_logs"

    @patch("qubx.restorers.state.psycopg")
    def test_with_no_tables(self, mock_psycopg):
        conn = MagicMock()
        mock_psycopg.connect.return_value = conn
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # No tables exist
        cursor.fetchall.return_value = []

        restorer = PostgresStateRestorer(
            strategy_name=self._strategy_name,
            postgres_uri=self._postgres_uri,
        )
        result = restorer.restore_state()

        assert isinstance(result, RestoredState)
        assert len(result.positions) == 0
        assert len(result.balances) == 0
        conn.close.assert_called_once()

    @patch("qubx.restorers.state.psycopg")
    def test_with_sample_data(self, mock_psycopg, mock_lookup):
        conn = MagicMock()
        mock_psycopg.connect.return_value = conn
        ts = datetime.now(timezone.utc) - timedelta(hours=1)

        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # We need to handle multiple sequential cursor calls.
        # The state restorer: 1) checks tables, then delegates to individual restorers.
        # Each restorer creates its own cursor via conn.cursor().
        # We track call count to return appropriate data.

        call_count = {"n": 0}

        def side_effect_fetchall():
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                # Table existence check
                return [("qubx_logs_positions",), ("qubx_logs_signals",), ("qubx_logs_balance",)]
            elif n == 2:
                # Position restorer: fetch positions
                return [("BTCUSDT", "BINANCE.UM", "SWAP", 1.0, 90000.0, 50.0, 91000.0, 5.0, 10.0, ts)]
            elif n == 3:
                # Signal restorer: fetch signals (returns dicts via _load_data)
                return []
            elif n == 4:
                # Target restorer: fetch targets
                return []
            elif n == 5:
                # Balance restorer: fetch balances
                return [("BINANCE.UM", "USDT", 10000.0, 500.0)]
            return []

        fetchone_count = {"n": 0}

        def side_effect_fetchone():
            fetchone_count["n"] += 1
            return ("run-123",)

        cursor.fetchall.side_effect = side_effect_fetchall
        cursor.fetchone.side_effect = side_effect_fetchone
        cursor.description = []

        restorer = PostgresStateRestorer(
            strategy_name=self._strategy_name,
            postgres_uri=self._postgres_uri,
        )
        result = restorer.restore_state()

        assert isinstance(result, RestoredState)
        assert len(result.positions) == 1
        assert len(result.balances) == 1

        # Check position
        btc_pos = None
        for instrument, position in result.positions.items():
            if instrument.symbol == "BTCUSDT":
                btc_pos = position
        assert btc_pos is not None
        assert btc_pos.quantity == 1.0
        assert btc_pos.position_avg_price == 90000.0

        # Check balance
        assert result.balances[0].currency == "USDT"
        assert result.balances[0].total == 10000.0
        assert result.balances[0].free == 9500.0

        conn.close.assert_called_once()


# --- Logger tests ---

class TestPostgresLogsWriter:
    """Test that PostgresLogsWriter can be instantiated and registered in the factory."""

    def test_factory_registration(self):
        from qubx.loggers.factory import LOGS_WRITER_REGISTRY
        assert "PostgresLogsWriter" in LOGS_WRITER_REGISTRY

    def test_import(self):
        from qubx.loggers import PostgresLogsWriter
        assert PostgresLogsWriter is not None

    @patch("qubx.loggers.postgres.ThreadPool")
    @patch("qubx.loggers.postgres.ConnectionPool")
    def test_write_data_calls_thread_pool(self, mock_pool_cls, mock_tp_cls):
        from qubx.loggers.postgres import PostgresLogsWriter

        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool
        mock_tp = MagicMock()
        mock_tp_cls.return_value = mock_tp

        # Mock the connection for _ensure_tables
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        writer = PostgresLogsWriter(
            account_id="test_acc",
            strategy_id="test_strat",
            run_id="run-1",
            postgres_uri="postgresql://localhost/test",
        )

        data = [
            {
                "timestamp": "2025-01-01T00:00:00",
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "pnl_quoted": 100.0,
                "funding_pnl_quoted": 5.0,
                "realized_pnl_quoted": 50.0,
                "quantity": 1.0,
                "notional": 90000.0,
                "avg_position_price": 90000.0,
                "current_price": 91000.0,
                "market_value_quoted": 91000.0,
                "commissions_quoted": 10.0,
            }
        ]

        writer.write_data("positions", data)

        mock_tp.apply_async.assert_called_once()

    @patch("qubx.loggers.postgres.ThreadPool")
    @patch("qubx.loggers.postgres.ConnectionPool")
    def test_write_data_ignores_empty(self, mock_pool_cls, mock_tp_cls):
        from qubx.loggers.postgres import PostgresLogsWriter

        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool
        mock_tp = MagicMock()
        mock_tp_cls.return_value = mock_tp

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        writer = PostgresLogsWriter(
            account_id="test_acc",
            strategy_id="test_strat",
            run_id="run-1",
            postgres_uri="postgresql://localhost/test",
        )

        writer.write_data("positions", [])

        mock_tp.apply_async.assert_not_called()
