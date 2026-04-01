"""
End-to-end tests for PostgreSQL logger and restorers.

Requires a running PostgreSQL instance at localhost:5432 with
user=xlydian, password=xlydian, database=qubx_test_e2e.

Tests write data via PostgresLogsWriter, then read it back via
Postgres restorers, verifying the full round-trip.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import psycopg
import pytest

from qubx.core.basics import Instrument, MarketType, RestoredState
from qubx.loggers.postgres import PostgresLogsWriter
from qubx.restorers.balance import PostgresBalanceRestorer
from qubx.restorers.position import PostgresPositionRestorer
from qubx.restorers.signal import PostgresSignalRestorer
from qubx.restorers.state import PostgresStateRestorer

POSTGRES_URI = "postgresql://xlydian:xlydian@localhost:5432/qubx_test_e2e"
TABLE_PREFIX = "e2e_test"
STRATEGY_NAME = "e2e_test_strategy"
ACCOUNT_ID = "e2e_account"
RUN_ID = "e2e_run_001"


def _create_mock_instruments():
    return {
        "BTCUSDT": Instrument(
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
        "ETHUSDT": Instrument(
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


def _mock_find_symbol(exchange, symbol, market_type=None):
    instruments = _create_mock_instruments()
    inst = instruments.get(symbol)
    if inst and inst.exchange == exchange:
        return inst
    return None


@pytest.fixture(scope="module")
def mock_lookup():
    with (
        patch("qubx.restorers.position.lookup") as mock_pos,
        patch("qubx.restorers.signal.lookup") as mock_sig,
    ):
        mock_pos.find_symbol = _mock_find_symbol
        mock_sig.find_symbol = _mock_find_symbol
        yield


@pytest.fixture(scope="module")
def pg_writer():
    """Create a PostgresLogsWriter, write test data, then clean up."""
    writer = PostgresLogsWriter(
        account_id=ACCOUNT_ID,
        strategy_id=STRATEGY_NAME,
        run_id=RUN_ID,
        postgres_uri=POSTGRES_URI,
        table_prefix=TABLE_PREFIX,
        pool_size=2,
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # --- Write positions ---
    writer.write_data(
        "positions",
        [
            {
                "timestamp": now,
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "pnl_quoted": 150.0,
                "funding_pnl_quoted": 3.5,
                "realized_pnl_quoted": 75.0,
                "quantity": 0.5,
                "notional": 45000.0,
                "avg_position_price": 89000.0,
                "current_price": 90000.0,
                "market_value_quoted": 45000.0,
                "commissions_quoted": 12.0,
            },
            {
                "timestamp": now,
                "symbol": "ETHUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "pnl_quoted": 20.0,
                "funding_pnl_quoted": 1.0,
                "realized_pnl_quoted": 10.0,
                "quantity": 2.0,
                "notional": 6000.0,
                "avg_position_price": 2900.0,
                "current_price": 3000.0,
                "market_value_quoted": 6000.0,
                "commissions_quoted": 3.0,
            },
        ],
    )

    # --- Write portfolio ---
    writer.write_data(
        "portfolio",
        [
            {
                "timestamp": now,
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "pnl_quoted": 150.0,
                "quantity": 0.5,
                "realized_pnl_quoted": 75.0,
                "avg_position_price": 89000.0,
                "current_price": 90000.0,
                "market_value_quoted": 45000.0,
                "exchange_time": now,
                "commissions_quoted": 12.0,
                "cumulative_funding": 3.5,
            },
        ],
    )

    # --- Write executions ---
    writer.write_data(
        "executions",
        [
            {
                "timestamp": now,
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "side": "buy",
                "filled_qty": 0.5,
                "price": 89000.0,
                "commissions": 12.0,
                "commissions_quoted": "USDT",
                "order_id": "order-001",
                "order_type": "TAKER",
            },
        ],
    )

    # --- Write signals ---
    writer.write_data(
        "signals",
        [
            {
                "timestamp": now,
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "signal": 1.0,
                "reference_price": 89000.0,
                "price": 89000.0,
                "take": None,
                "stop": None,
                "group": "main",
                "comment": "e2e test signal",
                "service": False,
                "options": {"reason": "test"},
            },
        ],
    )

    # --- Write targets ---
    writer.write_data(
        "targets",
        [
            {
                "timestamp": now,
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "target_position": 0.5,
                "entry_price": 89000.0,
                "take_price": 95000.0,
                "stop_price": 85000.0,
                "options": {"mode": "test"},
            },
        ],
    )

    # --- Write balance ---
    writer.write_data(
        "balance",
        [
            {
                "timestamp": now,
                "exchange": "BINANCE.UM",
                "currency": "USDT",
                "total": 50000.0,
                "locked": 5000.0,
            },
            {
                "timestamp": now,
                "exchange": "BINANCE.UM",
                "currency": "BTC",
                "total": 0.5,
                "locked": 0.0,
            },
        ],
    )

    # Wait for async thread pool writes to complete
    writer._thread_pool.close()
    writer._thread_pool.join()

    yield writer

    # Cleanup: drop all e2e tables
    conn = psycopg.connect(POSTGRES_URI, autocommit=True)
    with conn.cursor() as cur:
        for log_type in ["positions", "portfolio", "executions", "signals", "targets", "balance"]:
            cur.execute(psycopg.sql.SQL("DROP TABLE IF EXISTS {table}").format(
                table=psycopg.sql.Identifier(f"{TABLE_PREFIX}_{log_type}")
            ))
    conn.close()
    writer._pool.close()


@pytest.fixture(scope="module")
def pg_conn(pg_writer):
    """Get a connection to the test database (after data is written)."""
    conn = psycopg.connect(POSTGRES_URI, autocommit=True)
    yield conn
    conn.close()


# --- Verify data was written ---

@pytest.mark.e2e
class TestPostgresE2EDataWritten:

    def test_positions_table_has_rows(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_PREFIX}_positions")
            assert cur.fetchone()[0] == 2

    def test_portfolio_table_has_rows(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_PREFIX}_portfolio")
            assert cur.fetchone()[0] == 1

    def test_executions_table_has_rows(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_PREFIX}_executions")
            assert cur.fetchone()[0] == 1

    def test_signals_table_has_rows(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_PREFIX}_signals")
            assert cur.fetchone()[0] == 1

    def test_targets_table_has_rows(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_PREFIX}_targets")
            assert cur.fetchone()[0] == 1

    def test_balance_table_has_rows(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_PREFIX}_balance")
            assert cur.fetchone()[0] == 2

    def test_metadata_columns(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(
                f"SELECT run_id, account_id, strategy_name FROM {TABLE_PREFIX}_positions LIMIT 1"
            )
            row = cur.fetchone()
            assert row[0] == RUN_ID
            assert row[1] == ACCOUNT_ID
            assert row[2] == STRATEGY_NAME

    def test_index_exists(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = %s AND indexname LIKE %s",
                (f"{TABLE_PREFIX}_positions", "idx_%_strategy_ts"),
            )
            assert cur.fetchone() is not None


# --- Verify restorers can read back the data ---

@pytest.mark.e2e
class TestPostgresE2EPositionRestorer:

    def test_restore_positions(self, pg_conn, mock_lookup):
        restorer = PostgresPositionRestorer(
            strategy_name=STRATEGY_NAME,
            connection=pg_conn,
            table_name=f"{TABLE_PREFIX}_positions",
        )
        positions = restorer.restore_positions()

        assert len(positions) == 2

        btc_pos = None
        eth_pos = None
        for inst, pos in positions.items():
            if inst.symbol == "BTCUSDT":
                btc_pos = pos
            elif inst.symbol == "ETHUSDT":
                eth_pos = pos

        assert btc_pos is not None
        assert btc_pos.quantity == 0.5
        assert btc_pos.position_avg_price == 89000.0
        assert btc_pos.r_pnl == 75.0
        assert btc_pos.commissions == 12.0

        assert eth_pos is not None
        assert eth_pos.quantity == 2.0
        assert eth_pos.position_avg_price == 2900.0


@pytest.mark.e2e
class TestPostgresE2ESignalRestorer:

    def test_restore_signals(self, pg_conn, mock_lookup):
        restorer = PostgresSignalRestorer(
            strategy_name=STRATEGY_NAME,
            connection=pg_conn,
            signals_table_name=f"{TABLE_PREFIX}_signals",
            targets_table_name=f"{TABLE_PREFIX}_targets",
        )
        signals = restorer.restore_signals()

        assert len(signals) == 1
        btc_signals = list(signals.values())[0]
        assert len(btc_signals) == 1
        assert btc_signals[0].signal == 1.0
        assert btc_signals[0].price == 89000.0
        assert btc_signals[0].group == "main"
        assert btc_signals[0].comment == "e2e test signal"

    def test_restore_targets(self, pg_conn, mock_lookup):
        restorer = PostgresSignalRestorer(
            strategy_name=STRATEGY_NAME,
            connection=pg_conn,
            signals_table_name=f"{TABLE_PREFIX}_signals",
            targets_table_name=f"{TABLE_PREFIX}_targets",
        )
        targets = restorer.restore_targets()

        assert len(targets) == 1
        btc_targets = list(targets.values())[0]
        assert len(btc_targets) == 1
        assert btc_targets[0].target_position_size == 0.5
        assert btc_targets[0].entry_price == 89000.0
        assert btc_targets[0].take_price == 95000.0
        assert btc_targets[0].stop_price == 85000.0


@pytest.mark.e2e
class TestPostgresE2EBalanceRestorer:

    def test_restore_balances(self, pg_conn):
        restorer = PostgresBalanceRestorer(
            strategy_name=STRATEGY_NAME,
            connection=pg_conn,
            table_name=f"{TABLE_PREFIX}_balance",
        )
        balances = restorer.restore_balances()

        assert len(balances) == 2
        by_currency = {b.currency: b for b in balances}

        assert "USDT" in by_currency
        assert by_currency["USDT"].total == 50000.0
        assert by_currency["USDT"].locked == 5000.0
        assert by_currency["USDT"].free == 45000.0

        assert "BTC" in by_currency
        assert by_currency["BTC"].total == 0.5
        assert by_currency["BTC"].locked == 0.0
        assert by_currency["BTC"].free == 0.5


@pytest.mark.e2e
class TestPostgresE2EStateRestorer:

    def test_restore_full_state(self, pg_writer, mock_lookup):
        restorer = PostgresStateRestorer(
            strategy_name=STRATEGY_NAME,
            postgres_uri=POSTGRES_URI,
            table_prefix=TABLE_PREFIX,
        )
        state = restorer.restore_state()

        assert isinstance(state, RestoredState)
        assert isinstance(state.time, np.datetime64)

        # Positions
        assert len(state.positions) == 2
        btc_pos = None
        for inst, pos in state.positions.items():
            if inst.symbol == "BTCUSDT":
                btc_pos = pos
        assert btc_pos is not None
        assert btc_pos.quantity == 0.5

        # Signals
        assert len(state.instrument_to_signal_positions) == 1

        # Targets
        assert len(state.instrument_to_target_positions) == 1

        # Balances
        assert len(state.balances) == 2
        by_currency = {b.currency: b for b in state.balances}
        assert by_currency["USDT"].total == 50000.0
