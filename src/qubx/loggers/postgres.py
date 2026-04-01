from multiprocessing.pool import ThreadPool
from typing import Any

from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from qubx import logger
from qubx.core.loggers import LogsWriter

# Column definitions for each log type: (column_name, sql_type)
_TABLE_SCHEMAS: dict[str, list[tuple[str, str]]] = {
    "positions": [
        ("timestamp", "TIMESTAMPTZ NOT NULL"),
        ("symbol", "TEXT NOT NULL"),
        ("exchange", "TEXT NOT NULL"),
        ("market_type", "TEXT NOT NULL"),
        ("pnl_quoted", "DOUBLE PRECISION"),
        ("funding_pnl_quoted", "DOUBLE PRECISION"),
        ("realized_pnl_quoted", "DOUBLE PRECISION"),
        ("quantity", "DOUBLE PRECISION"),
        ("notional", "DOUBLE PRECISION"),
        ("avg_position_price", "DOUBLE PRECISION"),
        ("current_price", "DOUBLE PRECISION"),
        ("market_value_quoted", "DOUBLE PRECISION"),
        ("commissions_quoted", "DOUBLE PRECISION"),
    ],
    "portfolio": [
        ("timestamp", "TIMESTAMPTZ NOT NULL"),
        ("symbol", "TEXT NOT NULL"),
        ("exchange", "TEXT NOT NULL"),
        ("market_type", "TEXT NOT NULL"),
        ("pnl_quoted", "DOUBLE PRECISION"),
        ("quantity", "DOUBLE PRECISION"),
        ("realized_pnl_quoted", "DOUBLE PRECISION"),
        ("avg_position_price", "DOUBLE PRECISION"),
        ("current_price", "DOUBLE PRECISION"),
        ("market_value_quoted", "DOUBLE PRECISION"),
        ("exchange_time", "TIMESTAMPTZ"),
        ("commissions_quoted", "DOUBLE PRECISION"),
        ("cumulative_funding", "DOUBLE PRECISION"),
    ],
    "executions": [
        ("timestamp", "TIMESTAMPTZ NOT NULL"),
        ("symbol", "TEXT NOT NULL"),
        ("exchange", "TEXT NOT NULL"),
        ("market_type", "TEXT NOT NULL"),
        ("side", "TEXT NOT NULL"),
        ("filled_qty", "DOUBLE PRECISION"),
        ("price", "DOUBLE PRECISION"),
        ("commissions", "DOUBLE PRECISION"),
        ("commissions_quoted", "TEXT"),
        ("order_id", "TEXT"),
        ("order_type", "TEXT"),
    ],
    "signals": [
        ("timestamp", "TIMESTAMPTZ NOT NULL"),
        ("symbol", "TEXT NOT NULL"),
        ("exchange", "TEXT NOT NULL"),
        ("market_type", "TEXT NOT NULL"),
        ("signal", "TEXT"),
        ("reference_price", "DOUBLE PRECISION"),
        ("price", "DOUBLE PRECISION"),
        ("take", "DOUBLE PRECISION"),
        ("stop", "DOUBLE PRECISION"),
        ("group_name", "TEXT"),
        ("comment", "TEXT"),
        ("service", "BOOLEAN"),
        ("options", "JSONB"),
    ],
    "targets": [
        ("timestamp", "TIMESTAMPTZ NOT NULL"),
        ("symbol", "TEXT NOT NULL"),
        ("exchange", "TEXT NOT NULL"),
        ("market_type", "TEXT NOT NULL"),
        ("target_position", "DOUBLE PRECISION"),
        ("entry_price", "DOUBLE PRECISION"),
        ("take_price", "DOUBLE PRECISION"),
        ("stop_price", "DOUBLE PRECISION"),
        ("options", "JSONB"),
    ],
    "balance": [
        ("timestamp", "TIMESTAMPTZ NOT NULL"),
        ("exchange", "TEXT NOT NULL"),
        ("currency", "TEXT NOT NULL"),
        ("total", "DOUBLE PRECISION"),
        ("locked", "DOUBLE PRECISION"),
    ],
}

# Mapping from log data dict keys to column names (only where they differ)
_COLUMN_RENAMES: dict[str, dict[str, str]] = {
    "signals": {"group": "group_name"},
}

# Columns that use JSONB type and need Jsonb() wrapper
_JSONB_COLUMNS: set[str] = {
    name for schema in _TABLE_SCHEMAS.values() for name, col_type in schema if col_type == "JSONB"
}


def _table_name(prefix: str, log_type: str) -> str:
    return f"{prefix}_{log_type}"


class PostgresLogsWriter(LogsWriter):
    """
    PostgreSQL implementation of LogsWriter interface.
    Writes log data to typed PostgreSQL tables asynchronously using psycopg v3 and a connection pool.
    """

    def __init__(
        self,
        account_id: str,
        strategy_id: str,
        run_id: str,
        postgres_uri: str = "postgresql://localhost:5432/qubx_logs",
        table_prefix: str = "qubx_logs",
        pool_size: int = 4,
    ) -> None:
        super().__init__(account_id, strategy_id, run_id)
        self.table_prefix = table_prefix
        self._pool = ConnectionPool(postgres_uri, min_size=1, max_size=pool_size)
        self._thread_pool = ThreadPool(pool_size)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for log_type, columns in _TABLE_SCHEMAS.items():
                    tname = _table_name(self.table_prefix, log_type)
                    col_defs = ",\n    ".join(f"{name} {col_type}" for name, col_type in columns)
                    cur.execute(
                        sql.SQL(
                            """
                            CREATE TABLE IF NOT EXISTS {table} (
                                id BIGSERIAL PRIMARY KEY,
                                run_id TEXT NOT NULL,
                                account_id TEXT NOT NULL,
                                strategy_name TEXT NOT NULL,
                                created_at TIMESTAMPTZ DEFAULT NOW(),
                                {columns}
                            )
                            """
                        ).format(
                            table=sql.Identifier(tname),
                            columns=sql.SQL(col_defs),
                        )
                    )
                    # Index on strategy_name + timestamp for common queries
                    idx_name = f"idx_{tname}_strategy_ts"
                    cur.execute(
                        sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} (strategy_name, timestamp)").format(
                            idx=sql.Identifier(idx_name),
                            table=sql.Identifier(tname),
                        )
                    )
            conn.commit()

    def _remap_keys(self, log_type: str, row: dict[str, Any]) -> dict[str, Any]:
        renames = _COLUMN_RENAMES.get(log_type)
        if not renames:
            return row
        return {renames.get(k, k): v for k, v in row.items()}

    def _do_write(self, log_type: str, data: list[dict[str, Any]]) -> None:
        tname = _table_name(self.table_prefix, log_type)
        schema_cols = [name for name, _ in _TABLE_SCHEMAS[log_type]]
        meta_cols = ["run_id", "account_id", "strategy_name"]
        all_cols = meta_cols + schema_cols

        rows = []
        for raw_row in data:
            row = self._remap_keys(log_type, raw_row)
            values = [self.run_id, self.account_id, self.strategy_id]
            for col in schema_cols:
                val = row.get(col)
                # Convert numpy datetime / stringified timestamps
                if col in ("timestamp", "exchange_time") and val is not None:
                    val = str(val)
                # Convert signal value to string
                if col == "signal" and val is not None:
                    val = str(val)
                # Wrap dicts in Jsonb for JSONB columns
                if col in _JSONB_COLUMNS and isinstance(val, dict):
                    val = Jsonb(val)
                values.append(val)
            rows.append(tuple(values))

        col_ids = sql.SQL(", ").join(sql.Identifier(c) for c in all_cols)
        placeholders = sql.SQL(", ").join(sql.Placeholder() * len(all_cols))
        query = sql.SQL("INSERT INTO {table} ({cols}) VALUES ({vals})").format(
            table=sql.Identifier(tname),
            cols=col_ids,
            vals=placeholders,
        )

        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(query, rows)
                conn.commit()
        except Exception:
            logger.exception(f"PostgresLogsWriter: failed to write {len(rows)} rows to {tname}")

    def write_data(self, log_type: str, data: list[dict[str, Any]]) -> None:
        if data:
            self._thread_pool.apply_async(self._do_write, (log_type, data))

    def flush_data(self) -> None:
        pass

    def close(self) -> None:
        self._thread_pool.close()
        self._thread_pool.join()
        self._pool.close()
