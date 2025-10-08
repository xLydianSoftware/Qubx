from typing import Any

import pandas as pd
import psycopg as pg
from questdb.ingress import IngressError, Sender


class QuestDBClient:
    """
    A helper class for interacting with QuestDB.
    """

    def __init__(
        self,
        host: str = "nebula",
        port: int = 8812,
        user: str = "admin",
        password: str = "quest",
        dbname: str | None = None,
    ):
        """
        Initialize the QuestDB client.

        Args:
            host: The hostname of the QuestDB server
            port: The port number for QuestDB PostgreSQL interface
            user: The username for authentication
            password: The password for authentication
            dbname: Optional database name
        """
        conn_str = f"user={user} password={password} host={host} port={port}"
        if dbname:
            conn_str += f" dbname={dbname}"

        self.conn_str = conn_str

    @property
    def http_connection_string(self) -> str:
        """Get HTTP connection string for QuestDB ingress."""
        # Extract host from conn_str
        host = "localhost"  # default
        for part in self.conn_str.split():
            if part.startswith("host="):
                host = part.split("=", 1)[1]
                break
        return f"http::addr={host}:9000;"

    def query(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a pandas DataFrame.

        Args:
            query: The SQL query to execute
            params: Optional parameters for the query

        Returns:
            A pandas DataFrame containing the query results
        """
        with pg.connect(self.conn_str) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:  # Check if the query returns data
                    column_names = [desc.name for desc in cursor.description]
                    records = cursor.fetchall()
                    return pd.DataFrame(records, columns=column_names)
                return pd.DataFrame()

    def execute(self, query: str, params: dict[str, Any] | None = None) -> int:
        """
        Execute a SQL statement that doesn't return data (INSERT, UPDATE, etc.).

        Args:
            query: The SQL query to execute
            params: Optional parameters for the query

        Returns:
            The number of rows affected
        """
        with pg.connect(self.conn_str) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)  # type: ignore
                conn.commit()
                return cursor.rowcount

    def insert_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Insert DataFrame into QuestDB table using the HTTP ingress API.

        Args:
            df: DataFrame to insert. Must have a timestamp index or 'timestamp' column
            table_name: Name of the target QuestDB table

        Raises:
            IngressError: If ingestion fails
            ValueError: If DataFrame doesn't have proper timestamp information
        """
        try:
            with Sender.from_conf(self.http_connection_string) as sender:
                # Check if DataFrame has a proper timestamp index
                if isinstance(df.index, pd.DatetimeIndex) and df.index.name:
                    # Use the timestamp index directly
                    sender.dataframe(df.reset_index(), table_name=table_name, at=df.index.name)
                elif "timestamp" in df.columns:
                    # If timestamp is a column, use it directly
                    sender.dataframe(df, table_name=table_name, at="timestamp")
                else:
                    # Try to use index name if it exists
                    if df.index.name:
                        sender.dataframe(df.reset_index(), table_name=table_name, at=df.index.name)
                    else:
                        raise ValueError("DataFrame must have either a named timestamp index or a 'timestamp' column")
                sender.flush()
        except IngressError as e:
            raise IngressError(f"Failed to insert DataFrame into {table_name}: {e}")

    @staticmethod
    def get_table_name(exchange: str, market: str, symbol: str | None, table_type: str) -> str:
        """
        Generate table name following the exchange.market.symbol.type pattern.

        Args:
            exchange: Exchange name (e.g., 'binance')
            market: Market type (e.g., 'umfutures', 'spot')
            symbol: Symbol name (e.g., 'btcusdt'), can be None for market-wide tables
            table_type: Type of data (e.g., 'candle_1m', 'trade', 'orderbook')

        Returns:
            Formatted table name
        """
        parts = [exchange.lower(), market.lower()]
        if symbol:
            parts.append(symbol.lower())
        parts.append(table_type.lower())
        return ".".join(parts)

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
