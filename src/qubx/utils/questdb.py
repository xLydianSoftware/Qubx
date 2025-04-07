from typing import Any, Dict, List, Optional, Union

import pandas as pd
import psycopg as pg
from psycopg.sql import SQL, Composed


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
        dbname: Optional[str] = None,
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

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
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

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> int:
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
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
