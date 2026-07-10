"""
Utility functions for the restorers module.
"""

import re
from pathlib import Path
from typing import Optional

from psycopg import sql


def find_latest_run_folder(base_dir: str | Path) -> Optional[Path]:
    """
    Find the latest run folder in the given directory.

    The function looks for folders matching the pattern 'run_YYYYMMDDHHMMSS'
    and returns the path to the most recent one.

    Args:
        base_dir: The base directory to search in.

    Returns:
        The path to the latest run folder, or None if no run folders are found.
    """
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return None

    # Find all directories matching the run pattern
    run_pattern = re.compile(r"run_\d{14}")
    run_folders = []

    for item in base_path.iterdir():
        if item.is_dir() and run_pattern.match(item.name):
            run_folders.append(item)

    if not run_folders:
        return None

    # Sort by folder name (which contains the timestamp)
    # The latest run will have the highest timestamp
    latest_run = sorted(run_folders, key=lambda x: x.name, reverse=True)[0]

    return latest_run


def latest_run_id(cur, table: str, strategy_name: str, since) -> str | None:
    """Return the most recent run_id for *strategy_name* in *table*, within the
    *since* lower bound. None if the table has no matching rows."""
    cur.execute(
        sql.SQL(
            "SELECT run_id FROM {table} WHERE strategy_name = %s AND timestamp >= %s "
            "ORDER BY timestamp DESC LIMIT 1"
        ).format(table=sql.Identifier(table)),
        (strategy_name, since),
    )
    row = cur.fetchone()
    return row[0] if row else None


def canonical_run_id(cur, tables: list[str], strategy_name: str, since) -> str | None:
    """Return the run_id of the most recent row across all *tables* (the previous
    run), within the *since* lower bound. None if no tables or no matching rows."""
    if not tables:
        return None
    parts, params = [], []
    for table in tables:
        parts.append(
            sql.SQL(
                "SELECT run_id, timestamp FROM {table} WHERE strategy_name = %s AND timestamp >= %s"
            ).format(table=sql.Identifier(table))
        )
        params.extend([strategy_name, since])
    query = sql.SQL("SELECT run_id FROM ({union}) u ORDER BY timestamp DESC LIMIT 1").format(
        union=sql.SQL(" UNION ALL ").join(parts)
    )
    cur.execute(query, params)
    row = cur.fetchone()
    return row[0] if row else None


def mongo_latest_run_id(collection, match: dict) -> str | None:
    """Return the most recent run_id in *collection* matching *match* (which must
    already carry strategy_name/log_type and any lookback bound). None if no docs."""
    doc = next(
        collection.find(match, {"run_id": 1, "timestamp": 1}).sort("timestamp", -1).limit(1),
        None,
    )
    return doc["run_id"] if doc else None


def mongo_canonical_run_id(sources: list[tuple]) -> str | None:
    """Return the run_id of the most recent doc across all *sources* (the previous
    run), shared across log types. Each source is a ``(collection, match)`` pair.
    None if no source has a matching doc."""
    best_ts, best_run = None, None
    for collection, match in sources:
        doc = next(
            collection.find(match, {"run_id": 1, "timestamp": 1}).sort("timestamp", -1).limit(1),
            None,
        )
        if doc is not None and (best_ts is None or doc["timestamp"] > best_ts):
            best_ts, best_run = doc["timestamp"], doc["run_id"]
    return best_run
