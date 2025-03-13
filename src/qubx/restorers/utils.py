"""
Utility functions for the restorers module.
"""

import re
from pathlib import Path
from typing import Optional


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
