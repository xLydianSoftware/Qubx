"""
Pytest configuration for the exporters tests.
"""

import os
from pathlib import Path

import pytest

from tests.qubx.exporters.utils.mocks import MockAccountViewer


def load_env_file(file_path):
    """Load environment variables from a file."""
    if not Path(file_path).exists():
        return {}

    env_vars = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            env_vars[key.strip()] = value.strip().strip("\"'")
    return env_vars


@pytest.fixture
def account_viewer():
    """Fixture for a mock account viewer."""
    return MockAccountViewer()


@pytest.fixture
def slack_webhook_url():
    """
    Get the Slack webhook URL for integration tests.

    First checks if SLACK_WEBHOOK_URL is set in environment variables.
    If not, tries to load it from .env.integration file in the project root.

    Returns:
        str: The Slack webhook URL or None if not found
    """
    # First check environment variables
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook_url:
        return webhook_url

    # If not found, try to load from .env.integration file
    # Find the project root (assuming it's where the .env.integration file is)
    current_dir = Path(__file__).parent
    while current_dir.name != "Qubx" and current_dir.parent != current_dir:
        current_dir = current_dir.parent

    # Go up one more level to get to the project root
    env_file = current_dir / ".env.integration"

    # Load environment variables from the file
    env_vars = load_env_file(env_file)
    return env_vars.get("SLACK_WEBHOOK_URL")
