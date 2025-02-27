"""
Pytest configuration for the exporters tests.
"""

import os
import socket
import time
from pathlib import Path

import pytest

from tests.qubx.exporters.utils.mocks import MockAccountViewer


def is_github_actions():
    """Check if running in GitHub Actions environment."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


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


def pytest_configure(config):
    """Configure pytest for the exporters tests."""
    # Set the docker-compose file path if not in GitHub Actions
    if not is_github_actions():
        config.option.docker_compose_file = os.path.join(os.path.dirname(__file__), "docker-compose.yml")


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


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    """Get the docker-compose file path."""
    return getattr(pytestconfig.option, "docker_compose_file", None)


@pytest.fixture(scope="session")
def docker_ip():
    """Get the IP address to use for Docker containers."""
    # Use localhost for simplicity
    return "localhost"


@pytest.fixture(scope="session")
def docker_services(docker_ip, docker_compose_file):
    """Start the Docker services."""
    # If running in GitHub Actions, Redis is provided as a service
    if is_github_actions():
        yield DockerServices(docker_ip)
        return

    # Otherwise, start Redis using Docker Compose
    import subprocess

    # Start the services
    subprocess.run(["docker", "compose", "-f", docker_compose_file, "up", "-d"], check=True)

    # Yield to allow tests to run
    yield DockerServices(docker_ip)

    # Stop the services
    subprocess.run(["docker", "compose", "-f", docker_compose_file, "down"], check=True)


@pytest.fixture(scope="session")
def redis_service(docker_ip, docker_services):
    """Start a Redis container or use GitHub Actions service."""
    if is_github_actions():
        # In GitHub Actions, Redis is available at localhost:6379
        return "redis://localhost:6379/0"

    # For local development, start Redis using Docker Compose
    # Start the Redis service
    docker_services.start("redis")

    # Wait for the Redis service to be ready
    public_port = docker_services.wait_for_service("redis", 6379)

    # Return the connection URL
    return f"redis://{docker_ip}:{public_port}/0"


class DockerServices:
    """Helper class for Docker services."""

    def __init__(self, docker_ip):
        self.docker_ip = docker_ip

    def start(self, service_name):
        """Start a service."""
        # Service is already started by docker compose up
        pass

    def wait_for_service(self, service_name, port, timeout=30):
        """Wait for a service to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect((self.docker_ip, port))
                sock.close()
                return port
            except (socket.error, socket.timeout):
                time.sleep(1)

        raise TimeoutError(f"Timed out waiting for {service_name} on {self.docker_ip}:{port}")
