"""
Pytest configuration for the exporters integration tests.
"""

import os
import socket
import time

import pytest
import redis

from qubx.core.basics import AssetType, Instrument, MarketType
from tests.qubx.exporters.utils.mocks import MockAccountViewer


def is_github_actions():
    """Check if running in GitHub Actions environment."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


@pytest.fixture
def account_viewer():
    """Fixture for a mock account viewer."""
    return MockAccountViewer()


@pytest.fixture
def instruments():
    """Fixture for test instruments."""
    return [
        Instrument(
            symbol="BTC-USDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTCUSDT",
            tick_size=0.01,
            lot_size=0.00001,
            min_size=0.0001,
        ),
        Instrument(
            symbol="ETH-USDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETHUSDT",
            tick_size=0.01,
            lot_size=0.00001,
            min_size=0.0001,
        ),
    ]


@pytest.fixture(scope="session")
def docker_compose_file():
    """Get the docker-compose file path."""
    # Use the docker-compose.yml from the current directory
    return os.path.join(os.path.dirname(__file__), "docker-compose.yml")


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
    time.sleep(1)

    # Return the connection URL
    return f"redis://{docker_ip}:{public_port}/0"


@pytest.fixture
def clear_redis_streams(redis_service):
    """Fixture to clear Redis streams before each test."""
    r = redis.from_url(redis_service)

    # Clear all streams used in tests
    stream_keys = [
        "strategy:test_strategy:signals",
        "strategy:test_strategy:targets",
        "strategy:test_strategy:position_changes",
    ]

    for key in stream_keys:
        try:
            # Delete the stream if it exists
            r.delete(key)
        except Exception:
            pass

    yield

    # Clean up after tests
    for key in stream_keys:
        try:
            r.delete(key)
        except Exception:
            pass


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
