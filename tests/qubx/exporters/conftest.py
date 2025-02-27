"""
Pytest configuration for the exporters tests.
"""

import os
import socket
import time

import pytest


def is_github_actions():
    """Check if running in GitHub Actions environment."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


def pytest_configure(config):
    """Configure pytest for the exporters tests."""
    # Set the docker-compose file path if not in GitHub Actions
    if not is_github_actions():
        config.option.docker_compose_file = os.path.join(os.path.dirname(__file__), "docker-compose.yml")


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
