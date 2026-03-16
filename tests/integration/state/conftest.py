"""
Pytest configuration for state persistence integration tests.
"""

import os
import socket
import time

import pytest
import redis as redis_lib


def is_github_actions():
    """Check if running in GitHub Actions environment."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


def is_redis_available(host: str = "localhost", port: int = 6379) -> bool:
    """Check if Redis is already running and accessible."""
    try:
        r = redis_lib.from_url(f"redis://{host}:{port}/0")
        r.ping()
        return True
    except (redis_lib.ConnectionError, redis_lib.RedisError):
        return False


@pytest.fixture(scope="session")
def docker_compose_file():
    """Get the docker-compose file path."""
    return os.path.join(os.path.dirname(__file__), "docker-compose.yml")


@pytest.fixture(scope="session")
def docker_ip():
    """Get the IP address to use for Docker containers."""
    return "localhost"


class DockerServices:
    """Helper class for Docker services."""

    def __init__(self, docker_ip):
        self.docker_ip = docker_ip

    def start(self, service_name):
        """Start a service."""
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


@pytest.fixture(scope="session")
def redis_service(docker_ip, docker_compose_file):
    """Get Redis connection URL, reusing existing Redis if available."""
    redis_url = f"redis://{docker_ip}:6379/0"
    started_docker = False

    # Check if Redis is already running (CI, existing container, or local install)
    if not is_github_actions() and not is_redis_available(docker_ip, 6379):
        # Start Redis via docker-compose
        import subprocess

        subprocess.run(["docker", "compose", "-f", docker_compose_file, "up", "-d"], check=True)
        started_docker = True

        # Wait for Redis to be ready
        services = DockerServices(docker_ip)
        services.wait_for_service("redis", 6379)
        time.sleep(1)

    yield redis_url

    # Cleanup only if we started docker
    if started_docker:
        import subprocess

        subprocess.run(["docker", "compose", "-f", docker_compose_file, "down"], check=True)


@pytest.fixture
def clear_state_keys(redis_service):
    """Clear state keys before and after each test."""
    r = redis_lib.from_url(redis_service)

    # Clear before test
    for key in r.scan_iter("state:test_strategy:*"):
        r.delete(key)

    yield

    # Clear after test
    for key in r.scan_iter("state:test_strategy:*"):
        r.delete(key)
