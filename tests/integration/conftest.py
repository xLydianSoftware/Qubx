"""
Shared pytest fixtures for integration tests that depend on Redis.

Per-suite conftests under ``tests/integration/<suite>/`` override / extend this
with suite-specific fixtures (cleanup, mocks, sample data, etc.).
"""

import os
import socket
import time

import pytest
import redis as redis_lib


def is_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def is_redis_available(host: str = "localhost", port: int = 6379) -> bool:
    """Return True if a Redis server responds to PING at the given address."""
    try:
        r = redis_lib.from_url(f"redis://{host}:{port}/0")
        r.ping()
        return True
    except (redis_lib.ConnectionError, redis_lib.RedisError):
        return False


@pytest.fixture(scope="session")
def docker_ip() -> str:
    return "localhost"


@pytest.fixture(scope="session")
def docker_compose_file() -> str:
    return os.path.join(os.path.dirname(__file__), "docker-compose.yml")


@pytest.fixture(scope="session")
def redis_service(docker_ip, docker_compose_file):
    """
    Session-scoped Redis URL.

    Behavior:
    - In GitHub Actions, Redis is provided as a service container; we just
      return the URL.
    - Locally, reuse an already-running Redis on ``docker_ip:6379`` if present
      (e.g. a local dev container), otherwise start one via docker-compose and
      tear it down after the session.
    """
    redis_url = f"redis://{docker_ip}:6379/0"
    started_docker = False

    if not is_github_actions() and not is_redis_available(docker_ip, 6379):
        import subprocess

        subprocess.run(["docker", "compose", "-f", docker_compose_file, "up", "-d"], check=True)
        started_docker = True

        start = time.time()
        while time.time() - start < 30:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect((docker_ip, 6379))
                sock.close()
                break
            except (socket.error, socket.timeout):
                time.sleep(1)
        else:
            raise TimeoutError(f"Timed out waiting for redis on {docker_ip}:6379")
        time.sleep(1)

    yield redis_url

    if started_docker:
        import subprocess

        subprocess.run(["docker", "compose", "-f", docker_compose_file, "down"], check=True)
