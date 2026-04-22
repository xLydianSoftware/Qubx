"""
Pytest configuration for Redis-backed rate limiting integration tests.
"""

import os
import socket
import time

import pytest
import redis as redis_lib


def is_github_actions():
    return os.environ.get("GITHUB_ACTIONS") == "true"


def is_redis_available(host: str = "localhost", port: int = 6379) -> bool:
    try:
        r = redis_lib.from_url(f"redis://{host}:{port}/0")
        r.ping()
        return True
    except (redis_lib.ConnectionError, redis_lib.RedisError):
        return False


@pytest.fixture(scope="session")
def docker_compose_file():
    return os.path.join(os.path.dirname(__file__), "docker-compose.yml")


@pytest.fixture(scope="session")
def docker_ip():
    return "localhost"


@pytest.fixture(scope="session")
def redis_service(docker_ip, docker_compose_file):
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


@pytest.fixture
def clear_rate_limit_keys(redis_service):
    r = redis_lib.from_url(redis_service)
    for key in r.scan_iter("qubx:test:rl:*"):
        r.delete(key)
    yield
    for key in r.scan_iter("qubx:test:rl:*"):
        r.delete(key)
