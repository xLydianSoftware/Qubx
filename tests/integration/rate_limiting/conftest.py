"""
Pytest configuration for Redis-backed rate limiting integration tests.

Redis lifecycle (``redis_service``) is provided by ``tests/integration/conftest.py``.
"""

import pytest
import redis as redis_lib


@pytest.fixture
def clear_rate_limit_keys(redis_service):
    r = redis_lib.from_url(redis_service)
    for key in r.scan_iter("qubx:test:rl:*"):
        r.delete(key)
    yield
    for key in r.scan_iter("qubx:test:rl:*"):
        r.delete(key)
