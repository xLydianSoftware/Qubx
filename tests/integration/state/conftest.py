"""
Pytest configuration for state persistence integration tests.

Redis lifecycle (``redis_service``) is provided by ``tests/integration/conftest.py``.
"""

import pytest
import redis as redis_lib


@pytest.fixture
def clear_state_keys(redis_service):
    """Clear state keys before and after each test."""
    r = redis_lib.from_url(redis_service)

    for key in r.scan_iter("state:test_strategy:*"):
        r.delete(key)

    yield

    for key in r.scan_iter("state:test_strategy:*"):
        r.delete(key)
