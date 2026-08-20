# tests/qubx/state/test_safe_integration.py
"""Integration: SafeStatePersistence vs a real redis that gets frozen mid-run.

``docker pause`` freezes the server process WITHOUT closing TCP connections —
the closest reproduction of the 2026-08-19 half-open-socket incident.
Requires docker; runs only under `-m integration`.
"""
import json
import subprocess
import time
import uuid

import pytest

pytestmark = pytest.mark.integration

CONTAINER = f"qubx-test-redis-{uuid.uuid4().hex[:8]}"
PORT = 63790


@pytest.fixture(scope="module")
def redis_container():
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", CONTAINER, "-p", f"{PORT}:6379", "redis:7-alpine"],
        check=True, capture_output=True,
    )
    time.sleep(1.0)
    yield CONTAINER
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


def _pause():
    subprocess.run(["docker", "pause", CONTAINER], check=True, capture_output=True)


def _unpause():
    subprocess.run(["docker", "unpause", CONTAINER], check=True, capture_output=True)


def test_event_loop_liveness_through_redis_freeze(redis_container):
    from qubx.state.redis import RedisStatePersistence
    from qubx.state.safe import SafeStatePersistence

    backend = RedisStatePersistence(
        redis_url=f"redis://localhost:{PORT}/0", strategy_name="itest", socket_timeout=1.0, socket_connect_timeout=1.0
    )
    sp = SafeStatePersistence(backend, retry_backoff_s=(0.5, 1.0))
    sp.validate_startup(deadline_s=10.0)

    sp.save("k", {"phase": 1})
    deadline = time.monotonic() + 5.0
    while sp.last_success_age() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert sp.last_success_age() is not None

    _pause()
    try:
        t0 = time.monotonic()
        sp.save("k", {"phase": 2})           # must return instantly despite frozen server
        assert time.monotonic() - t0 < 0.05
        assert sp.load("k") == {"phase": 2}  # read-your-writes while frozen
        time.sleep(3.0)
        assert sp.last_success_age() > 2.0   # staleness visibly grows
    finally:
        _unpause()

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if json.loads(backend._redis.get("state:itest:k") or "null") == {"phase": 2}:
            break
        time.sleep(0.2)
    else:
        pytest.fail("pending write was not flushed after recovery")
    sp.stop()


def test_cold_start_against_frozen_redis_fails_fast(redis_container):
    from qubx.core.exceptions import StatePersistenceUnavailable
    from qubx.state.redis import RedisStatePersistence
    from qubx.state.safe import SafeStatePersistence

    _pause()
    try:
        backend = RedisStatePersistence(
            redis_url=f"redis://localhost:{PORT}/0", strategy_name="itest2",
            socket_timeout=1.0, socket_connect_timeout=1.0,
        )
        sp = SafeStatePersistence(backend)
        t0 = time.monotonic()
        with pytest.raises(StatePersistenceUnavailable):
            sp.validate_startup(deadline_s=5.0)
        assert time.monotonic() - t0 < 15.0  # bounded, not TCP-retransmission minutes
        sp.stop()
    finally:
        _unpause()
