# tests/qubx/rate_limiting/test_resilient_integration.py
"""Integration: ResilientRateLimitBackend vs a real redis that gets frozen mid-run.

``docker pause`` freezes the server process WITHOUT closing TCP connections —
the closest reproduction of the 2026-08-19 half-open-socket incident.
Requires docker; runs only under `-m integration`, and skips (rather than
erroring) when docker itself is unavailable.
"""

import shutil
import subprocess
import time
import uuid

import pytest

CONTAINER = f"qubx-rl-itest-redis-{uuid.uuid4().hex[:8]}"
PORT = 63791


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "ps"], capture_output=True).returncode == 0


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _docker_available(), reason="docker is not available in this environment"),
]


def _wait_until_ready(deadline_s: float = 10.0) -> None:
    """Poll `redis-cli ping` instead of a fixed sleep — a loaded host or a cold image
    can take longer than a guessed constant to accept connections."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        result = subprocess.run(["docker", "exec", CONTAINER, "redis-cli", "ping"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip() == "PONG":
            return
        time.sleep(0.1)
    raise RuntimeError(f"redis container {CONTAINER} did not become ready within {deadline_s}s")


@pytest.fixture(scope="module")
def redis_container():
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", CONTAINER, "-p", f"{PORT}:6379", "redis:7-alpine"],
        check=True,
        capture_output=True,
    )
    _wait_until_ready()
    yield CONTAINER
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)  # works on paused containers too


def _pause():
    subprocess.run(["docker", "pause", CONTAINER], check=True, capture_output=True)


def _unpause():
    subprocess.run(["docker", "unpause", CONTAINER], check=True, capture_output=True)


@pytest.mark.asyncio
async def test_resilient_backend_survives_a_redis_freeze(redis_container):
    import redis.asyncio as aioredis

    from qubx.rate_limiting.redis_backend import RedisBackend
    from qubx.rate_limiting.resilient import ResilientRateLimitBackend

    url = f"redis://localhost:{PORT}/0"
    backend = ResilientRateLimitBackend(RedisBackend(url, socket_timeout=5.0, socket_connect_timeout=2.0))
    control = aioredis.from_url(url, decode_responses=True)

    try:
        # 1. Healthy primary: acquire succeeds via redis, and a control client can see the key.
        waited = await backend.acquire("ratelimit:itest", 1.0, 10.0, 5.0)
        assert waited == 0.0
        assert await control.exists("ratelimit:itest") == 1
        assert backend._broken is False

        _pause()
        try:
            # 2. Frozen primary: the socket timeout bounds the hang, and the breaker opens —
            #    the call is served by the local fallback, well under the 15-min incident hang.
            t0 = time.monotonic()
            await backend.acquire("ratelimit:itest", 1.0, 10.0, 5.0)
            elapsed = time.monotonic() - t0
            assert elapsed < 8.0, f"acquire took {elapsed:.2f}s while redis was paused"
            assert backend._broken is True

            # 3. Breaker open: the next call must not pay the timeout tax again.
            t1 = time.monotonic()
            await backend.acquire("ratelimit:itest", 1.0, 10.0, 5.0)
            assert time.monotonic() - t1 < 0.5
        finally:
            _unpause()

        # 4. Recovered primary: force the cooldown to expire and confirm the probe routes
        #    back to redis (control client observes last_refill advance).
        before = await control.hget("ratelimit:itest", "last_refill")
        backend._broken_until = 0.0
        await backend.acquire("ratelimit:itest", 1.0, 10.0, 5.0)
        assert backend._broken is False
        after = await control.hget("ratelimit:itest", "last_refill")
        assert after is not None and after != before
    finally:
        await backend.close()
        await control.aclose()
