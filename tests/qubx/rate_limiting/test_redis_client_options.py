"""Mirrors tests/qubx/state/test_redis_client_options.py for the rate-limit RedisBackend.

RedisBackend creates its redis client lazily, per event loop, inside
``_scripts_for_current_loop`` — so these tests must run inside a running loop.
"""

from unittest.mock import MagicMock, patch

import pytest

from qubx.rate_limiting.redis_backend import RedisBackend


@pytest.mark.asyncio
async def test_client_created_with_bounded_failure_defaults():
    with patch("redis.asyncio.from_url", return_value=MagicMock()) as from_url:
        RedisBackend("redis://x")._scripts_for_current_loop()
    kwargs = from_url.call_args.kwargs
    assert kwargs["socket_connect_timeout"] == 2.0
    assert kwargs["socket_timeout"] == 5.0
    assert kwargs["socket_keepalive"] is True
    assert kwargs["health_check_interval"] == 30
    assert kwargs["decode_responses"] is True
    assert kwargs["single_connection_client"] is True


@pytest.mark.asyncio
async def test_timeouts_overridable():
    with patch("redis.asyncio.from_url", return_value=MagicMock()) as from_url:
        RedisBackend("redis://x", socket_timeout=1.5)._scripts_for_current_loop()
    assert from_url.call_args.kwargs["socket_timeout"] == 1.5
