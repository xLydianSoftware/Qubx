from unittest.mock import MagicMock, patch

from qubx.state.redis import RedisStatePersistence


def test_client_created_with_bounded_failure_defaults():
    with patch("qubx.state.redis.redis.from_url", return_value=MagicMock()) as from_url:
        RedisStatePersistence(redis_url="redis://localhost:6379/0", strategy_name="s")
    kwargs = from_url.call_args.kwargs
    assert kwargs["socket_connect_timeout"] == 2.0
    assert kwargs["socket_timeout"] == 5.0
    assert kwargs["socket_keepalive"] is True
    assert kwargs["health_check_interval"] == 30


def test_timeouts_overridable():
    with patch("qubx.state.redis.redis.from_url", return_value=MagicMock()) as from_url:
        RedisStatePersistence(redis_url="redis://localhost:6379/0", strategy_name="s", socket_timeout=1.5)
    assert from_url.call_args.kwargs["socket_timeout"] == 1.5
