import threading
import time
from unittest.mock import MagicMock, patch

from qubx.exporters.redis_streams import RedisStreamsExporter


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _make_exporter(**kwargs):
    with patch("qubx.exporters.redis_streams.redis.from_url", return_value=MagicMock()) as from_url:
        exp = RedisStreamsExporter(redis_url="redis://localhost:6379/0", strategy_name="s", **kwargs)
    return exp, from_url


def test_client_has_bounded_failure_defaults():
    _, from_url = _make_exporter()
    kwargs = from_url.call_args.kwargs
    assert kwargs["socket_connect_timeout"] == 2.0
    assert kwargs["socket_timeout"] == 5.0
    assert kwargs["socket_keepalive"] is True


def test_stream_writes_preserve_fifo_order():
    exp, _ = _make_exporter()
    seen: list[int] = []
    for i in range(50):
        exp._worker.submit(seen.append, i)
    exp._worker.stop(flush_timeout_s=2.0)
    assert seen == list(range(50))


def test_queue_bounded_under_hung_backend():
    exp, _ = _make_exporter(max_queue=5)
    gate = threading.Event()
    started = threading.Event()

    def occupy() -> None:
        started.set()
        gate.wait(5.0)

    exp._worker.submit(occupy)  # occupy the worker
    assert _wait(started.is_set)  # ensure it's dequeued (in-flight) before the flood arrives
    t0 = time.monotonic()
    for i in range(50):
        exp._worker.submit(lambda: None)
    assert time.monotonic() - t0 < 0.5
    assert exp._worker.dropped >= 44
    gate.set()
    exp.stop()
