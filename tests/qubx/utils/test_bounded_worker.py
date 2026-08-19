import threading
import time

from qubx.utils.threading import BoundedWorker


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_executes_in_fifo_order_and_stop_flushes():
    out: list[int] = []
    w = BoundedWorker("t1", maxlen=100)
    for i in range(20):
        w.submit(out.append, i)
    w.stop(flush_timeout_s=2.0)
    assert out == list(range(20))


def test_submit_never_blocks_and_drops_oldest_when_full():
    gate = threading.Event()
    started = threading.Event()
    out: list[int] = []

    def task(i: int) -> None:
        started.set()
        gate.wait(5.0)
        out.append(i)

    w = BoundedWorker("t2", maxlen=3)
    w.submit(task, 0)
    assert _wait(started.is_set)  # ensure item 0 is dequeued (in-flight) before the rest arrive
    t0 = time.monotonic()
    for i in range(1, 10):  # 1 in-flight (blocked on gate), 3 queued max, rest dropped-oldest
        w.submit(task, i)
    assert time.monotonic() - t0 < 0.5  # submit never blocked
    assert _wait(lambda: w.dropped >= 6)
    gate.set()
    w.stop(flush_timeout_s=2.0)
    # the in-flight item plus the LAST 3 queued survive; older ones were dropped
    assert out[0] == 0 and out[-3:] == [7, 8, 9]
    assert w.dropped == 6


def test_task_exception_does_not_kill_worker():
    out: list[str] = []
    w = BoundedWorker("t3", maxlen=10)
    w.submit(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    w.submit(out.append, "alive")
    w.stop(flush_timeout_s=2.0)
    assert out == ["alive"]


def test_submit_after_stop_is_noop():
    w = BoundedWorker("t4", maxlen=10)
    w.stop()
    w.submit(lambda: None)  # must not raise
    assert w.queued == 0
