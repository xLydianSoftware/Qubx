"""
The thread-blocking side of TokenBucketRateLimiter.
"""

import threading
import time

from qubx.utils.rate_limiter import TokenBucketRateLimiter


def test_threads_share_one_budget():
    """
    Eight threads taking twenty tokens from a bucket of 4 at 20/s must take about the time the
    refill needs, not the time one thread alone would take.
    """
    limiter = TokenBucketRateLimiter(capacity=4, refill_rate=20.0, name="t")
    done: list[float] = []

    def worker():
        for _ in range(2):
            limiter.acquire_blocking()
        done.append(time.monotonic())

    start = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = max(done) - start
    # - 16 tokens, 4 free, 12 at 20/s = 0.6s
    assert 0.4 < elapsed < 2.0, elapsed


def test_timeout_refuses_and_hands_the_reservation_back():
    limiter = TokenBucketRateLimiter(capacity=1, refill_rate=1.0, name="t")
    assert limiter.acquire_blocking() is True
    before = limiter.get_available_tokens()

    assert limiter.acquire_blocking(timeout=0.01) is False
    # - a refused call must not leave the bucket poorer than it found it
    assert limiter.get_available_tokens() >= before - 0.05


def test_draining_makes_the_next_caller_wait():
    limiter = TokenBucketRateLimiter(capacity=10, refill_rate=50.0, name="t")
    limiter.set_tokens(-5)

    start = time.monotonic()
    limiter.acquire_blocking()
    assert time.monotonic() - start >= 0.1
