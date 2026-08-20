"""Regression tests for the 2026-07-28 ccxt connector deadlock.

Every test here uses a REAL background event loop: the bug is a thread/loop interaction and mocks
cannot reproduce it. Every blocking call is run behind a hard deadline so a regression fails the
test instead of hanging CI forever.
"""

import asyncio
import concurrent.futures
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from qubx.connectors.ccxt.connection_manager import ConnectionManager
from qubx.connectors.ccxt.subscription_manager import SubscriptionManager
from qubx.utils.misc import BackgroundEventLoop


def run_with_deadline(fn, timeout: float) -> float:
    """Run ``fn`` on a daemon thread and fail the test if it does not return in ``timeout``."""
    outcome: dict[str, BaseException] = {}

    def _run() -> None:
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 - re-raised on the test thread below
            outcome["error"] = e

    thread = threading.Thread(target=_run, daemon=True, name="deadline-runner")
    started = time.monotonic()
    thread.start()
    thread.join(timeout)
    elapsed = time.monotonic() - started

    if thread.is_alive():
        pytest.fail(f"call did not return within {timeout}s - unbounded wait regression")
    if "error" in outcome:
        raise outcome["error"]
    return elapsed


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


async def _hang_forever() -> None:
    await asyncio.Event().wait()


async def _ping() -> str:
    return "pong"


def _drain(loop: BackgroundEventLoop) -> None:
    """Cancel whatever is still parked on the loop and stop it without ever blocking teardown."""

    async def cancel_all() -> None:
        current = asyncio.current_task()
        for task in asyncio.all_tasks():
            if task is not current:
                task.cancel()

    try:
        loop.submit(cancel_all()).result(timeout=2.0)
        time.sleep(0.05)
    except Exception:
        pass
    # BackgroundEventLoop.stop() joins its thread unbounded, and a loop these tests deliberately
    # wedge never runs call_soon_threadsafe - stopping it off-thread keeps a regression a failing
    # test rather than a hung CI job. The loop thread is a daemon, so leaving it is harmless.
    stopper = threading.Thread(target=loop.stop, daemon=True, name="loop-stopper")
    stopper.start()
    stopper.join(2.0)


@pytest.fixture
def bg_loop():
    loop = BackgroundEventLoop("DeadlockTestLoop")
    yield loop
    _drain(loop)


@pytest.fixture
def exchange_manager(bg_loop):
    return SimpleNamespace(exchange=SimpleNamespace(asyncio_loop=bg_loop.loop), rate_limiter=None)


def make_manager(exchange_manager, **kwargs) -> ConnectionManager:
    params = dict(
        exchange_id="TEST.F",
        exchange_manager=exchange_manager,
        subscription_manager=SubscriptionManager(),
        cleanup_timeout=0.3,
    )
    params.update(kwargs)
    return ConnectionManager(**params)  # type: ignore[arg-type]


@pytest.fixture
def manager(exchange_manager):
    return make_manager(exchange_manager)


class TestWaitIsBounded:
    def test_future_from_run_coroutine_threadsafe_is_never_running(self, bg_loop):
        """Documents the semantics the old `while future.running()` poll got wrong."""
        future = bg_loop.submit(_hang_forever())
        time.sleep(0.1)

        assert future.running() is False
        assert future.done() is False

        future.cancel()

    def test_wait_is_bounded_when_coroutine_never_completes(self, manager, bg_loop):
        future = bg_loop.submit(_hang_forever())
        time.sleep(0.05)

        elapsed = run_with_deadline(lambda: manager._wait(future, "hanging coroutine"), 2.0)

        assert elapsed < 2.0
        assert wait_until(future.cancelled)

    def test_wait_swallows_expected_cleanup_errors(self, manager, bg_loop):
        async def raising() -> None:
            raise RuntimeError("UnsubscribeError-like failure")

        future = bg_loop.submit(raising())

        # - must not propagate: UnsubscribeError and friends are expected during teardown
        run_with_deadline(lambda: manager._wait(future, "failing unsubscriber"), 2.0)

    def test_wait_returns_immediately_for_a_cancelled_future(self, manager, bg_loop):
        future = bg_loop.submit(_hang_forever())
        time.sleep(0.05)
        future.cancel()

        elapsed = run_with_deadline(lambda: manager._wait(future, "cancelled stream"), 2.0)

        assert elapsed < 0.3


class TestStopStreamIsBounded:
    def test_stop_stream_wait_true_returns_when_unsubscriber_hangs(self, manager, bg_loop):
        stream_name = "orderbook(0, 1):47:hanging"
        manager.enable_stream(stream_name)
        manager.register_stream_future(stream_name, bg_loop.submit(_hang_forever()))
        manager.set_stream_unsubscriber(stream_name, _hang_forever)

        elapsed = run_with_deadline(lambda: manager.stop_stream(stream_name, wait=True), 3.0)

        assert elapsed < 3.0
        assert manager.is_stream_enabled(stream_name) is False
        assert manager.get_stream_future(stream_name) is None
        assert manager.get_stream_unsubscriber(stream_name) is None

    def test_stop_stream_clears_registry_even_when_the_wait_times_out(self, manager, bg_loop):
        stream_name = "orderbook(0, 1):48:hanging"
        manager.enable_stream(stream_name)
        manager.register_stream_future(stream_name, bg_loop.submit(_hang_forever()))
        manager.set_stream_unsubscriber(stream_name, _hang_forever)

        run_with_deadline(lambda: manager.stop_stream(stream_name, wait=True), 3.0)

        assert stream_name not in manager._is_stream_enabled
        assert stream_name not in manager._stream_to_coro
        assert stream_name not in manager._stream_to_unsubscriber

    def test_registry_is_torn_down_before_the_blocking_waits(self, manager, bg_loop):
        """Teardown must precede the waits, or a timeout leaves a half-removed stream behind.

        Observed from inside the unsubscriber, i.e. *while* stop_stream is still blocked - checking
        after it returns passes under either ordering.
        """
        stream_name = "orderbook(0, 1):48:ordering"
        seen: dict[str, bool] = {}

        async def observing_unsubscriber() -> None:
            seen["coro"] = stream_name in manager._stream_to_coro
            seen["unsub"] = stream_name in manager._stream_to_unsubscriber
            seen["wanted"] = stream_name in manager._is_stream_enabled
            await asyncio.Event().wait()

        manager.enable_stream(stream_name)
        manager.register_stream_future(stream_name, bg_loop.submit(_hang_forever()))
        manager.set_stream_unsubscriber(stream_name, observing_unsubscriber)

        run_with_deadline(lambda: manager.stop_stream(stream_name, wait=True), 3.0)

        assert wait_until(lambda: "coro" in seen)
        assert seen == {"coro": False, "unsub": False, "wanted": False}

    def test_stop_stream_from_loop_thread_does_not_deadlock(self, exchange_manager, bg_loop):
        """Reproduction of the 00:02:28 self-deadlock that froze okx-am-agg for 18 days.

        Production-ish cleanup_timeout on purpose: with a short one an un-degraded path still
        returns quickly and the test would pass while the exchange loop is frozen for seconds.
        """
        manager = make_manager(exchange_manager, cleanup_timeout=5.0)
        stream_name = "orderbook(0, 1):47:self-deadlock"
        manager.enable_stream(stream_name)
        manager.register_stream_future(stream_name, bg_loop.submit(_hang_forever()))
        manager.set_stream_unsubscriber(stream_name, _hang_forever)

        elapsed: dict[str, float] = {}

        async def stop_from_the_loop() -> None:
            started = time.monotonic()
            manager.stop_stream(stream_name)  # wait=True, the production default
            elapsed["seconds"] = time.monotonic() - started

        with patch("qubx.connectors.ccxt.connection_manager.logger", MagicMock()) as log:
            stopping = bg_loop.submit(stop_from_the_loop())
            # - the loop must stay responsive *while* stop_stream runs on it: on the broken version
            #   it is parked on itself and this ping never gets a turn
            assert bg_loop.submit(_ping()).result(timeout=2.0) == "pong"
            stopping.result(timeout=5.0)
            degraded = " ".join(str(c) for c in log.info.call_args_list)

        assert elapsed["seconds"] < 1.0
        assert "degrading to non-blocking cleanup" in degraded
        # - and the unsubscriber was still scheduled, fire-and-forget
        assert manager.get_stream_unsubscriber(stream_name) is None

    def test_stop_stream_from_loop_thread_never_raises(self, exchange_manager, bg_loop):
        """Raising would propagate into callers that treat it as a fault and stop their poller."""
        manager = make_manager(exchange_manager, cleanup_timeout=5.0)
        stream_name = "quote:3:no-raise"
        manager.enable_stream(stream_name)
        manager.register_stream_future(stream_name, bg_loop.submit(_hang_forever()))
        manager.set_stream_unsubscriber(stream_name, _hang_forever)

        async def stop_from_the_loop() -> str:
            manager.stop_stream(stream_name)
            return "returned"

        assert bg_loop.submit(stop_from_the_loop()).result(timeout=5.0) == "returned"

    def test_the_degraded_path_still_schedules_the_unsubscriber(self, exchange_manager, bg_loop):
        """Degrading must not silently drop the venue-side unsubscribe."""
        manager = make_manager(exchange_manager, cleanup_timeout=5.0)
        stream_name = "quote:3:degraded-unsub"
        called = threading.Event()

        async def unsubscriber() -> None:
            called.set()

        manager.enable_stream(stream_name)
        manager.register_stream_future(stream_name, bg_loop.submit(_hang_forever()))
        manager.set_stream_unsubscriber(stream_name, unsubscriber)

        async def stop_from_the_loop() -> None:
            manager.stop_stream(stream_name)

        bg_loop.submit(stop_from_the_loop()).result(timeout=5.0)

        assert called.wait(2.0)


class TestLoopThreadDetection:
    def test_detects_the_exchange_loop_thread(self, manager, bg_loop):
        async def check() -> bool:
            return manager._on_exchange_loop_thread()

        assert bg_loop.submit(check()).result(timeout=2.0) is True

    def test_off_loop_thread_is_not_flagged(self, manager):
        assert manager._on_exchange_loop_thread() is False

    def test_a_different_loop_is_not_flagged(self, manager):
        other = BackgroundEventLoop("OtherLoop")
        try:

            async def check() -> bool:
                return manager._on_exchange_loop_thread()

            assert other.submit(check()).result(timeout=2.0) is False
        finally:
            _drain(other)


def test_cleanup_timeout_is_actually_used(exchange_manager, bg_loop):
    """The old poll loop made `cleanup_timeout` dead code; it must bound the wait now."""
    manager = make_manager(exchange_manager, cleanup_timeout=0.5)
    future = bg_loop.submit(_hang_forever())

    elapsed = run_with_deadline(lambda: manager._wait(future, "timing check"), 3.0)

    assert 0.4 < elapsed < 1.5
    assert isinstance(concurrent.futures.TimeoutError(), Exception)
