import asyncio

import pytest

from qubx.utils.misc import BackgroundEventLoop, run_sync


def test_background_loop_run_sync_returns_result():
    bel = BackgroundEventLoop(name="test-loop")
    try:

        async def add(a, b):
            await asyncio.sleep(0)
            return a + b

        assert bel.run_sync(add(2, 3)) == 5
    finally:
        bel.stop()


def test_run_sync_propagates_exception():
    bel = BackgroundEventLoop()
    try:

        async def boom():
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            bel.run_sync(boom())
    finally:
        bel.stop()


def test_run_sync_times_out():
    bel = BackgroundEventLoop()
    try:

        async def slow():
            await asyncio.sleep(5)

        with pytest.raises(TimeoutError):
            bel.run_sync(slow(), timeout=0.05)
    finally:
        bel.stop()


def test_run_sync_reentrancy_guard_raises():
    bel = BackgroundEventLoop()
    try:

        async def reenter():
            # called ON the loop thread → must raise, not deadlock
            return run_sync(bel.loop, asyncio.sleep(0))

        with pytest.raises(RuntimeError, match="own thread"):
            bel.run_sync(reenter())
    finally:
        bel.stop()


def test_stop_joins_thread():
    bel = BackgroundEventLoop(name="join-me")
    bel.stop()
    assert not bel._thread.is_alive()


def test_async_thread_loop_run_sync_and_submit():
    from qubx.utils.misc import AsyncThreadLoop

    bel = BackgroundEventLoop()
    try:
        atl = AsyncThreadLoop(bel.loop)

        async def mul(a, b):
            return a * b

        assert atl.run_sync(mul(3, 4)) == 12

        async def seven():
            return 7

        assert atl.submit(seven()).result(1) == 7
    finally:
        bel.stop()
