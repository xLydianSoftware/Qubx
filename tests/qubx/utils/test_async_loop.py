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


# --------------------------------------------------------------------------- #
# BulkRestLoop singleton (quantkit#106)
# --------------------------------------------------------------------------- #
class TestBulkRestLoop:
    def test_lazy_creation_and_singleton(self, monkeypatch):
        """No loop exists until first call; repeated calls return the same running loop."""
        from qubx.utils import misc

        monkeypatch.setattr(misc, "_bulk_rest_loop", None)

        first = misc.get_bulk_rest_loop()
        assert isinstance(first, BackgroundEventLoop)
        assert first.loop.is_running()
        assert first._thread.name == "BulkRestLoop"
        assert misc.get_bulk_rest_loop() is first

    def test_distinct_from_caller_loops(self):
        from qubx.utils.misc import get_bulk_rest_loop

        bel = BackgroundEventLoop(name="not-the-bulk-loop")
        try:
            assert get_bulk_rest_loop().loop is not bel.loop
        finally:
            bel.stop()

    def test_simulation_import_does_not_create_bulk_loop(self):
        """Live-only: importing the backtester (simulation entry point) must not spin up
        the BulkRestLoop — simulation never touches ccxt factories or loops. Run in a
        subprocess so loops created by other tests in this process can't pollute it."""
        import subprocess
        import sys

        code = (
            "import qubx.backtester  # noqa: F401\n"
            "from qubx.utils import misc\n"
            "assert misc._bulk_rest_loop is None, 'backtester import created the BulkRestLoop'\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
