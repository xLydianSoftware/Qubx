"""Unit tests for `_run_ws_loop`'s StreamHealth ledger wiring (A.1) and the
account-stream read-timeout heartbeat (A.2).

Mocked ccxt — no credentials, no real loop/thread. `_run_ws_loop` is driven directly
as an asyncio task, mirroring the `_subscribe_executions` drive pattern already used
by the ws-ready tests in test_ccxt_connector_reads.py: `conn.channel.control.is_set`
is toggled from inside the fake `watch` to end the loop deterministically.
"""

import asyncio
import contextlib
from unittest.mock import Mock

import pytest

from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.health.dummy import DummyHealthMonitor
from tests.qubx.core.utils_test import DummyTimeProvider


def _make_connector(*, health_monitor: Mock | None = None) -> tuple[CcxtConnector, Mock]:
    exchange = Mock()
    exchange.name = "binance"

    em = Mock()
    em.exchange = exchange
    em.register_recreation_callback = Mock()

    dp = Mock()
    channel = Mock()
    channel.control = Mock()
    channel.control.is_set = Mock(return_value=True)

    hm = health_monitor if health_monitor is not None else Mock()
    conn = CcxtConnector(
        exchange_name="BINANCE.UM",
        channel=channel,
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        data_provider=dp,
        health_monitor=hm,
    )
    return conn, hm


def test_default_health_monitor_is_dummy_and_safe() -> None:
    """No health_monitor passed at construction -> falls back to DummyHealthMonitor
    (the sim/no-op contract every IHealthMonitor implementation must satisfy)."""
    exchange = Mock()
    exchange.name = "binance"
    em = Mock()
    em.exchange = exchange
    conn = CcxtConnector(
        exchange_name="BINANCE.UM",
        channel=Mock(),
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        data_provider=Mock(),
    )
    assert isinstance(conn._health_monitor, DummyHealthMonitor)


@pytest.mark.asyncio
async def test_account_stream_timeout_records_drive_not_event() -> None:
    """A.2: a watch() that never resolves must time out at ACCOUNT_WATCH_TIMEOUT_S,
    record a drive each time around (heartbeat), never record an event, and keep
    looping rather than raising or exiting. Readiness stays optimistic across the
    heartbeat timeout — a quiet account stream is normal, not an error."""
    conn, hm = _make_connector()
    conn.ACCOUNT_WATCH_TIMEOUT_S = 0.02  # keep the test fast
    handle = Mock()

    block = asyncio.Event()

    async def _watch():
        await block.wait()  # never resolves within the timeout

    task = asyncio.ensure_future(
        conn._run_ws_loop(
            watch=_watch, handle=handle, stream="executions", mark_ready=True, is_account_stream=True
        )
    )
    await asyncio.sleep(conn.ACCOUNT_WATCH_TIMEOUT_S * 10)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert hm.record_stream_drive.call_count >= 2
    for call in hm.record_stream_drive.call_args_list:
        assert call.args == ("BINANCE.UM", "executions")
    hm.record_stream_event.assert_not_called()
    handle.assert_not_called()
    assert conn.is_ws_ready() is True


@pytest.mark.asyncio
async def test_account_stream_message_records_drive_then_event_then_handles() -> None:
    """A.1: a watch() that resolves with a message records the drive, then the event,
    then hands the message to `handle` — in that order."""
    conn, hm = _make_connector()
    calls: list[str] = []
    hm.record_stream_drive.side_effect = lambda *a: calls.append("drive")
    hm.record_stream_event.side_effect = lambda *a: calls.append("event")
    handle = Mock(side_effect=lambda *a: calls.append("handle"))

    async def _watch():
        conn.channel.control.is_set = Mock(return_value=False)  # stop after this iteration
        return [{"id": "1"}]

    await conn._run_ws_loop(
        watch=_watch, handle=handle, stream="executions", mark_ready=True, is_account_stream=True
    )

    assert calls == ["drive", "event", "handle"]
    hm.record_stream_drive.assert_called_once_with("BINANCE.UM", "executions")
    hm.record_stream_event.assert_called_once_with("BINANCE.UM", "executions")
    handle.assert_called_once_with({"id": "1"})


@pytest.mark.asyncio
async def test_market_data_stream_not_gated_by_account_timeout() -> None:
    """is_account_stream defaults to False: a market-data-style loop sharing
    _run_ws_loop must NOT be bounded by ACCOUNT_WATCH_TIMEOUT_S (current market-data
    behavior preserved) even when the timeout is shorter than the watch's real latency."""
    conn, hm = _make_connector()
    conn.ACCOUNT_WATCH_TIMEOUT_S = 0.01
    handle = Mock()

    async def _watch():
        await asyncio.sleep(conn.ACCOUNT_WATCH_TIMEOUT_S * 5)  # longer than the account timeout
        conn.channel.control.is_set = Mock(return_value=False)
        return [{"id": "1"}]

    await conn._run_ws_loop(watch=_watch, handle=handle, stream="orderbook", mark_ready=False)

    handle.assert_called_once_with({"id": "1"})
    hm.record_stream_event.assert_called_once_with("BINANCE.UM", "orderbook")
