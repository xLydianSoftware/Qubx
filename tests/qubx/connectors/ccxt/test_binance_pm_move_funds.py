"""Unit tests for ``BinancePmCcxtConnector.move_funds`` — futures→margin collection and
negative-balance repay. Offline: the exchange is a MagicMock, ``_spawn`` captures the venue
coroutine and the test awaits it itself.
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import ccxt
import pytest

from qubx.connectors.ccxt.exchanges.binance.connector import BinancePmCcxtConnector
from qubx.core.basics import CtrlChannel, WalletMove
from qubx.core.events import FundsMovedEvent
from tests.qubx.core.utils_test import DummyTimeProvider


def _pm_connector() -> tuple[BinancePmCcxtConnector, list, MagicMock]:
    exchange = MagicMock()

    em = Mock()
    em.exchange = exchange
    em.rate_limiter = None

    sent: list = []
    channel = Mock(spec=CtrlChannel)
    channel.send = Mock(side_effect=lambda e: sent.append(e))

    conn = BinancePmCcxtConnector(
        exchange_name="BINANCE.PM",
        channel=channel,
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        data_provider=Mock(),
    )
    captured: list = []
    conn._spawn = Mock(side_effect=lambda coro: captured.append(coro))
    conn._captured = captured  # type: ignore[attr-defined]
    conn.request_snapshot = Mock()
    return conn, sent, exchange


async def _drive(conn: BinancePmCcxtConnector) -> None:
    for coro in conn._captured:  # type: ignore[attr-defined]
        await coro
    conn._captured.clear()  # type: ignore[attr-defined]


def _moved(sent: list):
    (event,) = [e for e in sent if isinstance(e, FundsMovedEvent)]
    return event.moved


def test_declares_collect_and_repay():
    conn, _, _ = _pm_connector()
    assert conn.wallet_moves() == [WalletMove("futures_um", "margin", False), WalletMove("margin", "futures_um", False)]


@pytest.mark.parametrize(
    "src,dst,amount", [("margin", "futures_cm", None), ("futures_um", "margin", 10.0), ("spot", "margin", None)]
)
def test_unsupported_moves_raise_and_emit_nothing(src, dst, amount):
    conn, sent, _ = _pm_connector()
    with pytest.raises(ValueError):
        conn.move_funds("USDT", src, dst, amount)
    assert sent == []
    conn._spawn.assert_not_called()


async def test_collect_one_asset_calls_asset_collection():
    conn, sent, ex = _pm_connector()
    ex.papiPostAssetCollection = AsyncMock(return_value={"msg": "success"})
    move_id = conn.move_funds("usdt", "futures_um", "margin")
    await _drive(conn)
    ex.papiPostAssetCollection.assert_awaited_once_with({"asset": "USDT"})
    moved = _moved(sent)
    assert moved.move_id == move_id and moved.status == "DONE"
    assert (moved.exchange, moved.currency, moved.src, moved.dst) == ("BINANCE.PM", "usdt", "futures_um", "margin")
    assert moved.requested is None and moved.failure_reason is None
    conn.request_snapshot.assert_called_once_with(include_orders=False)


async def test_collect_everything_calls_auto_collection():
    conn, sent, ex = _pm_connector()
    ex.papiPostAutoCollection = AsyncMock(return_value={"msg": "success"})
    ex.papiPostAssetCollection = AsyncMock()
    conn.move_funds(None, "futures_um", "margin")
    await _drive(conn)
    ex.papiPostAutoCollection.assert_awaited_once()
    ex.papiPostAssetCollection.assert_not_awaited()
    moved = _moved(sent)
    assert moved.status == "DONE" and moved.currency is None


async def test_repay_calls_repay_futures_negative_balance():
    conn, sent, ex = _pm_connector()
    ex.papiPostRepayFuturesNegativeBalance = AsyncMock(return_value={"msg": "success"})
    conn.move_funds("USDT", "margin", "futures_um")
    await _drive(conn)
    ex.papiPostRepayFuturesNegativeBalance.assert_awaited_once()
    moved = _moved(sent)
    assert moved.status == "DONE" and moved.currency == "USDT"
    conn.request_snapshot.assert_called_once_with(include_orders=False)


async def test_venue_error_is_one_failed_event_and_still_refreshes():
    conn, sent, ex = _pm_connector()
    ex.papiPostAssetCollection = AsyncMock(side_effect=ccxt.ExchangeError('binance {"code":-5000}'))
    conn.move_funds("USDT", "futures_um", "margin")
    await _drive(conn)
    moved = _moved(sent)
    assert moved.status == "FAILED" and "-5000" in moved.failure_reason
    conn.request_snapshot.assert_called_once_with(include_orders=False)


async def test_non_success_msg_is_failed():
    conn, sent, ex = _pm_connector()
    ex.papiPostAssetCollection = AsyncMock(return_value={"msg": "no balance to collect"})
    conn.move_funds("USDT", "futures_um", "margin")
    await _drive(conn)
    moved = _moved(sent)
    assert moved.status == "FAILED" and moved.failure_reason == "no balance to collect"
    conn.request_snapshot.assert_called_once_with(include_orders=False)


async def test_response_without_msg_reports_the_response():
    conn, sent, ex = _pm_connector()
    ex.papiPostAssetCollection = AsyncMock(return_value={"code": -1})
    conn.move_funds("USDT", "futures_um", "margin")
    await _drive(conn)
    moved = _moved(sent)
    assert moved.status == "FAILED" and moved.failure_reason == str({"code": -1})


async def test_move_ids_are_unique():
    conn, _, _ = _pm_connector()
    first = conn.move_funds("USDT", "futures_um", "margin")
    second = conn.move_funds("USDT", "futures_um", "margin")
    for coro in conn._captured:  # type: ignore[attr-defined]
        coro.close()
    assert first.startswith("mv-") and first != second
