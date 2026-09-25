"""Unit tests for ``BinancePmCcxtConnector.repay_debt`` — margin loan + interest repayment via
``repayLoan``. Offline: the exchange is a MagicMock, ``_spawn`` captures the venue coroutine and
the test awaits it itself.
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import ccxt
import pytest

from qubx.connectors.ccxt.exchanges.binance.connector import BinancePmCcxtConnector
from qubx.core.basics import CtrlChannel
from qubx.core.events import DebtRepaidEvent
from tests.qubx.core.utils_test import DummyTimeProvider

FRAB_USDT_ROW = {
    "asset": "USDT",
    "totalWalletBalance": "5055.48548376",
    "crossMarginAsset": "144168.84518903",
    "crossMarginBorrowed": "0.0",
    "crossMarginInterest": "3.38935086",
    "umWalletBalance": "-139113.35970527",
    "negativeBalance": "0.0",
}


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


def _repaid(sent: list):
    (event,) = [e for e in sent if isinstance(e, DebtRepaidEvent)]
    return event.repaid


def test_declares_borrowed_and_interest():
    conn, _, _ = _pm_connector()
    assert conn.debt_repayments() == ["borrowed", "interest"]


@pytest.mark.parametrize("currency,amount", [("", None), ("USDT", 0.0), ("USDT", -1.0)])
def test_bad_arguments_raise_and_emit_nothing(currency, amount):
    conn, sent, _ = _pm_connector()
    with pytest.raises(ValueError):
        conn.repay_debt(currency, amount)
    assert sent == []
    conn._spawn.assert_not_called()


async def test_everything_owed_reads_the_balance_and_repays_the_venue_string():
    conn, sent, ex = _pm_connector()
    ex.papiGetBalance = AsyncMock(return_value=[{"asset": "BTC", "crossMarginInterest": "1"}, FRAB_USDT_ROW])
    ex.papiPostRepayLoan = AsyncMock(return_value={"tranId": 415031841643})
    repay_id = conn.repay_debt("usdt")
    await _drive(conn)
    ex.papiPostRepayLoan.assert_awaited_once_with({"asset": "USDT", "amount": "3.38935086"})
    repaid = _repaid(sent)
    assert repaid.repay_id == repay_id and repaid.status == "DONE"
    assert (repaid.exchange, repaid.currency, repaid.requested) == ("BINANCE.PM", "usdt", None)
    assert repaid.venue_ref == "415031841643" and repaid.failure_reason is None
    conn.request_snapshot.assert_called_once_with(include_orders=False)


async def test_borrowed_and_interest_are_summed_exactly():
    conn, _, ex = _pm_connector()
    row = {**FRAB_USDT_ROW, "crossMarginBorrowed": "100.1", "crossMarginInterest": "0.2"}
    ex.papiGetBalance = AsyncMock(return_value=[row])
    ex.papiPostRepayLoan = AsyncMock(return_value={"tranId": 1})
    conn.repay_debt("USDT")
    await _drive(conn)
    ex.papiPostRepayLoan.assert_awaited_once_with({"asset": "USDT", "amount": "100.3"})


async def test_explicit_amount_passes_through_without_a_balance_read():
    conn, sent, ex = _pm_connector()
    ex.papiGetBalance = AsyncMock()
    ex.papiPostRepayLoan = AsyncMock(return_value={"tranId": 7})
    conn.repay_debt("USDT", 2.5)
    await _drive(conn)
    ex.papiGetBalance.assert_not_awaited()
    ex.papiPostRepayLoan.assert_awaited_once_with({"asset": "USDT", "amount": "2.5"})
    repaid = _repaid(sent)
    assert repaid.status == "DONE" and repaid.requested == 2.5


async def test_nothing_owed_fails_without_a_repay_call():
    conn, sent, ex = _pm_connector()
    row = {**FRAB_USDT_ROW, "crossMarginInterest": "0.0"}
    ex.papiGetBalance = AsyncMock(return_value=[row])
    ex.papiPostRepayLoan = AsyncMock()
    conn.repay_debt("USDT")
    await _drive(conn)
    ex.papiPostRepayLoan.assert_not_awaited()
    repaid = _repaid(sent)
    assert repaid.status == "FAILED" and repaid.failure_reason == "nothing to repay"
    conn.request_snapshot.assert_called_once_with(include_orders=False)


async def test_currency_missing_from_the_balance_is_nothing_to_repay():
    conn, sent, ex = _pm_connector()
    ex.papiGetBalance = AsyncMock(return_value=[FRAB_USDT_ROW])
    ex.papiPostRepayLoan = AsyncMock()
    conn.repay_debt("BNB")
    await _drive(conn)
    ex.papiPostRepayLoan.assert_not_awaited()
    assert _repaid(sent).failure_reason == "nothing to repay"


async def test_venue_error_is_one_failed_event_and_still_refreshes():
    conn, sent, ex = _pm_connector()
    ex.papiPostRepayLoan = AsyncMock(side_effect=ccxt.ExchangeError('binance {"code":-3041}'))
    conn.repay_debt("USDT", 1.0)
    await _drive(conn)
    repaid = _repaid(sent)
    assert repaid.status == "FAILED" and "-3041" in repaid.failure_reason
    conn.request_snapshot.assert_called_once_with(include_orders=False)


async def test_balance_read_error_is_one_failed_event():
    conn, sent, ex = _pm_connector()
    ex.papiGetBalance = AsyncMock(side_effect=ccxt.NetworkError("timeout"))
    ex.papiPostRepayLoan = AsyncMock()
    conn.repay_debt("USDT")
    await _drive(conn)
    ex.papiPostRepayLoan.assert_not_awaited()
    repaid = _repaid(sent)
    assert repaid.status == "FAILED" and "timeout" in repaid.failure_reason
    conn.request_snapshot.assert_called_once_with(include_orders=False)


async def test_response_without_tran_id_reports_the_response():
    conn, sent, ex = _pm_connector()
    ex.papiPostRepayLoan = AsyncMock(return_value={"code": -1})
    conn.repay_debt("USDT", 1.0)
    await _drive(conn)
    repaid = _repaid(sent)
    assert repaid.status == "FAILED" and repaid.failure_reason == str({"code": -1})
    conn.request_snapshot.assert_called_once_with(include_orders=False)


async def test_repay_ids_are_unique():
    conn, _, _ = _pm_connector()
    first = conn.repay_debt("USDT", 1.0)
    second = conn.repay_debt("USDT", 1.0)
    for coro in conn._captured:  # type: ignore[attr-defined]
        coro.close()
    assert first.startswith("rp-") and first != second
