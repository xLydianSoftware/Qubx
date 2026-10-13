"""Amend emulated as cancel + recreate (connector._replace_async + replace.Replacement), on the
Binance PM connector whose spot (papi margin) orders cannot be modified in place."""

import asyncio
import dataclasses
from unittest.mock import AsyncMock, Mock

import ccxt
import pytest

from qubx.connectors.ccxt.exchanges.binance.connector import BinancePmCcxtConnector
from qubx.core.basics import CtrlChannel, Order, OrderSide, OrderStatus, OrderType
from qubx.core.events import (
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.lookups import lookup
from qubx.core.series import Quote
from tests.qubx.core.utils_test import DummyTimeProvider

CID = "qubx_TRXUSDT_1"


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _instr(exchange: str):
    instrument = lookup.find_symbol(exchange, "TRXUSDT")
    assert instrument is not None
    # - the venue's 5 USDT floor, pinned: the bundled instrument data may carry none
    return dataclasses.replace(instrument, min_notional=5.0)


SPOT = _instr("BINANCE")
PERP = _instr("BINANCE.UM")


def _connector(quote: Quote | None = None):
    ex = Mock()
    ex.has = {"editOrder": True}
    ex.cancel_order = AsyncMock(return_value={"id": "A", "status": "canceled", "filled": 0.0, "cost": 0.0})
    ex.create_order = AsyncMock(return_value={"id": "B", "clientOrderId": CID, "status": "open"})
    ex.edit_order = AsyncMock(return_value={"id": "A", "status": "open"})
    ex.fetch_order_with_client_order_id = AsyncMock()
    em = Mock()
    em.exchange = ex
    em.rate_limiter = None
    data_provider = Mock()
    data_provider.get_quote = Mock(return_value=quote or Quote(0, 0.330, 0.331, 1, 1))
    sent: list = []
    channel = Mock(spec=CtrlChannel)
    channel.send = Mock(side_effect=lambda e: sent.append(e))
    conn = BinancePmCcxtConnector(
        exchange_name="BINANCE.UM",
        markets=("BINANCE.UM", "BINANCE"),
        channel=channel,
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        data_provider=data_provider,
    )
    captured: list = []
    conn._spawn = Mock(side_effect=lambda coro: captured.append(coro))  # type: ignore[method-assign]
    conn._instrument_for_symbol = Mock(side_effect=lambda s: PERP if s.endswith(":USDT") else SPOT)  # type: ignore[method-assign]
    return conn, ex, sent, captured


def _order(
    instrument=SPOT, *, vid: str | None = "A", qty: float = 60.0, filled: float = 0.0, tif: str = "gtx"
) -> Order:
    return Order(
        client_order_id=CID,
        type=OrderType.LIMIT,
        instrument=instrument,
        quantity=qty,
        side=OrderSide.BUY,
        time_in_force=tif,
        status=OrderStatus.ACCEPTED,
        venue_order_id=vid,
        price=0.30,
        filled_quantity=filled,
    )


def _raw(vid: str, status: str, *, amount: float, filled: float = 0.0, avg: float | None = None, trades=None) -> dict:
    return {
        "id": vid,
        "clientOrderId": CID,
        "symbol": "TRX/USDT",
        "type": "limit",
        "side": "buy",
        "status": status,
        "amount": amount,
        "filled": filled,
        "remaining": amount - filled,
        "price": 0.30,
        "average": avg,
        "timeInForce": "GTC",
        "timestamp": 1784722758752,
        "lastUpdateTimestamp": 1784722758752,
        "trades": trades or [],
        "info": {},
    }


def _trade(tid: str, vid: str, amount: float, price: float = 0.30) -> dict:
    return {
        "id": tid,
        "order": vid,
        "timestamp": 1784722758752,
        "side": "buy",
        "amount": amount,
        "price": price,
        "takerOrMaker": "maker",
        "fee": {"cost": 0.0, "currency": "USDT"},
    }


def _drain(captured: list) -> None:
    while captured:
        run(captured.pop(0))


def _replace(conn, captured, order=None, **kw) -> None:
    conn.update_order(order or _order(), **({"price": 0.31} | kw))
    _drain(captured)


class TestRouting:
    def test_spot_amend_runs_as_cancel_and_recreate_with_the_same_client_id(self):
        conn, ex, sent, captured = _connector()
        _replace(conn, captured)
        ex.edit_order.assert_not_called()
        ex.cancel_order.assert_awaited_once()
        assert ex.cancel_order.await_args.args[0] == "A"
        payload = ex.create_order.await_args.kwargs
        assert payload["amount"] == 60.0 and payload["price"] == 0.31
        assert payload["params"]["clientOrderId"] == CID
        assert payload["params"].get("postOnly") is True
        (event,) = sent
        assert isinstance(event, OrderUpdatedEvent)
        assert event.client_order_id == CID and event.venue_order_id == "B" and event.new_price == 0.31

    def test_perp_still_amends_in_place(self):
        conn, ex, sent, captured = _connector()
        conn.update_order(_order(PERP), price=0.31)
        _drain(captured)
        ex.edit_order.assert_awaited_once()
        ex.cancel_order.assert_not_called()

    def test_unacked_order_is_refused_untouched(self):
        conn, ex, sent, captured = _connector()
        _replace(conn, captured, _order(vid=None))
        ex.cancel_order.assert_not_called()
        assert isinstance(sent[0], OrderUpdateRejectedEvent)

    def test_post_only_price_crossing_the_book_is_refused_before_the_cancel(self):
        conn, ex, sent, captured = _connector(quote=Quote(0, 0.330, 0.331, 1, 1))
        _replace(conn, captured, price=0.335)  # a post-only buy above the ask
        ex.cancel_order.assert_not_called()
        assert isinstance(sent[0], OrderUpdateRejectedEvent)


class TestPartialFills:
    def test_recreate_is_sized_from_the_cancel_responses_executed_quantity(self):
        # - local filled says 0 (fills still on the wire); the venue executed 20 on A
        conn, ex, sent, captured = _connector()
        ex.cancel_order.return_value = {"id": "A", "status": "canceled", "filled": 20.0, "cost": 6.0}
        _replace(conn, captured)
        assert ex.create_order.await_args.kwargs["amount"] == 40.0

    def test_b_reports_carry_the_lifetime_figures(self):
        conn, ex, sent, captured = _connector()
        ex.cancel_order.return_value = {"id": "A", "status": "canceled", "filled": 20.0, "cost": 6.0}
        _replace(conn, captured)
        sent.clear()
        conn._handle_ws_order(
            _raw("B", "open", amount=40.0, filled=10.0, avg=0.31, trades=[_trade("t2", "B", 10.0, 0.31)])
        )
        fill = next(e for e in sent if isinstance(e, OrderPartiallyFilledEvent))
        assert fill.venue_order_id == "B" and fill.fill.amount == 10.0
        b = conn._replacements[CID]
        assert b.filled_before == 20.0
        # - the terminal report: venue_filled_quantity is the lifetime figure, the record is released
        sent.clear()
        conn._handle_ws_order(
            _raw("B", "closed", amount=40.0, filled=40.0, avg=0.31, trades=[_trade("t3", "B", 30.0, 0.31)])
        )
        done = next(e for e in sent if isinstance(e, OrderFilledEvent))
        assert done.venue_filled_quantity == pytest.approx(60.0)
        assert done.venue_avg_price == pytest.approx((6.0 + 40.0 * 0.31) / 60.0)
        assert CID not in conn._replacements

    def test_fill_that_beats_the_cancel_is_the_orders_end(self):
        # - A filled before the cancel landed: the venue answers unknown order
        conn, ex, sent, captured = _connector()
        ex.cancel_order.side_effect = ccxt.OrderNotFound("binance {'code':-2011,'msg':'Unknown order sent.'}")
        conn.update_order(_order(), price=0.31)
        run(captured.pop(0))
        ex.create_order.assert_not_called()
        assert isinstance(sent[0], OrderUpdateRejectedEvent)
        assert captured, "a status fetch of the old order surfaces its terminal state"
        assert CID not in conn._replacements
        # - its FILLED report passes untouched
        sent.clear()
        conn._handle_ws_order(_raw("A", "closed", amount=60.0, filled=60.0, avg=0.30, trades=[_trade("t1", "A", 60.0)]))
        assert any(isinstance(e, OrderFilledEvent) and e.venue_order_id == "A" for e in sent)

    @pytest.mark.parametrize("filled", [60.0, 59.9])
    def test_nothing_placeable_left_ends_the_order_as_cancelled(self, filled):
        # - fully executed, or a sub-lot / sub-min-notional rest
        conn, ex, sent, captured = _connector()
        ex.cancel_order.return_value = {"id": "A", "status": "canceled", "filled": filled, "cost": filled * 0.3}
        _replace(conn, captured)
        ex.create_order.assert_not_called()
        (event,) = sent
        assert isinstance(event, OrderCanceledEvent) and event.client_order_id == CID


class TestOldOrderReports:
    def test_old_orders_cancel_is_dropped_while_the_replace_runs(self):
        conn, ex, sent, captured = _connector()

        async def cancel(*_a, **_k):
            # - the WS CANCELED of A lands before the REST cancel answers
            conn._handle_ws_order(_raw("A", "canceled", amount=60.0))
            return {"id": "A", "status": "canceled", "filled": 0.0, "cost": 0.0}

        ex.cancel_order.side_effect = cancel
        _replace(conn, captured)
        assert not any(isinstance(e, OrderCanceledEvent) for e in sent)
        assert isinstance(sent[-1], OrderUpdatedEvent)

    def test_late_reports_of_the_old_order_after_the_recreate(self):
        conn, ex, sent, captured = _connector()
        _replace(conn, captured)
        sent.clear()
        conn._handle_ws_order(_raw("A", "canceled", amount=60.0))
        conn._handle_ws_order(_raw("A", "open", amount=60.0))
        assert sent == []  # neither may point the order back at A or end it
        conn._handle_ws_order(_raw("A", "open", amount=60.0, filled=5.0, avg=0.3, trades=[_trade("t9", "A", 5.0)]))
        (fill,) = [e for e in sent if isinstance(e, OrderPartiallyFilledEvent)]
        assert fill.venue_order_id == "B" and fill.fill.trade_id == "t9"
        assert not any(isinstance(e, OrderAcceptedEvent) and e.venue_order_id == "A" for e in sent)

    def test_snapshot_open_order_of_b_carries_the_lifetime_quantity(self):
        conn, ex, sent, captured = _connector()
        ex.cancel_order.return_value = {"id": "A", "status": "canceled", "filled": 20.0, "cost": 6.0}
        _replace(conn, captured)
        (order,) = conn._merge_open_orders([_raw("B", "open", amount=40.0, filled=0.0)], [])
        assert order.venue_order_id == "B" and order.quantity == 60.0 and order.filled_quantity == 20.0


class TestRecreateFailures:
    def test_refused_recreate_ends_the_order(self):
        conn, ex, sent, captured = _connector()
        ex.create_order.side_effect = ccxt.InvalidOrder(
            "binance {'code':-2010,'msg':'Order would immediately match and take.'}"
        )
        _replace(conn, captured)
        (event,) = sent
        assert isinstance(event, OrderCanceledEvent) and event.venue_order_id == "A"
        assert CID not in conn._replacements

    def test_unknown_recreate_that_landed_is_adopted(self, monkeypatch):
        monkeypatch.setattr("qubx.connectors.ccxt.connector.asyncio.sleep", AsyncMock())
        conn, ex, sent, captured = _connector()
        ex.create_order.side_effect = ccxt.RequestTimeout("timeout")
        ex.fetch_order_with_client_order_id.return_value = _raw("B", "open", amount=60.0)
        _replace(conn, captured)
        assert isinstance(sent[0], OrderUpdatedEvent) and sent[0].venue_order_id == "B"

    def test_unknown_recreate_that_did_not_land_ends_the_order(self, monkeypatch):
        monkeypatch.setattr("qubx.connectors.ccxt.connector.asyncio.sleep", AsyncMock())
        conn, ex, sent, captured = _connector()
        ex.create_order.side_effect = ccxt.RequestTimeout("timeout")
        ex.fetch_order_with_client_order_id.return_value = _raw("A", "canceled", amount=60.0)  # newest is still A
        _replace(conn, captured)
        (event,) = sent
        assert isinstance(event, OrderCanceledEvent)

    def test_cancel_outcome_unknown_places_nothing(self):
        conn, ex, sent, captured = _connector()
        ex.cancel_order.side_effect = ccxt.RequestTimeout("timeout")
        conn.cancel_retry_interval = 0
        conn.max_cancel_retries = 1
        _replace(conn, captured)
        ex.create_order.assert_not_called()
        assert sent == []  # left PENDING_UPDATE for the reconciler
        assert CID not in conn._replacements  # A's own reports flow again


class TestCancelDuringReplace:
    def test_cancel_before_the_recreate_skips_it(self):
        conn, ex, sent, captured = _connector()

        async def cancel(*_a, **_k):
            conn.cancel_order(_order())  # the strategy cancels meanwhile
            run_now = captured.pop()
            await run_now
            return {"id": "A", "status": "canceled", "filled": 0.0, "cost": 0.0}

        ex.cancel_order.side_effect = cancel
        _replace(conn, captured)
        ex.create_order.assert_not_called()
        (event,) = sent
        assert isinstance(event, OrderCanceledEvent)

    def test_cancel_during_the_recreate_cancels_the_new_order(self):
        conn, ex, sent, captured = _connector()

        async def create(**_k):
            conn.cancel_order(_order())
            await captured.pop()
            return {"id": "B", "clientOrderId": CID, "status": "open"}

        ex.create_order.side_effect = create
        ex.cancel_order.side_effect = [
            {"id": "A", "status": "canceled", "filled": 0.0, "cost": 0.0},
            {"id": "B", "status": "canceled", "filled": 0.0, "cost": 0.0, "clientOrderId": CID},
        ]
        _replace(conn, captured)
        assert [c.args[0] for c in ex.cancel_order.await_args_list] == ["A", "B"]
        assert isinstance(sent[0], OrderUpdatedEvent)
        assert isinstance(sent[-1], OrderCanceledEvent) and sent[-1].venue_order_id == "B"

    def test_a_cancel_addressed_to_the_old_venue_id_goes_to_the_new_one(self):
        conn, ex, sent, captured = _connector()
        _replace(conn, captured)
        ex.cancel_order.reset_mock()
        ex.cancel_order.return_value = {"id": "B", "status": "canceled"}
        conn.cancel_order(_order(vid="A"))
        _drain(captured)
        assert ex.cancel_order.await_args.args[0] == "B"

    def test_second_amend_chains_on_the_recreated_order(self):
        conn, ex, sent, captured = _connector()
        ex.cancel_order.return_value = {"id": "A", "status": "canceled", "filled": 20.0, "cost": 6.0}
        _replace(conn, captured)
        ex.cancel_order.return_value = {"id": "B", "status": "canceled", "filled": 10.0, "cost": 3.1}
        ex.create_order.return_value = {"id": "C", "clientOrderId": CID, "status": "open"}
        _replace(conn, captured, _order(vid="B", filled=30.0), price=0.32)
        assert ex.cancel_order.await_args.args[0] == "B"
        assert ex.create_order.await_args.kwargs["amount"] == 30.0  # 60 - (20 + 10)
        record = conn._replacements[CID]
        assert record.current_vid == "C" and record.superseded == {"A", "B"} and record.filled_before == 30.0
