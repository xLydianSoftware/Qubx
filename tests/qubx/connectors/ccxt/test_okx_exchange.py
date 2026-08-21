"""Tests for OKX exchange registration and custom class."""

import asyncio
from unittest.mock import Mock

import ccxt
import ccxt.pro as cxp
import pytest
from ccxt.base.errors import ChecksumError

from qubx import logger
from qubx.connectors.ccxt.exchanges import EXCHANGE_ALIASES, OkxFutures
from qubx.connectors.ccxt.exchanges.okx.connector import OkxCcxtConnector
from qubx.connectors.ccxt.utils import ccxt_status_to_order_status
from qubx.core.basics import OrderStatus
from qubx.core.basics import CtrlChannel


def run(coro):
    # NOT asyncio.run: that clears the thread's current event loop on exit, breaking
    # later tests in the same worker that rely on asyncio.get_event_loop()
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _swap_market() -> dict:
    """Minimal OKX perpetual market dict that satisfies ccxt's lookups."""
    return {
        "id": "BTC-USDT-SWAP",
        "symbol": "BTC/USDT:USDT",
        "base": "BTC",
        "quote": "USDT",
        "settle": "USDT",
        "baseId": "BTC",
        "quoteId": "USDT",
        "settleId": "USDT",
        "type": "swap",
        "spot": False,
        "margin": False,
        "swap": True,
        "future": False,
        "option": False,
        "contract": True,
        "linear": True,
        "inverse": False,
        "subType": "linear",
        "active": True,
        "taker": 0.0005,
        "maker": 0.0002,
        "contractSize": 0.01,
        "expiry": None,
        "expiryDatetime": None,
        "strike": None,
        "optionType": None,
        "precision": {"amount": 0.01, "price": 0.1},
        "limits": {
            "amount": {"min": 0.01, "max": None},
            "price": {"min": None, "max": None},
            "cost": {"min": None, "max": None},
        },
        "info": {},
        "created": None,
    }


@pytest.fixture
def offline_okx():
    exchange = OkxFutures()
    exchange.set_markets([_swap_market()])
    return exchange


class TestOkxRegistration:
    def test_exchange_alias_exists(self):
        assert EXCHANGE_ALIASES["okx.f"] == "okx_futures"

    def test_custom_class_registered_in_ccxt(self):
        assert cxp.okx_futures is OkxFutures
        assert "okx_futures" in cxp.exchanges

    def test_defaults_to_swap_in_net_mode(self):
        options = OkxFutures().describe()["options"]
        assert options["defaultType"] == "swap"
        assert options["positionSide"] == "net"


class TestOrderbookChecksumMessage:
    """
    A checksum mismatch on a book SNAPSHOT reaches ccxt with symbol=None, because its snapshot
    branch passes no market and the payload carries no instId. The base implementation
    concatenates the symbol and raises TypeError, which skips the subscription cleanup and
    leaves the waiter unresolved — the stream then stalls with nothing raised.
    """

    def test_base_raises_on_missing_symbol(self):
        with pytest.raises(TypeError):
            cxp.okx().orderbook_checksum_message(None)

    def test_message_survives_missing_symbol(self):
        message = OkxFutures().orderbook_checksum_message(None)
        assert "okx" in message

    def test_message_keeps_the_symbol_when_present(self):
        assert "BTC/USDT:USDT" in OkxFutures().orderbook_checksum_message("BTC/USDT:USDT")

    def test_error_is_constructible_without_a_symbol(self):
        exchange = OkxFutures()
        assert isinstance(ChecksumError(exchange.orderbook_checksum_message(None)), ChecksumError)


class TestProtectiveStopRouting:
    """
    Measured live on 2026-08-21, same account/instrument/size, one parameter apart:
    ``triggerPrice`` + ``reduceOnly`` -> 51205 "Reduce Only is not available.";
    ``stopLossPrice`` + ``reduceOnly`` -> accepted. So a reduce-only stop has to go out as
    OKX's conditional algo type, which ccxt selects by the parameter name.
    """

    def test_reduce_only_stop_becomes_a_conditional_order(self, offline_okx):
        request = offline_okx.create_order_request(
            "BTC/USDT:USDT", "market", "sell", 0.12, None, {"triggerPrice": 62256.1, "reduceOnly": True}
        )
        assert request["ordType"] == "conditional"
        assert request["slTriggerPx"] == "62256.1"
        assert request["slOrdPx"] == "-1"
        assert "triggerPx" not in request

    def test_plain_stop_stays_a_trigger_order(self, offline_okx):
        request = offline_okx.create_order_request(
            "BTC/USDT:USDT", "market", "sell", 0.12, None, {"triggerPrice": 62256.1}
        )
        assert request["ordType"] == "trigger"
        assert request["triggerPx"] == "62256.1"

    def test_a_reduce_only_order_without_a_level_is_untouched(self, offline_okx):
        params = {"reduceOnly": True}
        assert offline_okx._route_protective_stop(params) == params

    def test_the_caller_params_are_not_mutated(self, offline_okx):
        params = {"triggerPrice": 62256.1, "reduceOnly": True}
        offline_okx._route_protective_stop(params)
        assert params["triggerPrice"] == 62256.1


class TestAlgoOrderParsing:
    """
    ccxt returns the raw ``ordType`` as the order type, so an algo order reads as
    "trigger"/"conditional" — neither is an OrderType, and the connector routes a cancel to
    the algo book by order type. A conditional also leaves ccxt's triggerPrice empty.
    """

    def _parse(self, exchange, raw: dict) -> dict:
        return exchange.parse_order({"instId": "BTC-USDT-SWAP", "side": "sell", "sz": "0.12", **raw})

    def test_conditional_reads_as_stop_market_with_its_trigger(self, offline_okx):
        parsed = self._parse(offline_okx, {"ordType": "conditional", "slTriggerPx": "62256.1", "slOrdPx": "-1"})
        assert parsed["type"] == "stop_market"
        assert parsed["triggerPrice"] == 62256.1

    def test_conditional_with_a_limit_price_reads_as_stop_limit(self, offline_okx):
        parsed = self._parse(offline_okx, {"ordType": "conditional", "slTriggerPx": "62256.1", "slOrdPx": "62200"})
        assert parsed["type"] == "stop_limit"

    def test_trigger_reads_as_stop_market(self, offline_okx):
        parsed = self._parse(offline_okx, {"ordType": "trigger", "triggerPx": "62256.1", "orderPx": "-1"})
        assert parsed["type"] == "stop_market"
        assert parsed["triggerPrice"] == 62256.1

    def test_a_regular_order_is_left_alone(self, offline_okx):
        parsed = self._parse(offline_okx, {"ordType": "limit", "px": "73000"})
        assert parsed["type"] == "limit"


class TestTriggerOrderListing:
    """
    Qubx's snapshot asks for trigger orders with ``params={"trigger": True}``. ccxt turns that
    into OKX's pending-algo call with ``ordType="trigger"`` — one type only — so a conditional
    stop would never be listed and reconcile would not see it.
    """

    @staticmethod
    def _record_calls(monkeypatch) -> list:
        calls = []

        async def stub(self, symbol=None, since=None, limit=None, params={}):
            calls.append(params)
            return [{"id": params.get("ordType", "regular")}]

        monkeypatch.setattr(cxp.okx, "fetch_open_orders", stub)
        return calls

    def test_trigger_request_asks_for_both_algo_types(self, offline_okx, monkeypatch):
        calls = self._record_calls(monkeypatch)
        orders = run(offline_okx.fetch_open_orders(params={"trigger": True}))
        assert [c["ordType"] for c in calls] == ["trigger", "conditional"]
        assert [o["id"] for o in orders] == ["trigger", "conditional"]

    def test_an_explicit_ord_type_is_respected(self, offline_okx, monkeypatch):
        calls = self._record_calls(monkeypatch)
        run(offline_okx.fetch_open_orders(params={"trigger": True, "ordType": "oco"}))
        assert [c["ordType"] for c in calls] == ["oco"]

    def test_regular_listing_is_a_single_untouched_call(self, offline_okx, monkeypatch):
        calls = self._record_calls(monkeypatch)
        run(offline_okx.fetch_open_orders("BTC/USDT:USDT"))
        assert calls == [{}]


class TestOrderStatusMapping:
    """
    OKX's submit ack is `{ordId, clOrdId, tag, sCode, sMsg}` — no state — so every order
    logged "Unknown ccxt order status 'None'". ACCEPTED is the right mapping for an ack;
    it is the warning that was wrong.
    """

    @staticmethod
    def _warnings_while(call) -> list[str]:
        # - caplog does not see loguru records, so the sink has to be loguru's own
        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            call()
        finally:
            logger.remove(sink_id)
        return messages

    def test_absent_status_maps_to_accepted_without_warning(self):
        assert self._warnings_while(lambda: ccxt_status_to_order_status(None)) == []
        assert ccxt_status_to_order_status(None) is OrderStatus.ACCEPTED

    def test_an_unrecognized_status_still_warns(self):
        warnings = self._warnings_while(lambda: ccxt_status_to_order_status("teleported"))
        assert len(warnings) == 1
        assert "teleported" in warnings[0]
        assert ccxt_status_to_order_status("teleported") is OrderStatus.ACCEPTED

    def test_known_statuses_are_unchanged(self):
        assert ccxt_status_to_order_status("closed") is OrderStatus.FILLED
        assert ccxt_status_to_order_status("canceled") is OrderStatus.CANCELED
        assert ccxt_status_to_order_status("open", {"status": "partially_filled"}) is OrderStatus.PARTIALLY_FILLED


class TestAlgoOrderStream:
    """
    OKX pushes trigger/conditional orders on channel "orders-algo"; plain `watch_orders` covers
    "orders" only. Without the extra stream a stop's terminal state never arrives over the
    socket — a cancelled stop sat in PENDING_CANCEL for 43s on a live close, until the next
    order-bearing snapshot resolved it.
    """

    @staticmethod
    def _connector() -> "OkxCcxtConnector":
        exchange_manager = Mock()
        exchange_manager.exchange = Mock()
        return OkxCcxtConnector(
            exchange_name="OKX.F",
            channel=Mock(spec=CtrlChannel),
            time_provider=Mock(),
            exchange_manager=exchange_manager,
            data_provider=Mock(),
        )

    def _recorded_streams(self, monkeypatch) -> list[dict]:
        calls = []
        monkeypatch.setattr(OkxCcxtConnector, "_run_ws_loop", lambda self, **kwargs: calls.append(kwargs))
        self._connector()._account_streams()
        return calls

    def test_three_streams_are_started(self, monkeypatch):
        assert [c["stream"] for c in self._recorded_streams(monkeypatch)] == [
            "orders",
            "my_trades",
            "orders_algo",
        ]

    def test_the_algo_stream_asks_for_trigger_orders(self, monkeypatch):
        algo = self._recorded_streams(monkeypatch)[-1]
        assert algo["watch"].keywords == {"params": {"trigger": True}}

    def test_only_the_plain_order_stream_owns_liveness(self, monkeypatch):
        streams = self._recorded_streams(monkeypatch)
        assert [c["mark_ready"] for c in streams] == [True, False, False]

    def test_algo_orders_go_through_the_same_handler(self, monkeypatch):
        streams = self._recorded_streams(monkeypatch)
        assert streams[-1]["handle"].__func__ is streams[0]["handle"].__func__


class _FakeWsClient:
    """The two things ccxt's orderbook error path touches on a client."""

    def __init__(self, message_hash: str):
        self.subscriptions = {message_hash: True}
        self.rejected: list = []
        self.resolved: list = []

    def reject(self, error, message_hash=None):
        self.rejected.append(error)

    def resolve(self, result, message_hash=None):
        self.resolved.append(result)


def _snapshot(checksum: int) -> dict:
    return {
        "arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
        "action": "snapshot",
        "data": [
            {
                "asks": [["78000.1", "1", "0", "1"]],
                "bids": [["77999.9", "2", "0", "1"]],
                "ts": "1787305636893",
                "checksum": checksum,
                "seqId": 1,
                "prevSeqId": -1,
            }
        ],
    }


class TestOrderBookChecksumFailure:
    """
    Prod, 2026-08-04: a ping-pong timeout, then the reconnect's first snapshot failed its
    checksum and ccxt raised TypeError while building the error message. The raise happens
    before the cleanup, so the subscription was never dropped and the waiter never rejected —
    the stream went quiet with nothing raised to the connection manager.

    ccxt's snapshot branch passes no market and the payload has no instId, so the symbol
    resolves to None. Both master and 4.5.50 still do this.
    """

    MESSAGE_HASH = "books:BTC/USDT:USDT"

    def test_the_base_class_raises_typeerror_instead_of_the_checksum_error(self):
        exchange = cxp.okx()
        exchange.set_markets([_swap_market()])
        with pytest.raises(TypeError):
            exchange.handle_order_book(_FakeWsClient(self.MESSAGE_HASH), _snapshot(checksum=1))

    def test_a_bad_checksum_rejects_the_waiter(self, offline_okx):
        client = _FakeWsClient(self.MESSAGE_HASH)
        offline_okx.handle_order_book(client, _snapshot(checksum=1))
        assert len(client.rejected) == 1
        assert isinstance(client.rejected[0], ChecksumError)

    def test_a_bad_checksum_drops_the_subscription_and_the_stale_book(self, offline_okx):
        client = _FakeWsClient(self.MESSAGE_HASH)
        offline_okx.handle_order_book(client, _snapshot(checksum=1))
        assert self.MESSAGE_HASH not in client.subscriptions
        assert "BTC/USDT:USDT" not in offline_okx.orderbooks

    def test_the_error_is_retried_by_the_connection_manager(self):
        # - ChecksumError is a NetworkError, which listen_to_stream retries; the TypeError was
        #   not raised out of the watch at all, so nothing retried
        assert issubclass(ChecksumError, ccxt.NetworkError)

    def test_a_good_snapshot_keeps_the_book(self, offline_okx):
        client = _FakeWsClient(self.MESSAGE_HASH)
        offline_okx.handle_order_book(client, _snapshot(checksum=1))
        good = _snapshot(checksum=1)
        payload = "77999.9:2:78000.1:1"
        good["data"][0]["checksum"] = offline_okx.crc32(payload, True)
        client = _FakeWsClient(self.MESSAGE_HASH)
        offline_okx.handle_order_book(client, good)
        assert client.rejected == []
        assert "BTC/USDT:USDT" in offline_okx.orderbooks
