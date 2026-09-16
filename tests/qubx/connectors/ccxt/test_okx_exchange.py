"""Tests for OKX exchange registration and custom class."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import ccxt
import ccxt.pro as cxp
import pytest
from ccxt.base.errors import ChecksumError

from qubx import logger
from qubx.connectors.ccxt.exchanges import EXCHANGE_ALIASES, OkxFutures
from qubx.connectors.ccxt.exchanges.okx.connector import OkxCcxtConnector
from qubx.connectors.ccxt.utils import ccxt_status_to_order_status
from qubx.core.basics import OrderStatus
from qubx.core.basics import CtrlChannel, Instrument, MarketType


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
        self._checksums_on(exchange)
        with pytest.raises(TypeError):
            exchange.handle_order_book(_FakeWsClient(self.MESSAGE_HASH), _snapshot(checksum=1))

    @staticmethod
    def _checksums_on(exchange) -> None:
        # - OkxFutures ships with them off (see TestOrderBookChecksumDisabled); these two
        #   cover the guard for anyone who turns them back on, and for other venues
        exchange.options["watchOrderBook"]["checksum"] = True

    def test_a_bad_checksum_rejects_the_waiter(self, offline_okx):
        self._checksums_on(offline_okx)
        client = _FakeWsClient(self.MESSAGE_HASH)
        offline_okx.handle_order_book(client, _snapshot(checksum=1))
        assert len(client.rejected) == 1
        assert isinstance(client.rejected[0], ChecksumError)

    def test_a_bad_checksum_drops_the_subscription_and_the_stale_book(self, offline_okx):
        self._checksums_on(offline_okx)
        client = _FakeWsClient(self.MESSAGE_HASH)
        offline_okx.handle_order_book(client, _snapshot(checksum=1))
        assert self.MESSAGE_HASH not in client.subscriptions
        assert "BTC/USDT:USDT" not in offline_okx.orderbooks

    def test_by_default_a_bad_checksum_is_ignored(self, offline_okx):
        # - what keeps the stream alive: nothing is raised, nothing is torn down
        client = _FakeWsClient(self.MESSAGE_HASH)
        offline_okx.handle_order_book(client, _snapshot(checksum=1))
        assert client.rejected == []
        assert self.MESSAGE_HASH in client.subscriptions
        assert "BTC/USDT:USDT" in offline_okx.orderbooks

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


class TestLeverageReads:
    """
    ccxt's okx has neither `fetchLeverages` nor `fetchLeverageTiers` — the two whole-universe
    calls the base poller uses — so nothing fills the cache and the getters have nothing to
    read until a batch lands. The reads must not block: on the 5s state snapshot the caller is
    the ProcessorThread, once per universe instrument.

    Measured on mainnet with ccxt 4.5.50: `account/leverage-info` takes up to 20 comma-separated
    instIds and 20 of them cost 270ms against 279ms for one, so the configured leverage is read
    in batches; the venue cap is in the market metadata already loaded, so it costs nothing.
    """

    @staticmethod
    def _connector(exchange: Mock) -> OkxCcxtConnector:
        exchange_manager = Mock()
        exchange_manager.exchange = exchange
        exchange_manager.rate_limiter = None
        connector = OkxCcxtConnector(
            exchange_name="OKX.F",
            channel=Mock(spec=CtrlChannel),
            time_provider=Mock(),
            exchange_manager=exchange_manager,
            data_provider=Mock(),
        )
        # No _run_sync stub: a read that blocked again would reach the Mock exchange's loop
        # and fail loudly instead of quietly passing.
        captured: list = []
        connector._spawn = lambda coro: captured.append(coro)  # type: ignore[method-assign]
        connector._captured = captured  # type: ignore[attr-defined]
        connector._leverage_flush_debounce_s = 0.0
        return connector

    @staticmethod
    def _drive(connector: OkxCcxtConnector) -> None:
        """Run the flushes the reads scheduled, the way the exchange loop would."""
        scheduled = list(connector._captured)  # type: ignore[attr-defined]
        connector._captured.clear()  # type: ignore[attr-defined]
        for coro in scheduled:
            run(coro)

    @staticmethod
    def _discard(connector: OkxCcxtConnector) -> None:
        """Drop flushes a test deliberately never drives. An open coroutine surfaces as a
        `never awaited` RuntimeWarning from pytest's gc, attributed to a later, unrelated test."""
        for coro in connector._captured:  # type: ignore[attr-defined]
            coro.close()
        connector._captured.clear()  # type: ignore[attr-defined]

    @staticmethod
    def _symbol(base: str) -> str:
        return f"{base}/USDT:USDT"

    @staticmethod
    def _inst_id(base: str) -> str:
        return f"{base}-USDT-SWAP"

    _RAW_SWAP = {
        "instType": "SWAP",
        "baseCcy": "",
        "quoteCcy": "",
        "settleCcy": "USDT",
        "ctVal": "1",
        "ctMult": "1",
        "optType": "",
        "stk": "",
        "listTime": "1700000000000",
        "expTime": "",
        "tickSz": "0.001",
        "lotSz": "1",
        "minSz": "1",
        "ctType": "linear",
        "state": "live",
    }

    @classmethod
    def _markets(cls, *bases: str, lever: str | None = "125") -> dict:
        """Markets as ccxt's own okx parser produces them, not as we imagine it does.

        ``lever=None`` is the venue publishing no cap: ccxt normalises that to
        ``limits.leverage.max == 1.0``, which is why the getter reads ``info.lever``.
        """
        parser = ccxt.okx()
        markets = {}
        for base in bases:
            raw = dict(
                cls._RAW_SWAP,
                instId=cls._inst_id(base),
                uly=f"{base}-USDT",
                instFamily=f"{base}-USDT",
                ctValCcy=base,
            )
            if lever is not None:
                raw["lever"] = lever
            market = parser.parse_market(raw)
            markets[market["symbol"]] = market
        return markets

    @classmethod
    def _rows(cls, *bases: str, lever: str = "5", pos_side: str = "net") -> dict:
        return {"code": "0", "data": [{"instId": cls._inst_id(b), "posSide": pos_side, "lever": lever} for b in bases]}

    @classmethod
    def _exchange(cls, *bases: str, **overrides) -> Mock:
        exchange = Mock()
        exchange.has = {}
        exchange.fetch_positions = AsyncMock(return_value=[])
        exchange.markets = cls._markets(*(bases or ("BTC",)))
        exchange.privateGetAccountLeverageInfo = AsyncMock(return_value=cls._rows(*(bases or ("BTC",))))
        for k, v in overrides.items():
            setattr(exchange, k, v)
        return exchange

    @staticmethod
    def _instrument(base: str = "BTC") -> Instrument:
        return Instrument(
            symbol=f"{base}USDT",
            market_type=MarketType.SWAP,
            exchange="OKX.F",
            base=base,
            quote="USDT",
            settle="USDT",
            exchange_symbol=f"{base}-USDT-SWAP",
            tick_size=0.1,
            lot_size=0.01,
            min_size=0.01,
        )

    @staticmethod
    def _asked_ids(exchange: Mock) -> list[list[str]]:
        """The instId lists of every leverage-info call, in order."""
        return [call.args[0]["instId"].split(",") for call in exchange.privateGetAccountLeverageInfo.await_args_list]

    # -- the cap: market metadata, never the venue ------------------------------- #

    @pytest.mark.parametrize("lever, expected", [("125", 125.0), ("50", 50.0)])
    def test_the_cap_comes_from_the_market_with_no_venue_call(self, lever, expected):
        exchange = self._exchange()
        exchange.markets = self._markets("BTC", lever=lever)
        connector = self._connector(exchange)

        assert connector.get_max_instrument_leverage(self._instrument()) == expected
        assert connector._captured == []
        exchange.privateGetAccountLeverageInfo.assert_not_awaited()

    def test_a_market_without_a_published_cap_reads_none(self):
        """ccxt defaults an absent `lever` to a cap of 1.0. Reading that would have the write
        path clamp a 3x default down to 1x and call it the venue's decision, where None means
        "unknown, send it unclamped"."""
        exchange = self._exchange()
        exchange.markets = self._markets("BTC", lever=None)
        connector = self._connector(exchange)
        assert exchange.markets["BTC/USDT:USDT"]["limits"]["leverage"]["max"] == 1.0

        assert connector.get_max_instrument_leverage(self._instrument()) is None

    def test_a_market_with_an_empty_cap_reads_none(self):
        exchange = self._exchange()
        exchange.markets = self._markets("BTC", lever="")
        connector = self._connector(exchange)

        assert connector.get_max_instrument_leverage(self._instrument()) is None

    def test_an_unloaded_market_reads_none(self):
        exchange = self._exchange()
        exchange.markets = {}
        connector = self._connector(exchange)

        assert connector.get_max_instrument_leverage(self._instrument()) is None

    def test_the_write_path_still_clamps_to_the_cap(self):
        """The cap is no longer in `_leverage_cache` at all, so the clamp has to reach the
        getter — otherwise every OKX request would go to the venue unclamped."""
        exchange = self._exchange()
        exchange.set_leverage = AsyncMock(return_value={})
        connector = self._connector(exchange)

        connector.set_instrument_leverage(self._instrument(), 500.0)
        self._drive(connector)

        exchange.set_leverage.assert_awaited_once_with(125, "BTC/USDT:USDT")

    # -- configured leverage: batched ------------------------------------------- #

    def test_the_first_read_answers_none_without_a_venue_call(self):
        exchange = self._exchange()
        connector = self._connector(exchange)

        assert connector.get_instrument_leverage(self._instrument()) is None
        exchange.privateGetAccountLeverageInfo.assert_not_awaited()
        self._discard(connector)

    def test_one_tick_of_misses_leaves_as_a_single_call(self):
        bases = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
        exchange = self._exchange(*bases)
        connector = self._connector(exchange)

        for base in bases:
            assert connector.get_instrument_leverage(self._instrument(base)) is None
        assert len(connector._captured) == 1

        self._drive(connector)

        assert self._asked_ids(exchange) == [sorted(self._inst_id(b) for b in bases)]
        assert connector.get_instrument_leverage(self._instrument("DOGE")) == 5.0

    def test_more_than_twenty_misses_split_into_chunks_of_twenty(self):
        """21 instIds is error 50025 "Parameter instId count exceeds the limit 20"."""
        bases = [f"C{i:02d}" for i in range(25)]
        exchange = self._exchange(*bases)
        connector = self._connector(exchange)

        for base in bases:
            connector.get_instrument_leverage(self._instrument(base))
        self._drive(connector)

        assert [len(ids) for ids in self._asked_ids(exchange)] == [20, 5]
        assert sorted(sum(self._asked_ids(exchange), [])) == sorted(self._inst_id(b) for b in bases)
        assert len(connector._leverage_cache) == 25

    def test_a_miss_during_a_flush_lands_in_the_next_batch(self):
        exchange = self._exchange("BTC", "ETH")
        connector = self._connector(exchange)
        connector.get_instrument_leverage(self._instrument("BTC"))

        # the ETH read arrives while the venue is answering the BTC batch
        async def _answer(params):
            inst_ids = params["instId"].split(",")
            if self._inst_id("BTC") in inst_ids:
                connector.get_instrument_leverage(self._instrument("ETH"))
            return self._rows(*(i.split("-")[0] for i in inst_ids))

        exchange.privateGetAccountLeverageInfo = AsyncMock(side_effect=_answer)
        self._drive(connector)

        assert self._asked_ids(exchange) == [[self._inst_id("BTC")], [self._inst_id("ETH")]]
        assert connector.get_instrument_leverage(self._instrument("ETH")) == 5.0
        assert connector._leverage_pending == set()
        assert connector._leverage_flush_scheduled is False

    @pytest.mark.parametrize(
        "rows, expected",
        [
            ([("short", "7"), ("long", "3")], 3.0),
            ([("long", "3"), ("short", "7")], 3.0),
            ([("net", "5")], 5.0),
            ([("short", "7")], 7.0),
        ],
        ids=["short-then-long", "long-then-short", "net", "short-only"],
    )
    def test_the_rows_collapse_to_one_configured_value(self, rows, expected):
        """One row per (instId, posSide): "net" on a one-way account, "long"/"short" hedged.
        The long side wins whichever order the venue lists them in."""
        exchange = self._exchange()
        exchange.privateGetAccountLeverageInfo = AsyncMock(
            return_value={
                "code": "0",
                "data": [{"instId": self._inst_id("BTC"), "posSide": side, "lever": lever} for side, lever in rows],
            }
        )
        connector = self._connector(exchange)
        connector.get_instrument_leverage(self._instrument())
        self._drive(connector)

        assert connector.get_instrument_leverage(self._instrument()) == expected

    def test_a_symbol_the_venue_returned_no_row_for_is_asked_once(self):
        exchange = self._exchange("BTC", "ETH")
        exchange.privateGetAccountLeverageInfo = AsyncMock(return_value=self._rows("BTC"))
        connector = self._connector(exchange)
        for base in ("BTC", "ETH"):
            connector.get_instrument_leverage(self._instrument(base))
        self._drive(connector)

        assert connector.get_instrument_leverage(self._instrument("ETH")) is None
        assert connector._captured == []
        # a cache entry all the same, so the hourly refresh is what retries it
        assert "ETH/USDT:USDT" in connector._leverage_cache

    def test_a_read_after_the_batch_costs_no_venue_call(self):
        exchange = self._exchange()
        connector = self._connector(exchange)
        connector.get_instrument_leverage(self._instrument())
        self._drive(connector)

        for _ in range(3):
            assert connector.get_instrument_leverage(self._instrument()) == 5.0
        assert connector._captured == []
        assert exchange.privateGetAccountLeverageInfo.await_count == 1

    def test_an_adopted_write_does_not_suppress_the_read(self):
        """`set_instrument_leverage` seats a cache entry of its own; gating on the cache would
        leave the read path silent for every instrument the default-leverage apply touched."""
        exchange = self._exchange()
        exchange.set_leverage = AsyncMock(return_value={})
        connector = self._connector(exchange)
        connector.set_instrument_leverage(self._instrument(), 10.0)
        self._drive(connector)
        assert connector._leverage_cache["BTC/USDT:USDT"].configured == 10

        assert connector.get_instrument_leverage(self._instrument()) == 10.0
        connector._leverage_cache.pop("BTC/USDT:USDT")
        assert connector.get_instrument_leverage(self._instrument()) is None
        assert len(connector._captured) == 1

    def test_a_failing_call_leaves_the_symbols_unprobed_and_backs_off(self):
        """Unprobed so a blip recovers, but behind a backoff so a permanent fault (a key with
        no account read, an account type that rejects cross) costs one call a minute rather
        than one per symbol per 5s tick."""
        exchange = self._exchange(privateGetAccountLeverageInfo=AsyncMock(side_effect=ccxt.NotSupported("nope")))
        connector = self._connector(exchange)
        connector.get_instrument_leverage(self._instrument())
        self._drive(connector)

        assert connector._leverage_probed == set()
        assert connector._leverage_cache == {}
        assert connector._leverage_flush_backoff_until - time.monotonic() == pytest.approx(60.0, abs=1.0)

        connector.get_instrument_leverage(self._instrument())
        assert connector._captured == []

        connector._leverage_flush_backoff_until = 0.0
        connector.get_instrument_leverage(self._instrument())
        assert len(connector._captured) == 1
        self._discard(connector)
        assert exchange.privateGetAccountLeverageInfo.await_count == 1

    def test_a_symbol_already_queued_still_starts_a_flush(self):
        """The queue is never consulted to decide whether to schedule: a symbol left in it by
        the race below would otherwise be skipped forever, with no task left to fetch it."""
        exchange = self._exchange()
        connector = self._connector(exchange)
        connector._leverage_pending.add(self._symbol("BTC"))

        connector.get_instrument_leverage(self._instrument())
        assert len(connector._captured) == 1

        self._drive(connector)
        assert connector.get_instrument_leverage(self._instrument()) == 5.0

    def test_a_symbol_queued_during_the_drain_is_flushed_without_another_miss(self):
        """The drain's last look at the queue and the flag going down are two steps, and a miss
        landing between them sees the flag still set and does not spawn. Reproduced with a queue
        that answers the drain empty and then holds the late symbol — an interleaving one thread
        cannot otherwise produce."""

        class _RacingQueue(set):
            def __init__(self, late: str) -> None:
                super().__init__()
                self._late: str | None = late

            def __iter__(self):
                if self._late is None:
                    return super().__iter__()
                late, self._late = self._late, None
                empty = iter(())  # what the drain's emptiness check sees
                super().add(late)  # the miss that lands before the flag goes down
                return empty

        exchange = self._exchange()
        connector = self._connector(exchange)
        connector._leverage_pending = _RacingQueue(self._symbol("BTC"))
        connector._ensure_flush_scheduled()

        self._drive(connector)  # the flush finds nothing, then re-schedules for the late symbol
        assert len(connector._captured) == 1
        self._drive(connector)

        assert connector.get_instrument_leverage(self._instrument()) == 5.0
        assert connector._leverage_flush_scheduled is False

    def test_a_loop_that_is_gone_neither_raises_nor_wedges_the_symbol(self):
        """The read runs on the ProcessorThread inside the 5s snapshot: raising there loses the
        tick for every exchange, and leaving the flush flagged loses it for good."""
        exchange = self._exchange()
        connector = self._connector(exchange)
        connector._spawn = Mock(side_effect=RuntimeError("Event loop is closed"))

        assert connector.get_instrument_leverage(self._instrument()) is None
        assert connector._leverage_pending == set()
        assert connector._leverage_flush_scheduled is False
        assert connector._leverage_probed == set()

        connector._spawn = lambda coro: connector._captured.append(coro)
        connector.get_instrument_leverage(self._instrument())
        self._drive(connector)
        assert connector.get_instrument_leverage(self._instrument()) == 5.0

    def test_a_flush_that_finishes_inside_the_schedule_leaves_nothing_flagged(self):
        """The loop thread can run the whole flush before the caller's next bytecode; flagging
        after the spawn would let the flush's clear land first and strand the flag."""
        exchange = self._exchange()
        connector = self._connector(exchange)
        connector._spawn = run  # completes the coroutine before returning

        assert connector.get_instrument_leverage(self._instrument()) is None

        assert connector._leverage_pending == set()
        assert connector._leverage_flush_scheduled is False
        assert connector.get_instrument_leverage(self._instrument()) == 5.0

    def test_a_cancelled_flush_leaves_the_symbols_unprobed(self):
        """Probed has to mean "a call came back": the hourly refresh iterates the cache, so a
        symbol marked probed with no cache entry would be retried by neither it nor the read."""
        exchange = self._exchange(privateGetAccountLeverageInfo=AsyncMock(side_effect=asyncio.CancelledError))
        connector = self._connector(exchange)
        connector.get_instrument_leverage(self._instrument())

        with pytest.raises(asyncio.CancelledError):
            self._drive(connector)

        assert connector._leverage_probed == set()
        assert connector._leverage_pending == set()
        assert connector._leverage_flush_scheduled is False
        assert connector._leverage_cache == {}

    # -- the hourly refresh ------------------------------------------------------ #

    def test_the_refresh_batches_the_symbols_the_cache_holds(self):
        bases = [f"C{i:02d}" for i in range(22)]
        exchange = self._exchange(*bases)
        connector = self._connector(exchange)
        for base in bases:
            connector.get_instrument_leverage(self._instrument(base))
        self._drive(connector)
        exchange.privateGetAccountLeverageInfo = AsyncMock(return_value=self._rows(*bases, lever="9"))

        run(connector._refresh_leverage_cache())

        assert [len(ids) for ids in self._asked_ids(exchange)] == [20, 2]
        assert connector.get_instrument_leverage(self._instrument("C21")) == 9.0

    def test_the_refresh_reads_nothing_when_the_cache_is_empty(self):
        exchange = self._exchange()
        connector = self._connector(exchange)
        run(connector._refresh_leverage_cache())
        exchange.privateGetAccountLeverageInfo.assert_not_awaited()


class TestOrderBookChecksumDisabled:
    """
    OKX sends `checksum: 0` on every books message (measured 2026-08-21, BTC and ETH swaps,
    snapshot and updates), and a crc32 is never 0 — so ccxt's check fails on the first update
    after the snapshot, always. Below ccxt 4.5.55 that ended the stream 1.4s after subscribing
    on the prod reversals config: the error reached the connection manager, whose retry
    re-awaits the same watch and never returns. ccxt deleted the check in 4.5.55.
    """

    def test_okx_futures_turns_the_checksum_off(self):
        assert OkxFutures().describe()["options"]["watchOrderBook"]["checksum"] is False

    def test_the_base_class_leaves_it_on(self):
        # - so the override cannot quietly stop mattering on a ccxt that still checksums
        base = cxp.okx().describe()["options"].get("watchOrderBook", {})
        assert base.get("checksum", True) is not False

    def test_the_book_depth_is_untouched(self):
        options = OkxFutures().describe()["options"]["watchOrderBook"]
        assert options.get("depth") == cxp.okx().describe()["options"].get("watchOrderBook", {}).get("depth")
