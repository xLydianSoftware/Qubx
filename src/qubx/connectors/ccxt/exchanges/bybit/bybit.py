import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

import ccxt.pro as cxp
from ccxt.base.errors import ArgumentsRequired, BadRequest, OrderNotFound
from ccxt.base.types import Liquidation, Num, Order, OrderSide, OrderType, Position, Str, Strings

from ...adapters.polling_adapter import PollingConfig, PollingToWebSocketAdapter
from ...utils import info_float
from ..base import CcxtFuturePatchMixin

# Bybit answers a redundant set_leverage with this code rather than no-opping.
_LEVERAGE_NOT_MODIFIED = "110043"

# a stop rests on the far side of the market from its own side: BUY above, SELL below
_TRIGGER_DIRECTION = {"buy": "ascending", "sell": "descending"}

# a symbol-less positions/orders read defaults to settleCoin=USDT, so anything settled in
# another coin reads as absent; the endpoints take one settle coin per call
_SETTLE_COINS = ("USDT", "USDC")

# Bybit ranks the ADL queue 1..5, 0 = not ranked; the framework scale is Binance's 0..4,
# higher = closer to the front of the queue.
_ADL_MIN_RANK = 1
_ADL_MAX_RANK = 5

# a crossing post-only order comes back cancelled with this reject-reason family
_POST_ONLY_REFUSAL = "EC_PostOnly"


def _adl_level(position: Any) -> int | None:
    """Bybit's ``adlRankIndicator`` on the framework scale.

    A rank outside 1..5 is dropped, not clamped: 0 means "not ranked", not "safest".
    """
    rank = info_float(position, "adlRankIndicator")
    if rank is None or not (_ADL_MIN_RANK <= rank <= _ADL_MAX_RANK):
        return None
    return int(rank) - _ADL_MIN_RANK


FUNDING_RATE_DEFAULT_POLL_MINUTES = 5


class BybitF(CcxtFuturePatchMixin, cxp.bybit):
    """Bybit perpetual futures: ccxt patches for the calls Bybit does not accept as written."""

    # the only depths Bybit's WS book serves; anything else is a BadRequest at subscribe
    _WS_DEPTHS = (1, 50, 200, 1000)

    def __init__(self, config=None):
        super().__init__(config or {})
        self._funding_rate_adapter: PollingToWebSocketAdapter | None = None

    def describe(self):
        return self.deep_extend(
            super().describe(),
            {
                "has": {
                    "unWatchBidsAsks": True,
                    "watchFundingRates": True,
                    "watchLiquidationsForSymbols": True,
                },
                "options": {
                    "fetchOrder": {"acknowledged": True},
                    # caps the cursor walk armed by params={"paginate": True}; read by the
                    # pagination driver only, so it cannot re-arm on the recursive call
                    "fetchOpenOrders": {"paginationCalls": 20},
                    "fetchPositions": {"paginationCalls": 20},
                },
            },
        )

    def _settle_filtered(self, params: dict) -> bool:
        return any(params.get(k) is not None for k in ("settleCoin", "baseCoin", "symbol"))

    async def _per_settle_coin(self, fetch: Callable[[dict], Awaitable[list]], params: dict) -> list:
        """One paginated call per settle coin, concatenated.

        The row caps — /v5/order/realtime serves 20, ccxt pins /v5/position/list at 200 — need
        the cursor walk, and ``paginate`` must ride params, not options: the recursive call
        carries ``settleCoin``, so it lands back on the filtered branch and ends at ``super()``.
        """
        legs = await asyncio.gather(
            *(fetch(self.extend(params, {"settleCoin": coin, "paginate": True})) for coin in _SETTLE_COINS)
        )
        return [row for leg in legs for row in leg]

    async def fetch_positions(self, symbols: Strings = None, params={}) -> list[Any]:
        if symbols or self._settle_filtered(params):
            return await super().fetch_positions(symbols, params)
        return await self._per_settle_coin(partial(super(BybitF, self).fetch_positions, symbols), params)

    def parse_position(self, position, market=None) -> Position:
        """Keep the venue's own maintenance margin and put the ADL rank on the unified key.

        ccxt reads ``positionMM`` and then replaces it with ``|liqPrice - bustPrice| * size``
        whenever a liquidation price is set; a UTA account reports ``bustPrice=""``, so the
        product is None and the framework falls back to a flat 5% of notional.

        ``info`` is ccxt's passthrough of the raw row, which is where ``ccxt_convert_position``
        reads ``adl`` — Bybit spells the same figure ``adlRankIndicator`` on a 1-based scale.
        """
        parsed = super().parse_position(position, market)
        if parsed.get("maintenanceMargin") is None:
            venue_mm = info_float(position, "positionMM")
            if venue_mm is not None:
                parsed["maintenanceMargin"] = venue_mm
        adl = _adl_level(position)
        if adl is not None:
            parsed["info"] = {**position, "adl": adl}
        return parsed

    async def fetch_open_orders(
        self, symbol: Str = None, since: Num = None, limit: Num = None, params={}
    ) -> list[Order]:
        if symbol is not None or self._settle_filtered(params):
            return await super().fetch_open_orders(symbol, since, limit, params)
        return await self._per_settle_coin(partial(super(BybitF, self).fetch_open_orders, symbol, since, limit), params)

    def create_order_request(
        self,
        symbol: str,
        type: OrderType,
        side: OrderSide,
        amount: float,
        price: Num = None,
        params={},
        isUTA=True,
    ):
        return super().create_order_request(
            symbol, type, side, amount, price, self._with_trigger_direction(side, params), isUTA
        )

    @staticmethod
    def _with_trigger_direction(side: str, params: dict) -> dict:
        """ccxt refuses every non-spot trigger order that does not carry ``triggerDirection``."""
        if params.get("triggerDirection") is not None:
            return params
        if params.get("triggerPrice") is None and params.get("stopPrice") is None:
            return params
        return {**params, "triggerDirection": _TRIGGER_DIRECTION[side.lower()]}

    def parse_order(self, order, market=None):
        """Report a post-only refusal as rejected, not cancelled.

        Bybit accepts a crossing post-only order and then cancels it with
        ``rejectReason=EC_PostOnlyWillTakeLiquidity`` instead of refusing it synchronously.
        """
        parsed = super().parse_order(order, market)
        if parsed.get("status") == "canceled":
            reason = self.safe_string(parsed.get("info") or {}, "rejectReason", "")
            if reason.startswith(_POST_ONLY_REFUSAL):
                parsed["status"] = "rejected"
        return parsed

    async def set_leverage(self, leverage: int, symbol: Str = None, params={}):
        """Treat a redundant set as the silent no-op every other venue answers with."""
        try:
            return await super().set_leverage(leverage, symbol, params)
        except BadRequest as e:
            if _LEVERAGE_NOT_MODIFIED not in str(e):
                raise
            return {}

    def cancel_order_request(self, id: str, symbol: Str = None, params={}):
        link_id = self.safe_string_2(params, "orderLinkId", "clientOrderId")
        if link_id is None:
            return super().cancel_order_request(id, symbol, params)
        request = super().cancel_order_request(id, symbol, self.omit(params, ["clientOrderId", "orderLinkId"]))
        request.pop("orderId", None)
        request["orderLinkId"] = link_id
        return request

    async def fetch_order(self, id: str, symbol: Str = None, params={}) -> Order:
        """Look up an order, retrying without the trigger filter before believing it is gone.

        ``trigger``/``stop`` becomes ``orderFilter=StopOrder``, which holds only UNTRIGGERED
        conditionals — a fired stop moves to the regular book but still reads back STOP_MARKET.
        """
        try:
            return await self._fetch_order(id, symbol, params)
        except OrderNotFound:
            if not self.safe_bool_2(params, "trigger", "stop", False):
                raise
        return await self._fetch_order(id, symbol, self.omit(params, ["trigger", "stop"]))

    async def _fetch_order(self, id: str, symbol: Str, params: dict) -> Order:
        """Route a client-id lookup through the order-list endpoints, the only ones that accept
        ``orderLinkId``."""
        link_id = self.safe_string_2(params, "orderLinkId", "clientOrderId")
        if link_id is None:
            return await super().fetch_order(id, symbol, params)
        rest = self.omit(params, ["clientOrderId", "orderLinkId"])
        for fetch in (self.fetch_open_orders, self.fetch_canceled_and_closed_orders):
            rows = await fetch(symbol, None, None, self.extend(rest, {"orderLinkId": link_id}))
            # match explicitly rather than trusting the venue-side filter
            for row in rows:
                if row.get("clientOrderId") == link_id:
                    return row
        raise OrderNotFound(f"{self.id} order with orderLinkId {link_id} was not found")

    async def watch_bids_asks(self, symbols: Strings = None, params={}) -> dict:
        """Take quotes off ``tickers``, not ``orderbook.1``.

        ccxt's ``watchBidsAsks`` writes its one-level book into the same
        ``self.orderbooks[symbol]`` the depth subscription fills, collapsing the L2 book.
        """
        tickers = await self.watch_tickers(symbols, params)
        return {s: t for s, t in tickers.items() if t.get("bid") is not None and t.get("ask") is not None}

    async def un_watch_bids_asks(self, symbols: Strings = None, params={}) -> Any:
        """Release the tickers topic ``watch_bids_asks`` subscribed; upstream has no unwatch."""
        return await self.un_watch_tickers(symbols, params)

    async def watch_funding_rates(
        self, symbols: list[str] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Present ``fetch_funding_rates`` as a watch, on a poll.

        Bybit has no funding WS channel, and ``tickers`` carries the rate on its snapshot frame
        only — a topic ``watch_bids_asks`` already owns.
        """
        params = params or {}
        await self.load_markets()
        poll_minutes = params.get("poll_interval_minutes", FUNDING_RATE_DEFAULT_POLL_MINUTES)

        if self._funding_rate_adapter is None:
            self._funding_rate_adapter = PollingToWebSocketAdapter(
                fetch_method=self._fetch_funding_rates_for,
                symbols=symbols or [],
                params=params,
                config=PollingConfig(poll_interval_seconds=poll_minutes * 60),
            )
        elif symbols is not None:
            await self._funding_rate_adapter.update_symbols(symbols)

        rates = await self._funding_rate_adapter.get_next_data()
        # ccxt's bybit parser writes the next funding time as `fundingTimestamp`;
        # ccxt_convert_funding_rate reads `nextFundingTime`
        return {
            symbol: {**info, "nextFundingTime": info.get("fundingTimestamp")}
            for symbol, info in (rates or {}).items()
            if isinstance(info, dict)
        }

    async def _fetch_funding_rates_for(self, symbols: list[str], **params) -> dict[str, Any]:
        # bybit's fetch does market(symbols[0]) on an empty list -> IndexError
        return await self.fetch_funding_rates(symbols or None, params)

    async def un_watch_funding_rates(self, symbols: list[str] | None = None) -> None:
        """The handler's cleanup calls this with no arguments, meaning "stop everything"."""
        adapter = self._funding_rate_adapter
        if adapter is None:
            return
        if symbols:
            await adapter.remove_symbols(symbols)
            if adapter.is_watching():
                return
        await adapter.stop()
        self._funding_rate_adapter = None

    def _ws_depth(self, limit: int | None) -> int:
        """Round a requested depth up to one Bybit serves; 50 is ccxt's linear default, pinned
        here so the unwatch topic matches the subscribed one."""
        return next((d for d in self._WS_DEPTHS if d >= (limit or 50)), self._WS_DEPTHS[-1])

    async def watch_order_book_for_symbols(self, symbols: list[str], limit: int | None = None, params={}):
        return await super().watch_order_book_for_symbols(symbols, self._ws_depth(limit), params)

    async def un_watch_order_book_for_symbols(self, symbols: list[str], limit: int | None = None, params={}) -> Any:
        """Upstream takes the depth via params and otherwise defaults to 500, an illegal topic."""
        return await super().un_watch_order_book_for_symbols(
            symbols, self.extend(params, {"limit": self._ws_depth(limit)})
        )

    async def un_watch_order_book(self, symbol: str, limit: int | None = None, params={}) -> Any:
        # upstream passes params as the second positional, which is our limit
        return await self.un_watch_order_book_for_symbols([symbol], limit, params)

    async def watch_liquidations_for_symbols(
        self, symbols: list[str], since: int | None = None, limit: int | None = None, params={}
    ) -> list[Liquidation]:
        """Subscribe ``allLiquidation`` for many symbols at once: ccxt ships only the
        single-symbol watch, and no unwatch for liquidations on any venue."""
        if not symbols:
            raise ArgumentsRequired(f"{self.id} watchLiquidationsForSymbols() requires a non-empty array of symbols")
        await self.load_markets()
        symbols = self.market_symbols(symbols)
        url = await self.get_url_by_market_type(symbols[0], False, "watchLiquidations", params)
        params = self.clean_params(params)
        method, params = self.handle_option_and_params(params, "watchLiquidations", "method", "allLiquidation")
        topics = [method + "." + self.market(symbol)["id"] for symbol in symbols]
        hashes = ["liquidations::" + symbol for symbol in symbols]
        liquidations = await self.watch_topics(url, hashes, topics, params)
        if self.newUpdates:
            return liquidations
        return self.filter_by_symbols_since_limit(self.liquidations, symbols, since, limit, True)

    def parse_ws_liquidation(self, liquidation, market=None) -> Liquidation:
        """Report the side of the liquidating ORDER, as other venues do.

        ccxt passes ``'S'`` as ``safe_string_lower``'s default value instead of as a second key,
        and Bybit's ``S`` is the POSITION side — the inverse of Binance's ``forceOrder``.
        """
        parsed = super().parse_ws_liquidation(liquidation, market)
        position_side = self.safe_string_lower_2(liquidation, "side", "S")
        parsed["side"] = "sell" if position_side == "buy" else "buy"
        return parsed
