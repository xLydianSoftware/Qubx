import asyncio
import socket
import ssl
import sys

import aiohttp
import ccxt.pro as cxp

from ..base import CcxtFuturePatchMixin


class OkxFutures(CcxtFuturePatchMixin, cxp.okx):
    """
    OKX perpetual futures exchange class.

    Sets defaultType to 'swap' for perpetual contracts and applies
    the CcxtFuturePatchMixin for race condition fix.
    Forces IPv4 connections because OKX API key IP whitelisting
    typically only covers IPv4 addresses.
    """

    def describe(self):
        return self.deep_extend(
            super().describe(),
            {
                "options": {
                    "defaultType": "swap",
                    "positionSide": "net",
                },
            },
        )

    def create_order_request(self, symbol: str, type, side, amount: float, price=None, params={}):
        return super().create_order_request(symbol, type, side, amount, price, self._route_protective_stop(params))

    @staticmethod
    def _route_protective_stop(params: dict) -> dict:
        """
        Send a reduce-only stop as OKX's conditional algo order instead of a trigger order.

        OKX answers 51205 "Reduce Only is not available." to ``reduceOnly`` on
        ``ordType=trigger`` and accepts it on ``ordType=conditional`` — measured live, same
        account, instrument and size, one parameter apart. ccxt picks the algo type from the
        parameter name, so moving the level onto ``stopLossPrice`` yields ``slTriggerPx`` with
        ``slOrdPx=-1``, i.e. close at market when the level trades.
        """
        level = params.get("triggerPrice", params.get("stopPrice"))
        if level is None or not params.get("reduceOnly"):
            return params
        rerouted = {k: v for k, v in params.items() if k not in ("triggerPrice", "stopPrice")}
        rerouted["stopLossPrice"] = level
        return rerouted

    async def fetch_open_orders(self, symbol: str | None = None, since=None, limit=None, params={}) -> list:
        """
        List both algo types when the caller asks for trigger orders.

        OKX's pending-algo endpoint takes a single ``ordType`` and ccxt defaults it to
        "trigger", so a reduce-only stop — which goes out as "conditional" — would be absent
        from every snapshot and read as an order the framework does not know about.
        """
        wants_trigger = self.safe_value_2(params, "stop", "trigger")
        if not wants_trigger or self.safe_string(params, "ordType") is not None:
            return await super().fetch_open_orders(symbol, since, limit, params)
        orders = []
        for ord_type in ("trigger", "conditional"):
            orders.extend(await super().fetch_open_orders(symbol, since, limit, {**params, "ordType": ord_type}))
        return orders

    def parse_order(self, order: dict, market=None) -> dict:
        """
        Report OKX algo orders as the framework's stop types, with the trigger as their price.

        ccxt hands the raw ``ordType`` back as the order type, so an algo order reads as
        "trigger"/"conditional" — neither is an ``OrderType``, and a cancel routed by order
        type then misses the venue's algo book entirely. A conditional also leaves ccxt's
        ``triggerPrice`` empty because that field is read from ``triggerPx`` only.
        """
        parsed = super().parse_order(order, market)
        if self.safe_string(order, "ordType") not in ("trigger", "conditional", "oco"):
            return parsed
        limit_price = self.safe_string_2(order, "orderPx", "slOrdPx")
        parsed["type"] = "stop_market" if limit_price in (None, "", "-1") else "stop_limit"
        if parsed.get("triggerPrice") is None:
            parsed["triggerPrice"] = self.safe_number_n(order, ["triggerPx", "slTriggerPx", "tpTriggerPx"])
        return parsed

    def handle_order_book_message(self, client, message, orderbook, messageHash, market=None):
        """
        Give ccxt the market it does not pass on the snapshot path.

        The per-item payload carries no ``instId`` — that sits in ``arg`` — and ccxt's snapshot
        branch calls this without a market, so the symbol resolves to None. It is needed to drop
        the stale book on a checksum failure, and to build the error at all.
        """
        if market is None and orderbook is not None:
            market = self.markets.get(orderbook.get("symbol"))
        return super().handle_order_book_message(client, message, orderbook, messageHash, market)

    def orderbook_checksum_message(self, symbol: str | None) -> str:
        """
        Build the checksum-failure message even when ccxt could not resolve the symbol.

        ccxt's snapshot branch calls ``handle_order_book_message`` without a market and the
        per-item payload carries no ``instId``, so a checksum mismatch on a snapshot resolves
        the symbol to None and the base implementation raises TypeError while building the
        error. That exception skips the subscription cleanup and the waiter is never rejected,
        so the stream stalls with nothing raised to the connection manager.
        """
        return super().orderbook_checksum_message(symbol if symbol is not None else self.id)

    def open(self):
        if self.asyncio_loop is None:
            if sys.version_info >= (3, 7):
                self.asyncio_loop = asyncio.get_running_loop()
            else:
                self.asyncio_loop = asyncio.get_event_loop()
            self.throttler.loop = self.asyncio_loop  # type: ignore

        if self.ssl_context is None:
            # Create our SSL context object with our CA cert file
            self.ssl_context = ssl.create_default_context(cafile=self.cafile) if self.verify else self.verify
            if self.ssl_context and self.safe_bool(self.options, "include_OS_certificates", False):
                os_default_paths = ssl.get_default_verify_paths()
                if os_default_paths.cafile and os_default_paths.cafile != self.cafile:
                    self.ssl_context.load_verify_locations(cafile=os_default_paths.cafile)

        if self.own_session and self.session is None:
            # Pass this SSL context to aiohttp and create a TCPConnector
            self.tcp_connector = aiohttp.TCPConnector(
                ssl=self.ssl_context, loop=self.asyncio_loop, enable_cleanup_closed=True, family=socket.AF_INET
            )
            self.session = aiohttp.ClientSession(
                loop=self.asyncio_loop, connector=self.tcp_connector, trust_env=self.aiohttp_trust_env
            )
