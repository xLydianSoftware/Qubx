import asyncio
import socket
import ssl

import aiohttp
import ccxt.pro as cxp
from ccxt.base.errors import ExchangeClosedByUser

from qubx import logger

from ..base import CcxtFuturePatchMixin


class OkxFutures(CcxtFuturePatchMixin, cxp.okx):
    """
    OKX perpetual futures exchange class.

    Sets defaultType to 'swap' for perpetual contracts and applies
    the CcxtFuturePatchMixin for race condition fix.
    Forces IPv4 connections because OKX API key IP whitelisting
    typically only covers IPv4 addresses.
    """

    def __init__(self, config=None):
        super().__init__(config or {})
        self._funding_read_warned = False

    def describe(self):
        return self.deep_extend(
            super().describe(),
            {
                "options": {
                    "defaultType": "swap",
                    "positionSide": "net",
                    # - OKX sends `checksum: 0` on every books message. Measured 2026-08-21 on
                    #   wss://ws.okx.com:8443/ws/v5/public, BTC-USDT-SWAP and ETH-USDT-SWAP,
                    #   snapshot and updates alike:
                    #       action=snapshot checksum=0 seqId=335226848531 prevSeqId=-1
                    #       action=update   checksum=0 seqId=335226848654 prevSeqId=335226848531
                    #   A crc32 is never 0, so ccxt's check fails on the first update after the
                    #   snapshot, always. Building the payload from the venue's own raw strings
                    #   gives the same 846098866 as ccxt's rendering of it — the book is fine,
                    #   there is simply nothing to compare against. Below ccxt 4.5.55 that failure
                    #   ends the stream for good: the error reaches the connection manager, its
                    #   retry re-awaits the same watch and never returns (twice on the prod
                    #   reversals config, dead 1.4s after subscribing, quotes frozen six minutes
                    #   later). ccxt deleted the check in 4.5.55; this carries that to older
                    #   builds and is a no-op on newer ones. seqId/prevSeqId continuity still runs
                    #   and is the real guard on this stream.
                    "watchOrderBook": {"checksum": False},
                    # - ccxt moved the default from tickers to bbo-tbt after 4.5.50 (tick-by-tick L1,
                    #   ~10ms vs 100ms); quotes stay on the tickers channel they have always used
                    "watchBidsAsks": {"channel": "tickers"},
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

    async def fetch_balance(self, params={}):
        """Trading-account balance with the funding-account rows (``GET /api/v5/asset/balances``)
        grafted onto ``info["funding"]``; None when that read fails."""
        balances = await super().fetch_balance(params)
        funding = None
        try:
            funding = (await self.privateGetAssetBalances()).get("data")
        except Exception as e:  # noqa: BLE001 — the funding leg is decoration; never sink the snapshot
            if self._funding_read_warned:
                logger.debug("[okx] funding balance read failed: {}", e)
            else:
                self._funding_read_warned = True
                logger.warning("[okx] funding balance read failed, funding wallet unreported: {}", e)
        info = balances.get("info")
        if isinstance(info, dict):
            info["funding"] = funding
        return balances

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

    def open(self, lazy=False):
        if lazy and self.closed_by_user:
            raise ExchangeClosedByUser(self.id + " instance was closed by the user")
        self.closed_by_user = False
        if self.asyncio_loop is None:
            self.asyncio_loop = asyncio.get_running_loop()
            self.throttler.loop = self.asyncio_loop  # type: ignore

        if self.ssl_context is None:
            # Create our SSL context object with our CA cert file
            self.ssl_context = ssl.create_default_context(cafile=self.cafile) if self.verify else self.verify
            if self.ssl_context and self.safe_bool(self.options, "include_OS_certificates", False):
                os_default_paths = ssl.get_default_verify_paths()
                if os_default_paths.cafile and os_default_paths.cafile != self.cafile:
                    self.ssl_context.load_verify_locations(cafile=os_default_paths.cafile)

        if self.own_session and self.session is None:
            # - upstream's AF_UNSPEC + happy_eyeballs_delay=0 left out: IPv4 only (see class doc), and
            #   connection attempts to the resolved addresses stay sequential as before
            self.tcp_connector = aiohttp.TCPConnector(
                ssl=self.ssl_context, loop=self.asyncio_loop, enable_cleanup_closed=True, family=socket.AF_INET
            )
            self.session = aiohttp.ClientSession(
                loop=self.asyncio_loop, connector=self.tcp_connector, trust_env=self.aiohttp_trust_env
            )
