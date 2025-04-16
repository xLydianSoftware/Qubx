import asyncio

from qubx.connectors.ccxt.account import CcxtAccountProcessor
from qubx.connectors.ccxt.utils import (
    ccxt_convert_order_info,
    ccxt_extract_deals_from_exec,
    ccxt_find_instrument,
)
from qubx.core.basics import CtrlChannel


class BitfinexAccountProcessor(CcxtAccountProcessor):
    async def _subscribe_executions(self, name: str, channel: CtrlChannel):
        _symbol_to_instrument = {}

        async def _watch_orders():
            orders = await self.exchange.watch_orders()
            for order in orders:
                instrument = ccxt_find_instrument(order["symbol"], self.exchange, _symbol_to_instrument)
                order = ccxt_convert_order_info(instrument, order)
                channel.send((instrument, "order", order, False))

        async def _watch_my_trades():
            trades = await self.exchange.watch_my_trades()
            for trade in trades:  # type: ignore
                instrument = ccxt_find_instrument(trade["symbol"], self.exchange, _symbol_to_instrument)
                deals = ccxt_extract_deals_from_exec({"trades": [trade]})
                channel.send((instrument, "deals", deals, False))

        await asyncio.gather(
            self._listen_to_stream(
                subscriber=_watch_orders,
                exchange=self.exchange,
                channel=channel,
                name=name,
            ),
            self._listen_to_stream(
                subscriber=_watch_my_trades,
                exchange=self.exchange,
                channel=channel,
                name=name,
            ),
        )
