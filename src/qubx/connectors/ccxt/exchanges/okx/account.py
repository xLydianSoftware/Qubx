import asyncio

from qubx import logger
from qubx.connectors.ccxt.account import CcxtAccountProcessor
from qubx.connectors.ccxt.utils import (
    ccxt_convert_order_info,
    ccxt_extract_deals_from_exec,
    ccxt_find_instrument,
)
from qubx.core.basics import CtrlChannel


class OkxAccountProcessor(CcxtAccountProcessor):
    """
    OKX-specific account processor.

    OKX's watch_orders does NOT include trades/fills in the response,
    so we must subscribe to both channels separately:
    - orders channel: for order status updates (watch_orders)
    - trades channel: for trade/fill updates (watch_my_trades)
    """

    async def _subscribe_executions(self, name: str, channel: CtrlChannel):
        logger.info("<yellow>[OKX]</yellow> Subscribing to executions (orders + trades)")
        _symbol_to_instrument = {}

        async def _watch_orders():
            orders = await self.exchange_manager.exchange.watch_orders()
            for order in orders:
                instrument = ccxt_find_instrument(
                    order["symbol"], self.exchange_manager.exchange, _symbol_to_instrument
                )
                order = ccxt_convert_order_info(instrument, order)
                channel.send((instrument, "order", order, False))

        async def _watch_my_trades():
            trades = await self.exchange_manager.exchange.watch_my_trades()
            for trade in trades:
                instrument = ccxt_find_instrument(
                    trade["symbol"], self.exchange_manager.exchange, _symbol_to_instrument
                )
                deals = ccxt_extract_deals_from_exec({"trades": [trade]})
                channel.send((instrument, "deals", deals, False))

        await asyncio.gather(
            self._listen_to_stream(
                subscriber=_watch_orders,
                exchange=self.exchange_manager.exchange,
                channel=channel,
                name=f"{name}_orders",
            ),
            self._listen_to_stream(
                subscriber=_watch_my_trades,
                exchange=self.exchange_manager.exchange,
                channel=channel,
                name=f"{name}_trades",
            ),
        )

    async def _subscribe_instruments(self, instruments: list) -> None:
        """Skip SPOT instruments on OKX.F - futures connector only handles swap contracts."""
        await super()._subscribe_instruments([i for i in instruments if i.is_futures()])

    async def _init_spot_positions(self) -> None:
        logger.debug("Skipping spot position initialization for OKX")
