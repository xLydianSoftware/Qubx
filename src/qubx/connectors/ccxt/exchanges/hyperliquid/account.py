import asyncio

from qubx import logger
from qubx.connectors.ccxt.account import CcxtAccountProcessor
from qubx.connectors.ccxt.utils import (
    ccxt_convert_order_info,
    ccxt_extract_deals_from_exec,
    ccxt_find_instrument,
)
from qubx.core.basics import CtrlChannel, Instrument
from qubx.utils.time import now_utc, timestamp_to_ms


class HyperliquidAccountProcessor(CcxtAccountProcessor):
    """
    Hyperliquid-specific account processor.

    Hyperliquid uses separate WebSocket channels:
    - orderUpdates: for order status (watch_orders)
    - userFills: for trade/fill updates (watch_my_trades)

    Unlike Binance, Hyperliquid's watch_orders does NOT include trades,
    so we must subscribe to both channels separately.
    """

    async def _subscribe_instruments(self, instruments: list[Instrument]):
        """Override to filter out instruments from other exchanges (e.g., spot vs futures)."""
        # Filter instruments to only those belonging to this exchange
        exchange_name = self.exchange_manager.exchange.name
        matching_instruments = [instr for instr in instruments if instr.exchange == exchange_name]

        if len(matching_instruments) < len(instruments):
            skipped = [instr for instr in instruments if instr.exchange != exchange_name]
            logger.debug(f"Skipping subscription for {len(skipped)} instruments from other exchanges: {skipped}")

        # Call parent with filtered instruments
        if matching_instruments:
            await super()._subscribe_instruments(matching_instruments)

    async def _subscribe_executions(self, name: str, channel: CtrlChannel):
        logger.info("<yellow>[Hyperliquid]</yellow> Subscribing to executions")
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
            trades = await self.exchange_manager.exchange.watch_my_trades(since=timestamp_to_ms(now_utc()))
            for trade in trades:  # type: ignore
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
