"""
Trade data type handler for CCXT data provider.

Handles subscription and warmup for trade data.
"""

from typing import Set

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, dt_64

from ..subscription_config import SubscriptionConfiguration
from ..utils import (
    ccxt_convert_trade,
    ccxt_find_instrument,
    create_market_type_batched_subscriber,
    instrument_to_ccxt_symbol,
)
from .base import BaseDataTypeHandler


class TradeDataHandler(BaseDataTypeHandler):
    """Handler for trade data subscription and processing."""

    @property
    def data_type(self) -> str:
        return "trade"

    def prepare_subscription(
        self, name: str, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **params
    ) -> SubscriptionConfiguration:
        """
        Prepare trade subscription configuration.

        Args:
            name: Stream name for this subscription
            sub_type: Parsed subscription type ("trade")
            channel: Control channel for managing subscription lifecycle
            instruments: Set of instruments to subscribe to
            
        Returns:
            SubscriptionConfiguration with subscriber and unsubscriber functions
        """
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_trades(instruments_batch: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments_batch]
            trades = await self._exchange.watch_trades_for_symbols(symbols)

            exch_symbol = trades[0]["symbol"]
            instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)

            for trade in trades:
                converted_trade = ccxt_convert_trade(trade)
                self._data_provider._health_monitor.record_data_arrival(sub_type, dt_64(converted_trade.time, "ns"))
                channel.send((instrument, sub_type, converted_trade, False))

        async def un_watch_trades(instruments_batch: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments_batch]
            await self._exchange.un_watch_trades_for_symbols(symbols)

        # Return subscription configuration instead of calling _listen_to_stream directly
        return SubscriptionConfiguration(
            subscriber_func=create_market_type_batched_subscriber(watch_trades, instruments),
            unsubscriber_func=create_market_type_batched_subscriber(un_watch_trades, instruments),
            stream_name=name,
            requires_market_type_batching=True,
        )

    async def warmup(self, instruments: Set[Instrument], channel: CtrlChannel, warmup_period: str, **params) -> None:
        """
        Fetch historical trade data for warmup during backtesting.

        Args:
            instruments: Set of instruments to warm up
            channel: Control channel for sending warmup data
            warmup_period: Period to warm up (e.g., "30d", "1000h")
        """
        for instrument in instruments:
            trades = await self._exchange.fetch_trades(
                instrument.symbol, since=self._data_provider._time_msec_nbars_back(warmup_period)
            )

            logger.debug(f"<yellow>{self._exchange_id}</yellow> Loaded {len(trades)} trades for {instrument}")

            channel.send(
                (
                    instrument,
                    DataType.TRADE,
                    [ccxt_convert_trade(trade) for trade in trades],
                    True,  # historical data
                )
            )
