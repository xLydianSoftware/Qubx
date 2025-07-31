"""
OrderBook data type handler for CCXT data provider.

Handles subscription and warmup for orderbook data with support for both
single instrument and multi-instrument approaches.
"""

import asyncio
from typing import Set

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, dt_64

from ..subscription_config import SubscriptionConfiguration
from ..utils import (
    ccxt_convert_orderbook,
    ccxt_find_instrument,
    create_market_type_batched_subscriber,
    instrument_to_ccxt_symbol,
)
from .base import BaseDataTypeHandler


class OrderBookDataHandler(BaseDataTypeHandler):
    """Handler for orderbook data subscription and processing."""

    @property
    def data_type(self) -> str:
        return "orderbook"

    def prepare_subscription(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        tick_size_pct: float = 0.01,
        depth: int = 200,
        **params,
    ) -> SubscriptionConfiguration:
        """
        Prepare orderbook subscription configuration.

        Args:
            name: Stream name for this subscription
            sub_type: Parsed subscription type ("orderbook")
            channel: Control channel for managing subscription lifecycle
            instruments: Set of instruments to subscribe to
            tick_size_pct: Tick size percentage for orderbook levels
            depth: Number of orderbook levels to subscribe to
            
        Returns:
            SubscriptionConfiguration with subscriber and unsubscriber functions
        """
        # Use exchange-specific approach based on capabilities
        if self._exchange.has.get("watchOrderBookForSymbols", False):
            return self._prepare_subscription_for_instruments(name, sub_type, channel, instruments, tick_size_pct, depth)
        else:
            # Fall back to individual instrument subscriptions
            return self._prepare_subscription_for_individual_instruments(name, sub_type, channel, instruments, tick_size_pct, depth)

    def _prepare_subscription_for_instruments(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        tick_size_pct: float,
        depth: int,
    ) -> SubscriptionConfiguration:
        """Prepare subscription configuration for multiple instruments using bulk API."""
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_orderbook(instruments_batch: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments_batch]
            ccxt_ob = await self._exchange.watch_order_book_for_symbols(symbols)

            exch_symbol = ccxt_ob["symbol"]
            instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)

            ob = ccxt_convert_orderbook(
                ccxt_ob,
                instrument,
                levels=depth,
                tick_size_pct=tick_size_pct,
                current_timestamp=self._data_provider.time_provider.time(),
            )
            if ob is None:
                return

            self._data_provider._health_monitor.record_data_arrival(sub_type, dt_64(ob.time, "ns"))

            # Generate synthetic quote if no quote subscription exists
            if not self._data_provider.has_subscription(instrument, DataType.QUOTE):
                quote = ob.to_quote()
                self._data_provider._last_quotes[instrument] = quote

            channel.send((instrument, sub_type, ob, False))

        async def un_watch_orderbook(instruments_batch: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments_batch]
            await self._exchange.un_watch_order_book_for_symbols(symbols)

        return SubscriptionConfiguration(
            subscriber_func=create_market_type_batched_subscriber(watch_orderbook, instruments),
            unsubscriber_func=create_market_type_batched_subscriber(un_watch_orderbook, instruments),
            stream_name=name,
            requires_market_type_batching=True,
        )

    def _prepare_subscription_for_individual_instruments(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        tick_size_pct: float,
        depth: int,
    ) -> SubscriptionConfiguration:
        """Prepare subscription configuration for individual instruments (fallback approach)."""
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}

        async def watch_orderbook_individual(instruments_batch: list[Instrument]):
            # Handle multiple instruments by subscribing to each individually
            tasks = []
            for instrument in instruments_batch:
                ccxt_symbol = _instr_to_ccxt_symbol[instrument]
                
                async def watch_single_instrument():
                    ccxt_ob = await self._exchange.watch_order_book(ccxt_symbol)
                    ob = ccxt_convert_orderbook(
                        ccxt_ob,
                        instrument,
                        levels=depth,
                        tick_size_pct=tick_size_pct,
                        current_timestamp=self._data_provider.time_provider.time(),
                    )
                    if ob is None:
                        return

                    self._data_provider._health_monitor.record_data_arrival(sub_type, dt_64(ob.time, "ns"))

                    # Generate synthetic quote if no quote subscription exists
                    if not self._data_provider.has_subscription(instrument, DataType.QUOTE):
                        quote = ob.to_quote()
                        self._data_provider._last_quotes[instrument] = quote

                    channel.send((instrument, sub_type, ob, False))
                
                tasks.append(watch_single_instrument())
            
            # Run all individual subscriptions concurrently
            await asyncio.gather(*tasks)

        async def un_watch_orderbook_individual(instruments_batch: list[Instrument]):
            tasks = []
            for instrument in instruments_batch:
                ccxt_symbol = _instr_to_ccxt_symbol[instrument]
                
                async def unwatch_single_instrument():
                    if hasattr(self._exchange, "un_watch_order_book"):
                        await self._exchange.un_watch_order_book(ccxt_symbol)
                
                tasks.append(unwatch_single_instrument())
            
            await asyncio.gather(*tasks)

        return SubscriptionConfiguration(
            subscriber_func=create_market_type_batched_subscriber(watch_orderbook_individual, instruments),
            unsubscriber_func=create_market_type_batched_subscriber(un_watch_orderbook_individual, instruments),
            stream_name=name,
            requires_market_type_batching=True,
        )

    async def warmup(self, instruments: Set[Instrument], channel: CtrlChannel, **params) -> None:
        """
        Orderbook warmup is typically not needed as it represents current state.

        Args:
            instruments: Set of instruments to warm up
            channel: Control channel for sending warmup data
        """
        # Orderbook data is real-time state, no historical warmup needed
        logger.debug(f"<yellow>{self._exchange_id}</yellow> Orderbook warmup skipped (real-time data only)")
        pass
