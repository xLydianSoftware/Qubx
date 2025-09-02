"""
OrderBook data type handler for CCXT data provider.

Handles subscription and warmup for orderbook data with support for both
single instrument and multi-instrument approaches.
"""

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

    def _process_orderbook(
        self, ccxt_ob: dict, instrument: Instrument, sub_type: str, channel, depth: int, tick_size_pct: float
    ):
        """
        Process an orderbook with synthetic quote generation and health monitoring.

        This method handles the common logic for processing orderbook data that's shared between
        bulk and individual subscription approaches.

        Args:
            ccxt_ob: CCXT orderbook dictionary
            instrument: Instrument this orderbook belongs to
            sub_type: Subscription type string
            channel: Control channel to send data through
            depth: Number of orderbook levels
            tick_size_pct: Tick size percentage for orderbook levels

        Returns:
            True if orderbook was processed and sent, False if orderbook was None
        """
        # Convert CCXT orderbook to Qubx format
        ob = ccxt_convert_orderbook(
            ccxt_ob,
            instrument,
            levels=depth,
            tick_size_pct=tick_size_pct,
            current_timestamp=self._data_provider.time_provider.time(),
        )
        if ob is None:
            return False

        # Generate synthetic quote if no quote subscription exists
        if not self._data_provider.has_subscription(instrument, DataType.QUOTE):
            quote = ob.to_quote()
            self._data_provider._last_quotes[instrument] = quote

        # Notify all listeners
        self._data_provider.notify_data_arrival(sub_type, dt_64(ob.time, "ns"))
        
        channel.send((instrument, sub_type, ob, False))
        return True

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
        if self._exchange_manager.exchange.has.get("watchOrderBookForSymbols", False):
            return self._prepare_subscription_for_instruments(
                name, sub_type, channel, instruments, tick_size_pct, depth
            )
        else:
            # Fall back to individual instrument subscriptions
            return self._prepare_subscription_for_individual_instruments(
                name, sub_type, channel, instruments, tick_size_pct, depth
            )

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
            ccxt_ob = await self._exchange_manager.exchange.watch_order_book_for_symbols(symbols)

            exch_symbol = ccxt_ob["symbol"]
            instrument = ccxt_find_instrument(exch_symbol, self._exchange_manager.exchange, _symbol_to_instrument)

            # Use private processing method to avoid duplication
            self._process_orderbook(ccxt_ob, instrument, sub_type, channel, depth, tick_size_pct)

        async def un_watch_orderbook(instruments_batch: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments_batch]
            await self._exchange_manager.exchange.un_watch_order_book_for_symbols(symbols)

        return SubscriptionConfiguration(
            subscription_type=sub_type,
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
        """
        Prepare subscription configuration for individual instruments.
        
        Creates separate subscriber functions for each instrument to enable independent
        WebSocket streams without waiting for all instruments. This follows the same
        pattern as the OHLC handler for proper individual stream management.
        """
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}

        individual_subscribers = {}
        individual_unsubscribers = {}

        for instrument in instruments:
            ccxt_symbol = _instr_to_ccxt_symbol[instrument]

            # Create individual subscriber for this instrument using closure
            def create_individual_subscriber(inst=instrument, symbol=ccxt_symbol, exchange_id=self._exchange_id):
                async def individual_subscriber():
                    try:
                        # Watch orderbook for single instrument
                        ccxt_ob = await self._exchange_manager.exchange.watch_order_book(symbol)
                        
                        # Use private processing method to avoid duplication
                        self._process_orderbook(ccxt_ob, inst, sub_type, channel, depth, tick_size_pct)
                        
                    except Exception as e:
                        logger.error(
                            f"<yellow>{exchange_id}</yellow> Error in individual orderbook subscription for {inst.symbol}: {e}"
                        )
                        raise  # Let connection manager handle retries

                return individual_subscriber

            individual_subscribers[instrument] = create_individual_subscriber()

            # Create individual unsubscriber if exchange supports it
            un_watch_method = getattr(self._exchange_manager.exchange, "un_watch_order_book", None)
            if un_watch_method is not None and callable(un_watch_method):
                
                def create_individual_unsubscriber(symbol=ccxt_symbol, exchange_id=self._exchange_id):
                    async def individual_unsubscriber():
                        try:
                            await self._exchange_manager.exchange.un_watch_order_book(symbol)
                        except Exception as e:
                            logger.error(f"<yellow>{exchange_id}</yellow> Error unsubscribing orderbook for {symbol}: {e}")

                    return individual_unsubscriber

                individual_unsubscribers[instrument] = create_individual_unsubscriber()

        return SubscriptionConfiguration(
            subscription_type=sub_type,
            instrument_subscribers=individual_subscribers,
            instrument_unsubscribers=individual_unsubscribers if individual_unsubscribers else None,
            stream_name=name,
            requires_market_type_batching=False,
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
