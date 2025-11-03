"""
Trade data type handler for CCXT data provider.

Handles subscription and warmup for trade data.
"""

from typing import Set

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, dt_64
from qubx.core.series import Quote

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

    def _process_trade(self, trades: list, instrument: Instrument, sub_type: str, channel: CtrlChannel):
        """
        Process trades with synthetic quote generation.

        This method handles the common logic for processing trade data that's shared between
        bulk and individual subscription approaches.

        Args:
            trades: List of CCXT trade dictionaries
            instrument: Instrument these trades belong to
            sub_type: Subscription type string
            channel: Control channel to send data through
        """
        for trade in trades:
            converted_trade = ccxt_convert_trade(trade)
            channel.send((instrument, sub_type, converted_trade, False))

        # Generate synthetic quote if no quote/orderbook subscription exists
        if len(trades) > 0 and not (
            self._data_provider.has_subscription(instrument, DataType.ORDERBOOK)
            or self._data_provider.has_subscription(instrument, DataType.QUOTE)
        ):
            last_trade = trades[-1]
            converted_trade = ccxt_convert_trade(last_trade)
            _price = converted_trade.price
            _time = converted_trade.time
            _s2 = instrument.tick_size / 2.0
            _bid, _ask = _price - _s2, _price + _s2
            self._data_provider._last_quotes[instrument] = Quote(_time, _bid, _ask, 0.0, 0.0)

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
        # Use exchange-specific approach based on capabilities
        if self._exchange_manager.exchange.has.get("watchTradesForSymbols", False):
            return self._prepare_subscription_for_instruments(name, sub_type, channel, instruments)
        else:
            # Fall back to individual instrument subscriptions
            return self._prepare_subscription_for_individual_instruments(name, sub_type, channel, instruments)

    def _prepare_subscription_for_instruments(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
    ) -> SubscriptionConfiguration:
        """Prepare subscription configuration for multiple instruments using bulk API."""
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_trades(instruments_batch: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments_batch]
            trades = await self._exchange_manager.exchange.watch_trades_for_symbols(symbols)

            exch_symbol = trades[0]["symbol"]
            instrument = ccxt_find_instrument(exch_symbol, self._exchange_manager.exchange, _symbol_to_instrument)

            # Use private processing method to avoid duplication
            self._process_trade(trades, instrument, sub_type, channel)

        async def un_watch_trades(instruments_batch: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments_batch]
            await self._exchange_manager.exchange.un_watch_trades_for_symbols(symbols)

        return SubscriptionConfiguration(
            subscription_type=sub_type,
            subscriber_func=create_market_type_batched_subscriber(watch_trades, instruments),
            unsubscriber_func=create_market_type_batched_subscriber(un_watch_trades, instruments),
            stream_name=name,
            requires_market_type_batching=True,
        )

    def _prepare_subscription_for_individual_instruments(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
    ) -> SubscriptionConfiguration:
        """
        Prepare subscription configuration for individual instruments.

        Creates separate subscriber functions for each instrument to enable independent
        WebSocket streams without waiting for all instruments. This follows the same
        pattern as the orderbook handler for proper individual stream management.
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
                        # Watch trades for single instrument
                        trades = await self._exchange_manager.exchange.watch_trades(symbol)

                        # Use private processing method to avoid duplication
                        self._process_trade(trades, inst, sub_type, channel)

                    except Exception as e:
                        logger.error(
                            f"<yellow>{exchange_id}</yellow> Error in individual trade subscription for {inst.symbol}: {e}"
                        )
                        raise  # Let connection manager handle retries

                return individual_subscriber

            individual_subscribers[instrument] = create_individual_subscriber()

            # Create individual unsubscriber if exchange supports it
            un_watch_method = getattr(self._exchange_manager.exchange, "un_watch_trades", None)
            if un_watch_method is not None and callable(un_watch_method):

                def create_individual_unsubscriber(symbol=ccxt_symbol, exchange_id=self._exchange_id):
                    async def individual_unsubscriber():
                        try:
                            await self._exchange_manager.exchange.un_watch_trades(symbol)
                        except Exception as e:
                            logger.error(f"<yellow>{exchange_id}</yellow> Error unsubscribing trades for {symbol}: {e}")

                    return individual_unsubscriber

                individual_unsubscribers[instrument] = create_individual_unsubscriber()

        return SubscriptionConfiguration(
            subscription_type=sub_type,
            instrument_subscribers=individual_subscribers,
            instrument_unsubscribers=individual_unsubscribers if individual_unsubscribers else None,
            stream_name=name,
            requires_market_type_batching=False,
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
            trades = await self._exchange_manager.exchange.fetch_trades(
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
