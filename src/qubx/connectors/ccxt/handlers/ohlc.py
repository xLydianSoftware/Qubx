"""
OHLC data type handler for CCXT data provider.

Handles subscription and warmup for OHLC (candlestick) data.
"""

import asyncio
from typing import Set

import pandas as pd

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument
from qubx.core.exceptions import NotSupported
from qubx.core.series import Bar, Quote

from ..subscription_config import SubscriptionConfiguration
from ..utils import ccxt_find_instrument, create_market_type_batched_subscriber, instrument_to_ccxt_symbol
from .base import BaseDataTypeHandler


class OhlcDataHandler(BaseDataTypeHandler):
    """Handler for OHLC (candlestick) data subscription and processing."""

    @property
    def data_type(self) -> str:
        return "ohlc"

    def prepare_subscription(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        timeframe: str = "1m",
        **params,
    ) -> SubscriptionConfiguration:
        """
        Prepare OHLC subscription configuration.

        Args:
            name: Stream name for this subscription
            sub_type: Parsed subscription type ("ohlc")
            channel: Control channel for managing subscription lifecycle
            instruments: Set of instruments to subscribe to
            timeframe: Timeframe for OHLC data (e.g., "1m", "5m", "1h")

        Returns:
            SubscriptionConfiguration with subscriber function
        """
        # Check if exchange supports bulk OHLCV watching and branch logic early
        # Note: Bulk subscriptions work great for static instrument sets but can have issues
        # with dynamic changes. For production use with static instruments, bulk is preferred.
        supports_bulk = self._exchange_manager.exchange.has.get("watchOHLCVForSymbols", False)
        supports_single = self._exchange_manager.exchange.has.get("watchOHLCV", False)

        # Add a parameter to force individual subscriptions for better development/testing experience
        force_individual = params.get("force_individual_subscriptions", False)
        if force_individual:
            supports_bulk = False
            logger.debug(
                f"<yellow>{self._exchange_id}</yellow> Forcing individual subscriptions (force_individual_subscriptions=True)"
            )

        if supports_bulk:
            return self._prepare_bulk_ohlcv_subscriptions(name, sub_type, channel, instruments, timeframe, **params)
        elif supports_single:
            return self._prepare_individual_ohlcv_subscriptions(
                name, sub_type, channel, instruments, timeframe, **params
            )
        else:
            raise NotSupported(f"No bulk or single OHLCV watching supported for {self._exchange_id}")

    async def warmup(
        self, instruments: Set[Instrument], channel: CtrlChannel, warmup_period: str, timeframe: str = "1m", **params
    ) -> None:
        """
        Fetch historical OHLC data for warmup during backtesting.

        Args:
            instruments: Set of instruments to warm up
            channel: Control channel for sending warmup data
            warmup_period: Period to warm up (e.g., "30d", "1000h")
            timeframe: Timeframe for OHLC data (e.g., "1m", "5m", "1h")
        """
        nbarsback = pd.Timedelta(warmup_period) // pd.Timedelta(timeframe)
        exch_timeframe = self._data_provider._get_exch_timeframe(timeframe)

        for instrument in instruments:
            start = self._data_provider._time_msec_nbars_back(timeframe, nbarsback)
            ccxt_symbol = instrument_to_ccxt_symbol(instrument)
            ohlcv = await self._exchange_manager.exchange.fetch_ohlcv(ccxt_symbol, exch_timeframe, since=start, limit=nbarsback + 1)

            logger.debug(f"<yellow>{self._exchange_id}</yellow> {instrument}: loaded {len(ohlcv)} {timeframe} bars")

            channel.send(
                (
                    instrument,
                    DataType.OHLC[timeframe],
                    [self._convert_ohlcv_to_bar(oh) for oh in ohlcv],
                    True,  # historical data
                )
            )

    async def get_historical_ohlc(self, instrument: Instrument, timeframe: str, nbarsback: int) -> list[Bar]:
        """
        Get historical OHLC data for a single instrument (used by get_ohlc method).

        Args:
            instrument: Instrument to fetch data for
            timeframe: Timeframe for OHLC data (e.g., "1m", "5m", "1h")
            nbarsback: Number of bars to fetch

        Returns:
            List of Bar objects with historical OHLC data
        """
        assert nbarsback >= 1
        ccxt_symbol = instrument_to_ccxt_symbol(instrument)
        since = self._data_provider._time_msec_nbars_back(timeframe, nbarsback)
        exch_timeframe = self._data_provider._get_exch_timeframe(timeframe)

        # Retrieve OHLC data from exchange
        # TODO: check if nbarsback > max_limit (1000) we need to do more requests
        ohlcv_data = await self._exchange_manager.exchange.fetch_ohlcv(ccxt_symbol, exch_timeframe, since=since, limit=nbarsback + 1)

        # Convert to Bar objects using utility method
        bars = []
        for oh in ohlcv_data:
            bars.append(self._convert_ohlcv_to_bar(oh))

        return bars

    def _prepare_bulk_ohlcv_subscriptions(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        timeframe: str = "1m",
        **params,
    ) -> SubscriptionConfiguration:
        """
        Prepare bulk OHLCV subscriptions when exchange supports bulk watching.
        """
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}
        _exchange_timeframe = self._data_provider._get_exch_timeframe(timeframe)

        async def watch_ohlcv(instruments_batch: list[Instrument]):
            _symbol_timeframe_pairs = [[_instr_to_ccxt_symbol[i], _exchange_timeframe] for i in instruments_batch]
            try:
                ohlcv = await self._exchange_manager.exchange.watch_ohlcv_for_symbols(_symbol_timeframe_pairs)

                # ohlcv is symbol -> timeframe -> list[timestamp, open, high, low, close, volume]
                for exch_symbol, _data in ohlcv.items():
                    instrument = ccxt_find_instrument(exch_symbol, self._exchange_manager.exchange, _symbol_to_instrument)
                    for _, ohlcvs in _data.items():
                        for oh in ohlcvs:
                            # Use private processing method to avoid duplication
                            self._process_ohlcv_bar(oh, instrument, sub_type, channel, timeframe, ohlcvs)
            except Exception as e:
                # Handle specific CCXT subscription errors more gracefully
                from ccxt.base.errors import UnsubscribeError

                if isinstance(e, UnsubscribeError):
                    # UnsubscribeError means there's a state conflict in CCXT - this is common with dynamic changes
                    logger.warning(f"<yellow>{self._exchange_id}</yellow> Bulk OHLCV subscription state conflict: {e}")
                    # Wait a moment for CCXT state to settle, then return empty data to continue gracefully
                    await asyncio.sleep(0.1)
                    return  # Return without data - connection manager will retry
                elif "InvalidStateError" in str(type(e)) or "invalid state" in str(e):
                    # CCXT Future.race race condition - multiple futures trying to set result simultaneously
                    logger.warning(
                        f"<yellow>{self._exchange_id}</yellow> CCXT Future.race InvalidStateError during bulk subscription: {e}"
                    )
                    logger.info(
                        f"<yellow>{self._exchange_id}</yellow> This is a known CCXT race condition during resubscription - continuing gracefully"
                    )
                    # Wait briefly for CCXT internal state to settle
                    await asyncio.sleep(0.2)
                    return  # Return without data - connection manager will retry automatically
                else:
                    # For other errors, log and re-raise
                    logger.error(f"<yellow>{self._exchange_id}</yellow> Bulk OHLCV subscription failed: {e}")
                    raise

        async def un_watch_ohlcv(instruments_batch: list[Instrument]):
            symbol_timeframe_pairs = [[_instr_to_ccxt_symbol[i], _exchange_timeframe] for i in instruments_batch]
            if hasattr(self._exchange_manager.exchange, "un_watch_ohlcv_for_symbols"):
                try:
                    # Wrap the unsubscription call with timeout to prevent hanging
                    result = await asyncio.wait_for(
                        self._exchange_manager.exchange.un_watch_ohlcv_for_symbols(symbol_timeframe_pairs), timeout=5.0
                    )
                    logger.debug(
                        f"<yellow>{self._exchange_id}</yellow> Successfully unsubscribed from {len(instruments_batch)} instruments"
                    )
                    return result

                except Exception as e:
                    logger.error(f"<yellow>{self._exchange_id}</yellow> Bulk OHLCV unsubscription failed: {e}")
                    # Don't crash on unsubscription error
                    pass

        # Use bulk subscription approach
        return SubscriptionConfiguration(
            subscription_type=sub_type,
            subscriber_func=create_market_type_batched_subscriber(watch_ohlcv, instruments),
            unsubscriber_func=create_market_type_batched_subscriber(un_watch_ohlcv, instruments),  # type: ignore
            stream_name=name,
            requires_market_type_batching=True,
        )

    def _prepare_individual_ohlcv_subscriptions(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        timeframe: str = "1m",
        **params,
    ) -> SubscriptionConfiguration:
        """
        Prepare individual OHLCV subscriptions when exchange doesn't support bulk watching.

        Creates separate subscriber functions for each instrument to enable independent
        WebSocket streams without waiting for all instruments.
        """
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _exchange_timeframe = self._data_provider._get_exch_timeframe(timeframe)

        individual_subscribers = {}
        individual_unsubscribers = {}

        for instrument in instruments:
            ccxt_symbol = _instr_to_ccxt_symbol[instrument]

            # Create individual subscriber for this instrument using closure
            def create_individual_subscriber(inst=instrument, symbol=ccxt_symbol, exchange_id=self._exchange_id):
                async def individual_subscriber():
                    while True:
                        try:
                            # Watch OHLCV for single instrument
                            ohlcv_data = await self._exchange_manager.exchange.watch_ohlcv(symbol, _exchange_timeframe)

                            # Process the OHLCV data using private method
                            if ohlcv_data:
                                for oh in ohlcv_data:
                                    # Use private processing method to avoid duplication
                                    self._process_ohlcv_bar(oh, inst, sub_type, channel, timeframe, ohlcv_data)

                        except Exception as e:
                            logger.error(
                                f"<yellow>{exchange_id}</yellow> Error in individual OHLCV subscription for {inst.symbol}: {e}"
                            )
                            # Continue the loop to maintain the subscription
                            await asyncio.sleep(1)  # Brief pause before retry

                return individual_subscriber

            individual_subscribers[instrument] = create_individual_subscriber()

            # Create individual unsubscriber if exchange supports it
            if hasattr(self._exchange_manager.exchange, "un_watch_ohlcv") and callable(getattr(self._exchange_manager.exchange, "un_watch_ohlcv", None)):

                def create_individual_unsubscriber(symbol=ccxt_symbol, exchange_id=self._exchange_id):
                    async def individual_unsubscriber():
                        try:
                            _unwatch = getattr(self._exchange_manager.exchange, "un_watch_ohlcv")
                            await _unwatch(symbol, _exchange_timeframe)
                        except Exception as e:
                            logger.error(f"<yellow>{exchange_id}</yellow> Error unsubscribing OHLCV for {symbol}: {e}")

                    return individual_unsubscriber

                individual_unsubscribers[instrument] = create_individual_unsubscriber()

        return SubscriptionConfiguration(
            subscription_type=sub_type,
            instrument_subscribers=individual_subscribers,
            instrument_unsubscribers=individual_unsubscribers if individual_unsubscribers else None,
            stream_name=name,
        )

    def _process_ohlcv_bar(
        self,
        oh: list,
        instrument: Instrument,
        sub_type: str,
        channel,
        timeframe: str,
        ohlcv_data_for_quotes: list | None = None,
    ):
        """
        Process a single OHLCV bar with timestamp handling, health monitoring, and synthetic quotes.

        This method handles the common logic for processing OHLCV data that's shared between
        bulk and individual subscription approaches.

        Args:
            oh: Single OHLCV array [timestamp, open, high, low, close, volume, ...]
            instrument: Instrument this bar belongs to
            sub_type: Subscription type string
            channel: Control channel to send data through
            timeframe: Timeframe string (e.g., "1m", "5m", "1h")
            ohlcv_data_for_quotes: Optional full OHLCV data list for synthetic quote generation
        """
        # Get current time and original bar timestamp
        current_time = self._data_provider.time_provider.time()
        current_timestamp_ns = current_time.astype("datetime64[ns]").view("int64")

        bar = self._convert_ohlcv_to_bar(oh)

        # Use current time for health monitoring with robust conversion
        current_timestamp_ms = current_timestamp_ns // 1_000_000
        health_timestamp = pd.Timestamp(current_timestamp_ms, unit="ms").asm8
        
        # Notify all listeners
        self._data_provider.notify_data_arrival(sub_type, health_timestamp)

        # Send the bar
        channel.send((instrument, sub_type, bar, False))  # not historical bar

        # Generate synthetic quotes if no orderbook/quote subscription exists
        if not (
            self._data_provider.has_subscription(instrument, DataType.ORDERBOOK)
            or self._data_provider.has_subscription(instrument, DataType.QUOTE)
        ):
            # Use provided OHLCV data or fall back to current bar
            quote_data = ohlcv_data_for_quotes or [oh]
            if quote_data:
                _price = quote_data[-1][4]  # Close price
                _s2 = instrument.tick_size / 2.0
                _bid, _ask = _price - _s2, _price + _s2
                self._data_provider._last_quotes[instrument] = Quote(current_timestamp_ns, _bid, _ask, 0.0, 0.0)

    def _convert_ohlcv_to_bar(self, oh: list) -> Bar:
        """
        Convert OHLCV array data to Bar object with proper field mapping.

        Args:
            oh: OHLCV array data from exchange

        Returns:
            Bar object with properly mapped fields
        """
        # Extended OHLCV data processing

        # OHLCV data mapping with inline conditionals for variable field lengths
        # oh[0-5] = standard OHLCV (timestamp, open, high, low, close, volume)
        # oh[6] = quote_volume (if available)
        # oh[7] = trade_count (if available)
        # oh[8] = taker_buy_base_volume (if available)
        # oh[9] = taker_buy_quote_volume (if available)
        return Bar(
            oh[0] * 1_000_000,  # timestamp
            oh[1],  # open
            oh[2],  # high
            oh[3],  # low
            oh[4],  # close
            oh[5],  # volume (base asset)
            bought_volume=oh[8] if len(oh) > 8 else 0.0,  # taker buy base volume
            volume_quote=oh[6] if len(oh) > 6 else 0.0,  # quote asset volume
            bought_volume_quote=oh[9] if len(oh) > 9 else 0.0,  # taker buy quote volume
            trade_count=int(oh[7]) if len(oh) > 7 else 0,  # trade count
        )
