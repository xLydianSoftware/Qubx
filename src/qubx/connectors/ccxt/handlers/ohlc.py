"""
OHLC data type handler for CCXT data provider.

Handles subscription and warmup for OHLC (candlestick) data.
"""

from typing import Set

import pandas as pd

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument
from qubx.core.series import Bar, Quote
from qubx.utils.time import convert_tf_str_td64

from ..subscription_config import SubscriptionConfiguration
from ..utils import ccxt_find_instrument, create_market_type_batched_subscriber, instrument_to_ccxt_symbol
from .base import BaseDataTypeHandler


class OhlcDataHandler(BaseDataTypeHandler):
    """Handler for OHLC (candlestick) data subscription and processing."""

    @property
    def data_type(self) -> str:
        return "ohlc"

    def _convert_ohlcv_to_bar(self, oh: list) -> Bar:
        """
        Convert OHLCV array data to Bar object with proper field mapping.
        
        Args:
            oh: OHLCV array data from exchange
            
        Returns:
            Bar object with properly mapped fields
        """
        # Extended OHLCV data processing
        
        if len(oh) > 6:
            # Extended OHLCV data from Binance (10 fields total)
            # oh[0-5] = standard OHLCV (timestamp, open, high, low, close, volume)
            # oh[6] = quote asset volume  
            # oh[7] = number of trades
            # oh[8] = taker buy base asset volume (bought_volume)
            # oh[9] = taker buy quote asset volume (bought_volume_quote)
            # Debug: Check if Bar is being created with correct extended data
            bar = Bar(
                oh[0] * 1_000_000,  # timestamp
                oh[1],  # open
                oh[2],  # high
                oh[3],  # low
                oh[4],  # close
                oh[5],  # volume (base asset)
                oh[6],  # volume_quote (quote asset volume)
                oh[8],  # bought_volume (taker buy base asset volume)
                oh[9],  # bought_volume_quote (taker buy quote asset volume)
                float(oh[7]),  # trade_count (number of trades) - now using double
            )
            
            # Clean up debug logging
            
            return bar
        else:
            # Standard OHLCV data
            return Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[5])

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
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _exchange_timeframe = self._data_provider._get_exch_timeframe(timeframe)
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_ohlcv(instruments_batch: list[Instrument]):
            _symbol_timeframe_pairs = [[_instr_to_ccxt_symbol[i], _exchange_timeframe] for i in instruments_batch]
            ohlcv = await self._exchange.watch_ohlcv_for_symbols(_symbol_timeframe_pairs)

            # ohlcv is symbol -> timeframe -> list[timestamp, open, high, low, close, volume]
            for exch_symbol, _data in ohlcv.items():
                instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
                for _, ohlcvs in _data.items():
                    for oh in ohlcvs:
                        # Get current time and original bar timestamp
                        current_time = self._data_provider.time_provider.time()
                        current_timestamp_ns = current_time.astype('datetime64[ns]').view('int64')
                        bar_timestamp_ns = oh[0] * 1_000_000
                        
                        # Create bar with correct timestamp handling
                        bar = self._convert_ohlcv_to_bar(oh)
                        
                        # Determine if this bar belongs to previous timeframe
                        timeframe_td = convert_tf_str_td64(timeframe)
                        timeframe_ns = timeframe_td.astype('timedelta64[ns]').astype('int64')
                        current_bar_start = (current_timestamp_ns // timeframe_ns) * timeframe_ns
                        bar_start = (bar_timestamp_ns // timeframe_ns) * timeframe_ns
                        
                        if bar_start < current_bar_start:
                            # This is a late update for previous timeframe
                            # Set timestamp to 1ns before end of previous bar to ensure correct processing
                            bar.time = current_bar_start - 1
                            logger.debug(f"<yellow>{self._exchange_id}</yellow> Late bar update detected - adjusted timestamp for {instrument.symbol}")
                        else:
                            # This is current timeframe - use current time for consolidation handling
                            bar.time = current_timestamp_ns
                        
                        # Use current time for health monitoring with robust conversion
                        current_timestamp_ms = current_timestamp_ns // 1_000_000
                        health_timestamp = pd.Timestamp(current_timestamp_ms, unit="ms").asm8
                        self._data_provider._health_monitor.record_data_arrival(sub_type, health_timestamp)

                        channel.send(
                            (
                                instrument,
                                sub_type,
                                bar,
                                False,  # not historical bar
                            )
                        )

                    # Generate synthetic quotes if no orderbook/quote subscription exists
                    if not (
                        self._data_provider.has_subscription(instrument, DataType.ORDERBOOK)
                        or self._data_provider.has_subscription(instrument, DataType.QUOTE)
                    ):
                        _price = ohlcvs[-1][4]
                        _s2 = instrument.tick_size / 2.0
                        _bid, _ask = _price - _s2, _price + _s2
                        self._data_provider._last_quotes[instrument] = Quote(oh[0] * 1_000_000, _bid, _ask, 0.0, 0.0)

        # Return subscription configuration instead of calling _listen_to_stream directly
        return SubscriptionConfiguration(
            subscriber_func=create_market_type_batched_subscriber(watch_ohlcv, instruments),
            unsubscriber_func=None,  # OHLC subscriptions don't support unsubscribe properly
            stream_name=name,
            requires_market_type_batching=True,
        )

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
            ohlcv = await self._exchange.fetch_ohlcv(
                instrument.symbol, exch_timeframe, since=start, limit=nbarsback + 1
            )

            logger.debug(f"<yellow>{self._exchange_id}</yellow> {instrument}: loaded {len(ohlcv)} {timeframe} bars")
            
            # Debug: Check warmup data format vs live data format
            if len(ohlcv) > 0:
                sample_bar = ohlcv[0]
                logger.info(f"Warmup OHLCV sample length: {len(sample_bar)}, data: {sample_bar}")
                if len(sample_bar) >= 10:
                    logger.info(f"Warmup extended fields: vol_quote={sample_bar[6]}, trades={sample_bar[7]}, buy_vol={sample_bar[8]}, buy_vol_quote={sample_bar[9]}")
                else:
                    logger.warning(f"Warmup data is standard OHLCV only (length {len(sample_bar)}), no extended fields!")

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
        symbol = instrument.symbol
        since = self._data_provider._time_msec_nbars_back(timeframe, nbarsback)
        exch_timeframe = self._data_provider._get_exch_timeframe(timeframe)

        # Retrieve OHLC data from exchange
        # TODO: check if nbarsback > max_limit (1000) we need to do more requests
        ohlcv_data = await self._exchange.fetch_ohlcv(symbol, exch_timeframe, since=since, limit=nbarsback + 1)

        # Convert to Bar objects using utility method
        bars = []
        for oh in ohlcv_data:
            bars.append(self._convert_ohlcv_to_bar(oh))

        return bars
