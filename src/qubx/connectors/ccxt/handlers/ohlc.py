"""
OHLC data type handler for CCXT data provider.

Handles subscription and warmup for OHLC (candlestick) data.
"""

from typing import Set

import pandas as pd

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, dt_64
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
                        timestamp_ns = oh[0] * 1_000_000
                        self._data_provider._health_monitor.record_data_arrival(sub_type, dt_64(timestamp_ns, "ns"))

                        channel.send(
                            (
                                instrument,
                                sub_type,
                                Bar(
                                    timestamp_ns,
                                    oh[1],
                                    oh[2],
                                    oh[3],
                                    oh[4],
                                    oh[6],
                                    bought_volume=oh[7] if len(oh) > 7 else 0,
                                    # trade_count=int(oh[8]) if len(oh) > 8 else 0,
                                ),
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

            channel.send(
                (
                    instrument,
                    DataType.OHLC[timeframe],
                    [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in ohlcv],
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
        # TODO: how to get quoted volumes ?
        ohlcv_data = await self._exchange.fetch_ohlcv(symbol, exch_timeframe, since=since, limit=nbarsback + 1)

        # Convert to Bar objects
        bars = []
        for oh in ohlcv_data:
            if len(oh) > 6:
                bar = Bar(
                    oh[0] * 1_000_000,  # timestamp
                    oh[1],  # open
                    oh[2],  # high
                    oh[3],  # low
                    oh[4],  # close
                    oh[6],  # volume (use quote volume if available)
                    bought_volume=oh[7] if len(oh) > 7 else 0,
                    trade_count=int(oh[8]) if len(oh) > 8 else 0,
                )
            else:
                bar = Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[5])
            bars.append(bar)

        return bars
