"""
Quote data type handler for CCXT data provider.

Handles subscription and warmup for quote (bid/ask) data.
"""

from typing import Set

from qubx import logger
from qubx.core.basics import CtrlChannel, Instrument, dt_64

from ..subscription_config import SubscriptionConfiguration
from ..utils import (
    ccxt_convert_ticker,
    ccxt_find_instrument,
    create_market_type_batched_subscriber,
    instrument_to_ccxt_symbol,
)
from .base import BaseDataTypeHandler


class QuoteDataHandler(BaseDataTypeHandler):
    """Handler for quote (bid/ask) data subscription and processing."""

    @property
    def data_type(self) -> str:
        return "quote"

    def prepare_subscription(
        self, name: str, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **params
    ) -> SubscriptionConfiguration:
        """
        Prepare quote subscription configuration.

        Args:
            name: Stream name for this subscription
            sub_type: Parsed subscription type ("quote")
            channel: Control channel for managing subscription lifecycle
            instruments: Set of instruments to subscribe to
            
        Returns:
            SubscriptionConfiguration with subscriber and unsubscriber functions
        """
        # Check if exchange supports bid/ask watching
        if not self._exchange.has.get("watchBidsAsks", False):
            logger.warning(f"<yellow>{self._exchange_id}</yellow> watchBidsAsks is not supported for {name}")
            self._data_provider.unsubscribe(sub_type, list(instruments))
            # Return a dummy configuration that does nothing
            return SubscriptionConfiguration(
                subscriber_func=lambda: None,
                unsubscriber_func=None,
                stream_name=name,
            )

        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_quote(instruments_batch: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments_batch]
            ccxt_tickers: dict[str, dict] = await self._exchange.watch_bids_asks(symbols)

            for exch_symbol, ccxt_ticker in ccxt_tickers.items():
                instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
                quote = ccxt_convert_ticker(ccxt_ticker)

                # Only emit if quote is newer than the last one
                last_quote = self._data_provider._last_quotes[instrument]
                if last_quote is None or quote.time > last_quote.time:
                    self._data_provider._health_monitor.record_data_arrival(sub_type, dt_64(quote.time, "ns"))
                    self._data_provider._last_quotes[instrument] = quote
                    channel.send((instrument, sub_type, quote, False))

        async def un_watch_quote(instruments_batch: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments_batch]
            if hasattr(self._exchange, "un_watch_bids_asks"):
                await getattr(self._exchange, "un_watch_bids_asks")(symbols)
            else:
                await self._exchange.un_watch_tickers(symbols)

        # Return subscription configuration instead of calling _listen_to_stream directly
        return SubscriptionConfiguration(
            subscriber_func=create_market_type_batched_subscriber(watch_quote, instruments),
            unsubscriber_func=create_market_type_batched_subscriber(un_watch_quote, instruments),
            stream_name=name,
            requires_market_type_batching=True,
        )

    async def warmup(self, instruments: Set[Instrument], channel: CtrlChannel, **params) -> None:
        """
        Quote warmup is typically not needed as it represents current market state.

        Args:
            instruments: Set of instruments to warm up
            channel: Control channel for sending warmup data
        """
        # Quote data is real-time market state, no historical warmup needed
        logger.debug(f"<yellow>{self._exchange_id}</yellow> Quote warmup skipped (real-time data only)")
        pass
