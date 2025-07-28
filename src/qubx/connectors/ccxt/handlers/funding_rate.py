"""
Funding rate data type handler for CCXT data provider.

Handles subscription and warmup for funding rate data.
"""

from typing import Set

from qubx import logger
from qubx.core.basics import CtrlChannel, Instrument, dt_64

from ..exceptions import CcxtSymbolNotRecognized
from ..subscription_config import SubscriptionConfiguration
from ..utils import ccxt_convert_funding_rate, ccxt_find_instrument
from .base import BaseDataTypeHandler


class FundingRateDataHandler(BaseDataTypeHandler):
    """Handler for funding rate data subscription and processing."""

    @property
    def data_type(self) -> str:
        return "funding_rate"

    def prepare_subscription(
        self, name: str, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **params
    ) -> SubscriptionConfiguration:
        """
        Prepare funding rate subscription configuration.

        Args:
            name: Stream name for this subscription
            sub_type: Parsed subscription type ("funding_rate")
            channel: Control channel for managing subscription lifecycle
            instruments: Set of instruments to subscribe to
            
        Returns:
            SubscriptionConfiguration with subscriber and unsubscriber functions
        """
        # Capture instruments at subscription time to avoid empty symbols during unsubscription
        symbols = [instr.symbol for instr in instruments]

        async def watch_funding_rates():
            try:
                # Use symbols captured at subscription time
                funding_rates = await self._exchange.watch_funding_rates(symbols)
                current_time = self._data_provider.time_provider.time()

                # Send individual funding rate updates per instrument
                for symbol, info in funding_rates.items():
                    try:
                        instrument = ccxt_find_instrument(symbol, self._exchange)
                        funding_rate = ccxt_convert_funding_rate(info)

                        self._data_provider._health_monitor.record_data_arrival(sub_type, dt_64(current_time, "s"))

                        logger.debug(
                            f"<yellow>{self._exchange_id}</yellow> Sending funding rate update for {instrument.symbol}: rate={funding_rate.rate}"
                        )
                        channel.send((instrument, sub_type, funding_rate, False))

                    except CcxtSymbolNotRecognized:
                        continue

            except Exception as e:
                logger.exception(e)
                # Re-raise to trigger retry logic in _listen_to_stream
                raise

        async def un_watch_funding_rates():
            unwatch = getattr(self._exchange, "un_watch_funding_rates", lambda: None)()
            if unwatch is not None:
                await unwatch

        # Return subscription configuration instead of calling _listen_to_stream directly
        return SubscriptionConfiguration(
            subscriber_func=watch_funding_rates,
            unsubscriber_func=un_watch_funding_rates,
            stream_name=name,
            requires_market_type_batching=False,
        )

    async def warmup(self, instruments: Set[Instrument], channel: CtrlChannel, **params) -> None:
        """
        Funding rate warmup is typically not needed as it represents current rates.

        Args:
            instruments: Set of instruments to warm up
            channel: Control channel for sending warmup data
        """
        # Funding rate data is typically current state, no historical warmup needed
        logger.debug(f"<yellow>{self._exchange_id}</yellow> Funding rate warmup skipped (current data only)")
        pass
