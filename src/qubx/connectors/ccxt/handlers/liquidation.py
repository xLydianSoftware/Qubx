"""
Liquidation data type handler for CCXT data provider.

Handles subscription and warmup for liquidation event data.
"""

from typing import Set

from qubx import logger
from qubx.core.basics import CtrlChannel, Instrument, dt_64

from ..exceptions import CcxtLiquidationParsingError
from ..subscription_config import SubscriptionConfiguration
from ..utils import (
    ccxt_convert_liquidation,
    ccxt_find_instrument,
    create_market_type_batched_subscriber,
    instrument_to_ccxt_symbol,
)
from .base import BaseDataTypeHandler


class LiquidationDataHandler(BaseDataTypeHandler):
    """Handler for liquidation event data subscription and processing."""

    @property
    def data_type(self) -> str:
        return "liquidation"

    def prepare_subscription(
        self, name: str, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **params
    ) -> SubscriptionConfiguration:
        """
        Prepare liquidation subscription configuration.

        Args:
            name: Stream name for this subscription
            sub_type: Parsed subscription type ("liquidation")
            channel: Control channel for managing subscription lifecycle
            instruments: Set of instruments to subscribe to
            
        Returns:
            SubscriptionConfiguration with subscriber and unsubscriber functions
        """
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_liquidation(instruments_batch: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments_batch]
            liquidations = await self._exchange.watch_liquidations_for_symbols(symbols)

            for liquidation in liquidations:
                try:
                    exch_symbol = liquidation["symbol"]
                    instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
                    liquidation_event = ccxt_convert_liquidation(liquidation)

                    self._data_provider._health_monitor.record_data_arrival(
                        sub_type, dt_64(liquidation_event.time, "ns")
                    )
                    channel.send((instrument, sub_type, liquidation_event, False))

                except CcxtLiquidationParsingError:
                    logger.debug(f"<yellow>{self._exchange_id}</yellow> Could not parse liquidation {liquidation}")
                    continue

        async def un_watch_liquidation(instruments_batch: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments_batch]
            unwatch = getattr(self._exchange, "un_watch_liquidations_for_symbols", lambda _: None)(symbols)
            if unwatch is not None:
                await unwatch

        # Return subscription configuration instead of calling _listen_to_stream directly
        return SubscriptionConfiguration(
            subscriber_func=create_market_type_batched_subscriber(watch_liquidation, instruments),
            unsubscriber_func=create_market_type_batched_subscriber(un_watch_liquidation, instruments),
            stream_name=name,
            requires_market_type_batching=True,
        )

    async def warmup(self, instruments: Set[Instrument], channel: CtrlChannel, **params) -> None:
        """
        Liquidation warmup is not supported by CCXT as these are real-time events.

        Args:
            instruments: Set of instruments to warm up
            channel: Control channel for sending warmup data
        """
        # Fetching of liquidations for warmup is not supported by CCXT
        logger.debug(f"<yellow>{self._exchange_id}</yellow> Liquidation warmup not supported (real-time events only)")
        pass
