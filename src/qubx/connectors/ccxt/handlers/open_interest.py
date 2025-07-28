"""
Open Interest data type handler for CCXT data provider.

Handles subscription and warmup for open interest data using REST API polling.
"""

import asyncio
import datetime
from asyncio.exceptions import CancelledError
from typing import Set

from qubx import logger
from qubx.core.basics import CtrlChannel, Instrument, dt_64

from ..exceptions import CcxtSymbolNotRecognized
from ..subscription_config import SubscriptionConfiguration
from ..utils import ccxt_convert_open_interest, ccxt_find_instrument
from .base import BaseDataTypeHandler


class OpenInterestDataHandler(BaseDataTypeHandler):
    """Handler for open interest data subscription and processing using REST polling."""

    @property
    def data_type(self) -> str:
        return "open_interest"

    def prepare_subscription(
        self, name: str, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **params
    ) -> SubscriptionConfiguration:
        """
        Prepare open interest subscription configuration using REST API polling.

        Binance doesn't provide open interest via WebSocket, so we poll every 5 minutes at exact boundaries.
        Polls at 00, 05, 10, 15, 20, etc. minutes of each hour.

        Args:
            name: Stream name for this subscription
            sub_type: Parsed subscription type ("open_interest")
            channel: Control channel for managing subscription lifecycle
            instruments: Set of instruments to subscribe to
            
        Returns:
            SubscriptionConfiguration with subscriber function and no unsubscriber (REST polling)
        """

        def get_current_5min_boundary():
            """Get the current 5-minute boundary timestamp for data timestamping"""
            now = datetime.datetime.now(datetime.timezone.utc)
            current_minute = (now.minute // 5) * 5
            boundary_time = now.replace(minute=current_minute, second=0, microsecond=0)
            return int(boundary_time.timestamp() * 1_000_000_000)  # Convert to nanoseconds

        def should_poll_now():
            """Check if we should poll now (every 5 minutes at boundaries)"""
            now = datetime.datetime.now(datetime.timezone.utc)
            minute_mod = now.minute % 5
            is_boundary = minute_mod == 0
            is_early_in_boundary = now.second < 30
            should_poll = is_boundary and is_early_in_boundary
            return should_poll

        async def poll_open_interest_once():
            """Single polling operation - called repeatedly by _listen_to_stream"""

            async def cancellation_aware_sleep(seconds: float):
                """Sleep that can be interrupted by subscription cancellation"""
                elapsed = 0.0
                check_interval = 0.5  # Check every 500ms

                while elapsed < seconds:
                    remaining = min(check_interval, seconds - elapsed)
                    await asyncio.sleep(remaining)
                    elapsed += remaining

                    # Check if subscription was cancelled
                    if not self._data_provider._connection_manager.is_stream_enabled(name):
                        raise CancelledError("Subscription cancelled")

            try:
                # Check for cancellation immediately at start of each poll cycle
                if not self._data_provider._connection_manager.is_stream_enabled(name):
                    raise CancelledError("Subscription cancelled")

                # Get current subscribed symbols (refreshed each call - handles universe changes!)
                subscribed_instruments = self._data_provider.get_subscribed_instruments(sub_type)
                symbols = [instr.symbol for instr in subscribed_instruments] if subscribed_instruments else []

                if not symbols:
                    await cancellation_aware_sleep(5)
                    return

                # Check if it's time to poll (every 5 minutes)
                if not should_poll_now():
                    await cancellation_aware_sleep(5)
                    return

                # Use 5-minute boundary timestamp for all data in this poll
                boundary_timestamp_ns = get_current_5min_boundary()

                # Fetch open interest for all symbols
                for symbol in symbols:
                    try:
                        # Fetch open interest data via REST API
                        oi_data = await self._exchange.fetch_open_interest(symbol)

                        # If USD value is missing, fetch mark price to calculate it
                        if oi_data.get("openInterestValue") is None and oi_data.get("openInterestAmount", 0) > 0:
                            try:
                                ticker = await self._exchange.fetch_ticker(symbol)
                                mark_price = ticker.get("last") or ticker.get("close", 0)

                                if mark_price > 0:
                                    calculated_usd = float(oi_data["openInterestAmount"]) * mark_price
                                    oi_data["openInterestValue"] = calculated_usd
                            except Exception as price_error:
                                logger.warning(
                                    f"<yellow>{self._exchange_id}</yellow> Failed to fetch mark price for {symbol}: {price_error}"
                                )

                        # Override timestamp to use 5-minute boundary
                        oi_data["timestamp"] = (
                            boundary_timestamp_ns // 1_000_000
                        )  # Convert to milliseconds for ccxt_convert

                        instrument = ccxt_find_instrument(symbol, self._exchange)
                        open_interest = ccxt_convert_open_interest(symbol, oi_data)
                        self._data_provider._health_monitor.record_data_arrival(
                            sub_type, dt_64(boundary_timestamp_ns, "ns")
                        )

                        # Send individual update per instrument
                        channel.send((instrument, sub_type, open_interest, False))

                    except CcxtSymbolNotRecognized:
                        logger.warning(f"<yellow>{self._exchange_id}</yellow> Symbol not recognized: {symbol}")
                        continue
                    except Exception as e:
                        logger.error(
                            f"<yellow>{self._exchange_id}</yellow> Error fetching open interest for {symbol}: {type(e).__name__}: {e}"
                        )
                        continue

                # After successful poll, sleep longer until next check (but still cancellation-aware)
                await cancellation_aware_sleep(100)

            except CancelledError:
                raise  # Re-raise to exit _listen_to_stream
            except Exception as e:
                logger.error(
                    f"<yellow>{self._exchange_id}</yellow> ❌ CRITICAL ERROR in poll_open_interest_once: {type(e).__name__}: {e}"
                )
                logger.exception(e)  # Full stack trace
                # Sleep before retry
                await cancellation_aware_sleep(10)

        # Return subscription configuration instead of calling _listen_to_stream directly
        return SubscriptionConfiguration(
            subscriber_func=poll_open_interest_once,
            unsubscriber_func=None,  # No cleanup needed for REST polling
            stream_name=name,
            requires_market_type_batching=False,
        )

    async def warmup(self, instruments: Set[Instrument], channel: CtrlChannel, **params) -> None:
        """
        Open interest warmup is typically not needed as it represents current state.

        Args:
            instruments: Set of instruments to warm up
            channel: Control channel for sending warmup data
        """
        # Open interest data is current state, no historical warmup needed
        logger.debug(f"<yellow>{self._exchange_id}</yellow> Open interest warmup skipped (current data only)")
        pass
