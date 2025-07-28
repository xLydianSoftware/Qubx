"""
Open Interest data type handler for CCXT data provider.

Handles subscription and warmup for open interest data using REST API polling.
"""

import asyncio
from asyncio.exceptions import CancelledError
from typing import Set

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import CtrlChannel, Instrument
from qubx.utils.time import floor_t64

from ..exceptions import CcxtSymbolNotRecognized
from ..subscription_config import SubscriptionConfiguration
from ..utils import ccxt_convert_open_interest, ccxt_find_instrument
from .base import BaseDataTypeHandler


class OpenInterestDataHandler(BaseDataTypeHandler):
    """Handler for open interest data subscription and processing using REST polling."""
    
    # Polling configuration constants
    POLL_INTERVAL_MINUTES = 5  # Poll every 5 minutes
    BOUNDARY_TOLERANCE_SECONDS = 30  # Poll within 30 seconds of boundary
    SHORT_SLEEP_SECONDS = 5  # Sleep when not polling time
    LONG_SLEEP_SECONDS = 100  # Sleep after successful poll
    ERROR_SLEEP_SECONDS = 10  # Sleep after error
    CANCELLATION_CHECK_INTERVAL = 0.5  # Check for cancellation every 500ms

    @property
    def data_type(self) -> str:
        return "open_interest"

    def prepare_subscription(
        self, name: str, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **params
    ) -> SubscriptionConfiguration:
        """
        Prepare open interest subscription configuration using REST API polling.

        Binance doesn't provide open interest via WebSocket, so we poll at regular intervals
        using proper time boundaries. Uses qubx time provider and utilities for accurate timing.

        Args:
            name: Stream name for this subscription
            sub_type: Parsed subscription type ("open_interest")
            channel: Control channel for managing subscription lifecycle
            instruments: Set of instruments to subscribe to
            
        Returns:
            SubscriptionConfiguration with subscriber function and no unsubscriber (REST polling)
        """

        def get_current_5min_boundary():
            """Get the current polling boundary timestamp for data timestamping"""
            current_time = self._data_provider.time_provider.time()
            boundary_time = floor_t64(current_time, np.timedelta64(self.POLL_INTERVAL_MINUTES, 'm'))
            return boundary_time.astype('datetime64[ns]').view('int64')

        def should_poll_now():
            """Check if we should poll now (within tolerance of polling boundary)"""
            current_time = self._data_provider.time_provider.time()
            boundary_time = floor_t64(current_time, np.timedelta64(self.POLL_INTERVAL_MINUTES, 'm'))
            
            # Check if we're within tolerance of the boundary
            time_diff = current_time - boundary_time
            seconds_since_boundary = time_diff / np.timedelta64(1, 's')
            
            return seconds_since_boundary < self.BOUNDARY_TOLERANCE_SECONDS

        # Track if this is the first poll for initial data
        first_poll = True

        async def poll_open_interest_once():
            """Single polling operation - called repeatedly by _listen_to_stream"""
            nonlocal first_poll

            async def cancellation_aware_sleep(seconds: float):
                """Sleep that can be interrupted by subscription cancellation"""
                elapsed = 0.0

                while elapsed < seconds:
                    remaining = min(self.CANCELLATION_CHECK_INTERVAL, seconds - elapsed)
                    await asyncio.sleep(remaining)
                    elapsed += remaining

                    # Check if subscription was cancelled
                    if not self._data_provider._connection_manager.is_stream_enabled(name):
                        raise CancelledError("Subscription cancelled")

            async def fetch_open_interest_data():
                """Fetch open interest data for all subscribed symbols"""
                # Get current subscribed symbols (refreshed each call - handles universe changes!)
                subscribed_instruments = self._data_provider.get_subscribed_instruments(sub_type)
                symbols = [instr.symbol for instr in subscribed_instruments] if subscribed_instruments else []

                if not symbols:
                    return

                # Use current time for first poll, boundary time for scheduled polls
                if first_poll:
                    current_time = self._data_provider.time_provider.time()
                    timestamp_ns = current_time.astype('datetime64[ns]').view('int64')
                else:
                    timestamp_ns = get_current_5min_boundary()

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

                        # Override timestamp - use current time for first poll, boundary time for regular polls
                        # Convert nanoseconds to milliseconds properly (ns -> ms)
                        timestamp_ms = timestamp_ns // 1_000_000
                        oi_data["timestamp"] = timestamp_ms

                        instrument = ccxt_find_instrument(symbol, self._exchange)
                        open_interest = ccxt_convert_open_interest(symbol, oi_data)
                        
                        # Use pandas for robust timestamp conversion for health monitoring
                        health_timestamp = pd.Timestamp(timestamp_ms, unit="ms").asm8
                        self._data_provider._health_monitor.record_data_arrival(
                            sub_type, health_timestamp
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

            try:
                # Check for cancellation immediately at start of each poll cycle
                if not self._data_provider._connection_manager.is_stream_enabled(name):
                    raise CancelledError("Subscription cancelled")

                # Always perform initial query immediately on first poll
                if first_poll:
                    logger.debug(f"<yellow>{self._exchange_id}</yellow> Performing initial open interest query")
                    await fetch_open_interest_data()
                    first_poll = False
                    # After initial fetch, sleep until next scheduled poll
                    await cancellation_aware_sleep(self.SHORT_SLEEP_SECONDS)
                    return

                # For subsequent polls, only fetch if it's time to poll
                if should_poll_now():
                    await fetch_open_interest_data()
                    # After successful poll, sleep longer until next check
                    await cancellation_aware_sleep(self.LONG_SLEEP_SECONDS)
                else:
                    # Not time to poll yet, sleep briefly and check again
                    await cancellation_aware_sleep(self.SHORT_SLEEP_SECONDS)

            except CancelledError:
                raise  # Re-raise to exit _listen_to_stream
            except Exception as e:
                logger.error(
                    f"<yellow>{self._exchange_id}</yellow> ❌ CRITICAL ERROR in poll_open_interest_once: {type(e).__name__}: {e}"
                )
                logger.exception(e)  # Full stack trace
                # Sleep before retry
                await cancellation_aware_sleep(self.ERROR_SLEEP_SECONDS)

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
