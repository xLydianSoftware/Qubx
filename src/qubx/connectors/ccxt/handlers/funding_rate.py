"""
Funding rate data type handler for CCXT data provider.

Handles subscription and warmup for funding rate data.
"""

from typing import Set

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, FundingPayment, FundingRate, Instrument, dt_64

from ..exceptions import CcxtSymbolNotRecognized
from ..subscription_config import SubscriptionConfiguration
from ..utils import ccxt_convert_funding_rate, ccxt_find_instrument, instrument_to_ccxt_symbol
from .base import BaseDataTypeHandler


class FundingRateDataHandler(BaseDataTypeHandler):
    """Handler for funding rate data subscription and processing.

    Supports both funding_rate and funding_payment subscriptions:
    - funding_rate: Real-time rate updates
    - funding_payment: Calculated payment events when funding intervals elapse
    """

    def __init__(self, data_provider, exchange, exchange_id: str):
        super().__init__(data_provider, exchange, exchange_id)
        # Store funding rate history for payment emission logic
        self._pending_funding_rates: dict[str, dict] = {}  # Store rates per instrument
        
        # Unified stream management with reference counting
        self._unified_stream_name: str | None = None  # Single stream name for all funding subscriptions
        self._reference_count = 0  # Track how many funding subscriptions are active
        self._active_subscriptions: set[str] = set()  # Track which subscription types are active
        self._subscription_channels: dict[str, CtrlChannel] = {}  # Map subscription type to channel
        self._subscription_instruments: set[Instrument] = set()  # All instruments across subscriptions

    @property
    def data_type(self) -> str:
        return "funding_rate"

    def prepare_subscription(
        self, name: str, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **_params
    ) -> SubscriptionConfiguration:
        """
        Prepare funding rate or funding payment subscription configuration.
        
        Each subscription appears normal to the orchestrator but shares the same WebSocket stream.
        """
        # Register this subscription 
        self._active_subscriptions.add(sub_type)
        self._subscription_channels[sub_type] = channel
        self._subscription_instruments.update(instruments)
        
        # All funding subscriptions share the same underlying stream name pattern
        # but each subscription appears independent to the orchestrator
        self._reference_count += 1
        
        # Convert to CCXT symbols (captured for this subscription)
        symbols = [instrument_to_ccxt_symbol(instr) for instr in instruments]
        if _params.pop("__all__", False):
            symbols = None

        async def watch_funding_shared():
            """
            Each funding subscription gets its own apparent subscriber, but they all
            call the same underlying WebSocket method and share data processing.
            """
            try:
                # All funding types use the same exchange call
                if _params:
                    funding_rates = await self._exchange.watch_funding_rates(symbols, _params)
                else:
                    funding_rates = await self._exchange.watch_funding_rates(symbols)
                
                current_time = self._data_provider.time_provider.time()
                
                # Process funding rate updates (shared logic for all funding types)
                for symbol, info in funding_rates.items():
                    try:
                        instrument = ccxt_find_instrument(symbol, self._exchange)
                        funding_rate = ccxt_convert_funding_rate(info)

                        # Record health for all active subscription types
                        for active_sub in self._active_subscriptions:
                            self._data_provider._health_monitor.record_data_arrival(active_sub, dt_64(current_time, "s"))

                        # Handle funding payments if needed
                        if self._should_emit_payment(instrument, funding_rate):
                            payment = self._create_funding_payment(instrument)
                            if DataType.FUNDING_PAYMENT in self._active_subscriptions:
                                payment_channel = self._subscription_channels.get(DataType.FUNDING_PAYMENT)
                                if payment_channel:
                                    payment_channel.send((instrument, DataType.FUNDING_PAYMENT, payment, False))

                        # Send funding rates if needed  
                        if DataType.FUNDING_RATE in self._active_subscriptions:
                            rate_channel = self._subscription_channels.get(DataType.FUNDING_RATE)
                            if rate_channel:
                                rate_channel.send((instrument, DataType.FUNDING_RATE, funding_rate, False))

                    except CcxtSymbolNotRecognized:
                        continue

            except Exception as e:
                logger.error(f"[FUNDING] Exception in {sub_type} subscriber: {e}")
                raise

        async def cleanup_funding_shared():
            """Each subscription cleans up its own state with reference counting."""
            self._active_subscriptions.discard(sub_type)
            self._subscription_channels.pop(sub_type, None)
            self._reference_count -= 1
            
            # Only clean up exchange resources when all funding subscriptions are gone
            if self._reference_count <= 0:
                self._subscription_instruments.clear()
                
                # Call exchange unwatch
                unwatch_func = getattr(self._exchange, "un_watch_funding_rates", None)
                if unwatch_func and callable(unwatch_func):
                    try:
                        unwatch_result = unwatch_func()
                        if unwatch_result:
                            await unwatch_result
                    except Exception as e:
                        logger.warning(f"Exception during funding unwatch: {e}")

        # Each subscription returns its own config with the orchestrator-provided name
        # This makes each subscription appear independent to the orchestrator
        config = SubscriptionConfiguration(
            subscription_type=sub_type,
            subscriber_func=watch_funding_shared,
            unsubscriber_func=cleanup_funding_shared,
            stream_name=name,  # Use orchestrator-provided name (different for each subscription)
        )
        
        return config

    def _should_emit_payment(self, instrument: Instrument, rate: FundingRate) -> bool:
        """
        Determine if a funding payment should be emitted.

        Uses the "rate with payment time" approach: emit payment when next_funding_time
        changes, using the rate that was active during the previous funding period.

        Args:
            instrument: The trading instrument
            rate: Current funding rate update

        Returns:
            bool: True if payment should be emitted
        """
        key = instrument.symbol

        # Get last stored funding info
        last_info = self._pending_funding_rates.get(key)

        # Always store current rate with its payment time
        self._pending_funding_rates[key] = {
            "rate": rate,
            "payment_time": rate.next_funding_time,
            "stored_at": rate.time,
        }

        # If this is first update, don't emit
        if last_info is None:
            logger.debug(
                f"Stored first funding rate for {instrument.symbol}: rate={rate.rate:.6f}, next_funding={rate.next_funding_time}"
            )
            return False

        # Emit if next_funding_time has advanced (new funding period started)
        if rate.next_funding_time > last_info["payment_time"]:
            # Store payment info for _create_funding_payment
            self._pending_funding_rates[f"{key}_payment"] = {
                "rate": last_info["rate"].rate,
                "time": last_info["payment_time"],  # Use the actual payment time
                "interval_hours": self._extract_interval_hours(last_info["rate"].interval),
            }

            logger.debug(
                f"Funding payment trigger for {instrument.symbol}: "
                f"rate={last_info['rate'].rate:.6f}, "
                f"payment_time={last_info['payment_time']}, "
                f"new_next_funding={rate.next_funding_time}"
            )
            return True

        return False

    def _create_funding_payment(self, instrument: Instrument) -> FundingPayment:
        """
        Create funding payment using stored payment info.

        Args:
            instrument: The trading instrument

        Returns:
            FundingPayment: The funding payment event
        """
        payment_key = f"{instrument.symbol}_payment"
        payment_info = self._pending_funding_rates.get(payment_key)

        if payment_info:
            return FundingPayment(
                time=payment_info["time"],
                funding_rate=payment_info["rate"],
                funding_interval_hours=payment_info["interval_hours"],
            )
        else:
            # Fallback - shouldn't happen in normal operation
            logger.warning(f"No payment info stored for {instrument.symbol}, using current rate")
            current_info = self._pending_funding_rates.get(instrument.symbol)
            if current_info:
                rate = current_info["rate"]
                return FundingPayment(
                    time=rate.time,
                    funding_rate=rate.rate,
                    funding_interval_hours=self._extract_interval_hours(rate.interval),
                )
            else:
                # Last resort fallback
                raise ValueError(f"No funding rate data available for {instrument.symbol}")

    def _extract_interval_hours(self, interval: str) -> int:
        """
        Extract hours from interval string (e.g., '8h' -> 8).

        Args:
            interval: Interval string from funding rate

        Returns:
            int: Hours as integer
        """
        if isinstance(interval, str) and interval.endswith("h"):
            return int(interval[:-1])
        elif isinstance(interval, str) and interval.isdigit():
            return int(interval)
        else:
            # Default to 8 hours for unknown formats (most common)
            logger.warning(f"Unknown funding interval format: {interval}, defaulting to 8h")
            return 8

    async def warmup(self, _instruments: Set[Instrument], **_params) -> None:
        """
        Funding rate warmup is typically not needed as it represents current rates.
        Funding payment warmup is handled by the backtester's simulated data.

        Args:
            _instruments: Set of instruments to warm up (unused)
            **_params: Additional parameters (unused for funding rates)
        """
        # Funding rate data is typically current state, no historical warmup needed
        # Funding payment data should come from historical data in backtesting
        pass
    
    def cleanup_subscription(self, sub_type: str) -> None:
        """Clean up a specific subscription type.
        
        Args:
            sub_type: The subscription type to clean up
        """
        self._active_subscriptions.discard(sub_type)
        self._subscription_channels.pop(sub_type, None)
        
        # If no more subscriptions, reset everything
        if not self._active_subscriptions:
            self._subscription_instruments.clear()
            logger.debug("All funding subscriptions cleaned up")
