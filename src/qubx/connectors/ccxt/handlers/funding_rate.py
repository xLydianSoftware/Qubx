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

    @property
    def data_type(self) -> str:
        return "funding_rate"

    def prepare_subscription(
        self, name: str, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **_params
    ) -> SubscriptionConfiguration:
        """
        Prepare funding rate or funding payment subscription configuration.

        Args:
            name: Stream name for this subscription
            sub_type: Parsed subscription type ("funding_rate" or "funding_payment")
            channel: Control channel for managing subscription lifecycle
            instruments: Set of instruments to subscribe to
            
        Returns:
            SubscriptionConfiguration with subscriber and unsubscriber functions
        """
        logger.debug(f"Preparing {sub_type} subscription for {len(instruments)} instruments")
        
        # Convert to CCXT symbol format
        symbols = [instrument_to_ccxt_symbol(instr) for instr in instruments]
        
        # Determine if this is a funding payment subscription
        is_payment_subscription = (sub_type == DataType.FUNDING_PAYMENT)

        async def watch_funding_rates():
            try:
                # Use symbols captured at subscription time
                funding_rates = await self._exchange.watch_funding_rates(symbols)
                current_time = self._data_provider.time_provider.time()

                # Process individual funding rate updates per instrument
                for symbol, info in funding_rates.items():
                    try:
                        instrument = ccxt_find_instrument(symbol, self._exchange)
                        funding_rate = ccxt_convert_funding_rate(info)

                        self._data_provider._health_monitor.record_data_arrival(sub_type, dt_64(current_time, "s"))

                        if is_payment_subscription:
                            # For funding payment subscriptions, check if we should emit a payment
                            if self._should_emit_payment(instrument, funding_rate):
                                payment = self._create_funding_payment(instrument)
                                channel.send((instrument, DataType.FUNDING_PAYMENT, payment, False))
                                logger.debug(f"Emitted funding payment for {instrument.symbol}: rate={payment.funding_rate:.6f}")
                        else:
                            # For funding rate subscriptions, send the rate directly
                            channel.send((instrument, DataType.FUNDING_RATE, funding_rate, False))

                    except CcxtSymbolNotRecognized:
                        continue

            except Exception as e:
                logger.exception(e)
                # Re-raise to trigger retry logic in _listen_to_stream
                raise

        async def un_watch_funding_rates():
            unwatch_func = getattr(self._exchange, "un_watch_funding_rates", None)
            if unwatch_func is not None and callable(unwatch_func):
                unwatch_result = unwatch_func()
                if unwatch_result is not None:
                    await unwatch_result

        # Return subscription configuration
        return SubscriptionConfiguration(
            subscriber_func=watch_funding_rates,
            unsubscriber_func=un_watch_funding_rates,
            stream_name=name,
            requires_market_type_batching=False,
        )

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
            'rate': rate,
            'payment_time': rate.next_funding_time,
            'stored_at': rate.time
        }
        
        # If this is first update, don't emit
        if last_info is None:
            logger.debug(f"Stored first funding rate for {instrument.symbol}: rate={rate.rate:.6f}, next_funding={rate.next_funding_time}")
            return False
        
        # Emit if next_funding_time has advanced (new funding period started)
        if rate.next_funding_time > last_info['payment_time']:
            # Store payment info for _create_funding_payment
            self._pending_funding_rates[f"{key}_payment"] = {
                'rate': last_info['rate'].rate,
                'time': last_info['payment_time'],  # Use the actual payment time
                'interval_hours': self._extract_interval_hours(last_info['rate'].interval)
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
                time=payment_info['time'],
                funding_rate=payment_info['rate'],
                funding_interval_hours=payment_info['interval_hours']
            )
        else:
            # Fallback - shouldn't happen in normal operation
            logger.warning(f"No payment info stored for {instrument.symbol}, using current rate")
            current_info = self._pending_funding_rates.get(instrument.symbol)
            if current_info:
                rate = current_info['rate']
                return FundingPayment(
                    time=rate.time,
                    funding_rate=rate.rate,
                    funding_interval_hours=self._extract_interval_hours(rate.interval)
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
        if isinstance(interval, str) and interval.endswith('h'):
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
