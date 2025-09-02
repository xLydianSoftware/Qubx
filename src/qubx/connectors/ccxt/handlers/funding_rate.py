"""
Funding rate data type handler for CCXT data provider.

Simplified unified approach: both funding_rate and funding_payment subscriptions
use the same WebSocket stream and always emit both data types when appropriate.
"""

from typing import Set

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, FundingPayment, FundingRate, Instrument, dt_64

from ..exceptions import CcxtSymbolNotRecognized
from ..subscription_config import SubscriptionConfiguration
from ..utils import ccxt_convert_funding_rate, ccxt_find_instrument, instrument_to_ccxt_symbol
from .base import BaseDataTypeHandler


class FundingRateDataHandler(BaseDataTypeHandler):
    """
    Unified funding rate handler for both funding_rate and funding_payment subscriptions.

    Always subscribes to funding rates and emits both FundingRate and FundingPayment
    data when appropriate, regardless of which subscription type was requested.
    """

    def __init__(self, data_provider, exchange_manager, exchange_id: str):
        super().__init__(data_provider, exchange_manager, exchange_id)
        # Store funding rate history for payment emission logic
        self._pending_funding_rates: dict[str, dict] = {}  # Store rates per instrument

    @property
    def data_type(self) -> str:
        return "funding_rate"

    def prepare_subscription(
        self, name: str, sub_type: str, channel: CtrlChannel, instruments: Set[Instrument], **_params
    ) -> SubscriptionConfiguration:
        """
        Prepare unified funding subscription configuration.

        Both funding_rate and funding_payment subscriptions use the same underlying
        WebSocket stream and emit both data types when appropriate.
        """
        # Convert to CCXT symbols
        symbols = [instrument_to_ccxt_symbol(instr) for instr in instruments]
        if _params.pop("__all__", False):
            symbols = None

        async def watch_funding():
            """Unified subscriber that handles both funding rates and payments."""
            try:
                if _params:
                    funding_rates = await self._exchange_manager.exchange.watch_funding_rates(symbols, _params)  # type: ignore
                else:
                    funding_rates = await self._exchange_manager.exchange.watch_funding_rates(symbols)  # type: ignore

                current_time = self._data_provider.time_provider.time()

                # Process funding rate updates if we got valid data
                if funding_rates:
                    for symbol, info in funding_rates.items():
                        try:
                            instrument = ccxt_find_instrument(symbol, self._exchange_manager.exchange)
                            funding_rate = ccxt_convert_funding_rate(info)

                            # Notify all listeners
                            self._data_provider.notify_data_arrival(DataType.FUNDING_RATE, dt_64(current_time, "s"))

                            # Always emit funding rate
                            channel.send((instrument, DataType.FUNDING_RATE, funding_rate, False))

                            # Emit payment if funding interval changed
                            if self._should_emit_payment(instrument, funding_rate):
                                payment = self._create_funding_payment(instrument)
                                channel.send((instrument, DataType.FUNDING_PAYMENT, payment, False))

                        except CcxtSymbolNotRecognized:
                            continue

            except Exception as e:
                logger.error(f"Exception in funding subscriber: {e}")
                raise

        async def cleanup_funding():
            """Simple cleanup - just call exchange unwatch if available."""
            unwatch_func = getattr(self._exchange_manager.exchange, "un_watch_funding_rates", None)
            if unwatch_func and callable(unwatch_func):
                try:
                    unwatch_result = unwatch_func()
                    if unwatch_result:
                        await unwatch_result
                except Exception as e:
                    logger.warning(f"Exception during funding unwatch: {e}")

        # Return unified subscription configuration
        return SubscriptionConfiguration(
            subscription_type=sub_type,
            subscriber_func=watch_funding,
            unsubscriber_func=cleanup_funding,
            stream_name=name,
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
            "rate": rate,
            "payment_time": rate.next_funding_time,
            "stored_at": rate.time,
        }

        # If this is first update, don't emit
        if last_info is None:
            return False

        # Emit if next_funding_time has advanced (new funding period started)
        if rate.next_funding_time > last_info["payment_time"]:
            # Store payment info for _create_funding_payment
            self._pending_funding_rates[f"{key}_payment"] = {
                "rate": last_info["rate"].rate,
                "time": last_info["payment_time"],  # Use the actual payment time
                "interval_hours": self._extract_interval_hours(last_info["rate"].interval),
            }

            # logger.debug(
            #     f"Funding payment trigger for {instrument.symbol}: "
            #     f"rate={last_info['rate'].rate:.6f}, "
            #     f"payment_time={last_info['payment_time']}, "
            #     f"new_next_funding={rate.next_funding_time}"
            # )
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

    async def warmup(self, instruments: Set[Instrument], **params) -> None:
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
