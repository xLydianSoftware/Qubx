"""Market stats handler for Lighter WebSocket messages"""

import time
from typing import TYPE_CHECKING, Any

from qubx import logger
from qubx.core.basics import FundingPayment, FundingRate, Instrument, OpenInterest, dt_64
from qubx.utils.time import floor_t64

from .base import BaseHandler

if TYPE_CHECKING:
    from ..instruments import LighterInstrumentLoader


class MarketStatsHandler(BaseHandler[dict[Instrument, list[FundingRate | OpenInterest | FundingPayment]]]):
    """
    Handler for market stats messages.
    It creates open interest, funding rate, and funding payment data objects.

    Message format (example):
    ```json
    {
        'channel': 'market_stats:all',
        'market_stats': {'0': {'market_id': 0,
        'index_price': '4021.32',
        'mark_price': '4019.38',
        'open_interest': '176612968.404318',
        'open_interest_limit': '72057594037927936.000000',
        'funding_clamp_small': '0.0500',
        'funding_clamp_big': '4.0000',
        'last_trade_price': '4019.04',
        'current_funding_rate': '0.0004',
        'funding_rate': '-0.0002',
        'funding_timestamp': 1760972400002,
        'daily_base_token_volume': 377551.1336,
        'daily_quote_token_volume': 1515105897.814945,
        'daily_price_low': 3907.78,
        'daily_price_high': 4082.24,
        'daily_price_change': 0.9686203189212543}},
        'type': 'update/market_stats'
    }
    ```

    Returns:
        dict[Instrument, list[FundingRate | OpenInterest | FundingPayment]]
        Maps each instrument to list of data objects generated from the message.

    Note on funding rates:
        - 'current_funding_rate': Active rate (used for FundingRate.rate)
        - 'funding_rate': Rate that was paid at last funding_timestamp
        - 'funding_timestamp': When last payment occurred (in the past)
        - Lighter uses 1-hour funding intervals
    """

    FUNDING_INTERVAL_HOURS = 1  # Lighter uses 1-hour funding
    FUNDING_RATE_INTERVAL = "1min"
    OPEN_INTEREST_INTERVAL = "5min"

    def __init__(self, instrument_loader: "LighterInstrumentLoader"):
        """
        Initialize market stats handler.

        Args:
            instrument_loader: LighterInstrumentLoader for market_id → Instrument mapping
        """
        super().__init__()
        self.instrument_loader: LighterInstrumentLoader = instrument_loader

        # Track funding state per market_id for payment detection
        self._last_funding_timestamps: dict[int, int] = {}  # market_id → funding_timestamp (ms)

        # Buffer for interval-based emission (stores last message in current interval)
        self._funding_rate_buffer: dict[int, dict] = {}  # market_id → {data, raw_time}
        self._open_interest_buffer: dict[int, dict] = {}  # market_id → {data, raw_time}

        # Track last floored boundary we emitted (None = no emission yet)
        self._funding_rate_last_boundary: dict[int, dt_64 | None] = {}
        self._open_interest_last_boundary: dict[int, dt_64 | None] = {}

    def can_handle(self, message: dict[str, Any]) -> bool:
        """Check if message is market_stats update"""
        return (
            message.get("channel", "").startswith("market_stats:")
            and message.get("type") in ["update/market_stats", "subscribed/market_stats"]
            and "market_stats" in message
        )

    def _handle_impl(
        self, message: dict[str, Any]
    ) -> dict[Instrument, list[FundingRate | OpenInterest | FundingPayment]] | None:
        """
        Process all markets in message.

        Args:
            message: Raw Lighter market stats message

        Returns:
            Dictionary mapping Instrument to list of data objects, or None if no valid data

        Raises:
            ValueError: If message format is invalid
        """
        market_stats = message.get("market_stats")
        if not market_stats:
            raise ValueError("Missing market_stats in message")

        # Get message timestamp (use current time if not in message)
        msg_timestamp_ms = message.get("timestamp")
        if msg_timestamp_ms:
            msg_time_ns = int(msg_timestamp_ms * 1_000_000)  # ms → ns
            msg_time = dt_64(msg_time_ns, "ns")
        else:
            msg_time_ns = int(time.time() * 1_000_000_000)
            msg_time = dt_64(msg_time_ns, "ns")

        results: dict[Instrument, list[FundingRate | OpenInterest | FundingPayment]] = {}

        # Process each market in the message
        for market_id_str, market_data in market_stats.items():
            try:
                market_id = int(market_id_str)
            except ValueError:
                logger.warning(f"Invalid market_id format: {market_id_str}, skipping")
                continue

            # Resolve market_id → Instrument
            instrument = self.instrument_loader.get_instrument_by_market_id(market_id)
            if instrument is None:
                logger.debug(f"Unknown market_id: {market_id}, skipping")
                continue

            # Process this market's data
            objects = self._process_market_data(market_id, instrument, market_data, msg_time)

            if objects:
                results[instrument] = objects

        return results if results else None

    def _process_market_data(
        self, market_id: int, instrument: Instrument, data: dict, msg_time: dt_64
    ) -> list[FundingRate | OpenInterest | FundingPayment]:
        """
        Process single market's stats data.

        Args:
            market_id: Lighter market ID
            instrument: Qubx Instrument
            data: Market stats data from message
            msg_time: Message timestamp as dt_64

        Returns:
            List of data objects (FundingRate, OpenInterest, FundingPayment)
        """
        objects: list[FundingRate | OpenInterest | FundingPayment] = []

        try:
            # Extract fields (all are strings in message)
            funding_rate_str = data.get("funding_rate")  # Past payment rate
            current_funding_rate_str = data.get("current_funding_rate")  # Active rate
            funding_timestamp_ms = data.get("funding_timestamp")  # When last payment occurred
            open_interest_str = data.get("open_interest")
            mark_price_str = data.get("mark_price")
            index_price_str = data.get("index_price")

            # Convert to appropriate types
            current_funding_rate = float(current_funding_rate_str) if current_funding_rate_str else None
            funding_timestamp = int(funding_timestamp_ms) if funding_timestamp_ms else None
            mark_price = float(mark_price_str) if mark_price_str else None
            index_price = float(index_price_str) if index_price_str else None

            # 1. Create FundingRate (active rate) - interval-based emission
            if current_funding_rate is not None and funding_timestamp is not None:
                # funding_timestamp is when LAST payment occurred
                # next payment is funding_timestamp + 1 hour
                funding_timestamp_ns = int(funding_timestamp * 1_000_000)
                next_funding_ns = funding_timestamp_ns + (self.FUNDING_INTERVAL_HOURS * 3600 * 1_000_000_000)

                # Prepare data for buffering
                funding_rate_data = {
                    "rate": current_funding_rate / 100.0,  # Convert from percentage to decimal
                    "next_funding_time": dt_64(next_funding_ns, "ns"),
                    "mark_price": mark_price,
                    "index_price": index_price,
                }

                # Check if we should emit based on interval
                should_emit, emission_time, data_to_emit = self._check_interval_emission(
                    market_id,
                    msg_time,
                    self._funding_rate_buffer,
                    self._funding_rate_last_boundary,
                    self.FUNDING_RATE_INTERVAL,
                )

                if should_emit and data_to_emit:
                    assert emission_time is not None, "emission_time should not be None when should_emit is True"
                    # Emit with boundary timestamp
                    funding_rate_obj = FundingRate(
                        time=emission_time,  # Boundary timestamp (when data becomes available)
                        rate=data_to_emit["rate"],
                        interval="1h",
                        next_funding_time=data_to_emit["next_funding_time"],
                        mark_price=data_to_emit.get("mark_price"),
                        index_price=data_to_emit.get("index_price"),
                    )
                    objects.append(funding_rate_obj)

                # Always buffer current data
                self._funding_rate_buffer[market_id] = {
                    "data": funding_rate_data,
                    "raw_time": msg_time,
                }

            # 2. Create OpenInterest - interval-based emission
            if open_interest_str and mark_price:
                open_interest = float(open_interest_str)
                open_interest_usd = open_interest * mark_price

                # Prepare data for buffering
                oi_data = {
                    "symbol": instrument.symbol,
                    "open_interest": open_interest,
                    "open_interest_usd": open_interest_usd,
                }

                # Check if we should emit based on interval
                should_emit, emission_time, data_to_emit = self._check_interval_emission(
                    market_id,
                    msg_time,
                    self._open_interest_buffer,
                    self._open_interest_last_boundary,
                    self.OPEN_INTEREST_INTERVAL,
                )

                if should_emit and data_to_emit:
                    assert emission_time is not None, "emission_time should not be None when should_emit is True"
                    # Emit with boundary timestamp
                    oi_obj = OpenInterest(
                        time=emission_time,  # Boundary timestamp (when data becomes available)
                        symbol=data_to_emit["symbol"],
                        open_interest=data_to_emit["open_interest"],
                        open_interest_usd=data_to_emit["open_interest_usd"],
                    )
                    objects.append(oi_obj)

                # Always buffer current data
                self._open_interest_buffer[market_id] = {
                    "data": oi_data,
                    "raw_time": msg_time,
                }

            # 3. Detect and create FundingPayment (when funding_timestamp changes)
            if funding_timestamp is not None and funding_rate_str is not None:
                if self._should_emit_funding_payment(market_id, funding_timestamp):
                    funding_rate_paid = float(funding_rate_str)
                    funding_timestamp_ns = int(funding_timestamp * 1_000_000)

                    payment = FundingPayment(
                        time=floor_t64(
                            dt_64(funding_timestamp_ns, "ns"), self.FUNDING_INTERVAL_HOURS
                        ),  # When payment occurred
                        funding_rate=funding_rate_paid / 100.0,  # Convert from percentage to decimal
                        funding_interval_hours=self.FUNDING_INTERVAL_HOURS,
                    )
                    objects.append(payment)

                # Update tracking
                self._last_funding_timestamps[market_id] = funding_timestamp

        except (ValueError, TypeError) as e:
            logger.error(f"Error processing market {market_id} ({instrument.symbol}) stats: {e}")
            return []

        return objects

    def _should_emit_funding_payment(self, market_id: int, current_funding_timestamp: int) -> bool:
        """
        Check if funding payment should be emitted.
        Emit when funding_timestamp advances (new hour started).

        Args:
            market_id: Market ID
            current_funding_timestamp: Current funding_timestamp from message (ms)

        Returns:
            True if payment should be emitted
        """
        last_timestamp = self._last_funding_timestamps.get(market_id)

        # Don't emit on first message (no previous state)
        if last_timestamp is None:
            return False

        # Emit if timestamp advanced (new funding period)
        return current_funding_timestamp > last_timestamp

    def _check_interval_emission(
        self,
        market_id: int,
        msg_time: dt_64,
        buffer_dict: dict[int, dict],
        last_boundary_dict: dict[int, dt_64 | None],
        interval: str,
    ) -> tuple[bool, dt_64 | None, dict | None]:
        """
        Check if we should emit based on interval boundary crossing.

        This implements interval-based emission with no lookahead bias:
        - First message: buffer only, don't emit
        - Boundary crossed: emit buffered data from previous interval with current boundary timestamp
        - Same interval: buffer only (overwrites previous)

        Args:
            market_id: Market ID
            msg_time: Current message timestamp
            buffer_dict: Buffer dictionary to check/update
            last_boundary_dict: Last boundary dictionary to check/update
            interval: Interval string (e.g., "1min", "5min")

        Returns:
            (should_emit, emission_timestamp, data_to_emit)
        """
        # Floor current message time to interval
        current_boundary = floor_t64(msg_time, interval)

        # Get last emitted boundary
        last_boundary = last_boundary_dict.get(market_id)

        # First message ever - buffer but don't emit
        if last_boundary is None:
            last_boundary_dict[market_id] = current_boundary
            return (False, None, None)

        # Check if we crossed a boundary
        if current_boundary > last_boundary:
            # Get buffered data from PREVIOUS interval
            buffered = buffer_dict.get(market_id)

            if buffered is None:
                # No data to emit (shouldn't happen, but handle gracefully)
                last_boundary_dict[market_id] = current_boundary
                return (False, None, None)

            # Emit buffered data with CURRENT boundary timestamp
            # (This is when the data becomes available, not when it was measured)
            last_boundary_dict[market_id] = current_boundary
            return (True, current_boundary, buffered["data"])

        # Still in same interval - buffer only
        return (False, None, None)
