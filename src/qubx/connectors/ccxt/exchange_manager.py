"""
ExchangeManager: Transparent wrapper for CCXT exchanges with automatic recreation.

Provides seamless exchange recreation without affecting consuming components.
"""

import asyncio
import threading
import time
from typing import Any, Callable, Optional

import pandas as pd

import ccxt.pro as cxp
from qubx import logger
from qubx.core.basics import dt_64
from qubx.core.interfaces import IDataArrivalListener

# Constants for better maintainability
DEFAULT_CHECK_INTERVAL_SECONDS = 60.0
DEFAULT_MAX_RECREATIONS = 5
DEFAULT_RESET_INTERVAL_HOURS = 6.0
SECONDS_PER_HOUR = 3600

# Custom stall detection thresholds (in seconds)
STALL_THRESHOLDS = {
    'funding_payment': 12 * SECONDS_PER_HOUR,    # 12 hours = 43,200s
    'open_interest': 30 * 60,                    # 30 minutes = 1,800s
    'orderbook': 5 * 60,                         # 5 minutes = 300s
    'trade': 60 * 60,                            # 60 minutes = 3,600s
    'liquidation': 7 * 24 * SECONDS_PER_HOUR,    # 7 days = 604,800s
    'ohlc': 15 * 60,                             # 15 minutes = 900s
}
DEFAULT_STALL_THRESHOLD_SECONDS = 2 * SECONDS_PER_HOUR  # 2 hours = 7,200s


class ExchangeManager(IDataArrivalListener):
    """
    Wrapper for CCXT Exchange that handles recreation internally with self-monitoring.
    
    Exposes the underlying exchange via .exchange property for explicit access.
    Self-monitors for data stalls and triggers recreation automatically.
    
    Key Features:
    - Explicit .exchange property for CCXT access
    - Self-contained stall detection and recreation triggering
    - Circuit breaker protection with recreation limits
    - Atomic exchange transitions during recreation
    - Background monitoring thread for stall detection
    """
    
    _exchange: cxp.Exchange  # Type hint that this is always a valid exchange

    def __init__(
        self,
        exchange_name: str,
        factory_params: dict[str, Any],
        initial_exchange: Optional[cxp.Exchange] = None,
        max_recreations: int = DEFAULT_MAX_RECREATIONS,
        reset_interval_hours: float = DEFAULT_RESET_INTERVAL_HOURS,
        check_interval_seconds: float = DEFAULT_CHECK_INTERVAL_SECONDS,
    ):
        """Initialize ExchangeManager with underlying CCXT exchange.
        
        Args:
            exchange_name: Exchange name for factory (e.g., "binance.um")
            factory_params: Parameters for get_ccxt_exchange() 
            initial_exchange: Pre-created exchange instance (from factory)
            max_recreations: Maximum recreation attempts before giving up
            reset_interval_hours: Hours between recreation count resets
            check_interval_seconds: How often to check for stalls (default: 60.0)
        """
        self._exchange_name = exchange_name
        self._factory_params = factory_params.copy()
        self._max_recreations = max_recreations
        self._reset_interval_hours = reset_interval_hours
        
        # Recreation state
        self._recreation_count = 0
        self._recreation_lock = threading.RLock()
        self._last_successful_reset = time.time()
        
        # Stall detection state
        self._check_interval = check_interval_seconds
        self._last_data_times: dict[str, float] = {}
        self._data_lock = threading.RLock()
        
        # Monitoring control
        self._monitoring_enabled = False
        self._monitor_thread = None
        
        # Recreation callback management
        self._recreation_callbacks: list[Callable[[], None]] = []
        
        # Use provided exchange or create new one
        if initial_exchange:
            self._exchange = initial_exchange
            # Setup exception handler on provided exchange
            self._setup_ccxt_exception_handler(self._exchange)
        else:
            self._exchange = self._create_exchange()

    def _create_exchange(self) -> cxp.Exchange:
        """Create new raw CCXT exchange instance using factory method."""
        try:
            # Import here to avoid circular import (factory → broker → exchange_manager)
            from .factory import get_ccxt_exchange
            
            # Create raw exchange using factory logic
            ccxt_exchange = get_ccxt_exchange(**self._factory_params)
            
            # Setup exception handler for the new exchange
            self._setup_ccxt_exception_handler(ccxt_exchange)
            
            logger.debug(f"Created new {self._exchange_name} exchange instance")
            return ccxt_exchange
            
        except Exception as e:
            logger.error(f"Failed to create {self._exchange_name} exchange: {e}")
            raise RuntimeError(f"Failed to create {self._exchange_name} exchange: {e}") from e
    
    def register_recreation_callback(self, callback: Callable[[], None]) -> None:
        """Register callback to be called after successful exchange recreation.
        
        Args:
            callback: Function to call after successful recreation (no parameters)
        """
        self._recreation_callbacks.append(callback)
        logger.debug(f"Registered recreation callback for {self._exchange_name}")

    def _call_recreation_callbacks(self) -> None:
        """Call all registered recreation callbacks after successful exchange recreation."""
        logger.debug(f"Calling {len(self._recreation_callbacks)} recreation callbacks for {self._exchange_name}")
        
        for callback in self._recreation_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in recreation callback for {self._exchange_name}: {e}")
                # Continue with other callbacks even if one fails

    def force_recreation(self) -> bool:
        """
        Force recreation due to data stalls (called by BaseHealthMonitor).
        
        Returns:
            True if recreation successful, False if failed/limit exceeded
        """
        with self._recreation_lock:
            # Check recreation limit
            if self._recreation_count >= self._max_recreations:
                logger.error(f"Cannot recreate {self._exchange_name}: recreation limit ({self._max_recreations}) exceeded")
                return False
            
            logger.info(f"Stall-triggered recreation for {self._exchange_name}")
            return self._recreate_exchange()
    
    def _recreate_exchange(self) -> bool:
        """Recreate the underlying exchange (must be called with _recreation_lock held)."""
        self._recreation_count += 1
        logger.warning(f"Recreating {self._exchange_name} exchange (attempt {self._recreation_count}/{self._max_recreations})")
        
        # Create new exchange
        try:
            new_exchange = self._create_exchange()
        except Exception as e:
            logger.error(f"Failed to recreate {self._exchange_name} exchange: {e}")
            return False
            
        # Atomically replace the exchange
        old_exchange = self._exchange
        self._exchange = new_exchange
        
        # Clean up old exchange
        try:
            if hasattr(old_exchange, 'close') and hasattr(old_exchange, 'asyncio_loop'):
                old_exchange.asyncio_loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(old_exchange.close())
                )
        except Exception as e:
            logger.warning(f"Error closing old {self._exchange_name} exchange: {e}")
        
        logger.info(f"Successfully recreated {self._exchange_name} exchange")
        
        # Call recreation callbacks after successful recreation
        self._call_recreation_callbacks()
        
        return True
        
    def reset_recreation_count_if_needed(self) -> None:
        """Reset recreation count periodically (called by monitoring loop)."""
        reset_interval_seconds = self._reset_interval_hours * SECONDS_PER_HOUR
        
        current_time = time.time()
        time_since_reset = current_time - self._last_successful_reset
        
        if time_since_reset >= reset_interval_seconds and self._recreation_count > 0:
            logger.info(f"Resetting recreation count for {self._exchange_name} (was {self._recreation_count})")
            self._recreation_count = 0
            self._last_successful_reset = current_time
    
    def on_data_arrival(self, event_type: str, event_time: dt_64) -> None:
        """Record data arrival for stall detection.
        
        Args:
            event_type: Type of data event (e.g., "ohlcv", "trade", "orderbook")
            event_time: Timestamp of the data event (unused for stall detection)
        """
        current_timestamp = time.time()
        with self._data_lock:
            self._last_data_times[event_type] = current_timestamp
    
    def _extract_ohlc_timeframe(self, event_type: str) -> Optional[str]:
        """Extract timeframe from OHLC event type like 'ohlc(1m)' -> '1m'."""
        if event_type.startswith('ohlc(') and event_type.endswith(')'):
            return event_type[5:-1]  # Simple slice: ohlc(1m) -> 1m
        return None

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds using pandas.Timedelta."""
        return int(pd.Timedelta(timeframe).total_seconds())

    def _get_stall_threshold(self, event_type: str) -> float:
        """Get stall threshold for specific event type.
        
        Extracts base data type from parameterized types like 'ohlc(1m)' -> 'ohlc'.
        """
        # Extract base data type (everything before first '(' if present)
        base_event_type = event_type.split('(')[0]
        return float(STALL_THRESHOLDS.get(base_event_type, DEFAULT_STALL_THRESHOLD_SECONDS))
            
    def start_monitoring(self) -> None:
        """Start background stall detection monitoring."""
        if self._monitoring_enabled:
            return
            
        self._monitoring_enabled = True
        self._monitor_thread = threading.Thread(target=self._stall_monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.debug(f"ExchangeManager: Started stall monitoring for {self._exchange_name}")
        
    def stop_monitoring(self) -> None:
        """Stop background stall detection monitoring."""
        self._monitoring_enabled = False
        if self._monitor_thread:
            self._monitor_thread = None
        logger.debug(f"ExchangeManager: Stopped stall monitoring for {self._exchange_name}")
        
    def _stall_monitor_loop(self) -> None:
        """Background thread that checks for data stalls and triggers self-recreation."""
        while self._monitoring_enabled:
            try:
                self._check_and_handle_stalls()
                self.reset_recreation_count_if_needed()
                time.sleep(self._check_interval)
            except Exception as e:
                logger.error(f"Error in ExchangeManager stall detection: {e}")
                time.sleep(self._check_interval)
                
    def _check_and_handle_stalls(self) -> None:
        """Check for stalls using custom thresholds per data type."""
        current_time = time.time()
        stalled_types = []
        
        with self._data_lock:
            for event_type, last_data_time in self._last_data_times.items():
                time_since_data = current_time - last_data_time
                threshold = self._get_stall_threshold(event_type)
                
                if time_since_data > threshold:
                    stalled_types.append((event_type, time_since_data))
                    
        if not stalled_types:
            return  # No stalls detected
            
        stall_info = ", ".join([f"{event_type}({int(time_since)}s)" for event_type, time_since in stalled_types])
        logger.error(f"Data stalls detected in {self._exchange_name}: {stall_info}")
        
        try:
            logger.info(f"Self-triggering recreation for {self._exchange_name} due to stalls...")
            if self.force_recreation():
                logger.info(f"Stall-triggered recreation successful for {self._exchange_name}")
                with self._data_lock:
                    current_timestamp = time.time()
                    for event_type in self._last_data_times.keys():
                        self._last_data_times[event_type] = current_timestamp
            else:
                logger.error(f"Stall-triggered recreation failed for {self._exchange_name}")
        except Exception as e:
            logger.error(f"Error during stall-triggered recreation: {e}")
    
    def _setup_ccxt_exception_handler(self, exchange: cxp.Exchange) -> None:
        """
        Set up global exception handler for the CCXT async loop to handle unretrieved futures.

        This prevents 'Future exception was never retrieved' warnings from CCXT's internal
        per-symbol futures that complete with UnsubscribeError during resubscription.
        
        Applied to every newly created exchange (initial and recreated).
        """
        asyncio_loop = exchange.asyncio_loop

        def handle_ccxt_exception(loop, context):
            """Handle unretrieved exceptions from CCXT futures."""
            exception = context.get("exception")

            # Handle expected CCXT UnsubscribeError during resubscription
            if exception and "UnsubscribeError" in str(type(exception)):
                return

            # Handle other CCXT-related exceptions quietly if they're in our exchange context
            if exception and any(
                keyword in str(exception) for keyword in [exchange.id, "ohlcv", "orderbook", "ticker"]
            ):
                return

            # For all other exceptions, use the default handler
            if hasattr(loop, "default_exception_handler"):
                loop.default_exception_handler(context)
            else:
                # Fallback logging if no default handler
                logger.warning(f"Unhandled asyncio exception: {context}")

        # Set the custom exception handler on the CCXT loop
        asyncio_loop.set_exception_handler(handle_ccxt_exception)

    # === Exchange Property Access === 
    # Explicit property to access underlying CCXT exchange
    
    @property
    def exchange(self) -> cxp.Exchange:
        """Access to the underlying CCXT exchange instance.
        
        Use this property to call CCXT methods: exchange_manager.exchange.fetch_ticker(symbol)
        
        Returns:
            The current CCXT exchange instance (may change after recreation)
        """
        return self._exchange
