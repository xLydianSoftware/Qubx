"""
ExchangeManager: Transparent wrapper for CCXT exchanges with automatic recreation.

Provides seamless exchange recreation without affecting consuming components.
"""

import asyncio
import threading
import time
from threading import Thread
from typing import Any, Dict, Optional, Tuple

import ccxt.pro as cxp
from qubx import logger
from qubx.core.interfaces import IDataArrivalListener
from qubx.core.basics import dt_64

# Constants for better maintainability
DEFAULT_STALL_THRESHOLD_SECONDS = 120.0
DEFAULT_CHECK_INTERVAL_SECONDS = 30.0
DEFAULT_MAX_RECREATIONS = 3
DEFAULT_RESET_INTERVAL_HOURS = 24.0
SECONDS_PER_HOUR = 3600

# Parameter names that should be filtered out when creating exchanges
FILTERED_PARAMS = {
    'exchange', 'api_key', 'secret', 'loop', 'use_testnet', 
    'max_recreations', 'reset_interval_hours', 'stall_threshold_seconds',
    'check_interval_seconds'
}


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
        factory_params: Dict[str, Any],
        initial_exchange: Optional[cxp.Exchange] = None,
        max_recreations: int = DEFAULT_MAX_RECREATIONS,
        reset_interval_hours: float = DEFAULT_RESET_INTERVAL_HOURS,
        stall_threshold_seconds: float = DEFAULT_STALL_THRESHOLD_SECONDS,
        check_interval_seconds: float = DEFAULT_CHECK_INTERVAL_SECONDS,
    ):
        """Initialize ExchangeManager with underlying CCXT exchange.
        
        Args:
            exchange_name: Exchange name for factory (e.g., "binance.um")
            factory_params: Parameters for get_ccxt_exchange() 
            initial_exchange: Pre-created exchange instance (from factory)
            max_recreations: Maximum recreation attempts before giving up
            reset_interval_hours: Hours between recreation count resets
            stall_threshold_seconds: Seconds without data before considering stalled (default: 120.0)
            check_interval_seconds: How often to check for stalls (default: 30.0)
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
        self._stall_threshold = stall_threshold_seconds
        self._check_interval = check_interval_seconds
        self._last_data_times: Dict[str, float] = {}
        self._data_lock = threading.RLock()
        
        # Monitoring control
        self._monitoring_enabled = False
        self._monitor_thread = None
        
        # Use provided exchange or create new one
        if initial_exchange:
            self._exchange = initial_exchange
            # Setup exception handler on provided exchange
            self._setup_ccxt_exception_handler(self._exchange)
        else:
            self._exchange = self._create_exchange()

    def _create_exchange(self) -> cxp.Exchange:
        """Create new raw CCXT exchange instance (not wrapped in ExchangeManager)."""
        try:
            params = self._factory_params.copy()
            exchange_name = params['exchange']
            
            # Extract and validate exchange configuration
            resolved_exchange_id = self._resolve_exchange_id(exchange_name, params)
            
            # Build exchange options
            options = self._build_exchange_options(exchange_name, params)
            
            # Create the actual CCXT exchange instance
            ccxt_exchange = self._instantiate_ccxt_exchange(resolved_exchange_id, options, params)
            
            # Apply post-creation configuration
            self._configure_exchange(ccxt_exchange, params)
            
            # Setup exception handler for the new exchange
            self._setup_ccxt_exception_handler(ccxt_exchange)
            
            logger.debug(f"Created new {self._exchange_name} exchange instance")
            return ccxt_exchange
            
        except Exception as e:
            logger.error(f"Failed to create {self._exchange_name} exchange: {e}")
            raise RuntimeError(f"Failed to create {self._exchange_name} exchange: {e}") from e
    
    def _resolve_exchange_id(self, exchange_name: str, params: Dict[str, Any]) -> str:
        """Resolve the CCXT exchange ID from exchange name and parameters."""
        # Import here to avoid circular import (exchanges → broker → exchange_manager)
        from .exchanges import EXCHANGE_ALIASES
        
        exchange_id = exchange_name.lower()
        if params.get("enable_mm", False):
            exchange_id = f"{exchange_id}.mm"

        exchange_id = EXCHANGE_ALIASES.get(exchange_id, exchange_id)

        if exchange_id not in cxp.exchanges:
            raise ValueError(f"Exchange {exchange_name} is not supported by ccxt.")
            
        return exchange_id
    
    def _extract_api_credentials(self, api_key: Optional[str], secret: Optional[str], params: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Extract API credentials from various parameter formats."""
        # Try to get API key from different parameter names
        if api_key is None:
            api_key = (params.pop("apiKey", None) or 
                      params.pop("key", None) or 
                      params.get("API_KEY"))
        
        # Try to get secret from different parameter names
        if secret is None:
            secret = (params.pop("secret", None) or 
                     params.pop("apiSecret", None) or 
                     params.get("API_SECRET") or 
                     params.get("SECRET"))
        
        return api_key, secret
    
    def _build_exchange_options(self, exchange_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build options dictionary for CCXT exchange instantiation."""
        options: Dict[str, Any] = {"name": exchange_name}
        
        # Handle asyncio loop configuration
        loop = params.get('loop')
        if loop is not None:
            options["asyncio_loop"] = loop
        else:
            loop = asyncio.new_event_loop()
            thread = Thread(target=loop.run_forever, daemon=True)
            thread.start()
            options["thread_asyncio_loop"] = thread  # type: ignore[assignment]
            options["asyncio_loop"] = loop  # type: ignore[assignment]

        # Add API credentials if available
        api_key, secret = self._extract_api_credentials(
            params.get('api_key'), 
            params.get('secret'), 
            params
        )
        if api_key and secret:
            options["apiKey"] = api_key
            options["secret"] = secret
            
        return options
    
    def _instantiate_ccxt_exchange(self, exchange_id: str, options: Dict[str, Any], params: Dict[str, Any]) -> cxp.Exchange:
        """Create the actual CCXT exchange instance."""
        # Filter out our custom parameters
        filtered_kwargs = {k: v for k, v in params.items() if k not in FILTERED_PARAMS}
        
        # Create the exchange instance
        return getattr(cxp, exchange_id)(options | filtered_kwargs)
    
    def _configure_exchange(self, exchange: cxp.Exchange, params: Dict[str, Any]) -> None:
        """Apply post-creation configuration to the exchange."""
        api_key = params.get('api_key')
        secret = params.get('secret')
        
        # Special handling for Hyperliquid
        if exchange.name and exchange.name.startswith("HYPERLIQUID") and api_key and secret:
            exchange.walletAddress = api_key
            exchange.privateKey = secret

        # Set sandbox mode if requested
        if params.get('use_testnet', False):
            exchange.set_sandbox_mode(True)

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
        """Check for stalls and trigger self-recreation if needed."""
        current_time = time.time()
        stalled_types = []
        
        with self._data_lock:
            for event_type, last_data_time in self._last_data_times.items():
                time_since_data = current_time - last_data_time
                if time_since_data > self._stall_threshold:
                    stalled_types.append((event_type, time_since_data))
                    
        if not stalled_types:
            return  # No stalls detected
            
        # Self-trigger recreation
        stall_info = ", ".join([f"{event_type}({int(time_since)}s)" for event_type, time_since in stalled_types])
        logger.error(f"Data stalls detected in {self._exchange_name}: {stall_info}")
        
        try:
            logger.info(f"Self-triggering recreation for {self._exchange_name} due to stalls...")
            if self.force_recreation():
                logger.info(f"Stall-triggered recreation successful for {self._exchange_name}")
                # Reset tracking times since exchange was recreated
                with self._data_lock:
                    self._last_data_times.clear()
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
