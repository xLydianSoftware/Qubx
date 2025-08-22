"""
ExchangeManager: Transparent wrapper for CCXT exchanges with automatic recreation.

Provides seamless exchange recreation without affecting consuming components.
"""

import asyncio
import threading
import time
from typing import Any, Dict, Optional

import ccxt.pro as cxp
from qubx import logger


class ExchangeManager:
    """
    Transparent wrapper for CCXT Exchange that handles recreation internally.
    
    All method calls and property access are delegated to the underlying exchange.
    Recreation is triggered externally by BaseHealthMonitor stall detection.
    
    Key Features:
    - Transparent delegation of ALL CCXT methods/properties
    - External recreation triggering (from health monitor)
    - Circuit breaker protection with recreation limits
    - Atomic asyncio_loop transitions during recreation
    - Zero impact on consuming code
    """
    
    _exchange: cxp.Exchange  # Type hint that this is always a valid exchange

    def __init__(
        self,
        exchange_name: str,
        factory_params: Dict[str, Any],
        initial_exchange: Optional[cxp.Exchange] = None,
        max_recreations: int = 3,
        reset_interval_hours: float = 24.0,
    ):
        """Initialize ExchangeManager with underlying CCXT exchange.
        
        Args:
            exchange_name: Exchange name for factory (e.g., "binance.um")
            factory_params: Parameters for get_ccxt_exchange() 
            initial_exchange: Pre-created exchange instance (from factory)
            max_recreations: Maximum recreation attempts before giving up
            reset_interval_hours: Hours between recreation count resets
        """
        self._exchange_name = exchange_name
        self._factory_params = factory_params.copy()
        self._max_recreations = max_recreations
        self._reset_interval_hours = reset_interval_hours
        
        # Recreation state
        self._recreation_count = 0
        self._recreation_lock = threading.RLock()
        self._last_successful_reset = time.time()
        
        # Use provided exchange or create new one
        self._exchange = initial_exchange or self._create_exchange()

    def _create_exchange(self) -> cxp.Exchange:
        """Create new CCXT exchange instance."""
        try:
            # Avoid circular import by importing here
            from .factory import get_ccxt_exchange
            # Remove enable_stability_manager to avoid recursive wrapping
            params = self._factory_params.copy()
            params.pop('enable_stability_manager', None)
            exchange = get_ccxt_exchange(**params)
            logger.debug(f"Created new {self._exchange_name} exchange instance")
            return exchange
        except Exception as e:
            logger.error(f"Failed to create {self._exchange_name} exchange: {e}")
            raise RuntimeError(f"Failed to create {self._exchange_name} exchange: {e}") from e

    def force_recreation_on_stall(self) -> bool:
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
        
    def reset_recreation_count_if_needed(self):
        """Reset recreation count periodically (called by health monitor)."""
        reset_interval_seconds = self._reset_interval_hours * 60 * 60
        
        current_time = time.time()
        time_since_reset = current_time - self._last_successful_reset
        
        if time_since_reset >= reset_interval_seconds and self._recreation_count > 0:
            logger.info(f"Resetting recreation count for {self._exchange_name} (was {self._recreation_count})")
            self._recreation_count = 0
            self._last_successful_reset = current_time

    # === Transparent Delegation === 
    # All CCXT exchange methods/properties are delegated to underlying exchange
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all method calls and property access to underlying exchange."""
        return getattr(self._exchange, name)
    
    # Explicitly delegate critical properties for clarity and performance
    @property
    def name(self) -> str:
        return str(self._exchange.name)
        
    @property  
    def id(self) -> str:
        return str(self._exchange.id)
        
    @property
    def asyncio_loop(self):
        """Critical: AsyncThreadLoop depends on this property."""
        return self._exchange.asyncio_loop
        
    @property
    def sandbox(self) -> bool:
        return getattr(self._exchange, 'sandbox', False)
        
    # Delegate close method explicitly for proper cleanup
    async def close(self):
        """Close the underlying exchange."""
        if hasattr(self._exchange, 'close'):
            await self._exchange.close()
