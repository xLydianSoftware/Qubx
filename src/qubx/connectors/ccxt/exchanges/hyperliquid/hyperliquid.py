import ccxt
import ccxt.pro as cxp
from typing import Dict, List, Optional, Any

from qubx import logger
from ...adapters.polling_adapter import PollingToWebSocketAdapter


class HyperliquidEnhanced:
    """
    Mixin class to enhance Hyperliquid with OHLCV parsing and funding rate subscriptions
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._funding_rate_adapter: Optional[PollingToWebSocketAdapter] = None
    
    def parse_ohlcv(self, ohlcv, market=None):
        """
        Override parse_ohlcv to include trade count data from Hyperliquid API
        
        Hyperliquid API returns:
        - 't': timestamp (start)
        - 'o': open price  
        - 'h': high price
        - 'l': low price
        - 'c': close price
        - 'v': volume (base)
        - 'n': trade count (number of trades)
        
        Returns extended OHLCV format: [timestamp, open, high, low, close, volume, 0, trade_count, 0, 0]
        Fields: [timestamp, open, high, low, close, volume, volume_quote, trade_count, bought_volume, bought_volume_quote]
        """
        return [
            self.safe_integer(ohlcv, 't'),     # timestamp
            self.safe_number(ohlcv, 'o'),      # open
            self.safe_number(ohlcv, 'h'),      # high  
            self.safe_number(ohlcv, 'l'),      # low
            self.safe_number(ohlcv, 'c'),      # close
            self.safe_number(ohlcv, 'v'),      # volume (base)
            0.0,                               # volume_quote (not provided by Hyperliquid)
            float(self.safe_integer(ohlcv, 'n') or 0),  # trade_count 
            0.0,                               # bought_volume (not provided by Hyperliquid)
            0.0,                               # bought_volume_quote (not provided by Hyperliquid)
        ]

    async def watch_funding_rates(self, symbols: Optional[List[str]] = None, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Watch funding rates using polling adapter.
        
        Args:
            symbols: List of symbols to watch (or None for all available)
            params: Additional parameters
            
        Yields:
            Funding rate data as it becomes available
        """
        if params is None:
            params = {}
            
        logger.info(f"<green>hyperliquid</green> Starting funding rate subscription for symbols: {symbols}")
        
        # Create and configure the adapter
        self._funding_rate_adapter = PollingToWebSocketAdapter(
            fetch_method=getattr(self, 'fetch_funding_rates'),  # Get the method from parent class
            poll_interval_seconds=300,  # 5 minutes (Hyperliquid updates hourly)
            symbols=symbols or [],
            params=params or {},
            adapter_id="hyperliquid_funding_rates"
        )
        
        # Stream data from the adapter
        async for funding_data in self._funding_rate_adapter.start_watching():
            yield funding_data
            
    async def un_watch_funding_rates(self, symbols: Optional[List[str]] = None) -> None:
        """
        Unwatch funding rates.
        
        Args:
            symbols: Specific symbols to unwatch, or None to stop all
        """
        if self._funding_rate_adapter:
            if symbols:
                # Remove specific symbols
                await self._funding_rate_adapter.remove_symbols(symbols)
                logger.debug(f"<green>hyperliquid</green> Removed funding rate subscription for symbols: {symbols}")
                
                # If no symbols left, cleanup adapter
                if not self._funding_rate_adapter.is_watching():
                    await self._funding_rate_adapter.stop()
                    self._funding_rate_adapter = None
                    logger.debug("<green>hyperliquid</green> Stopped funding rate adapter (no symbols left)")
            else:
                # Stop entire adapter
                await self._funding_rate_adapter.stop()
                self._funding_rate_adapter = None
                logger.info("<green>hyperliquid</green> Stopped funding rate subscription")


class Hyperliquid(HyperliquidEnhanced, ccxt.hyperliquid):
    """
    Enhanced Hyperliquid exchange (spot markets) with extended OHLCV parsing
    """
    pass


class HyperliquidF(HyperliquidEnhanced, cxp.hyperliquid):
    """
    Enhanced Hyperliquid futures exchange with extended OHLCV parsing
    
    This class provides the enhanced OHLCV parsing capabilities for Hyperliquid futures markets.
    """
    pass