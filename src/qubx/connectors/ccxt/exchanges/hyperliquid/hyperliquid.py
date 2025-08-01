from typing import Any, Dict, List, Optional

import ccxt.pro as cxp
from qubx import logger

from ...adapters.polling_adapter import PollingToWebSocketAdapter


class HyperliquidEnhanced(cxp.hyperliquid):
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
            self.safe_integer(ohlcv, "t"),  # timestamp
            self.safe_number(ohlcv, "o"),  # open
            self.safe_number(ohlcv, "h"),  # high
            self.safe_number(ohlcv, "l"),  # low
            self.safe_number(ohlcv, "c"),  # close
            self.safe_number(ohlcv, "v"),  # volume (base)
            0.0,  # volume_quote (not provided by Hyperliquid)
            float(self.safe_integer(ohlcv, "n") or 0),  # trade_count
            0.0,  # bought_volume (not provided by Hyperliquid)
            0.0,  # bought_volume_quote (not provided by Hyperliquid)
        ]

    async def watch_funding_rates(
        self, symbols: Optional[List[str]] = None, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Watch funding rates using polling adapter (CCXT awaitable pattern).

        Args:
            symbols: List of symbols to watch (or None for all available)
            params: Additional parameters

        Returns:
            Funding rate data for the next update
        """
        if params is None:
            params = {}

        # Load markets if not already loaded
        await self.load_markets()

        # Get polling interval from params - default to 5 minutes for funding rates
        poll_interval_minutes = params.get("poll_interval_minutes", 5)
        poll_interval_seconds = poll_interval_minutes * 60

        # Create adapter if it doesn't exist or if symbols changed
        if self._funding_rate_adapter is None:
            logger.debug(f"Starting funding rate adapter for {len(symbols or [])} symbols, poll interval: {poll_interval_minutes}min")
            
            self._funding_rate_adapter = PollingToWebSocketAdapter(
                fetch_method=self.fetch_funding_rates,
                poll_interval_seconds=poll_interval_seconds,
                symbols=symbols or [],
                params=params or {},
            )
            
            # Start the background polling
            await self._funding_rate_adapter.start_watching()
        else:
            # Update symbols if needed
            if symbols is not None:
                await self._funding_rate_adapter.update_symbols(symbols)

        # Get next data from the adapter (CCXT awaitable pattern)
        funding_data = await self._funding_rate_adapter.get_next_data()
        
        # Transform Hyperliquid format to match ccxt_convert_funding_rate expectations
        transformed_data = {}
        
        if isinstance(funding_data, dict):
            for symbol, rate_info in funding_data.items():
                if isinstance(rate_info, dict):
                    # Fix the format issues for ccxt_convert_funding_rate
                    transformed_info = rate_info.copy()
                    
                    # Fix timestamp: use fundingTimestamp if timestamp is None
                    if transformed_info.get('timestamp') is None:
                        transformed_info['timestamp'] = transformed_info.get('fundingTimestamp')
                    
                    # Fix nextFundingTime: use nextFundingTimestamp if available
                    if 'nextFundingTimestamp' in transformed_info:
                        transformed_info['nextFundingTime'] = transformed_info['nextFundingTimestamp']
                    elif 'nextFundingTime' not in transformed_info:
                        # Calculate next funding time if not available (1 hour from current funding time)
                        current_funding = transformed_info.get('fundingTimestamp')
                        if current_funding:
                            transformed_info['nextFundingTime'] = current_funding + (60 * 60 * 1000)  # +1 hour in ms
                    
                    transformed_data[symbol] = transformed_info
                else:
                    transformed_data[symbol] = rate_info
        else:
            transformed_data = funding_data
        
        return transformed_data

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
                logger.debug(f"Removed funding rate subscription for {len(symbols)} symbols")

                # If no symbols left, cleanup adapter
                if not self._funding_rate_adapter.is_watching():
                    await self._funding_rate_adapter.stop()
                    self._funding_rate_adapter = None
                    logger.debug("Stopped funding rate adapter (no symbols left)")
            else:
                # Stop entire adapter
                await self._funding_rate_adapter.stop()
                self._funding_rate_adapter = None
                logger.debug("Stopped funding rate subscription")


class Hyperliquid(HyperliquidEnhanced):
    """
    Enhanced Hyperliquid exchange (spot markets) with extended OHLCV parsing
    """

    pass


class HyperliquidF(HyperliquidEnhanced):
    """
    Enhanced Hyperliquid futures exchange with extended OHLCV parsing

    This class provides the enhanced OHLCV parsing capabilities for Hyperliquid futures markets.
    """

    pass
