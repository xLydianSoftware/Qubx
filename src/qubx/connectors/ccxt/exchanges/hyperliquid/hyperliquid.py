from typing import List

import ccxt
import ccxt.pro as cxp
from qubx import logger


class HyperliquidEnhanced:
    """
    Mixin class to enhance Hyperliquid OHLCV parsing with trade count data
    """
    
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


class Hyperliquid(HyperliquidEnhanced, ccxt.hyperliquid):
    """
    Enhanced Hyperliquid exchange (spot markets) with extended OHLCV parsing
    """
    pass


class HyperliquidF(HyperliquidEnhanced, cxp.hyperliquid):
    """
    Extended Hyperliquid exchange to provide watchOHLCVForSymbols support and enhanced OHLCV parsing
    
    Since Hyperliquid's CCXT implementation doesn't natively support watchOHLCVForSymbols,
    this override implements it by delegating to individual watchOHLCV calls.
    
    Also extends OHLCV parsing to include trade count data that Hyperliquid provides.
    """

    def describe(self):
        """
        Override to indicate support for watchOHLCVForSymbols
        """
        return self.deep_extend(
            super().describe(),
            {
                "has": {
                    "watchOHLCVForSymbols": False,  # Disable bulk watching to test individual streams
                },
            },
        )

    async def watch_ohlcv_for_symbols(self, symbols_and_timeframes: List[List[str]], since=None, limit=None, params={}):
        """
        Watches historical candlestick data for multiple symbols
        
        This implementation calls individual watchOHLCV for each symbol/timeframe and uses
        the existing CCXT caching mechanism. Each call returns the cached data for that symbol.
        
        :param list symbols_and_timeframes: list of [symbol, timeframe] pairs to watch
        :param int [since]: timestamp in ms of the earliest candle to fetch
        :param int [limit]: the maximum amount of candles to fetch
        :param dict params: extra parameters specific to the hyperliquid api endpoint
        :returns dict: OHLCV data in format {symbol: {timeframe: candles}}
        """
        await self.load_markets()
        
        result = {}
        
        for symbol_timeframe in symbols_and_timeframes:
            if len(symbol_timeframe) != 2:
                continue
            symbol, timeframe = symbol_timeframe
            
            # Validate symbol exists
            if symbol not in self.markets:
                logger.warning(f"Symbol {symbol} not found in markets")
                continue
                
            # Validate timeframe is supported  
            if timeframe not in self.timeframes:
                logger.warning(f"Timeframe {timeframe} not supported for {symbol}")
                continue
            
            try:
                # Call individual watchOHLCV - this maintains persistent connections
                # and returns immediately with cached data if available
                ohlcv_data = await self.watch_ohlcv(symbol, timeframe, since, limit, params)
                
                # Build result in expected format
                if symbol not in result:
                    result[symbol] = {}
                result[symbol][timeframe] = ohlcv_data
                
            except Exception as e:
                logger.error(f"Error watching OHLCV for {symbol}:{timeframe} - {e}")
                continue
        
        return result

    async def un_watch_ohlcv_for_symbols(self, symbols_and_timeframes: List[List[str]], params={}):
        """
        Unwatches historical candlestick data for multiple symbols
        :param list symbols_and_timeframes: list of [symbol, timeframe] pairs to unwatch
        :param dict params: extra parameters specific to the hyperliquid api endpoint
        """
        for symbol_timeframe in symbols_and_timeframes:
            if len(symbol_timeframe) != 2:
                continue
            symbol, timeframe = symbol_timeframe
            
            try:
                await self.un_watch_ohlcv(symbol, timeframe, params)
            except Exception as e:
                logger.error(f"Error unwatching {symbol}:{timeframe} - {e}")