import ccxt
import ccxt.pro as cxp


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
    Enhanced Hyperliquid futures exchange with extended OHLCV parsing
    
    This class provides the enhanced OHLCV parsing capabilities for Hyperliquid futures markets.
    """
    pass