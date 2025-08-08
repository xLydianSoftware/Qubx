import math
from typing import Any, Dict, List, Optional

import ccxt.pro as cxp
from qubx import logger

from ...adapters.polling_adapter import PollingConfig, PollingToWebSocketAdapter
from ..base import CcxtFuturePatchMixin

# Constants
FUNDING_RATE_DEFAULT_POLL_MINUTES = 5
FUNDING_RATE_HOUR_MS = 60 * 60 * 1000  # 1 hour in milliseconds


class HyperliquidEnhanced(CcxtFuturePatchMixin, cxp.hyperliquid):
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
        poll_interval_minutes = params.get("poll_interval_minutes", FUNDING_RATE_DEFAULT_POLL_MINUTES)
        poll_interval_seconds = poll_interval_minutes * 60

        # Create adapter if it doesn't exist or if symbols changed
        if self._funding_rate_adapter is None:
            logger.debug(f"Starting funding rate adapter, poll interval: {poll_interval_minutes}min")

            self._funding_rate_adapter = PollingToWebSocketAdapter(
                fetch_method=self.fetch_funding_rates,
                symbols=symbols or [],
                params=params or {},
                config=PollingConfig(poll_interval_seconds=poll_interval_seconds),
            )
        else:
            # Update symbols - will automatically trigger immediate poll if symbols changed
            if symbols is not None:
                await self._funding_rate_adapter.update_symbols(symbols)

        # Get next data from the adapter (CCXT awaitable pattern)
        funding_data = await self._funding_rate_adapter.get_next_data()

        # Transform Hyperliquid format to match ccxt_convert_funding_rate expectations
        transformed_data = {}
        current_time_ms = self.milliseconds()  # Get current time for unique timestamps
        symbol_index = 0  # Counter to ensure unique timestamps per symbol

        if isinstance(funding_data, dict):
            for symbol, rate_info in funding_data.items():
                if isinstance(rate_info, dict):
                    # Fix the format issues for ccxt_convert_funding_rate
                    transformed_info = rate_info.copy()

                    # Fix timestamp: use current time + small increment per symbol for uniqueness
                    # This provides unique timestamps per symbol and represents when we received the data
                    transformed_info["timestamp"] = current_time_ms + symbol_index

                    # Fix nextFundingTime: fundingTimestamp is already the next funding time
                    funding_timestamp = transformed_info.get("fundingTimestamp")
                    if funding_timestamp:
                        # fundingTimestamp from CCXT is already the next funding boundary
                        transformed_info["nextFundingTime"] = funding_timestamp
                    else:
                        # Fallback: calculate next hour boundary from current time
                        next_hour_ms = (math.floor(current_time_ms / FUNDING_RATE_HOUR_MS) + 1) * FUNDING_RATE_HOUR_MS
                        transformed_info["nextFundingTime"] = next_hour_ms

                    transformed_data[symbol] = transformed_info
                    symbol_index += 1  # Increment for next symbol
                else:
                    transformed_data[symbol] = rate_info
        else:
            transformed_data = funding_data

        return transformed_data

    def parse_order(self, order, market=None):
        """
        Override parse_order to handle HyperLiquid-specific order format.

        HyperLiquid WebSocket format:
        {
          "order": {
            "coin": "DOGE",
            "side": "B",
            "limitPx": "0.19",  // Present for limit orders
            "sz": "100.0",
            "oid": 123456,
            "timestamp": 1234567890,
            "origSz": "100.0",
            "cloid": "client_id"
          },
          "status": "open",
          "statusTimestamp": 1234567890
        }
        """
        # Call parent parser first
        parsed = super().parse_order(order, market)

        # Get HyperLiquid-specific order info
        info = order.get("info", {})
        if isinstance(info, dict):
            order_info = info.get("order", {})

            # Fix missing symbol first - this is critical for watch_orders
            if parsed.get("symbol") is None:
                # Try to construct symbol from coin
                coin = order_info.get("coin") if isinstance(order_info, dict) else None
                if coin:
                    # HyperLiquid uses USDC as quote currency
                    parsed["symbol"] = f"{coin}/USDC:USDC"

            # Fix order type based on HyperLiquid format
            if isinstance(order_info, dict):
                # If limitPx is present and not empty, it's a limit order
                limit_px = order_info.get("limitPx")
                if limit_px and limit_px != "" and limit_px != "0":
                    parsed["type"] = "limit"
                    # Ensure price is set from limitPx if not already set
                    if parsed.get("price") is None:
                        try:
                            parsed["price"] = float(limit_px)
                        except (ValueError, TypeError):
                            pass
                else:
                    parsed["type"] = "market"

                # Fix side mapping: B -> buy, S -> sell
                hl_side = order_info.get("side")
                if hl_side == "B":
                    parsed["side"] = "buy"
                elif hl_side == "S":
                    parsed["side"] = "sell"
                # Ensure side is never None/empty
                if not parsed.get("side"):
                    parsed["side"] = "buy" if hl_side == "B" else ("sell" if hl_side == "S" else "unknown")

                # Fix amount from sz if not set
                if parsed.get("amount") is None:
                    sz = order_info.get("sz") or order_info.get("origSz")
                    if sz:
                        try:
                            parsed["amount"] = float(sz)
                        except (ValueError, TypeError):
                            pass

                # Fix client order ID
                cloid = order_info.get("cloid")
                if cloid and parsed.get("clientOrderId") is None:
                    parsed["clientOrderId"] = cloid

            # Fix status from HyperLiquid status field
            hl_status = info.get("status")
            if hl_status and parsed.get("status") in [None, "open"]:
                # Map HyperLiquid status to standard CCXT status
                status_mapping = {
                    "open": "open",
                    "filled": "closed",
                    "canceled": "canceled",
                    "cancelled": "canceled",
                    "partial": "open",
                }
                mapped_status = status_mapping.get(hl_status.lower(), hl_status)
                parsed["status"] = mapped_status

        # Fallback if type is still None
        if parsed.get("type") is None:
            parsed["type"] = "limit"  # Default to limit for HyperLiquid

        return parsed

    def parse_trade(self, trade, market=None):
        """
        Override parse_trade to handle HyperLiquid-specific trade format.

        HyperLiquid WebSocket trade format:
        {
          "coin": "DOGE",
          "side": "B",
          "px": "0.19",
          "sz": "100.0",
          "hash": "0x...",
          "time": 1234567890,
          "tid": 123456,
          "users": ["user1", "user2"],
          "crossed": true,
          "fee": "0.001"
        }
        """
        # Call parent parser first
        parsed = super().parse_trade(trade, market)

        # Fix side mapping: B -> buy, S -> sell
        hl_side = trade.get("side")
        if hl_side == "B":
            parsed["side"] = "buy"
        elif hl_side == "S":
            parsed["side"] = "sell"

        # Fix price from px if not set
        if parsed.get("price") is None:
            px = trade.get("px")
            if px:
                try:
                    parsed["price"] = float(px)
                except (ValueError, TypeError):
                    pass

        # Fix amount from sz if not set
        if parsed.get("amount") is None:
            sz = trade.get("sz")
            if sz:
                try:
                    parsed["amount"] = float(sz)
                except (ValueError, TypeError):
                    pass

        # Fix timestamp from time if not set
        if parsed.get("timestamp") is None:
            time_val = trade.get("time")
            if time_val:
                try:
                    parsed["timestamp"] = int(time_val)
                except (ValueError, TypeError):
                    pass

        # Fix trade ID from tid if not set
        if parsed.get("id") is None:
            tid = trade.get("tid")
            if tid:
                parsed["id"] = str(tid)

        # Fix fee information if available
        fee_val = trade.get("fee")
        if fee_val and parsed.get("fee") is None:
            try:
                parsed["fee"] = {
                    "cost": float(fee_val),
                    "currency": None,  # HyperLiquid doesn't specify fee currency in trade
                }
            except (ValueError, TypeError):
                pass

        return parsed

    def safe_parse_order_with_info_fix(self, raw_order):
        """
        Apply HyperLiquid-specific fixes to raw order data before standard parsing.
        This ensures that ccxt_convert_order_info gets properly formatted data.
        """
        if not isinstance(raw_order, dict):
            return raw_order

        # Create a working copy
        order = raw_order.copy()
        info = order.get("info", {})

        if isinstance(info, dict):
            order_info = info.get("order", {})

            # Fix missing symbol from coin field
            if not order.get("symbol") and isinstance(order_info, dict):
                coin = order_info.get("coin")
                if coin:
                    order["symbol"] = f"{coin}/USDC:USDC"

            # Fix order type from limitPx
            if isinstance(order_info, dict):
                limit_px = order_info.get("limitPx")
                if limit_px and limit_px != "" and limit_px != "0":
                    order["type"] = "limit"
                elif order.get("type") is None:
                    order["type"] = "market"

            # Fix side mapping B/S -> buy/sell
            if isinstance(order_info, dict):
                hl_side = order_info.get("side")
                if hl_side == "B":
                    order["side"] = "buy"
                elif hl_side == "S":
                    order["side"] = "sell"

            # Fix amount from sz/origSz if missing
            if not order.get("amount") and isinstance(order_info, dict):
                sz = order_info.get("sz") or order_info.get("origSz")
                if sz:
                    try:
                        order["amount"] = float(sz)
                    except (ValueError, TypeError):
                        pass

        return order

    async def watch_orders(self, symbol=None, since=None, limit=None, params={}):
        """
        Override watch_orders to fix HyperLiquid order data before it gets processed.
        """
        # Call parent watch_orders first
        orders = await super().watch_orders(symbol, since, limit, params)

        # Fix each order using our HyperLiquid-specific parsing
        if isinstance(orders, list):
            for i in range(len(orders)):
                orders[i] = self.safe_parse_order_with_info_fix(orders[i])

        return orders

    async def create_orders(self, orders, params={}):
        """
        Override create_orders to properly handle HyperLiquid's order creation response.

        HyperLiquid returns minimal status information, so we need to reconstruct
        the full order details from the original request parameters.
        """
        # Store original order requests for reconstruction
        original_orders = []
        for order_request in orders:
            original_orders.append(
                {
                    "symbol": order_request.get("symbol"),
                    "type": order_request.get("type"),
                    "side": order_request.get("side"),
                    "amount": order_request.get("amount"),
                    "price": order_request.get("price"),
                    "params": order_request.get("params", {}),
                }
            )

        # Call parent create_orders to get the raw response
        response_orders = await super().create_orders(orders, params)

        # Reconstruct full order information by combining response with original request
        reconstructed_orders = []
        for i, response_order in enumerate(response_orders):
            if i < len(original_orders):
                original = original_orders[i]

                # Ensure response_order is a dictionary
                if isinstance(response_order, dict):
                    response_dict = response_order
                else:
                    response_dict = {}

                # Create a complete order object by merging response data with original request
                reconstructed_order = response_dict.copy()  # Start with response data

                # Override/add fields from original request
                reconstructed_order.update(
                    {
                        "symbol": original["symbol"],  # From original request
                        "type": original["type"].lower() if original["type"] else None,  # From original request
                        "side": original["side"].lower() if original["side"] else None,  # From original request
                        "price": float(original["price"]) if original["price"] else None,  # From original request
                        "amount": float(original["amount"]) if original["amount"] else None,  # From original request
                    }
                )

                reconstructed_orders.append(reconstructed_order)
            else:
                reconstructed_orders.append(response_order)

        return reconstructed_orders

    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        """
        Override create_order to ensure it uses our enhanced create_orders method.

        This ensures that both direct create_order calls and create_orders calls
        go through the same reconstruction logic.
        """
        # Convert single order to list format for create_orders
        order_request = {
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "params": params,
        }

        # Use our enhanced create_orders method
        orders = await self.create_orders([order_request], {})

        # Return the first (and only) order
        return orders[0] if orders else None

    async def create_order_ws(self, symbol, type, side, amount, price=None, params={}):
        """
        Override create_order_ws to fix the direct return from broker calls.

        This is the method called by the broker when enable_create_order_ws=True.
        The result is returned directly to ctx.trade(), so we need to fix it here.
        """
        # Call parent create_order_ws to get the raw response
        response_order = await super().create_order_ws(symbol, type, side, amount, price, params)

        # If response is None or not a dict, return as-is
        if not isinstance(response_order, dict):
            return response_order

        # Ensure symbol is set correctly
        if not response_order.get("symbol"):
            response_order["symbol"] = symbol

        # Ensure type is set correctly (from request parameters)
        if not response_order.get("type") or response_order.get("type") == "market":
            # Determine type based on whether price was provided
            if price is not None and price != 0:
                response_order["type"] = "limit"
            else:
                response_order["type"] = "market"

        # Ensure side is set correctly
        if not response_order.get("side"):
            response_order["side"] = side.lower()

        # Ensure amount is set correctly
        if not response_order.get("amount") or response_order.get("amount") == 0:
            response_order["amount"] = float(amount)

        # Ensure price is set correctly
        if not response_order.get("price") or response_order.get("price") == 0:
            if price is not None:
                response_order["price"] = float(price)

        return response_order

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
