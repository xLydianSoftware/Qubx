"""Trade handler for Lighter WebSocket messages"""

from typing import Any

from qubx.core.series import Trade

from .base import BaseHandler


class TradesHandler(BaseHandler[Trade]):
    """
    Handler for Lighter trade messages.

    Converts Lighter trade format to Qubx Trade objects.

    Lighter format:
    ```json
    {
      "channel": "trade:0",
      "type": "update/trade",
      "timestamp": ...,
      "trades": [{
        "trade_id": 212746654,
        "market_id": 1,
        "size": "0.00100",
        "price": "121043.9",
        "is_maker_ask": true,
        "timestamp": 1760041996404,
        ...
      }],
      "liquidation_trades": [...]  // Optional
    }
    ```

    Trade side logic:
    - is_maker_ask=true: maker selling (ask), taker buying -> side=1 (BUY)
    - is_maker_ask=false: maker buying (bid), taker selling -> side=-1 (SELL)
    """

    # Trade side constants (matching Qubx convention)
    SIDE_BUY = 1
    SIDE_SELL = -1

    def __init__(self, market_id: int):
        """
        Initialize trades handler.

        Args:
            market_id: Lighter market ID to handle
        """
        super().__init__()
        self.market_id = market_id

    def can_handle(self, message: dict[str, Any]) -> bool:
        """Check if message is trade for this market"""
        channel = message.get("channel", "")
        msg_type = message.get("type", "")

        # Check if it's a trade message for our market
        expected_channel = f"trade:{self.market_id}"
        is_trade_msg = msg_type in ["subscribed/trade", "update/trade"]

        return channel == expected_channel and is_trade_msg

    def _handle_impl(self, message: dict[str, Any]) -> list[Trade] | None:
        """
        Convert Lighter trade message to Qubx Trade objects.

        Args:
            message: Raw Lighter trade message

        Returns:
            List of Trade objects, or None if no trades in message

        Raises:
            ValueError: If message format is invalid
        """
        trades_list = []

        # Parse regular trades
        regular_trades = message.get("trades")
        if regular_trades:
            for trade_data in regular_trades:
                trade = self._parse_trade(trade_data)
                if trade:
                    trades_list.append(trade)

        # Parse liquidation trades
        liquidation_trades = message.get("liquidation_trades")
        if liquidation_trades:
            for trade_data in liquidation_trades:
                trade = self._parse_trade(trade_data)
                if trade:
                    trades_list.append(trade)

        return trades_list if trades_list else None

    def _parse_trade(self, trade_data: dict[str, Any]) -> Trade | None:
        """
        Parse a single Lighter trade to Qubx Trade.

        Args:
            trade_data: Single trade dict from Lighter

        Returns:
            Trade object, or None if invalid
        """
        try:
            # Extract fields
            trade_id = trade_data.get("trade_id")
            if trade_id is None:
                return None

            timestamp_ms = trade_data.get("timestamp")
            if timestamp_ms is None:
                return None

            price_str = trade_data.get("price")
            size_str = trade_data.get("size")
            is_maker_ask = trade_data.get("is_maker_ask")

            if price_str is None or size_str is None or is_maker_ask is None:
                return None

            # Convert types
            time_ns = int(timestamp_ms * 1_000_000)  # ms -> ns
            price = float(price_str)
            size = float(size_str)

            # Determine side from is_maker_ask
            # is_maker_ask=true: maker was on ask side, taker bought -> BUY
            # is_maker_ask=false: maker was on bid side, taker sold -> SELL
            side = self.SIDE_BUY if is_maker_ask else self.SIDE_SELL

            # Create Trade object
            return Trade(
                time=time_ns,
                price=price,
                size=size,
                side=side,
                trade_id=int(trade_id),
            )

        except (ValueError, TypeError):
            # Log error but don't raise - skip malformed trades
            return None
