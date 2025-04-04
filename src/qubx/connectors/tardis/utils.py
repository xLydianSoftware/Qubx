"""Utility functions for Tardis message conversion."""

import json
from typing import Any, Dict, Optional

import numpy as np

from qubx import logger
from qubx.core.basics import Instrument, dt_64
from qubx.core.series import OrderBook, Quote, Trade
from qubx.core.utils import recognize_time
from qubx.utils.orderbook import accumulate_orderbook_levels


def tardis_convert_trade(message: Dict[str, Any], instrument: Instrument) -> Optional[Trade]:
    """
    Convert a Tardis trade message to a QubX Trade.

    Args:
        message: Tardis trade message.
        instrument: The instrument the trade is for.

    Returns:
        Trade object or None if conversion fails.
    """
    try:
        # For single trade messages
        if "type" in message and message["type"] == "trade":
            # Parse timestamp using recognize_time
            timestamp = recognize_time(message["timestamp"])
            timestamp_ns = int(timestamp)

            side = 1 if message["side"].lower() == "buy" else -1
            return Trade(time=timestamp_ns, price=float(message["price"]), size=float(message["amount"]), side=side)

        # For trade message collections
        if not message.get("trades"):
            return None

        # Get the latest trade
        trade_data = message["trades"][-1]

        # Parse timestamp using recognize_time
        timestamp = recognize_time(trade_data["timestamp"])
        timestamp_ns = int(timestamp)

        # Determine side (1 for buy, -1 for sell)
        side = 1 if trade_data.get("side", "").lower() == "buy" else -1

        # Handle different field names (size vs amount)
        size = float(trade_data.get("size", trade_data.get("amount", 0)))

        return Trade(time=timestamp_ns, price=float(trade_data["price"]), size=size, side=side)
    except Exception as e:
        logger.error(f"Failed to convert Tardis trade message: {e}")
        return None


def tardis_convert_orderbook(
    message: Dict[str, Any],
    instr: Instrument,
    levels: int = 50,
    tick_size_pct: float = 0.01,
    sizes_in_quoted: bool = False,
    current_timestamp: Optional[dt_64] = None,
) -> Optional[OrderBook]:
    """
    Convert a Tardis orderbook snapshot to a QubX OrderBook with a fixed tick size.

    Parameters:
        message: Tardis orderbook message.
        instr: The instrument object containing market-specific details.
        levels: The number of levels to include in the order book. Default is 50.
        tick_size_pct: The tick size percentage. Default is 0.01%.
        sizes_in_quoted: Whether the size is in the quoted currency. Default is False.
        current_timestamp: Optional timestamp to use if message timestamp is None.

    Returns:
        OrderBook object or None if conversion fails.
    """
    try:
        # Get timestamp - use recognize_time for parsing
        if message.get("timestamp") is None:
            if current_timestamp is None:
                return None
            timestamp_ns = int(current_timestamp)
        else:
            # Use recognize_time which handles both string and unix timestamps
            timestamp = recognize_time(message["timestamp"])
            timestamp_ns = int(timestamp)

        # Extract bids and asks
        bids = []
        asks = []

        # Handle different orderbook message formats
        if "bids" in message and isinstance(message["bids"], list):
            if len(message["bids"]) > 0:
                # Check if bids/asks are dictionary format {"price": x, "amount": y} or array format [price, amount]
                if isinstance(message["bids"][0], dict):
                    # Dict format: [{"price": x, "amount": y}, ...]
                    bids = [
                        (float(b["price"]), float(b["amount"]))
                        for b in message["bids"]
                        if "price" in b and "amount" in b
                    ]
                    asks = [
                        (float(a["price"]), float(a["amount"]))
                        for a in message["asks"]
                        if "price" in a and "amount" in a
                    ]
                else:
                    # Array format: [[price, amount], ...]
                    bids = [(float(b[0]), float(b[1])) for b in message["bids"] if len(b) == 2]
                    asks = [(float(a[0]), float(a[1])) for a in message["asks"] if len(a) == 2]

        # Convert to numpy arrays
        raw_bids = np.array(bids, dtype=np.float64)
        raw_asks = np.array(asks, dtype=np.float64)

        if len(raw_bids) == 0 or len(raw_asks) == 0:
            return None

        # Determine tick size
        if tick_size_pct == 0:
            tick_size = instr.tick_size
        else:
            # Calculate mid price from the top of the book
            top_bid = raw_bids[0][0] if len(raw_bids) > 0 else 0
            top_ask = raw_asks[0][0] if len(raw_asks) > 0 else 0

            if top_bid == 0 or top_ask == 0:
                mid_price = top_bid or top_ask
            else:
                mid_price = (top_bid + top_ask) / 2

            # Calculate tick size as percentage of mid price
            tick_size = max(mid_price * tick_size_pct / 100, instr.tick_size)

        # Pre-allocate buffers for bids and asks
        bids_buffer = np.zeros(levels, dtype=np.float64)
        asks_buffer = np.zeros(levels, dtype=np.float64)

        # Accumulate bids and asks into the buffers
        top_bid, bids = accumulate_orderbook_levels(raw_bids, bids_buffer, tick_size, True, levels, sizes_in_quoted)
        top_ask, asks = accumulate_orderbook_levels(raw_asks, asks_buffer, tick_size, False, levels, sizes_in_quoted)

        # Create and return the OrderBook object
        return OrderBook(
            time=timestamp_ns,
            top_bid=top_bid,
            top_ask=top_ask,
            tick_size=tick_size,
            bids=bids,
            asks=asks,
        )
    except Exception as e:
        logger.error(f"Failed to convert Tardis orderbook message: {e}")
        logger.error(f"Message: {message}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def tardis_convert_quote(message: Dict[str, Any], instrument: Instrument) -> Optional[Quote]:
    """
    Convert a Tardis quote message to a QubX Quote.

    Args:
        message: Tardis quote message.
        instrument: The instrument the quote is for.

    Returns:
        Quote object or None if conversion fails.
    """
    try:
        # Handle book_snapshot with depth=1 format (quote messages)
        if message.get("type") == "book_snapshot" and message.get("depth") == 1:
            # Parse timestamp using recognize_time
            timestamp = recognize_time(message["timestamp"])
            timestamp_ns = int(timestamp)

            # Extract bid and ask from the first level
            bid_price = float(message["bids"][0]["price"]) if message.get("bids") and len(message["bids"]) > 0 else 0.0
            ask_price = float(message["asks"][0]["price"]) if message.get("asks") and len(message["asks"]) > 0 else 0.0

            bid_size = float(message["bids"][0]["amount"]) if message.get("bids") and len(message["bids"]) > 0 else 0.0
            ask_size = float(message["asks"][0]["amount"]) if message.get("asks") and len(message["asks"]) > 0 else 0.0

            return Quote(
                time=timestamp_ns,
                bid=bid_price,
                ask=ask_price,
                bid_size=bid_size,
                ask_size=ask_size,
            )

        # Original format with best_bid_price/best_ask_price
        if "best_bid_price" not in message or "best_ask_price" not in message:
            return None

        # Parse timestamp using recognize_time
        timestamp = recognize_time(message["timestamp"])
        timestamp_ns = int(timestamp)

        return Quote(
            time=timestamp_ns,
            bid=float(message["best_bid_price"]),
            ask=float(message["best_ask_price"]),
            bid_size=float(message.get("best_bid_size", 0.0)),
            ask_size=float(message.get("best_ask_size", 0.0)),
        )
    except Exception as e:
        logger.error(f"Failed to convert Tardis quote message: {e}")
        return None


def tardis_extract_timeframe(bar_name: str) -> str:
    """
    Extract timeframe from trade bar name.

    Args:
        bar_name: Name of the trade bar, e.g. "trade_bars_1m"

    Returns:
        Timeframe string (e.g. "1m")
    """
    # Extract the timeframe part (e.g. "1m" from "trade_bars_1m")
    if "trade_bars_" in bar_name:
        return bar_name.split("trade_bars_")[1]
    return ""


def tardis_parse_message(message_str: str) -> Dict[str, Any]:
    """
    Parse a Tardis message from JSON string.

    Args:
        message_str: JSON string representing a Tardis message.

    Returns:
        Parsed message as a dictionary.
    """
    try:
        return json.loads(message_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Tardis message: {e}")
        return {}
