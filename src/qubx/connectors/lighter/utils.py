"""Utility functions for Lighter connector"""

from typing import Any

import numpy as np

from qubx.core.basics import Instrument, Order, OrderSide, OrderStatus, dt_64
from qubx.core.series import OrderBook, Quote, Trade

from .constants import USDC_SCALE, LighterOrderSide


def lighter_symbol_to_qubx(symbol: str) -> str:
    """
    Convert Lighter symbol format to Qubx format.

    Lighter uses: BTC-USDC
    Qubx uses: BTC/USDC:USDC (for perpetual swaps)

    Args:
        symbol: Lighter symbol (e.g., "BTC-USDC")

    Returns:
        Qubx symbol format
    """
    # Replace hyphen with slash, add settle currency
    base, quote = symbol.split("-")
    return f"{base}/{quote}:{quote}"


def qubx_symbol_to_lighter(symbol: str) -> str:
    """
    Convert Qubx symbol format to Lighter format.

    Args:
        symbol: Qubx symbol (e.g., "BTC/USDC:USDC")

    Returns:
        Lighter symbol format (e.g., "BTC-USDC")
    """
    # Extract base and quote
    if ":" in symbol:
        symbol = symbol.split(":")[0]
    base, quote = symbol.split("/")
    return f"{base}-{quote}"


def lighter_order_side_to_qubx(side: str) -> OrderSide:
    """
    Convert Lighter order side to Qubx OrderSide.

    Args:
        side: Lighter side ("B" or "S")

    Returns:
        Qubx OrderSide (Literal["BUY", "SELL"])
    """
    if side == LighterOrderSide.BUY:
        return "BUY"
    elif side == LighterOrderSide.SELL:
        return "SELL"
    else:
        raise ValueError(f"Unknown Lighter order side: {side}")


def qubx_order_side_to_lighter(side: OrderSide) -> str:
    """
    Convert Qubx OrderSide to Lighter format.

    Args:
        side: Qubx OrderSide (Literal["BUY", "SELL"])

    Returns:
        Lighter side string
    """
    if side == "BUY":
        return LighterOrderSide.BUY
    elif side == "SELL":
        return LighterOrderSide.SELL
    else:
        raise ValueError(f"Unknown order side: {side}")


def lighter_price_to_float(price_str: str, decimals: int) -> float:
    """
    Convert Lighter integer price to float.

    Lighter uses integer representation: actual_price = int_price * 10^(-decimals)

    Args:
        price_str: Price as string
        decimals: Number of decimal places

    Returns:
        Float price
    """
    return float(price_str) * (10 ** -decimals)


def float_to_lighter_price(price: float, decimals: int) -> str:
    """
    Convert float price to Lighter integer format.

    Args:
        price: Float price
        decimals: Number of decimal places

    Returns:
        Price as string in Lighter format
    """
    return str(int(price * (10**decimals)))


def lighter_size_to_float(size_str: str, decimals: int) -> float:
    """
    Convert Lighter integer size to float.

    Args:
        size_str: Size as string
        decimals: Number of decimal places

    Returns:
        Float size
    """
    return float(size_str) * (10 ** -decimals)


def float_to_lighter_size(size: float, decimals: int) -> str:
    """
    Convert float size to Lighter integer format.

    Args:
        size: Float size
        decimals: Number of decimal places

    Returns:
        Size as string in Lighter format
    """
    return str(int(size * (10**decimals)))


def convert_lighter_orderbook(
    orderbook_data: dict, instrument: Instrument, timestamp_ns: int
) -> OrderBook:
    """
    Convert Lighter orderbook data to Qubx OrderBook.

    Args:
        orderbook_data: Raw orderbook data from Lighter
        instrument: Qubx instrument
        timestamp_ns: Timestamp in nanoseconds

    Returns:
        Qubx OrderBook object
    """
    asks = orderbook_data.get("asks", [])
    bids = orderbook_data.get("bids", [])

    # Convert to numpy arrays
    ask_prices = np.array([float(ask["price"]) for ask in asks], dtype=np.float64)
    ask_sizes = np.array([float(ask["size"]) for ask in asks], dtype=np.float64)
    bid_prices = np.array([float(bid["price"]) for bid in bids], dtype=np.float64)
    bid_sizes = np.array([float(bid["size"]) for bid in bids], dtype=np.float64)

    return OrderBook(
        time=np.datetime64(timestamp_ns, "ns"),
        ask_price=ask_prices,
        ask_size=ask_sizes,
        bid_price=bid_prices,
        bid_size=bid_sizes,
    )


def convert_lighter_trade(trade_data: dict, instrument: Instrument) -> Trade:
    """
    Convert Lighter trade data to Qubx Trade.

    Args:
        trade_data: Raw trade data from Lighter
        instrument: Qubx instrument

    Returns:
        Qubx Trade object
    """
    timestamp_ms = trade_data.get("timestamp", 0)
    price = float(trade_data.get("price", 0))
    size = float(trade_data.get("size", 0))
    side = trade_data.get("side", "B")

    return Trade(
        time=np.datetime64(timestamp_ms, "ms"),
        price=price,
        quantity=size,
        side=1 if side == "B" else -1,  # 1 for buy, -1 for sell
    )


def convert_lighter_quote(orderbook_data: dict, timestamp_ns: int) -> Quote:
    """
    Convert Lighter orderbook to Qubx Quote (best bid/ask).

    Args:
        orderbook_data: Raw orderbook data from Lighter
        timestamp_ns: Timestamp in nanoseconds

    Returns:
        Qubx Quote object
    """
    asks = orderbook_data.get("asks", [])
    bids = orderbook_data.get("bids", [])

    best_ask_price = float(asks[0]["price"]) if asks else np.nan
    best_ask_size = float(asks[0]["size"]) if asks else 0.0
    best_bid_price = float(bids[0]["price"]) if bids else np.nan
    best_bid_size = float(bids[0]["size"]) if bids else 0.0

    # Quote constructor: __init__(self, time, bid, ask, bid_size, ask_size)
    return Quote(
        np.datetime64(timestamp_ns, "ns"),
        best_bid_price,
        best_ask_price,
        best_bid_size,
        best_ask_size,
    )


def convert_lighter_order(order_data: dict, instrument: Instrument) -> Order:
    """
    Convert Lighter order data to Qubx Order.

    Args:
        order_data: Raw order data from Lighter
        instrument: Qubx instrument

    Returns:
        Qubx Order object
    """
    order_id = str(order_data.get("oid", ""))
    client_order_id = order_data.get("cloid")
    side_str = order_data.get("side", "B")
    price = float(order_data.get("limitPx", 0)) if order_data.get("limitPx") else None
    size = float(order_data.get("sz", 0))
    filled = float(order_data.get("origSz", size)) - size
    timestamp_ms = order_data.get("timestamp", 0)

    # Determine status
    status_str = order_data.get("status", "open")
    if status_str == "filled":
        status = OrderStatus.FILLED
    elif status_str == "canceled" or status_str == "cancelled":
        status = OrderStatus.CANCELLED
    elif filled > 0 and size > 0:
        status = OrderStatus.PARTIALLY_FILLED
    else:
        status = OrderStatus.OPEN

    return Order(
        id=order_id,
        instrument=instrument,
        side=lighter_order_side_to_qubx(side_str),
        order_type="limit" if price else "market",
        amount=size + filled,  # Original size
        price=price,
        filled=filled,
        remaining=size,
        status=status,
        timestamp=np.datetime64(timestamp_ms, "ms"),
        client_order_id=client_order_id,
    )
