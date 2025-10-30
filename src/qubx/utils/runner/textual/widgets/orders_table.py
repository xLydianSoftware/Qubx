"""Orders table widget for displaying live trading orders."""

from typing import Any

from textual.widgets import DataTable

from qubx import logger


class OrdersTable(DataTable):
    """DataTable widget for displaying orders with automatic updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor_type = "row"
        # Track mapping of order IDs to DataTable row keys for incremental updates
        self._row_keys: dict[str, Any] = {}  # order_id → RowKey
        self._col_keys: list[Any] = []  # Column keys from Textual

    def setup_columns(self):
        """Initialize table columns and store column keys."""
        self._col_keys = list(self.add_columns("Exchange", "Symbol", "Side", "Type", "Qty", "Price", "Status", "Time"))

    def update_orders(self, rows: list[dict]) -> None:
        """
        Update the orders table incrementally without clearing.

        Args:
            rows: List of order dictionaries with keys:
                - id: Order ID (unique identifier)
                - exchange: Exchange name
                - symbol: Instrument symbol
                - side: Order side (buy/sell)
                - type: Order type (limit/market)
                - qty: Order quantity
                - price: Order price (None for market orders)
                - filled: Filled quantity
                - status: Order status
                - time: Order timestamp
        """
        # Sanitize data first
        sanitized_rows = sanitize_orders_data(rows)

        # Build mapping of order IDs to order data
        new_orders = {r.get("id", ""): r for r in sanitized_rows if r.get("id")}

        # Determine which rows to add, update, or remove
        new_keys = set(new_orders.keys())
        existing_keys = set(self._row_keys.keys())

        keys_to_add = new_keys - existing_keys
        keys_to_update = new_keys & existing_keys
        keys_to_remove = existing_keys - new_keys

        # Remove stale orders
        for order_id in keys_to_remove:
            try:
                row_key = self._row_keys[order_id]
                self.remove_row(row_key)
                del self._row_keys[order_id]
            except Exception as e:
                logger.warning(f"Failed to remove order row {order_id}: {e}")

        # Update existing orders cell by cell (no visual flicker!)
        for order_id in keys_to_update:
            try:
                r = new_orders[order_id]
                row_key = self._row_keys[order_id]
                price_str = _format_order_price(r.get("price"), r.get("type", ""))
                time_str = _format_time(r.get("time", ""))

                # Update each column using stored column keys
                self.update_cell(row_key, self._col_keys[0], r.get("exchange", ""))
                self.update_cell(row_key, self._col_keys[1], r.get("symbol", "N/A"))
                self.update_cell(row_key, self._col_keys[2], r.get("side", ""))
                self.update_cell(row_key, self._col_keys[3], r.get("type", ""))
                self.update_cell(row_key, self._col_keys[4], _format_number(r.get("qty"), decimals=4))
                self.update_cell(row_key, self._col_keys[5], price_str)
                self.update_cell(row_key, self._col_keys[6], r.get("status", ""))
                self.update_cell(row_key, self._col_keys[7], time_str)
            except Exception as e:
                logger.warning(f"Failed to update order row {order_id}: {e}")

        # Add new orders (sorted by time, most recent first)
        sorted_new_keys = sorted(keys_to_add, key=lambda k: new_orders[k].get("time", ""), reverse=True)
        for order_id in sorted_new_keys:
            try:
                r = new_orders[order_id]
                price_str = _format_order_price(r.get("price"), r.get("type", ""))
                time_str = _format_time(r.get("time", ""))

                # Add row with order ID as key for future reference
                row_key = self.add_row(
                    r.get("exchange", ""),
                    r.get("symbol", "N/A"),
                    r.get("side", ""),
                    r.get("type", ""),
                    _format_number(r.get("qty"), decimals=4),
                    price_str,
                    r.get("status", ""),
                    time_str,
                    key=order_id,  # Use order ID as DataTable row key
                )
                self._row_keys[order_id] = row_key
            except Exception as e:
                logger.warning(f"Failed to add order row {order_id}: {e}")

    def _get_price_precision(self, row: dict) -> int:
        """Get price precision for formatting (default to 2 decimals)."""
        # Could be enhanced to use instrument precision if available
        return 2


def _format_number(value, decimals: int = 2) -> str:
    """
    Safely format a numeric value with fallback for invalid data.

    Args:
        value: Numeric value to format (or None)
        decimals: Number of decimal places

    Returns:
        Formatted string or "-" for invalid values
    """
    if value is None:
        return "-"

    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"

    # Check for NaN/Inf
    if num != num or num == float("inf") or num == float("-inf"):
        return "-"

    return f"{num:.{decimals}f}"


def _format_order_price(price, order_type: str) -> str:
    """
    Format order price, returning "Market" for market orders.

    Args:
        price: Order price (can be None)
        order_type: Order type (market/limit)

    Returns:
        Formatted price string or "Market"
    """
    if price is None or (isinstance(order_type, str) and order_type.lower() == "market"):
        return "Market"

    try:
        num = float(price)
        if num != num or num == float("inf") or num == float("-inf"):
            return "Market"
        return f"{num:.4f}"
    except (TypeError, ValueError):
        return "Market"


def _format_time(time_str: str) -> str:
    """
    Format timestamp string, truncating to datetime format.

    Args:
        time_str: Timestamp string

    Returns:
        Formatted timestamp (first 19 chars) or empty string
    """
    if not time_str:
        return ""

    try:
        # Truncate to datetime format (YYYY-MM-DD HH:MM:SS)
        return str(time_str)[:19]
    except Exception:
        return ""


def _sanitize_numeric(value, default: float = 0.0) -> float:
    """
    Sanitize a numeric value, replacing NaN/Inf/None with default.

    Args:
        value: Value to sanitize
        default: Default value for invalid inputs

    Returns:
        Sanitized numeric value
    """
    if value is None:
        return default

    try:
        num = float(value)
    except (TypeError, ValueError):
        return default

    if num != num or num == float("inf") or num == float("-inf"):
        return default

    return num


def sanitize_orders_data(orders: list[dict]) -> list[dict]:
    """
    Sanitize orders data by handling missing keys and invalid values.

    Args:
        orders: List of order dictionaries

    Returns:
        Sanitized list of order dictionaries
    """
    if not orders:
        return []

    sanitized = []
    for order in orders:
        try:
            sanitized.append(
                {
                    "id": order.get("id", ""),
                    "exchange": order.get("exchange", ""),
                    "symbol": order.get("symbol", "UNKNOWN"),
                    "side": order.get("side", ""),
                    "type": order.get("type", "LIMIT"),
                    "qty": _sanitize_numeric(order.get("qty"), 0.0),
                    "price": _sanitize_numeric(order.get("price")) if order.get("price") is not None else None,
                    "filled": _sanitize_numeric(order.get("filled"), 0.0),
                    "status": order.get("status", "UNKNOWN"),
                    "time": order.get("time", ""),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to sanitize order data: {e}")
            continue

    return sanitized
