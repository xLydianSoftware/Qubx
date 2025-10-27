"""Positions table widget for displaying live trading positions."""

from typing import Any

from textual.widgets import DataTable

from qubx import logger


class PositionsTable(DataTable):
    """DataTable widget for displaying positions with automatic updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor_type = "row"
        # Track mapping of position keys to DataTable row keys for incremental updates
        self._row_keys: dict[str, Any] = {}  # "exchange:symbol" → RowKey
        self._col_keys: list[Any] = []  # Column keys from Textual

    def setup_columns(self):
        """Initialize table columns and store column keys."""
        self._col_keys = list(self.add_columns("Exchange", "Symbol", "Side", "Qty", "Avg Px", "Last Px", "PnL", "Mkt Value"))

    def update_positions(self, rows: list[dict]) -> None:
        """
        Update the positions table incrementally without clearing.

        Args:
            rows: List of position dictionaries with keys:
                - exchange: Exchange name
                - symbol: Instrument symbol
                - side: Position side (LONG/SHORT/FLAT)
                - qty: Position quantity
                - avg_px: Average entry price
                - last_px: Last traded price
                - pnl: Total profit/loss
                - mkt_value: Market value
        """
        # Sanitize data first
        sanitized_rows = sanitize_positions_data(rows)

        # Build mapping of position keys to position data
        # Key format: "exchange:symbol"
        new_positions = {}
        for r in sanitized_rows:
            pos_key = f"{r.get('exchange', '')}:{r.get('symbol', '')}"
            if pos_key and pos_key != ":":
                new_positions[pos_key] = r

        # Determine which rows to add, update, or remove
        new_keys = set(new_positions.keys())
        existing_keys = set(self._row_keys.keys())

        keys_to_add = new_keys - existing_keys
        keys_to_update = new_keys & existing_keys
        keys_to_remove = existing_keys - new_keys

        # Remove closed positions
        for pos_key in keys_to_remove:
            try:
                row_key = self._row_keys[pos_key]
                self.remove_row(row_key)
                del self._row_keys[pos_key]
            except Exception as e:
                logger.warning(f"Failed to remove position row {pos_key}: {e}")

        # Update existing positions cell by cell (no visual flicker!)
        for pos_key in keys_to_update:
            try:
                r = new_positions[pos_key]
                row_key = self._row_keys[pos_key]

                # Update each column using stored column keys
                self.update_cell(row_key, self._col_keys[0], r.get("exchange", ""))
                self.update_cell(row_key, self._col_keys[1], r.get("symbol", "N/A"))
                self.update_cell(row_key, self._col_keys[2], r.get("side", ""))
                self.update_cell(row_key, self._col_keys[3], _format_number(r.get("qty"), decimals=4))
                self.update_cell(row_key, self._col_keys[4], _format_number(r.get("avg_px"), decimals=4))
                self.update_cell(row_key, self._col_keys[5], _format_number(r.get("last_px"), decimals=4))
                self.update_cell(row_key, self._col_keys[6], _format_number(r.get("pnl"), decimals=2, signed=True))
                self.update_cell(row_key, self._col_keys[7], _format_number(r.get("mkt_value"), decimals=3, signed=True))
            except Exception as e:
                logger.warning(f"Failed to update position row {pos_key}: {e}")

        # Add new positions (sorted by quantity, descending)
        sorted_new_keys = sorted(keys_to_add, key=lambda k: abs(new_positions[k].get("qty", 0)), reverse=True)
        for pos_key in sorted_new_keys:
            try:
                r = new_positions[pos_key]

                # Add row with position key for future reference
                row_key = self.add_row(
                    r.get("exchange", ""),
                    r.get("symbol", "N/A"),
                    r.get("side", ""),
                    _format_number(r.get("qty"), decimals=4),
                    _format_number(r.get("avg_px"), decimals=4),
                    _format_number(r.get("last_px"), decimals=4),
                    _format_number(r.get("pnl"), decimals=2, signed=True),
                    _format_number(r.get("mkt_value"), decimals=3, signed=True),
                    key=pos_key,  # Use exchange:symbol as DataTable row key
                )
                self._row_keys[pos_key] = row_key
            except Exception as e:
                logger.warning(f"Failed to add position row {pos_key}: {e}")


def _format_number(value, decimals: int = 2, signed: bool = False) -> str:
    """
    Safely format a numeric value with fallback for invalid data.

    Args:
        value: Numeric value to format (or None)
        decimals: Number of decimal places
        signed: Whether to include + sign for positive numbers

    Returns:
        Formatted string or "-" for invalid values
    """
    if value is None:
        return "-"

    # Convert to float if needed
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"

    # Check for NaN/Inf
    if num != num or num == float("inf") or num == float("-inf"):
        return "-"

    # Format with appropriate decimals
    formatted = f"{num:.{decimals}f}"

    # Add + sign for positive if requested
    if signed and num > 0:
        formatted = f"+{formatted}"

    return formatted


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

    # Check for NaN/Inf
    if num != num or num == float("inf") or num == float("-inf"):
        return default

    return num


def sanitize_positions_data(positions: list[dict]) -> list[dict]:
    """
    Sanitize positions data by handling missing keys and invalid values.

    Args:
        positions: List of position dictionaries

    Returns:
        Sanitized list of position dictionaries
    """
    if not positions:
        return []

    sanitized = []
    for pos in positions:
        try:
            # Ensure all required fields exist with safe defaults
            sanitized.append(
                {
                    "exchange": pos.get("exchange", ""),
                    "symbol": pos.get("symbol", "UNKNOWN"),
                    "side": pos.get("side", "FLAT"),
                    "qty": _sanitize_numeric(pos.get("qty"), 0.0),
                    "avg_px": _sanitize_numeric(pos.get("avg_px"), 0.0),
                    "last_px": _sanitize_numeric(pos.get("last_px"), 0.0),
                    "pnl": _sanitize_numeric(pos.get("pnl"), 0.0),
                    "mkt_value": _sanitize_numeric(pos.get("mkt_value"), 0.0),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to sanitize position data: {e}")
            continue

    return sanitized
