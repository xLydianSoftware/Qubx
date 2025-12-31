"""Positions table widget for displaying live trading positions."""

from typing import Any

from rich.text import Text
from textual.widgets import DataTable

from qubx import logger


class PositionsTable(DataTable):
    """DataTable widget for displaying positions with automatic updates."""

    BINDINGS = [
        ("ctrl+e", "sort_by_exchange", "Sort By Exchange"),
        ("ctrl+s", "sort_by_symbol", "Sort By Symbol"),
        ("ctrl+l", "sort_by_leverage", "Sort By Leverage"),
    ]

    _sorted_by: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor_type = "row"
        self._sorted_by = "Leverage"
        # Track mapping of position keys to DataTable row keys for incremental updates
        self._row_keys: dict[str, Any] = {}  # "exchange:symbol" → RowKey
        self._col_keys: list[Any] = []  # Column keys from Textual

    def setup_columns(self):
        """Initialize table columns and store column keys."""
        self._col_keys = list(
            self.add_columns(
                ("Exchange", "Exchange"),
                ("Symbol", "Symbol"),
                ("Leverage", "Leverage"),
                ("Qty", "Qty"),
                ("Avg Px", "Avg Px"),
                ("Last Px", "Last Px"),
                ("PnL", "PnL"),
                ("Mkt Value", "Mkt Value"),
            )
        )

    def update_positions(self, rows: list[dict]) -> None:
        """
        Update the positions table incrementally without clearing.

        Args:
            rows: List of position dictionaries with keys:
                - exchange: Exchange name
                - symbol: Instrument symbol
                - leverage: Position leverage as fractional value (e.g., 0.1 for 10%)
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
                side = self._get_side(r)

                # Update each column using stored column keys with color coding
                self.update_cell(row_key, self._col_keys[0], _colored_text(r.get("exchange", ""), side))
                self.update_cell(row_key, self._col_keys[1], _colored_text(r.get("symbol", "N/A"), side))
                self.update_cell(row_key, self._col_keys[2], _colored_text(_format_leverage(r.get("leverage")), side))
                self.update_cell(
                    row_key, self._col_keys[3], _colored_text(_format_number(r.get("qty"), decimals=4), side)
                )
                self.update_cell(
                    row_key, self._col_keys[4], _colored_text(_format_number(r.get("avg_px"), decimals=4), side)
                )
                self.update_cell(
                    row_key, self._col_keys[5], _colored_text(_format_number(r.get("last_px"), decimals=4), side)
                )
                self.update_cell(
                    row_key,
                    self._col_keys[6],
                    _colored_text(_format_number(r.get("pnl"), decimals=2, signed=True), side),
                )
                self.update_cell(
                    row_key,
                    self._col_keys[7],
                    _colored_text(_format_number(r.get("mkt_value"), decimals=3, signed=True), side),
                )
            except Exception as e:
                logger.warning(f"Failed to update position row {pos_key}: {e}")

        # Add new positions (sorted by absolute leverage, descending)
        sorted_new_keys = sorted(keys_to_add, key=lambda k: abs(new_positions[k].get("leverage", 0)), reverse=True)
        for pos_key in sorted_new_keys:
            try:
                r = new_positions[pos_key]
                side = self._get_side(r)

                # Add row with position key for future reference (with color coding)
                row_key = self.add_row(
                    _colored_text(r.get("exchange", ""), side),
                    _colored_text(r.get("symbol", "N/A"), side),
                    _colored_text(_format_leverage(r.get("leverage")), side),
                    _colored_text(_format_number(r.get("qty"), decimals=4), side),
                    _colored_text(_format_number(r.get("avg_px"), decimals=4), side),
                    _colored_text(_format_number(r.get("last_px"), decimals=4), side),
                    _colored_text(_format_number(r.get("pnl"), decimals=2, signed=True), side),
                    _colored_text(_format_number(r.get("mkt_value"), decimals=3, signed=True), side),
                    key=pos_key,  # Use exchange:symbol as DataTable row key
                )
                self._row_keys[pos_key] = row_key
            except Exception as e:
                logger.warning(f"Failed to add position row {pos_key}: {e}")

        self._sort()

    def _get_side(self, r: dict) -> str:
        leverage = r.get("leverage", 0.0)
        if leverage > 0:
            return "LONG"
        elif leverage < 0:
            return "SHORT"
        else:
            return "FLAT"

    def _sort(self) -> None:
        if self._sorted_by == "Leverage":
            self.sort(self._sorted_by, key=lambda x: abs(self._parse_number(x)), reverse=True)
        else:
            self.sort(self._sorted_by, key=self._parse_text)

    def _parse_text(self, value) -> str:
        """Extract text value from formatted Text object for sorting."""
        try:
            return str(value.plain) if hasattr(value, "plain") else str(value)
        except (ValueError, AttributeError):
            return ""

    def _parse_number(self, value) -> float:
        """Extract numeric value from formatted Text object for sorting."""
        try:
            text = str(value.plain) if hasattr(value, "plain") else str(value)
            # Remove % sign, + sign, commas and convert to float
            text = text.replace("%", "").replace("+", "").replace(",", "")
            return float(text)
        except (ValueError, AttributeError):
            return 0.0

    def action_sort_by_symbol(self) -> None:
        self._sorted_by = "Symbol"
        self._sort()

    def action_sort_by_leverage(self) -> None:
        self._sorted_by = "Leverage"
        self._sort()

    def action_sort_by_exchange(self) -> None:
        self._sorted_by = "Exchange"
        self._sort()


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
                    "leverage": _sanitize_numeric(pos.get("leverage"), 0.0),
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


def _format_leverage(value) -> str:
    """
    Format fractional leverage as percentage.

    Args:
        value: Fractional leverage value (e.g., 0.1 for 10%)

    Returns:
        Formatted percentage string (e.g., "10.0%") or "-" for invalid values
    """
    if value is None:
        return "-"

    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"

    if num != num or num == float("inf") or num == float("-inf"):
        return "-"

    # Convert to percentage with 1 decimal place
    return f"{num * 100:.1f}%"


def _colored_text(text: str, side: str) -> Text:
    """
    Create a colored Text object based on position side.

    Args:
        text: The text content
        side: Position side ("LONG", "SHORT", or "FLAT")

    Returns:
        Rich Text object with appropriate color
    """
    if side == "LONG":
        return Text(text, style="green")
    elif side == "SHORT":
        return Text(text, style="red")
    else:
        return Text(text)
