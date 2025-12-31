"""Quotes table widget for displaying live market data."""

from typing import Any

from rich.text import Text
from textual.widgets import DataTable

from qubx import logger


class QuotesTable(DataTable):
    """DataTable widget for displaying quotes with automatic updates and spread visualization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor_type = "row"
        # Track mapping of quote keys to DataTable row keys for incremental updates
        self._row_keys: dict[str, Any] = {}  # RowKey type from Textual
        self._col_keys: list[Any] = []  # Column keys from Textual

    def setup_columns(self):
        """Initialize table columns and store column keys."""
        self._col_keys = list(self.add_columns("Exchange", "Symbol", "Bid", "Ask", "Spread", "Spread%"))

    def update_quotes(self, quotes: dict[str, dict]) -> None:
        """
        Update the quotes table incrementally without clearing.

        Args:
            quotes: Dictionary mapping instrument keys to quote data with keys:
                - exchange: Exchange name
                - symbol: Instrument symbol
                - bid: Bid price
                - ask: Ask price
                - spread: Absolute spread (ask - bid)
                - spread_pct: Spread as percentage of bid
        """
        # Sanitize data first
        sanitized_quotes = sanitize_quotes_data(quotes)

        # Determine which rows to add, update, or remove
        new_keys = set(sanitized_quotes.keys())
        existing_keys = set(self._row_keys.keys())

        keys_to_add = new_keys - existing_keys
        keys_to_update = new_keys & existing_keys
        keys_to_remove = existing_keys - new_keys

        # Remove stale rows
        for key in keys_to_remove:
            try:
                row_key = self._row_keys[key]
                self.remove_row(row_key)
                del self._row_keys[key]
            except Exception as e:
                logger.warning(f"Failed to remove quote row for {key}: {e}")

        # Update existing rows cell by cell (no visual flicker!)
        for key in keys_to_update:
            try:
                q = sanitized_quotes[key]
                row_key = self._row_keys[key]
                spread_pct = q.get("spread_pct", 0.0)
                spread_visual = self._visualize_spread(spread_pct)

                # Update each column using stored column keys
                self.update_cell(row_key, self._col_keys[0], q.get("exchange", ""))
                self.update_cell(row_key, self._col_keys[1], q.get("symbol", "N/A"))
                self.update_cell(row_key, self._col_keys[2], _format_number(q.get("bid"), decimals=4))
                self.update_cell(row_key, self._col_keys[3], _format_number(q.get("ask"), decimals=4))
                self.update_cell(row_key, self._col_keys[4], _format_number(q.get("spread"), decimals=4))
                self.update_cell(row_key, self._col_keys[5], spread_visual)
            except Exception as e:
                logger.warning(f"Failed to update quote row for {key}: {e}")

        # Add new rows
        for key in sorted(
            keys_to_add, key=lambda k: (sanitized_quotes[k].get("exchange", ""), sanitized_quotes[k].get("symbol", ""))
        ):
            try:
                q = sanitized_quotes[key]
                spread_pct = q.get("spread_pct", 0.0)
                spread_visual = self._visualize_spread(spread_pct)

                # Add row with key for future reference
                row_key = self.add_row(
                    q.get("exchange", ""),
                    q.get("symbol", "N/A"),
                    _format_number(q.get("bid"), decimals=4),
                    _format_number(q.get("ask"), decimals=4),
                    _format_number(q.get("spread"), decimals=4),
                    spread_visual,
                    key=key,  # Use quote key as DataTable row key
                )
                self._row_keys[key] = row_key
            except Exception as e:
                logger.warning(f"Failed to add quote row for {key}: {e}")

    def _visualize_spread(self, spread_pct: float) -> Text:
        """
        Create a visual representation of the spread.

        Args:
            spread_pct: Spread as percentage

        Returns:
            Rich Text object with colored spread visualization
        """
        # Thresholds for spread quality (in basis points, 0.01% = 1bp)
        # Good: < 5bp (0.05%), Medium: 5-20bp, Wide: > 20bp
        if spread_pct < 0.05:
            color = "green"
            quality = "●●●"
        elif spread_pct < 0.20:
            color = "yellow"
            quality = "●●○"
        else:
            color = "red"
            quality = "●○○"

        text = Text()
        text.append(f"{spread_pct:.4f}% ", style=color)
        text.append(quality, style=color)
        return text

    def _get_price_precision(self, quote: dict) -> int:
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


def _sanitize_numeric(value, default: float | None = None) -> float | None:
    """
    Sanitize a numeric value, replacing NaN/Inf with default.

    Args:
        value: Value to sanitize
        default: Default value for invalid inputs (can be None)

    Returns:
        Sanitized numeric value or None
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


def sanitize_quotes_data(quotes: dict[str, dict]) -> dict[str, dict]:
    """
    Sanitize quotes data by handling missing keys and invalid values.

    Args:
        quotes: Dictionary mapping instrument keys to quote data

    Returns:
        Sanitized dictionary of quotes
    """
    if not quotes:
        return {}

    sanitized = {}
    for key, quote in quotes.items():
        try:
            # Calculate spread if not provided
            bid = _sanitize_numeric(quote.get("bid"))
            ask = _sanitize_numeric(quote.get("ask"))
            spread = _sanitize_numeric(quote.get("spread"))
            spread_pct = _sanitize_numeric(quote.get("spread_pct"), 0.0)

            # Calculate spread if we have bid and ask but no spread
            if spread is None and bid is not None and ask is not None:
                spread = ask - bid

            # Calculate spread_pct if we have spread and bid but no spread_pct
            if spread is not None and bid is not None and bid > 0 and spread_pct == 0.0:
                spread_pct = (spread / bid) * 100.0

            sanitized[key] = {
                "exchange": quote.get("exchange", ""),
                "symbol": quote.get("symbol", "UNKNOWN"),
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "spread_pct": spread_pct,
            }
        except Exception as e:
            logger.warning(f"Failed to sanitize quote data for {key}: {e}")
            continue

    return sanitized
