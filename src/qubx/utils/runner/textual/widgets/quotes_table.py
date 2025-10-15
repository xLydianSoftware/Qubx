"""Quotes table widget for displaying live market data."""

from rich.text import Text
from textual.widgets import DataTable


class QuotesTable(DataTable):
    """DataTable widget for displaying quotes with automatic updates and spread visualization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor_type = "row"

    def setup_columns(self):
        """Initialize table columns."""
        self.add_columns("Exchange", "Symbol", "Bid", "Ask", "Spread", "Spread%", "Last", "Volume")

    def update_quotes(self, quotes: dict[str, dict]) -> None:
        """
        Update the quotes table with new data.

        Args:
            quotes: Dictionary mapping instrument keys to quote data with keys:
                - exchange: Exchange name
                - symbol: Instrument symbol
                - bid: Bid price
                - ask: Ask price
                - spread: Absolute spread (ask - bid)
                - spread_pct: Spread as percentage of bid
                - last: Last traded price
                - volume: Trading volume
        """
        if not quotes:
            # Clear table if no quotes
            self.clear(columns=False)
            return

        # Clear rows only (keep column headers)
        self.clear(columns=False)

        # Sort by symbol
        sorted_items = sorted(quotes.items(), key=lambda x: (x[1].get("exchange", ""), x[1].get("symbol", "")))

        # Add rows
        for key, q in sorted_items:
            spread_pct = q.get("spread_pct", 0.0)
            spread_visual = self._visualize_spread(spread_pct)

            self.add_row(
                q.get("exchange", ""),
                q.get("symbol", ""),
                str(q.get("bid", "")) if q.get("bid") else "-",
                str(q.get("ask", "")) if q.get("ask") else "-",
                f"{q.get('spread', 0):.{self._get_price_precision(q)}f}",
                spread_visual,
                str(q.get("last", "")) if q.get("last") else "-",
                f"{q.get('volume', 0):.2f}" if q.get("volume") else "-",
            )

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
