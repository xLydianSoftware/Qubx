"""Orders table widget for displaying live trading orders."""

from textual.widgets import DataTable


class OrdersTable(DataTable):
    """DataTable widget for displaying orders with automatic updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor_type = "row"

    def setup_columns(self):
        """Initialize table columns."""
        self.add_columns("Exchange", "Symbol", "Side", "Type", "Qty", "Price", "Filled", "Status", "Time")

    def update_orders(self, rows: list[dict]) -> None:
        """
        Update the orders table with new data.

        Args:
            rows: List of order dictionaries with keys:
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
        # Always clear first to ensure stale data is removed
        self.clear(columns=False)

        if not rows:
            # Table is now empty
            return

        # Sort by time (most recent first)
        sorted_rows = sorted(rows, key=lambda r: r.get("time", ""), reverse=True)

        # Add rows
        for r in sorted_rows:
            price_str = f"{r['price']:.{self._get_price_precision(r)}f}" if r.get("price") else "Market"
            self.add_row(
                r.get("exchange", ""),
                r.get("symbol", ""),
                r.get("side", ""),
                r.get("type", ""),
                str(r.get("qty", "")),
                price_str,
                str(r.get("filled", "0")),
                r.get("status", ""),
                r.get("time", "")[:19] if r.get("time") else "",  # Truncate to datetime format
            )

    def _get_price_precision(self, row: dict) -> int:
        """Get price precision for formatting (default to 2 decimals)."""
        # Could be enhanced to use instrument precision if available
        return 2
