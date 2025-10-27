"""Positions table widget for displaying live trading positions."""

from textual.widgets import DataTable


class PositionsTable(DataTable):
    """DataTable widget for displaying positions with automatic updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor_type = "row"

    def setup_columns(self):
        """Initialize table columns."""
        self.add_columns("Exchange", "Symbol", "Side", "Qty", "Avg Px", "Last Px", "PnL", "Mkt Value")

    def update_positions(self, rows: list[dict]) -> None:
        """
        Update the positions table with new data.

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
        if not rows:
            return

        # Clear rows only (keep column headers)
        self.clear(columns=False)

        # Sort by market value (descending)
        sorted_rows = sorted(rows, key=lambda r: abs(r.get("qty", 0)), reverse=True)

        # Add rows
        for r in sorted_rows:
            self.add_row(
                r.get("exchange", ""),
                r["symbol"],
                r["side"],
                str(r["qty"]),
                str(r["avg_px"]),
                str(r["last_px"]),
                f"{r['pnl']:.2f}",
                f"{r['mkt_value']:.3f}",
            )


def sanitize_positions_data(positions: list[dict]) -> list[dict]:
    """
    Sanitize positions data by removing NaN and Inf values.

    Args:
        positions: List of position dictionaries

    Returns:
        Sanitized list of position dictionaries
    """
    sanitized = []
    for pos in positions:
        # Handle NaN/Inf in numeric fields
        pnl = pos.get("pnl", 0.0)
        if not isinstance(pnl, (int, float)) or (
            isinstance(pnl, float) and (pnl != pnl or pnl == float("inf") or pnl == float("-inf"))
        ):
            pnl = 0.0

        mkt_value = pos.get("mkt_value", 0.0)
        if not isinstance(mkt_value, (int, float)) or (
            isinstance(mkt_value, float)
            and (mkt_value != mkt_value or mkt_value == float("inf") or mkt_value == float("-inf"))
        ):
            mkt_value = 0.0

        sanitized.append(
            {
                "symbol": pos.get("symbol", ""),
                "side": pos.get("side", ""),
                "qty": pos.get("qty", 0.0),
                "avg_px": pos.get("avg_px", 0.0),
                "last_px": pos.get("last_px", 0.0),
                "pnl": pnl,
                "mkt_value": mkt_value,
            }
        )

    return sanitized
