"""Account summary bar widget showing per-exchange capital and leverage."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static


class AccountSummary(Widget):
    """Horizontal bar displaying per-exchange total capital, net and gross leverage."""

    DEFAULT_CSS = """
    AccountSummary {
        height: auto;
        dock: top;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="account-summary-bar"):
            yield Static("Waiting for data...", id="account-summary-content")

    def update_summary(self, summary: list[dict]) -> None:
        """Update the summary bar with per-exchange data.

        Args:
            summary: List of dicts with keys: exchange, total_capital, net_leverage, gross_leverage
        """
        if not summary:
            return

        parts = []
        for item in summary:
            exch = item.get("exchange", "?")
            capital = item.get("total_capital", 0.0)
            net_lev = item.get("net_leverage", 0.0)
            gross_lev = item.get("gross_leverage", 0.0)
            parts.append(
                f"[bold]{exch}[/bold]  "
                f"Capital: [cyan]${capital:,.2f}[/cyan]  "
                f"Net: [green]{net_lev:+.3f}x[/green]  "
                f"Gross: [yellow]{gross_lev:.3f}x[/yellow]"
            )

        content = self.query_one("#account-summary-content", Static)
        content.update("    ".join(parts))
