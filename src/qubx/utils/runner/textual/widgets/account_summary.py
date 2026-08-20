"""Account summary bar widget showing per-exchange capital and leverage."""

from datetime import datetime, timezone

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
            # - context status, between the capital figures and the clock
            yield Static("", id="account-summary-status")
            # - self-ticking wall clock, pushed to the right corner (content takes 1fr)
            yield Static("", id="account-summary-clock")

    def on_mount(self) -> None:
        self.set_interval(1.0, self._update_clock)
        self._update_clock()

    def _update_clock(self) -> None:
        self.query_one("#account-summary-clock", Static).update(datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))

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
                f"PnL: {self._money(item.get('pnl', 0.0))}  "
                f"Net: [green]{net_lev:+.3f}x[/green]  "
                f"Gross: [yellow]{gross_lev:.3f}x[/yellow]"
            )

        total_pnl = sum(item.get("pnl", 0.0) for item in summary)
        if len(summary) > 1:
            parts.insert(0, f"[bold]TOTAL[/bold] PnL: {self._money(total_pnl)}")

        content = self.query_one("#account-summary-content", Static)
        content.update("    ".join(parts))

    @staticmethod
    def _money(value: float) -> str:
        colour = "green" if value > 0 else ("red" if value < 0 else "white")
        sign = "-" if value < 0 else ""
        return f"[{colour}]{sign}${abs(value):,.2f}[/{colour}]"

    def update_status(self, status: dict | None) -> None:
        """Update the context-status chip.

        Args:
            status: {"status": "normal"|"degraded", "degradations": [{reason, scope, message}]}
        """
        widget = self.query_one("#account-summary-status", Static)
        if not status:
            widget.update("")
            return

        degradations = status.get("degradations") or []
        if status.get("status") != "degraded" or not degradations:
            widget.update("[green]● NORMAL[/green]")
            return

        # - one chip per held degradation: reason, and the exchange when it is scoped to one
        reasons = []
        for d in degradations:
            reason = d.get("reason", "?")
            scope = d.get("scope")
            reasons.append(f"{reason}@{scope}" if scope else reason)
        widget.update(f"[bold white on red] ⚠ DEGRADED: {', '.join(reasons)} [/]")
