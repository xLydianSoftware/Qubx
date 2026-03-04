import json
from typing import Any

import pandas as pd
from rich.console import RenderableType
from rich.table import Table
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    Markdown,
    Static,
    Tree,
)

from qubx import logger
from qubx.backtester import BacktestStorage
from qubx.cli.theme import QUBX_DARK
from qubx.core.metrics import TradingSessionResult, get_cum_pnl


def _e(text: object) -> str:
    """
    Escape text for Textual markup (Content.from_markup).

    Textual 6.x uses its own parser that treats ANY [WORD...] as a potential
    markup tag — including uppercase like [MCS_(fp=12,sp=36)] — which Rich's
    escape() misses (it only handles [a-z#/@...]).  Replacing [ with \\[ is
    the only safe approach for arbitrary user-supplied strings.
    """
    return str(text).replace("[", "\\[")


# - braille dot bit positions per row (top→bottom) for left and right columns
_BRAILLE_L = [0x01, 0x02, 0x04, 0x40]
_BRAILLE_R = [0x08, 0x10, 0x20, 0x80]


def _braille_chart(
    values: list[float],
    timestamps: list | None = None,
    width: int = 55,
    height: int = 12,
) -> str:
    """
    Render a 2D braille line chart with Y-axis labels and optional X date ticks.

    Each braille character encodes 2 x-columns × 4 y-rows, giving an effective
    resolution of (width*2) × (height*4) pixels.  Y increases upward so high
    values appear near the top of the chart.

    Args:
        values: Sequence of float data points to plot.
        timestamps: Optional sequence of timestamps for X-axis date labels.
        width: Chart width in braille characters.
        height: Chart height in braille characters.

    Returns:
        Multi-line string suitable for embedding in a Markdown code block.
    """
    if not values or len(values) < 2:
        return ""

    n_x = width * 2  # - total x-pixel columns
    n_y = height * 4  # - total y-pixel rows (0 = bottom, n_y-1 = top)
    n = len(values)

    # - resample uniformly to n_x columns
    sampled = [values[min(int(i * n / n_x), n - 1)] for i in range(n_x)]

    v_min, v_max = min(sampled), max(sampled)
    v_range = v_max - v_min if v_max != v_min else 1.0

    def to_ypx(v: float) -> int:
        # - maps value → y-pixel [0, n_y-1], 0 = bottom
        return max(0, min(n_y - 1, int((v - v_min) / v_range * (n_y - 1))))

    y_pixels = [to_ypx(v) for v in sampled]

    # - 2D canvas: canvas[char_row][char_col] accumulates braille bitmask
    canvas = [[0] * width for _ in range(height)]

    def set_px(x_px: int, y_px: int) -> None:
        # - char_row 0 = top of chart (high y_px), char_row height-1 = bottom
        char_col = x_px // 2
        char_row = height - 1 - (y_px // 4)
        # - within braille char: bit_row 0 = top dot, bit_row 3 = bottom dot
        bit_row = 3 - (y_px % 4)
        if 0 <= char_row < height and 0 <= char_col < width:
            if (x_px % 2) == 0:
                canvas[char_row][char_col] |= _BRAILLE_L[bit_row]
            else:
                canvas[char_row][char_col] |= _BRAILLE_R[bit_row]

    # - plot samples and vertically interpolate between adjacent points
    for x in range(n_x):
        set_px(x, y_pixels[x])
    for x in range(n_x - 1):
        y0, y1 = y_pixels[x], y_pixels[x + 1]
        if abs(y1 - y0) > 1:
            step = 1 if y1 > y0 else -1
            for y in range(y0 + step, y1, step):
                set_px(x, y)

    # - Y-axis label formatter
    def _fmt_y(v: float) -> str:
        if abs(v) >= 1_000_000:
            return f"{v / 1_000_000:+.1f}M"
        elif abs(v) >= 1_000:
            return f"{v / 1_000:+.1f}K"
        else:
            return f"{v:+.1f}"

    label_w = 8  # - chars reserved for the Y-axis label column

    # - render chart rows (char_row 0 = top)
    lines: list[str] = []
    for r in range(height):
        # - value at the top sub-pixel of this char row
        y_top_px = (height - 1 - r) * 4 + 3
        v_at_top = v_min + y_top_px / (n_y - 1) * v_range
        label = _fmt_y(v_at_top).rjust(label_w) if r % 3 == 0 else " " * label_w
        row_chars = "".join(chr(0x2800 + canvas[r][c]) for c in range(width))
        lines.append(f"{label} │{row_chars}")

    # - X-axis separator line
    lines.append(" " * label_w + " └" + "─" * width)

    # - X-axis date ticks (4 evenly spaced, only if timestamps supplied)
    if timestamps and len(timestamps) >= 2:
        try:
            ts = pd.to_datetime(timestamps)
            t0, t1 = ts[0], ts[-1]
            n_ticks = 4
            date_line = [" "] * (label_w + 1 + width)
            for ti in range(n_ticks):
                frac = ti / (n_ticks - 1)
                col = int(frac * (width - 1))
                t = t0 + (t1 - t0) * frac
                tick_label = t.strftime("%b'%y")
                pos = label_w + 1 + col
                # - shift left if label would overflow the right edge
                overflow = pos + len(tick_label) - len(date_line)
                if overflow > 0:
                    pos -= overflow
                for k, ch in enumerate(tick_label):
                    if 0 <= pos + k < len(date_line):
                        date_line[pos + k] = ch
            lines.append("".join(date_line))
        except Exception:
            pass

    return "\n".join(lines)


def _equity_chart(portfolio_df: pd.DataFrame, width: int = 55, height: int = 12) -> str | None:
    """
    Render a 2D braille equity curve from a portfolio parquet DataFrame.

    portfolio.parquet stores *incremental* per-bar PnL (not cumulative).
    We sum across symbols then cumsum to get the equity curve — matching the
    pattern used throughout metrics.py (calculate_total_pnl + .cumsum()).

    Total_PnL column (if pre-computed) is also incremental and needs cumsum.
    """
    if portfolio_df is None or portfolio_df.empty:
        return None

    pnl_cols = [c for c in portfolio_df.columns if c.endswith("_PnL")]
    if not pnl_cols:
        return None

    # - sum period PnL across all symbols, then cumsum → cumulative equity curve
    equity = portfolio_df[pnl_cols].sum(axis=1).cumsum()
    equity = equity.dropna()
    if len(equity) < 2:
        return None

    # - get timestamps: DuckDB returns timestamp as a plain column (not the index);
    #   equity.index is a RangeIndex in that case — pd.to_datetime([0,1,2,...]) would
    #   give Jan 1970, so we must prefer the actual timestamp column when present
    if "timestamp" in portfolio_df.columns:
        timestamps = pd.to_datetime(portfolio_df["timestamp"]).tolist()
    elif isinstance(portfolio_df.index, pd.DatetimeIndex):
        timestamps = equity.index.tolist()
    else:
        timestamps = None

    return _braille_chart(equity.tolist(), timestamps=timestamps, width=width, height=height)


def _build_markdown_report(result_info: dict[str, Any], equity_chart: str | None = None) -> str:
    """
    Build a markdown preview report from a flat BacktestStorage metadata row.

    Mirrors the structure of TradingSessionResult.to_markdown() but works
    directly from the search result dict — no need to load full parquet data.
    """

    def _s(key: str, default: str = "") -> str:
        """Safely extract a scalar string — handles None, NaN, numpy scalars."""
        v = result_info.get(key)
        if v is None:
            return default
        try:
            import math

            if isinstance(v, float) and math.isnan(v):
                return default
        except Exception:
            pass
        return str(v)

    def _f(key: str, default: float = 0.0) -> float:
        """Safely extract a float — handles None, NaN, numpy scalars."""
        v = result_info.get(key)
        try:
            f = float(v)  # type: ignore[arg-type]
            import math

            return default if math.isnan(f) else f
        except Exception:
            return default

    def _lst(key: str) -> list:
        """Safely extract a list — handles None, numpy arrays, JSON strings."""
        v = result_info.get(key)
        if v is None:
            return []
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return list(parsed) if parsed is not None else []
            except Exception:
                return [v] if v else []
        try:
            # - numpy arrays, pandas arrays, etc.
            return [x for x in v if x is not None]
        except Exception:
            return []

    name = _s("name", _s("backtest_id", "Unknown"))
    strategy_class = _s("strategy_class")
    author = _s("author")
    description = _s("description")
    backtest_id = _s("backtest_id")
    config_name = _s("config_name")
    is_variation = bool(result_info.get("is_variation", False))
    variation_params = _s("variation_params")

    # - timestamps
    def _ts(key: str, fmt: str = "%Y-%m-%d %H:%M") -> str:
        try:
            return pd.Timestamp(_s(key)).strftime(fmt)
        except Exception:
            return _s(key)

    created = _ts("creation_time", "%Y-%m-%d %H:%M")
    start = _ts("start", "%Y-%m-%d")
    stop = _ts("stop", "%Y-%m-%d")

    # - flat performance fields
    gain = _f("gain")
    cagr = _f("cagr")
    sharpe = _f("sharpe")
    qr = _f("qr")
    mdd_pct = _f("mdd_pct")
    mdd_usd = _f("mdd_usd")
    fees = _f("fees")
    execs = int(_f("execs"))

    # - drawdown timestamps (may not be present in all metadata versions)
    mdd_start = _ts("mdd_start", "%Y-%m-%d") if result_info.get("mdd_start") else "—"
    mdd_peak = _ts("mdd_peak", "%Y-%m-%d") if result_info.get("mdd_peak") else "—"
    mdd_recover = _ts("mdd_recover", "%Y-%m-%d") if result_info.get("mdd_recover") else "—"

    # - symbols, tags — DuckDB returns LIST columns as numpy arrays
    symbols = _lst("symbols")
    tags = _lst("tags")

    # - parameters (stored as JSON string in metadata)
    raw_params = result_info.get("parameters")
    try:
        params: dict = (
            json.loads(raw_params)
            if isinstance(raw_params, str)
            else (dict(raw_params) if raw_params is not None else {})
        )
    except Exception:
        params = {}

    # - capital
    capital = _s("capital")
    currency = _s("base_currency")
    commissions = _s("commissions")

    # ── header ──────────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(f"# {name}\n")

    meta_parts = []
    if author:
        meta_parts.append(f"**Author:** {author}")
    meta_parts.append(f"**Created:** {created}")
    meta_parts.append(f"**Interval:** {start} → {stop}")
    if is_variation and variation_params:
        meta_parts.append(f"**Variation:** `{variation_params}`")
    lines.append("  ".join(meta_parts) + "\n")

    lines.append(f"**Class:** `{strategy_class}`")
    if config_name:
        lines.append(f"  **Config:** `{config_name}`")
    lines.append(f"  **ID:** `{backtest_id}`\n")

    if capital:
        lines.append(f"**Capital:** {capital} {currency}  **Commissions:** {commissions}\n")

    if tags:
        lines.append("**Tags:** " + " ".join(f"`{t}`" for t in tags) + "\n")

    if description:
        lines.append("> " + "  \n> ".join(description.strip().split("\n")) + "\n")

    lines.append("---\n")

    # ── equity curve ─────────────────────────────────────────────────────────
    if equity_chart:
        lines.append("## Equity Curve\n")
        lines.append("```")
        lines.append(equity_chart)
        lines.append("```\n")

    # ── performance ─────────────────────────────────────────────────────────
    lines.append("## Performance\n")
    lines.append("| Gain | CAGR | Sharpe | QR | Fees | Executions |")
    lines.append("|-----:|-----:|-------:|---:|-----:|-----------:|")
    lines.append(f"| {gain:.3f} | {100 * cagr:.2f}% | {sharpe:.3f} | {qr:.3f} | {fees:.4f} | {execs} |\n")

    # ── drawdown ─────────────────────────────────────────────────────────────
    lines.append("## Drawdown\n")
    lines.append("| MDD % | MDD USD | Start | Peak | Recover |")
    lines.append("|------:|--------:|-------|------|---------|")
    lines.append(f"| {mdd_pct:.2f}% | {mdd_usd:.2f} | {mdd_start} | {mdd_peak} | {mdd_recover} |\n")

    # ── parameters ───────────────────────────────────────────────────────────
    if params:
        lines.append("## Parameters\n")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in params.items():
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")

    # ── instruments ──────────────────────────────────────────────────────────
    if symbols:
        lines.append("## Instruments\n")
        lines.append(" ".join(f"`{s}`" for s in sorted(symbols)) + "\n")

    return "\n".join(lines)


class BacktestTreeNode:
    """Represents a node in the backtest tree structure"""

    def __init__(self, name: str, path: str | None = None, storage: BacktestStorage | None = None):
        self.name = name
        self.path = path
        self.storage = storage
        self.children: list[BacktestTreeNode] = []
        self.backtest_results: list[dict[str, Any]] = []
        self.is_leaf = False

    def add_child(self, child: "BacktestTreeNode"):
        self.children.append(child)

    def add_backtest(self, result_info: dict[str, Any]):
        self.backtest_results.append(result_info)
        self.is_leaf = True


class BacktestResultsTree(Tree[BacktestTreeNode]):
    """Custom tree widget for backtest results"""

    def __init__(self, root_path: str):
        self.root_path = root_path
        self.storage = BacktestStorage(root_path)

        root_node = BacktestTreeNode("Backtest Results", root_path, self.storage)
        self._build_tree_structure(root_node)

        super().__init__(root_node.name, data=root_node)
        self._populate_tree(self.root, root_node)

    def _build_tree_structure(self, root_node: BacktestTreeNode):
        """Build tree structure from BacktestStorage search results."""
        df = self.storage.search()
        if df.empty:
            return

        # - deduplicate: glob may match _metadata.parquet at both the set level and
        #   inside each var_XXX/ subdir, producing duplicate rows per backtest_id
        df = df.drop_duplicates(subset=["backtest_id"])

        # - separate regular runs and variations
        is_var = df["is_variation"].fillna(False)
        regular = df[~is_var]
        variations = df[is_var]

        # - group regular results by short strategy class name
        strategy_groups: dict[str, list[dict]] = {}
        for _, row in regular.iterrows():
            row_dict = row.to_dict()
            # - for regular runs load_id == backtest_id
            row_dict["load_id"] = row_dict["backtest_id"]
            cls = str(row_dict.get("strategy_class", "Unknown")).split(".")[-1]
            strategy_groups.setdefault(cls, []).append(row_dict)

        for strategy_name, results in strategy_groups.items():
            strategy_node = BacktestTreeNode(f"{strategy_name} ({len(results)} results)", strategy_name, self.storage)
            root_node.add_child(strategy_node)
            for result in results:
                strategy_node.add_backtest(result)

        # - group variations by variation_name (the parent run dir, e.g. "strat/Cls/20240301_120000")
        # - NOTE: backtest_id for variation rows already contains the var_XXX suffix,
        #   so load_id == backtest_id directly (no extra appending needed)
        var_groups: dict[str, list[dict]] = {}
        for _, row in variations.iterrows():
            row_dict = row.to_dict()
            row_dict["load_id"] = row_dict["backtest_id"]
            parent_key = str(row_dict.get("variation_name", row_dict.get("backtest_id", "")))
            var_groups.setdefault(parent_key, []).append(row_dict)

        for parent_key, var_list in var_groups.items():
            name = var_list[0].get("name", parent_key).split(".")[0]
            var_node = BacktestTreeNode(f"📊 {name} ({len(var_list)} variations)", name, self.storage)
            root_node.add_child(var_node)
            for variation in var_list:
                var_node.add_backtest(variation)

    def _populate_tree(self, tree_node, data_node: BacktestTreeNode):
        """Populate the textual tree with data nodes"""
        for child in data_node.children:
            # - tree labels are also parsed as Rich markup — escape to be safe
            child_tree_node = tree_node.add(_e(child.name), data=child)
            if child.children:
                self._populate_tree(child_tree_node, child)


class MetricsTable(DataTable):
    """Data table for displaying backtest metrics with row selection"""

    def __init__(self):
        super().__init__()
        self.zebra_stripes = True
        self.cursor_type = "row"

    def populate_table(self, results_data: list[dict[str, Any]]):
        """Populate table with backtest results"""
        if not results_data:
            return

        self.clear(columns=True)

        columns = [
            "Name",
            "Strategy",
            "Created",
            "Author",
            "Gain",
            "CAGR",
            "Sharpe",
            "QR",
            "Max DD %",
            "MDD USD",
            "Fees",
            "Executions",
        ]

        for col in columns:
            self.add_column(col, key=col.lower().replace(" ", "_"))

        for result in results_data:
            # - metrics are flat columns in BacktestStorage rows (not nested under 'performance')
            # - user-data fields (name, strategy_class, author) must be _e()'d:
            #   variation names contain [...] which Textual parses as markup tags
            row_data = [
                _e(str(result.get("name", result.get("backtest_id", "Unknown")))),
                _e(str(result.get("strategy_class", "")).split(".")[-1]),
                pd.Timestamp(result.get("creation_time", "")).strftime("%Y-%m-%d %H:%M"),
                _e(str(result.get("author", ""))),
                f"{float(result.get('gain') or 0):.3f}",
                f"{float(result.get('cagr') or 0):.3f}",
                f"{float(result.get('sharpe') or 0):.3f}",
                f"{float(result.get('qr') or 0):.3f}",
                f"{float(result.get('mdd_pct') or 0):.3f}",
                f"{float(result.get('mdd_usd') or 0):.3f}",
                f"{float(result.get('fees') or 0):.3f}",
                f"{float(result.get('execs') or 0):.0f}",
            ]
            # - row key is load_id so tearsheet loading doesn't need a lookup by name
            self.add_row(*row_data, key=result.get("load_id", result.get("backtest_id", "")))

    def sort_by_column(self, column: str):
        """Sort table by specified column"""
        self.sort(column)


class EquityChart(Static):
    """Widget for displaying equity curves"""

    def __init__(self):
        super().__init__()
        self.results: list[TradingSessionResult] = []

    def set_results(self, results: list[TradingSessionResult]):
        """Set the trading results for chart display"""
        self.results = results
        self.refresh_chart()

    def refresh_chart(self):
        """Generate and display equity chart as text representation"""
        if not self.results:
            self.update("No results to display")
            return

        try:
            equity_data = get_cum_pnl(self.results, timeframe="1h")

            if isinstance(equity_data, pd.Series):
                equity_data = equity_data.to_frame()

            chart_text = self._create_text_chart(equity_data)
            self.update(chart_text)

        except Exception as e:
            logger.error(f"Failed to generate equity chart: {e}")
            self.update(f"Chart generation failed: {_e(str(e))}")

    def _create_text_chart(self, data: pd.DataFrame) -> RenderableType:
        """Create a simple text representation of equity curves"""
        if data.empty:
            return "No data available for chart"

        table = Table(title="[bold]Equity Summary[/]", border_style=QUBX_DARK.variables["border-green"])
        table.add_column("Strategy", style=QUBX_DARK.secondary)
        table.add_column("Start Return", style="dim")
        table.add_column("End Return", style=QUBX_DARK.primary)
        table.add_column("Max Return", style=QUBX_DARK.accent)
        table.add_column("Min Return", style=QUBX_DARK.error)

        for col in data.columns:
            series = data[col].dropna()
            if not series.empty:
                table.add_row(
                    str(col),
                    f"{series.iloc[0]:.2%}",
                    f"{series.iloc[-1]:.2%}",
                    f"{series.max():.2%}",
                    f"{series.min():.2%}",
                )

        return table


class BacktestBrowserApp(App):
    """Main TUI application for browsing backtest results"""

    DARK = True

    BINDINGS = [
        Binding("j", "vim_down", "Down", show=False),
        Binding("k", "vim_up", "Up", show=False),
        Binding("h", "vim_left", "Collapse/Left", show=False),
        Binding("l", "vim_right", "Expand/Right", show=False),
        Binding("G", "vim_end", "End", show=False),
        Binding("g", "vim_home", "Home", show=False),
    ]

    CSS = """
    /* - custom variable not in Textual's built-in theme token set;
       must be declared here so the CSS parser resolves it at parse time.
       Value is kept in sync with QUBX_DARK.variables["border-green"]. */
    $border-green: #1e3a1e;

    Screen { background: $background; }

    Header { background: $background; color: $primary; text-style: bold; }

    Footer { background: $background; color: $text-muted; }

    .box { height: 1fr; }

    #tree-container {
        width: 30%;
        border: solid $border-green;
        background: $background;
    }

    #tree-container Label {
        color: $secondary;
        text-style: bold;
        padding: 0 1;
    }

    Tree { background: $background; color: $text-muted; }
    Tree > .tree--guides { color: $border-green; }
    Tree > .tree--guides-hover { color: $primary; }
    Tree > .tree--guides-selected { color: $accent; }

    Tree:focus .tree--cursor,
    Tree .tree--cursor:hover {
        background: $boost;
        color: $accent;
        text-style: bold;
    }

    #content-container {
        width: 70%;
        border: solid $border-green;
        background: $background;
    }

    #metrics-table { height: 40%; background: $background; }

    DataTable { background: $background; color: $foreground; }

    DataTable > .datatable--header {
        background: $panel;
        color: $primary;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: $boost;
        color: $accent;
    }

    DataTable > .datatable--even-row {
        background: $surface;
    }

    DataTable > .datatable--odd-row {
        background: $panel;
    }

    #preview-pane {
        height: 60%;
        background: $background;
        border-top: solid $border-green;
        padding: 0 1;
        overflow-y: auto;
    }

    Markdown {
        background: $background;
        color: $foreground;
    }

    Markdown .markdown-h1 {
        color: $accent;
        text-style: bold;
    }

    Markdown .markdown-h2 {
        color: $primary;
        text-style: bold;
        border-bottom: solid $border-green;
    }

    Markdown .markdown-h3 {
        color: $secondary;
        text-style: bold;
    }

    Markdown .markdown-table-header {
        background: $panel;
        color: $primary;
    }

    Markdown .markdown-table-odd-row {
        background: $surface;
    }

    Markdown .markdown-table-even-row {
        background: $panel;
    }

    Markdown .markdown-hr {
        color: $border-green;
    }

    Markdown .markdown-code-inline {
        color: $secondary;
        background: $panel;
    }

    Markdown .markdown-blockquote {
        color: $text-muted;
        border-left: solid $border-green;
    }

    #controls {
        height: 3;
        background: $surface;
        border-bottom: solid $border-green;
    }

    Button {
        background: $panel;
        color: $secondary;
        border: solid $border-green;
        margin: 0 1;
    }

    Button:hover {
        background: $boost;
        color: $accent;
        border: solid $secondary;
    }

    Button:focus {
        background: $boost;
        border: solid $primary;
    }
    """

    def __init__(self, root_path: str):
        super().__init__()
        self.root_path = root_path
        self.current_results: list[dict[str, Any]] = []
        self.current_storage: BacktestStorage | None = None
        self._selected_result: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        """Compose the TUI layout"""
        yield Header()

        with Horizontal():
            with Vertical(id="tree-container", classes="box"):
                yield Label("Backtest Results Tree")
                yield BacktestResultsTree(self.root_path)

            with Vertical(id="content-container", classes="box"):
                with Horizontal(id="controls"):
                    yield Button("Sort by Sharpe", id="sort-sharpe")
                    yield Button("Sort by CAGR", id="sort-cagr")
                    yield Button("Sort by Gain", id="sort-gain")
                    yield Button("Open HTML", id="open-html")
                    yield Button("Refresh", id="refresh")

                table = MetricsTable()
                table.id = "metrics-table"
                yield table

                preview = Markdown("*Select a result from the table to preview.*", id="preview-pane")
                yield preview

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app when mounted"""
        self.register_theme(QUBX_DARK)
        self.theme = QUBX_DARK.name
        self.title = f"Qubx Backtest Browser - {self.root_path}"

    @on(Tree.NodeSelected)
    def handle_tree_selection(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection"""
        if event.node.data:
            node_data = event.node.data
            self.current_results = node_data.backtest_results
            self.current_storage = node_data.storage

            if self.current_results:
                self._update_table()
                # - move focus to the table so arrow keys / vim keys work immediately
                self.query_one("#metrics-table", MetricsTable).focus()

    @on(DataTable.RowSelected)
    async def handle_row_selection(self, event: DataTable.RowSelected) -> None:
        """Handle table row selection — render markdown preview"""
        if not self.current_storage or not self.current_results:
            return

        load_id = event.row_key.value if event.row_key else None
        if not load_id:
            return

        # - find result by load_id (stored as row key in the table)
        result_info = next((r for r in self.current_results if r.get("load_id") == load_id), None)
        if not result_info:
            logger.warning(f"Could not find result for load_id: {load_id}")
            return

        self._selected_result = result_info

        preview = self.query_one("#preview-pane", Markdown)
        try:
            # - load portfolio data for the equity chart (best-effort: skip on error)
            chart: str | None = None
            try:
                portfolio_df = self.current_storage.get_portfolio(load_id)
                # - derive chart width from preview pane's actual rendered width:
                #   subtract CSS padding (2) + Markdown code-block indent (~4) +
                #   Y-axis label overhead (label_w=8 + " │" = 10)
                chart_width = max(40, preview.size.width - 16)
                chart = _equity_chart(portfolio_df, width=chart_width, height=12)
            except Exception as e:
                logger.debug(f"Could not load portfolio for equity chart: {e}")

            report = _build_markdown_report(result_info, equity_chart=chart)
            await preview.update(report)
        except Exception as e:
            logger.error(f"Failed to build preview: {e}")
            await preview.update(f"*Preview error: {e}*")

    @on(Button.Pressed, "#sort-sharpe")
    def sort_by_sharpe(self) -> None:
        """Sort results by Sharpe ratio"""
        self._sort_table("sharpe")

    @on(Button.Pressed, "#sort-cagr")
    def sort_by_cagr(self) -> None:
        """Sort results by CAGR"""
        self._sort_table("cagr")

    @on(Button.Pressed, "#sort-gain")
    def sort_by_gain(self) -> None:
        """Sort results by total gain"""
        self._sort_table("gain")

    @on(Button.Pressed, "#open-html")
    def open_html_tearsheet(self) -> None:
        """Open full HTML tearsheet in browser for the selected result"""
        if self._selected_result:
            self._generate_html_tearsheet(self._selected_result)

    @on(Button.Pressed, "#refresh")
    def refresh_data(self) -> None:
        """Refresh the data from backtest storage"""
        try:
            tree_container = self.query_one("#tree-container")
            old_tree = self.query_one(BacktestResultsTree)
            old_tree.remove()

            new_tree = BacktestResultsTree(self.root_path)
            tree_container.mount(new_tree)

            self.current_results = []
            self.current_storage = new_tree.storage
            self._selected_result = None

            table = self.query_one("#metrics-table", MetricsTable)
            table.clear(columns=True)

        except Exception as e:
            logger.error(f"Failed to refresh data: {e}")

    def _sort_table(self, metric: str) -> None:
        """Sort table by specified metric"""
        if self.current_results:
            table = self.query_one("#metrics-table", MetricsTable)
            try:
                table.sort(metric, reverse=True)
            except Exception as e:
                logger.warning(f"Failed to sort by {metric}: {e}")
                # - fallback: sort results dict and repopulate
                self.current_results.sort(key=lambda x: float(x.get(metric) or 0), reverse=True)
                self._update_table()

    def _update_table(self) -> None:
        """Update the metrics table"""
        table = self.query_one("#metrics-table", MetricsTable)
        table.populate_table(self.current_results)

    # ── vim navigation ───────────────────────────────────────────────────────

    def action_vim_down(self) -> None:
        focused = self.focused
        if focused is None:
            return
        if isinstance(focused, (DataTable, Tree)):
            focused.action_cursor_down()
        else:
            focused.scroll_down(animate=False)

    def action_vim_up(self) -> None:
        focused = self.focused
        if focused is None:
            return
        if isinstance(focused, (DataTable, Tree)):
            focused.action_cursor_up()
        else:
            focused.scroll_up(animate=False)

    def action_vim_left(self) -> None:
        """Collapse tree node / move to parent; scroll left elsewhere."""
        focused = self.focused
        if focused is None:
            return
        if isinstance(focused, Tree):
            focused.action_cursor_parent()  # - move to parent node (vim: h = go up/left)
        elif isinstance(focused, DataTable):
            focused.action_cursor_left()
        else:
            focused.scroll_left(animate=False)

    def action_vim_right(self) -> None:
        """Expand/toggle tree node; scroll right elsewhere."""
        focused = self.focused
        if focused is None:
            return
        if isinstance(focused, Tree):
            focused.action_toggle_node()  # - expand/collapse node (vim: l = go into)
        elif isinstance(focused, DataTable):
            focused.action_cursor_right()
        else:
            focused.scroll_right(animate=False)

    def action_vim_end(self) -> None:
        """Jump to last row / scroll to end (G)."""
        focused = self.focused
        if focused is None:
            return
        if isinstance(focused, DataTable):
            focused.action_scroll_bottom()
        elif isinstance(focused, Tree):
            focused.action_scroll_end()
        else:
            focused.scroll_end(animate=False)

    def action_vim_home(self) -> None:
        """Jump to first row / scroll to top (g)."""
        focused = self.focused
        if focused is None:
            return
        if isinstance(focused, DataTable):
            focused.action_scroll_top()
        elif isinstance(focused, Tree):
            focused.action_scroll_home()
        else:
            focused.scroll_home(animate=False)

    # ── tearsheet ────────────────────────────────────────────────────────────

    def _generate_html_tearsheet(self, result_info: dict[str, Any]) -> None:
        """Generate full HTML tearsheet and open in browser"""
        try:
            import tempfile
            import webbrowser

            load_id = result_info.get("load_id", "")
            if not load_id or not self.current_storage:
                return

            trading_result = self.current_storage.load(load_id)
            html_content = trading_result.to_html()

            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
                # - IPython HTML objects expose content via .data; plain strings do not
                try:
                    content_str = str(html_content.data)
                except AttributeError:
                    content_str = str(html_content)
                f.write(content_str)
                temp_file_path = f.name

            webbrowser.open(f"file://{temp_file_path}")

        except Exception as e:
            logger.error(f"Failed to generate HTML tearsheet: {e}")


def run_backtest_browser(root_path: str):
    """Run the backtest browser TUI application"""
    try:
        app = BacktestBrowserApp(root_path)
        app.run()
    except Exception as e:
        logger.error(f"Failed to start backtest browser: {e}")
        raise
