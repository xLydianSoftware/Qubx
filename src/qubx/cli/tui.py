import json
import re
from typing import Any

import pandas as pd
from rich.console import RenderableType
from rich.table import Table
from rich.text import Text as RichText
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.coordinate import Coordinate
from textual.timer import Timer
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
from qubx.utils.results import SimulationResultsSaver

# - matches YYYYMMDD_HHMMSS timestamp directory names produced by SimulationResultsSaver
_TS_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


def _fv(v: Any, default: float = 0.0) -> float:
    """
    Safely coerce a metadata value to float, substituting ``default`` for
    ``None``, ``NaN``, and anything that cannot be parsed as a number.

    ``float(nan) or 0`` does NOT work — NaN is truthy in Python, so the
    ``or 0`` fallback is never reached.  This helper handles that correctly.
    """
    import math

    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _fmt_variation_params(result: dict[str, Any]) -> str:
    """
    Format variation parameters for compact table display.

    Resolution order:
    1. ``variation_params`` JSON field  →  ``fp=6, sp=34``
    2. ``[bracketed]`` content inside ``name``  →  ``xxx_(fp=6,sp=34)``
    3. ``variation_id`` as last resort  →  ``var_003``
    """
    raw = str(result.get("variation_params") or "").strip()
    if raw:
        try:
            params = json.loads(raw)
            if isinstance(params, dict) and params:
                return _e(", ".join(f"{k}={v}" for k, v in params.items()))
        except Exception:
            pass

    # - fallback: extract the [bracketed] part from the name field
    # - e.g. "strat_fff_ggg.[xxx_(fp=6,sp=34)]"  →  "xxx_(fp=6,sp=34)"
    name = str(result.get("name") or "")
    m = re.search(r"\[([^\]]+)\]", name)
    if m:
        return _e(m.group(1))

    # - last resort: raw variation_id
    return _e(str(result.get("variation_id") or ""))


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
        # - backtest_ids pending lazy metadata load (populated at tree build, cleared after first load)
        self.pending_ids: list[str] = []

    def add_child(self, child: "BacktestTreeNode"):
        self.children.append(child)

    def add_backtest(self, result_info: dict[str, Any]):
        self.backtest_results.append(result_info)
        self.is_leaf = True


class BacktestResultsTree(Tree[BacktestTreeNode]):
    """Custom tree widget for backtest results"""

    def __init__(self, root_path: str, storage_options: dict | None = None):
        self.root_path = root_path
        self.storage_options = storage_options
        self.storage = BacktestStorage(root_path, storage_options=storage_options)
        self._root_data = BacktestTreeNode("Backtest Results", root_path, self.storage)
        super().__init__(self._root_data.name, data=self._root_data)

    def on_mount(self) -> None:
        # - show placeholder immediately so user sees the tree is loading
        self.root.add_leaf("⏳ Loading…")
        self._load_data()

    @work(thread=True)
    def _load_data(self) -> None:
        # - runs in background thread — does the expensive storage.search()
        self._build_tree_structure(self._root_data)
        # - marshal back to the UI thread to update the widget
        self.app.call_from_thread(self._finish_loading)

    def _finish_loading(self) -> None:
        # - called on the UI thread once background load completes
        self.root.remove_children()
        self._populate_tree(self.root, self._root_data)
        self.root.expand()

    def _build_tree_structure(self, root_node: BacktestTreeNode):
        """
        Build 2-level tree structure by listing file paths only — zero parquet reads.

        Uses DuckDB glob() TABLE function to enumerate filesystem/S3 entries
        without reading any parquet data.  Metadata loads lazily on first leaf node
        selection (see BacktestBrowserApp._load_node_metadata).

        Path layout (new format)::

            {base}/{ShortClass}/{config-name}/{YYYYMMDD_HHMMSS}/_metadata.parquet
            {base}/{ShortClass}/{config-name}/{YYYYMMDD_HHMMSS}/var_XXX/_metadata.parquet  (old var)
            {base}/{ShortClass}/{config-name}/{YYYYMMDD_HHMMSS}/_metadata.parquet  (new var, N rows)

        Tree structure produced::

            Root
            ├── ShortClass  (parent node, expandable)
            │   ├── config-name-A (N runs)   ← leaf, lazy-loads on click
            │   └── config-name-B (M runs)   ← leaf, lazy-loads on click
            └── AnotherClass
                └── another-config (K runs)

        Variation sets are deduplicated to a single timestamp-level entry.
        Falls back to flat display when paths have fewer than 2 components
        before the timestamp (e.g. old-format backtests).
        """
        glob_pat = self.storage._glob(SimulationResultsSaver.METADATA_FILE)
        try:
            # - glob() is a DuckDB TABLE function: zero parquet I/O, just path listing
            paths_df = self.storage._conn.execute(f"SELECT file FROM glob('{glob_pat}')").df()
        except Exception as e:
            logger.warning(f"Failed to list backtest paths: {e}")
            return

        if paths_df.empty:
            return

        base = self.storage.base_path  # - always ends with /

        # - 2-level grouping: class_key → {config_name → set[run_id]}
        groups: dict[str, dict[str, set[str]]] = {}

        for file_path in paths_df["file"]:
            rel = str(file_path).removeprefix(base).strip("/")
            parts = rel.split("/")

            # - find the rightmost YYYYMMDD_HHMMSS component — this is the run root
            ts_idx = next(
                (i for i in range(len(parts) - 1, -1, -1) if _TS_DIR_RE.match(parts[i])),
                None,
            )
            if ts_idx is None:
                continue

            # - run_id = up to and including the timestamp dir (deduplicates var_XXX entries)
            run_id = "/".join(parts[: ts_idx + 1])

            if ts_idx >= 2:
                # - new format: ...prefix/ShortClass/config-name/timestamp
                # - config_name is immediately before the timestamp dir
                config_name = parts[ts_idx - 1]
                # - class_key is everything before config_name (may include extra path prefix)
                class_key = "/".join(parts[: ts_idx - 1])
            elif ts_idx == 1:
                # - only one component before timestamp — treat as flat (no class prefix)
                config_name = parts[0]
                class_key = ""
            else:
                # - timestamp at root level — shouldn't happen, skip
                continue

            groups.setdefault(class_key, {}).setdefault(config_name, set()).add(run_id)

        for class_key, config_groups in sorted(groups.items()):
            # - display name for the class node = last component of class_key (the ShortClass)
            class_display = class_key.split("/")[-1] if class_key else "root"
            total_runs = sum(len(run_ids) for run_ids in config_groups.values())

            class_node = BacktestTreeNode(f"{class_display} ({total_runs})", class_key, self.storage)
            # - class nodes are not leaves — they expand to show config names
            class_node.is_leaf = False

            for config_name, run_ids in sorted(config_groups.items()):
                leaf_path = f"{class_key}/{config_name}" if class_key else config_name
                leaf = BacktestTreeNode(f"{config_name} ({len(run_ids)})", leaf_path, self.storage)
                leaf.is_leaf = True
                # - newest-first; timestamp dirs are zero-padded so lexicographic sort works
                leaf.pending_ids = sorted(run_ids, reverse=True)
                class_node.add_child(leaf)

            root_node.add_child(class_node)

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

        # - detect whether this result set contains variation rows;
        # - if so, prepend a Params column instead of Name/Strategy
        # - (Name and Strategy are already visible in the left tree)
        has_variations = any(r.get("is_variation") for r in results_data)

        columns: list[str] = [
            "Gain",
            "CAGR",
            "Sharpe",
            "QR",
            "Max DD %",
            "MDD USD",
            "Fees",
            "Executions",
            "Created",
            "Author",
        ]
        if has_variations:
            columns.append("Params")

        for col in columns:
            self.add_column(col, key=col.lower().replace(" ", "_"))

        for result in results_data:
            gain_val = _fv(result.get("gain"))
            # - negative gain → dim red row; positive → default style
            row_style = "red" if gain_val < 0 else ""

            def _c(s: str) -> RichText:
                # - RichText doesn't parse markup so no _e() escaping needed
                return RichText(s, style=row_style)

            row_data: list[RichText] = [
                _c(f"{gain_val:.3f}"),
                _c(f"{_fv(result.get('cagr')):.3f}"),
                _c(f"{_fv(result.get('sharpe')):.3f}"),
                _c(f"{_fv(result.get('qr')):.3f}"),
                _c(f"{_fv(result.get('mdd_pct')):.3f}"),
                _c(f"{_fv(result.get('mdd_usd')):.3f}"),
                _c(f"{_fv(result.get('fees')):.3f}"),
                _c(f"{_fv(result.get('execs')):.0f}"),
                _c(pd.Timestamp(result.get("creation_time", "")).strftime("%Y-%m-%d %H:%M")),
                _c(str(result.get("author", ""))),
            ]
            if has_variations:
                row_data.append(_c(_fmt_variation_params(result)))
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
        Binding("q", "quit", "Quit", show=True),
        Binding("j", "vim_down", "Down", show=False),
        Binding("k", "vim_up", "Up", show=False),
        Binding("g", "vim_home", "Top", show=False),
        Binding("h", "vim_left", "◀ Tree", show=True),
        Binding("l", "vim_right", "▶ Table", show=True),
        Binding("S", "sort_by_key_sharpe", "Sort Sharpe", show=True),
        Binding("C", "sort_by_key_cagr", "Sort CAGR", show=True),
        Binding("G", "vim_end", "Sort Gain", show=True),  # - Tree: scroll end; Table: sort gain
        Binding("T", "sort_by_key_created", "Sort Created", show=True),
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

    def __init__(self, root_path: str, storage_options: dict | None = None):
        super().__init__()
        self.root_path = root_path
        self.storage_options = storage_options
        self.current_results: list[dict[str, Any]] = []
        self.current_storage: BacktestStorage | None = None
        self._selected_result: dict[str, Any] | None = None
        # - tracks which node is currently being loaded to discard stale callbacks
        self._loading_node: BacktestTreeNode | None = None
        # - cc (copy path) state machine
        self._pending_copy: bool = False
        self._pending_copy_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Compose the TUI layout"""
        yield Header()

        with Horizontal():
            with Vertical(id="tree-container", classes="box"):
                yield Label("Backtest Results Tree")
                yield BacktestResultsTree(self.root_path, storage_options=self.storage_options)

            with Vertical(id="content-container", classes="box"):
                with Horizontal(id="controls"):
                    yield Button("Sort by Sharpe", id="sort-sharpe")
                    yield Button("Sort by CAGR", id="sort-cagr")
                    yield Button("Sort by Gain", id="sort-gain")
                    yield Button("Sort by Created", id="sort-created")
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
        """Handle tree node selection — show cached results or trigger lazy metadata load."""
        if not event.node.data:
            return

        node_data: BacktestTreeNode = event.node.data
        self.current_storage = node_data.storage

        if node_data.backtest_results:
            # - already loaded: show immediately (cached after first click)
            self.current_results = node_data.backtest_results
            self._sort_table("creation_time")
            self.query_one("#metrics-table", MetricsTable).focus()

        elif node_data.pending_ids:
            # - first selection: show spinner on the table then kick off background load
            self._loading_node = node_data
            table = self.query_one("#metrics-table", MetricsTable)
            table.clear(columns=True)
            table.loading = True  # - built-in Textual spinner overlay on the table widget
            self._load_node_metadata(node_data)

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

    @work(thread=True)
    def _load_node_metadata(self, node_data: BacktestTreeNode) -> None:
        """
        Background worker: read parquet metadata for this group only.

        Uses a group-scoped glob  (ShortClass/name/**/_metadata.parquet) so DuckDB
        scans only this config's files — never the full storage tree.

        Handles both layouts transparently:
        - Regular run:   timestamp/_metadata.parquet          (1 row each)
        - Old variation: timestamp/var_XXX/_metadata.parquet  (1 row per var)
        - New variation: timestamp/_metadata.parquet          (N rows)
        """
        storage = node_data.storage
        if not storage:
            return

        # - targeted glob scoped to this group's subdirectory only
        group_glob = f"{storage.base_path}{node_data.path}/**/{SimulationResultsSaver.METADATA_FILE}"

        try:
            df = storage._conn.execute(
                f"SELECT * FROM read_parquet('{group_glob}', union_by_name=true) ORDER BY creation_time DESC"
            ).df()
        except Exception as e:
            logger.warning(f"Failed to load metadata for '{node_data.path}': {e}")
            df = pd.DataFrame()

        results: list[dict[str, Any]] = []
        if not df.empty and "backtest_id" in df.columns:
            df = df.drop_duplicates(subset=["backtest_id"])
            for _, row in df.iterrows():
                row_dict: dict[str, Any] = {str(k): v for k, v in row.items()}
                row_dict["load_id"] = row_dict["backtest_id"]
                results.append(row_dict)

        node_data.backtest_results = results
        node_data.pending_ids = []  # - free memory; signals "already loaded"
        self.app.call_from_thread(self._on_node_metadata_loaded, node_data)

    def _on_node_metadata_loaded(self, node_data: BacktestTreeNode) -> None:
        """UI-thread callback: update table after metadata load completes."""
        # - discard if user already navigated to a different node
        if node_data is not self._loading_node:
            return
        self._loading_node = None

        table = self.query_one("#metrics-table", MetricsTable)
        table.loading = False  # - dismiss spinner regardless of outcome

        self.current_results = node_data.backtest_results
        if self.current_results:
            self._sort_table("creation_time")
            table.focus()
        else:
            self.call_later(
                self.query_one("#preview-pane", Markdown).update,
                "*No completed results found for this group.*",
            )

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

    @on(Button.Pressed, "#sort-created")
    def sort_by_created(self) -> None:
        """Sort results by creation time"""
        self._sort_table("creation_time")

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

            new_tree = BacktestResultsTree(self.root_path, storage_options=self.storage_options)
            tree_container.mount(new_tree)

            self.current_results = []
            self.current_storage = new_tree.storage
            self._selected_result = None

            table = self.query_one("#metrics-table", MetricsTable)
            table.clear(columns=True)

        except Exception as e:
            logger.error(f"Failed to refresh data: {e}")

    def _sort_table(self, metric: str) -> None:
        """
        Sort current results by metric (descending) and repopulate the table.

        Always sorts the underlying dict list rather than using Textual's
        built-in DataTable.sort() — table cells are formatted strings so the
        built-in sort would compare lexicographically and cannot handle NaN.
        ``_fv`` substitutes 0.0 for NaN / None so those rows sink to the bottom.
        """
        if self.current_results:
            if metric == "creation_time":
                # - timestamps cannot go through _fv (float coercion); use pd.Timestamp
                self.current_results.sort(
                    key=lambda x: pd.Timestamp(x.get("creation_time") or "1970-01-01"),
                    reverse=True,
                )
            else:
                self.current_results.sort(key=lambda x: _fv(x.get(metric)), reverse=True)
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
        """In tree: collapse / go to parent.  In right pane: move focus back to tree."""
        focused = self.focused
        if focused is None:
            return
        if isinstance(focused, Tree):
            focused.action_cursor_parent()
        else:
            # - h from DataTable, Markdown preview, or any other right-pane widget
            self.query_one(BacktestResultsTree).focus()

    def action_vim_right(self) -> None:
        """Tree: expand node (l).  Right pane: cursor right / scroll right."""
        focused = self.focused
        if focused is None:
            return
        if isinstance(focused, Tree):
            # - if the node is a leaf (has results / pending), move focus to the table;
            # - otherwise expand/collapse the class-level parent node
            node = focused.cursor_node
            if node and node.data and (node.data.pending_ids or node.data.backtest_results):
                self.query_one("#metrics-table", MetricsTable).focus()
            else:
                focused.action_toggle_node()
        elif isinstance(focused, DataTable):
            focused.action_cursor_right()
        else:
            focused.scroll_right(animate=False)

    def action_vim_end(self) -> None:
        """Tree: scroll to end (G).  Table: sort by gain (G = Shift+G)."""
        focused = self.focused
        if focused is None:
            return
        if isinstance(focused, DataTable):
            # - G in right pane → sort by gain (more useful than scroll-to-bottom)
            self._sort_table("gain")
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

    def action_sort_by_key_sharpe(self) -> None:
        """Sort table by Sharpe ratio (S — right pane only)."""
        if not isinstance(self.focused, Tree):
            self._sort_table("sharpe")

    def action_sort_by_key_cagr(self) -> None:
        """Sort table by CAGR (C — right pane only)."""
        if not isinstance(self.focused, Tree):
            self._sort_table("cagr")

    def action_sort_by_key_created(self) -> None:
        """Sort table by creation time (T — right pane only)."""
        if not isinstance(self.focused, Tree):
            self._sort_table("creation_time")

    # ── cc copy path ─────────────────────────────────────────────────────────

    def on_key(self, event) -> None:
        """Handle key events for multi-key bindings (cc = copy path to clipboard)."""
        if event.key == "c":
            if not isinstance(self.focused, DataTable):
                # - only intercept c when DataTable is focused
                return
            if self._pending_copy:
                # - second 'c' within timeout: complete the copy
                self._pending_copy = False
                if self._pending_copy_timer:
                    self._pending_copy_timer.stop()
                    self._pending_copy_timer = None
                self._copy_current_path()
                event.prevent_default()
                event.stop()
            else:
                # - first 'c': enter pending state, start timeout
                self._pending_copy = True
                self._pending_copy_timer = self.set_timer(
                    0.5, self._cancel_pending_copy, name="cc_timeout"
                )
                event.prevent_default()
                event.stop()
        elif self._pending_copy:
            # - any other key cancels the pending copy
            self._cancel_pending_copy()

    def _cancel_pending_copy(self) -> None:
        """Cancel the pending cc copy operation (timeout or different key)."""
        self._pending_copy = False
        if self._pending_copy_timer:
            self._pending_copy_timer.stop()
            self._pending_copy_timer = None

    def _copy_current_path(self) -> None:
        """Copy the path of the currently highlighted table row to clipboard via OSC 52."""
        table = self.query_one("#metrics-table", MetricsTable)
        if table.row_count == 0:
            return

        try:
            cursor_row = table.cursor_row
            cell_key = table.coordinate_to_cell_key(Coordinate(cursor_row, 0))
            load_id = cell_key.row_key.value
        except Exception:
            self.notify("No row selected", severity="warning", timeout=2)
            return

        if not load_id:
            return

        # - load_id is the backtest_id = relative path from storage root
        self.copy_to_clipboard(str(load_id))
        self.notify(f"Copied: {load_id}", timeout=3)

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


def run_backtest_browser(root_path: str, storage_options: dict | None = None):
    """Run the backtest browser TUI application"""
    try:
        app = BacktestBrowserApp(root_path, storage_options=storage_options)
        app.run()
    except Exception as e:
        logger.error(f"Failed to start backtest browser: {e}")
        raise
