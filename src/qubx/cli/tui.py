import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    Placeholder,
    Static,
    Tree,
)

from qubx import logger
from qubx.backtester import BacktestsResultsManager
from qubx.core.metrics import TradingSessionResult, get_cum_pnl


class BacktestTreeNode:
    """Represents a node in the backtest tree structure"""

    def __init__(self, name: str, path: str | None = None, manager: BacktestsResultsManager | None = None):
        self.name = name
        self.path = path
        self.manager = manager
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
        self.manager = BacktestsResultsManager(root_path)

        # Build tree structure from backtest results
        root_node = BacktestTreeNode("Backtest Results", root_path, self.manager)
        self._build_tree_structure(root_node)

        super().__init__(root_node.name, data=root_node)
        self._populate_tree(self.root, root_node)

    def _build_tree_structure(self, root_node: BacktestTreeNode):
        """Build tree structure from backtest manager results"""
        # Group by strategy class for organization
        strategy_groups: dict[str, list[dict]] = {}

        for name, info in self.manager.results.items():
            strategy_class = info.get("strategy_class", "Unknown").split(".")[-1]
            if strategy_class not in strategy_groups:
                strategy_groups[strategy_class] = []
            info["display_name"] = name
            strategy_groups[strategy_class].append(info)

        # Create strategy nodes
        for strategy_name, results in strategy_groups.items():
            strategy_node = BacktestTreeNode(f"{strategy_name} ({len(results)} results)", strategy_name, self.manager)
            root_node.add_child(strategy_node)

            for result in results:
                strategy_node.add_backtest(result)

        # Add variations if any
        for idx, var_info in self.manager.variations.items():
            var_node = BacktestTreeNode(
                f"📊 {var_info['name']} ({len(var_info['variations'])} variations)", var_info["name"], self.manager
            )
            root_node.add_child(var_node)

            for variation in var_info["variations"]:
                var_node.add_backtest(variation)

    def _populate_tree(self, tree_node, data_node: BacktestTreeNode):
        """Populate the textual tree with data nodes"""
        for child in data_node.children:
            child_tree_node = tree_node.add(child.name, data=child)
            if child.children:
                self._populate_tree(child_tree_node, child)


class MetricsTable(DataTable):
    """Data table for displaying backtest metrics"""

    def __init__(self):
        super().__init__()
        self.zebra_stripes = True
        self.cursor_type = "row"

    def populate_table(self, results_data: list[dict[str, Any]]):
        """Populate table with backtest results"""
        if not results_data:
            return

        # Clear existing data
        self.clear()

        # Define columns based on metrics
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

        # Add rows
        for result in results_data:
            performance = result.get("performance", {})
            row_data = [
                result.get("display_name", result.get("name", "Unknown")),
                result.get("strategy_class", "").split(".")[-1],
                pd.Timestamp(result.get("creation_time", "")).strftime("%Y-%m-%d %H:%M"),
                result.get("author", ""),
                f"{performance.get('gain', 0):.3f}",
                f"{performance.get('cagr', 0):.3f}",
                f"{performance.get('sharpe', 0):.3f}",
                f"{performance.get('qr', 0):.3f}",
                f"{performance.get('max_dd_pct', 0):.3f}",
                f"{performance.get('mdd_usd', 0):.3f}",
                f"{performance.get('fees', 0):.3f}",
                f"{performance.get('execs', 0):.0f}",
            ]
            self.add_row(*row_data, key=result.get("name", ""))

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
            # Get equity data for all results
            equity_data = get_cum_pnl(self.results, timeframe="1h")

            if isinstance(equity_data, pd.Series):
                equity_data = equity_data.to_frame()

            # Create a simple text-based chart representation
            chart_text = self._create_text_chart(equity_data)
            self.update(chart_text)

        except Exception as e:
            logger.warning(f"Failed to generate equity chart: {e}")
            self.update(f"Chart generation failed: {e}")

    def _create_text_chart(self, data: pd.DataFrame) -> RenderableType:
        """Create a simple text representation of equity curves"""
        if data.empty:
            return "No data available for chart"

        # Create a simple summary table instead of complex chart
        table = Table(title="Equity Summary")
        table.add_column("Strategy", style="cyan")
        table.add_column("Start Return", style="green")
        table.add_column("End Return", style="green")
        table.add_column("Max Return", style="green")
        table.add_column("Min Return", style="red")

        for col in data.columns:
            series = data[col].dropna()
            if not series.empty:
                table.add_row(
                    str(col),
                    f"{series.iloc[0]:.2%}" if len(series) > 0 else "N/A",
                    f"{series.iloc[-1]:.2%}" if len(series) > 0 else "N/A",
                    f"{series.max():.2%}" if len(series) > 0 else "N/A",
                    f"{series.min():.2%}" if len(series) > 0 else "N/A",
                )

        return table


class BacktestBrowserApp(App):
    """Main TUI application for browsing backtest results"""

    CSS = """
    .box {
        height: 1fr;
    }
    
    #tree-container {
        width: 30%;
        border: solid $primary;
    }
    
    #content-container {
        width: 70%;
        border: solid $primary;
    }
    
    #metrics-table {
        height: 60%;
    }
    
    #equity-chart {
        height: 40%;
    }
    
    #controls {
        height: 3;
        background: $surface;
    }
    """

    def __init__(self, root_path: str):
        super().__init__()
        self.root_path = root_path
        self.current_view = "table"  # "table" or "chart"
        self.current_results: list[dict[str, Any]] = []
        self.current_manager: BacktestsResultsManager | None = None

    def compose(self) -> ComposeResult:
        """Compose the TUI layout"""
        yield Header()

        with Horizontal():
            with Vertical(id="tree-container", classes="box"):
                yield Label("Backtest Results Tree")
                yield BacktestResultsTree(self.root_path)

            with Vertical(id="content-container", classes="box"):
                with Horizontal(id="controls"):
                    yield Button("Table View", id="table-btn", variant="primary")
                    yield Button("Chart View", id="chart-btn")
                    yield Button("Sort by Sharpe", id="sort-sharpe")
                    yield Button("Sort by CAGR", id="sort-cagr")
                    yield Button("Sort by Gain", id="sort-gain")

                table = MetricsTable()
                table.id = "metrics-table"
                yield table

                chart = EquityChart()
                chart.id = "equity-chart"
                yield chart

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app when mounted"""
        self.title = f"Qubx Backtest Browser - {self.root_path}"
        # Hide chart initially
        self.query_one("#equity-chart").display = False

    @on(Tree.NodeSelected)
    def handle_tree_selection(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection"""
        if event.node.data:
            node_data = event.node.data
            self.current_results = node_data.backtest_results
            self.current_manager = node_data.manager

            if self.current_results:
                self._update_content_view()

    @on(Button.Pressed, "#table-btn")
    def show_table_view(self) -> None:
        """Switch to table view"""
        self.current_view = "table"
        self._update_content_view()
        # Note: button variant changes may not be supported in all textual versions

    @on(Button.Pressed, "#chart-btn")
    def show_chart_view(self) -> None:
        """Switch to chart view"""
        self.current_view = "chart"
        self._update_content_view()
        # Note: button variant changes may not be supported in all textual versions

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

    def _sort_table(self, metric: str) -> None:
        """Sort table by specified metric"""
        if self.current_view == "table" and self.current_results:
            # Sort current results by metric
            self.current_results.sort(key=lambda x: x.get("performance", {}).get(metric, 0), reverse=True)
            self._update_table()

    def _update_content_view(self) -> None:
        """Update the content area based on current view"""
        if self.current_view == "table":
            self.query_one("#metrics-table").display = True
            self.query_one("#equity-chart").display = False
            self._update_table()
        else:
            self.query_one("#metrics-table").display = False
            self.query_one("#equity-chart").display = True
            self._update_chart()

    def _update_table(self) -> None:
        """Update the metrics table"""
        table = self.query_one("#metrics-table", MetricsTable)
        table.populate_table(self.current_results)

    def _update_chart(self) -> None:
        """Update the equity chart"""
        if not self.current_manager or not self.current_results:
            return

        try:
            # Load actual TradingSessionResult objects for charting
            results = []
            for result_info in self.current_results[:10]:  # Limit to first 10 for performance
                try:
                    name = result_info.get("name", "")
                    if name:
                        trading_result = self.current_manager.load(name)
                        results.append(trading_result)
                except Exception as e:
                    logger.warning(f"Failed to load result {name}: {e}")

            chart = self.query_one("#equity-chart", EquityChart)
            chart.set_results(results)

        except Exception as e:
            logger.error(f"Failed to update chart: {e}")


def run_backtest_browser(root_path: str):
    """Run the backtest browser TUI application"""
    try:
        app = BacktestBrowserApp(root_path)
        app.run()
    except Exception as e:
        logger.error(f"Failed to start backtest browser: {e}")
        raise
