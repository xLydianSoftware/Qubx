import webbrowser
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
from rich.console import RenderableType
from rich.table import Table
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
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
    """Data table for displaying backtest metrics with row selection"""

    def __init__(self):
        super().__init__()
        self.zebra_stripes = True
        self.cursor_type = "row"

    def populate_table(self, results_data: list[dict[str, Any]]):
        """Populate table with backtest results"""
        if not results_data:
            return

        # Clear existing data and columns
        self.clear(columns=True)

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

        # Add columns only if not already present
        for col in columns:
            column_key = col.lower().replace(" ", "_")
            self.add_column(col, key=column_key)

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
            # Validate that all results are TradingSessionResult objects
            for i, result in enumerate(self.results):
                if not hasattr(result, "portfolio_log"):
                    logger.error(f"Result {i} ({type(result)}) does not have portfolio_log attribute")
                    self.update(f"Invalid result type at index {i}: {type(result)}")
                    return

            # Get equity data for all results
            equity_data = get_cum_pnl(self.results, timeframe="1h")

            if isinstance(equity_data, pd.Series):
                equity_data = equity_data.to_frame()

            # Create a simple text-based chart representation
            chart_text = self._create_text_chart(equity_data)
            self.update(chart_text)

        except Exception as e:
            logger.error(f"Failed to generate equity chart: {e}")
            self.update(f"Chart generation failed: {e}\nResults types: {[type(r) for r in self.results]}")

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


class TearsheetViewer(Static):
    """Widget for displaying tearsheet information"""

    def __init__(self):
        super().__init__()
        self.update(
            "Select a backtest result from the table to view its tearsheet.\nClick on any row to generate and open the tearsheet in your browser."
        )

    def show_tearsheet_info(self, result_name: str, file_path: str):
        """Show information about generated tearsheet"""
        self.update(
            f"Tearsheet generated for: {result_name}\n\nFile saved to: {file_path}\n\nThe tearsheet has been opened in your default browser.\nIf it didn't open automatically, you can manually open the file above."
        )


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
        height: 70%;
    }
    
    #tearsheet-info {
        height: 30%;
    }
    
    #controls {
        height: 3;
        background: $surface;
    }
    """

    def __init__(self, root_path: str):
        super().__init__()
        self.root_path = root_path
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
                    yield Button("Sort by Sharpe", id="sort-sharpe")
                    yield Button("Sort by CAGR", id="sort-cagr")
                    yield Button("Sort by Gain", id="sort-gain")
                    yield Button("Refresh", id="refresh")

                table = MetricsTable()
                table.id = "metrics-table"
                yield table

                tearsheet_viewer = TearsheetViewer()
                tearsheet_viewer.id = "tearsheet-info"
                yield tearsheet_viewer

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app when mounted"""
        self.title = f"Qubx Backtest Browser - {self.root_path}"

    @on(Tree.NodeSelected)
    def handle_tree_selection(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection"""
        if event.node.data:
            node_data = event.node.data
            self.current_results = node_data.backtest_results
            self.current_manager = node_data.manager

            if self.current_results:
                self._update_table()

    @on(DataTable.RowSelected)
    def handle_row_selection(self, event: DataTable.RowSelected) -> None:
        """Handle table row selection to show tearsheet"""
        if not self.current_manager or not self.current_results:
            return

        # Get the selected row key (which corresponds to the result name)
        row_key = event.row_key.value if event.row_key else None
        if not row_key:
            return

        # Find the corresponding result info
        result_info = None
        for result in self.current_results:
            if result.get("name") == row_key:
                result_info = result
                break

        if not result_info:
            logger.warning(f"Could not find result info for row key: {row_key}")
            return

        self._generate_tearsheet(result_info)

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

    @on(Button.Pressed, "#refresh")
    def refresh_data(self) -> None:
        """Refresh the data from the backtest manager"""
        try:
            # Get the current tree widget and its container
            tree_container = self.query_one("#tree-container")
            old_tree = self.query_one(BacktestResultsTree)

            # Remove the old tree
            old_tree.remove()

            # Create a new tree with refreshed data
            new_tree = BacktestResultsTree(self.root_path)
            tree_container.mount(new_tree)

            # Clear current results to force reselection
            self.current_results = []
            self.current_manager = new_tree.manager

            # Clear the table
            table = self.query_one("#metrics-table", MetricsTable)
            table.clear(columns=True)

            # Update tearsheet viewer
            tearsheet_viewer = self.query_one("#tearsheet-info", TearsheetViewer)
            tearsheet_viewer.update("Data refreshed. Select a strategy from the tree to view results.")

        except Exception as e:
            logger.error(f"Failed to refresh data: {e}")
            tearsheet_viewer = self.query_one("#tearsheet-info", TearsheetViewer)
            tearsheet_viewer.update(f"Failed to refresh data: {e}")

    def _sort_table(self, metric: str) -> None:
        """Sort table by specified metric"""
        if self.current_results:
            # Use DataTable's built-in sorting by column key
            column_key = metric.lower().replace(" ", "_")
            table = self.query_one("#metrics-table", MetricsTable)
            try:
                table.sort(column_key, reverse=True)  # Sort descending for better metrics first
            except Exception as e:
                logger.warning(f"Failed to sort by {metric}: {e}")
                # Fallback to manual sorting and repopulation
                self.current_results.sort(key=lambda x: x.get("performance", {}).get(metric, 0), reverse=True)
                self._update_table()

    def _update_table(self) -> None:
        """Update the metrics table"""
        table = self.query_one("#metrics-table", MetricsTable)
        table.populate_table(self.current_results)

    def _generate_tearsheet(self, result_info: dict[str, Any]) -> None:
        """Generate tearsheet for selected result"""
        try:
            import tempfile
            import webbrowser

            name = result_info.get("name", "")
            display_name = result_info.get("display_name", name)

            if not name or not self.current_manager:
                return

            # Load the trading result
            trading_result = self.current_manager.load(name)
            if isinstance(trading_result, list):
                trading_result = trading_result[0]  # Take first if multiple

            # Generate HTML tearsheet
            html_content = trading_result.to_html()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
                # Handle different types of HTML content and ensure it's a string
                if hasattr(html_content, "data"):
                    content_str = str(html_content.data)
                else:
                    content_str = str(html_content)
                f.write(content_str)
                temp_file_path = f.name

            # Open in browser
            webbrowser.open(f"file://{temp_file_path}")

            # Update the tearsheet viewer
            tearsheet_viewer = self.query_one("#tearsheet-info", TearsheetViewer)
            tearsheet_viewer.show_tearsheet_info(display_name, temp_file_path)

        except Exception as e:
            logger.error(f"Failed to generate tearsheet: {e}")
            tearsheet_viewer = self.query_one("#tearsheet-info", TearsheetViewer)
            tearsheet_viewer.update(f"Failed to generate tearsheet: {e}")


def run_backtest_browser(root_path: str):
    """Run the backtest browser TUI application"""
    try:
        app = BacktestBrowserApp(root_path)
        app.run()
    except Exception as e:
        logger.error(f"Failed to start backtest browser: {e}")
        raise
