import threading
import time
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from dash import Dash, ctx, dcc, html
from dash._jupyter import JupyterDisplayMode
from dash.dependencies import Input, Output
from IPython.display import clear_output
from plotly.subplots import make_subplots

from qubx import QubxLogConfig, logger
from qubx.backtester.simulator import SimulatedBroker
from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.core.basics import Instrument
from qubx.core.interfaces import IStrategy, IStrategyContext
from qubx.core.lookups import lookup
from qubx.core.series import OrderBook, TimeSeries
from qubx.pandaz import scols
from qubx.utils.charting.lookinglass import LookingGlass
from qubx.utils.runner import get_account_config

pio.templates.default = "plotly_dark"

TIMEFRAMES = ["1s", "1m", "5m", "15m", "1h", "4h", "1d"]


class TradingDashboard:
    ctx: IStrategyContext
    max_history: int

    def __init__(self, ctx: IStrategyContext, max_history: int = 10_000):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.ctx = ctx
        self.max_history = max_history
        self._symbol_to_instrument = {instr.symbol: instr for instr in ctx.instruments}

        # Setup layout with dark theme
        self.app.layout = html.Div(
            [
                html.H2("Trading Dashboard"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Symbol:"),
                                dcc.Dropdown(
                                    id="symbol-dropdown",
                                    options=[
                                        {"label": instr.symbol, "value": instr.symbol} for instr in self.ctx.instruments
                                    ],
                                    value=self.ctx.instruments[0].symbol,
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Timeframe:"),
                                dcc.Dropdown(
                                    id="timeframe",
                                    options=[{"label": tf, "value": tf} for tf in TIMEFRAMES],
                                    value="1s",
                                ),
                            ],
                            width=1,
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Pause", id="play-pause-button", color="primary", className="ms-2", n_clicks=0
                                ),
                            ],
                            width=1,
                        ),
                    ],
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="live-graph",
                        ),
                    ],
                ),
                dcc.Interval(id="interval-component", interval=5 * 1000, n_intervals=0, disabled=False),
            ],
            className="dash-bootstrap",
        )

        @self.app.callback(
            Output("interval-component", "disabled"),
            Output("play-pause-button", "children"),
            Input("play-pause-button", "n_clicks"),
            Input("interval-component", "disabled"),
        )
        def toggle_updates(n_clicks, disabled):
            if n_clicks > 0:
                disabled = not disabled
            return disabled, "Resume" if disabled else "Pause"

        @self.app.callback(
            Output("live-graph", "figure"),
            [
                Input("interval-component", "n_intervals"),
                Input("symbol-dropdown", "value"),
                Input("timeframe", "value"),
            ],
        )
        def update_graph(n: int, symbol: str, timeframe: str):
            if not self.ctx.is_running() or not self.ctx.is_fitted():
                logger.info(f"Strategy running: {self.ctx.is_running()}, Strategy fitted: {self.ctx.is_fitted()}")
                return {}

            instrument = self.ctx.query_instrument(symbol, "BINANCE.UM")
            if instrument is None:
                logger.error(f"Could not find instrument for symbol: {symbol}")
                return {}

            ohlc = self.ctx.ohlc(instrument, timeframe).loc[-self.max_history :]
            key_to_ind = self.ctx.strategy[symbol]  # type: ignore
            indicators = {key: ind.pd() for key, ind in key_to_ind.items()}
            fig = (
                LookingGlass(
                    ohlc,
                    indicators,
                    master_plot_height=800,
                    study_plot_height=100,
                )
                .look(title="")
                .hover(h=900)
            )
            return fig

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8050,
        mode: JupyterDisplayMode = "external",
        debug: bool = False,
        use_reloader: bool = False,
        **kwargs,
    ):
        self.app.run(
            debug=debug, host=host, port=str(port), jupyter_mode=mode, dev_tools_hot_reload=use_reloader, **kwargs
        )
