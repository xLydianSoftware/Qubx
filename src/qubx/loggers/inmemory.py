from typing import Any

import pandas as pd

from qubx import logger
from qubx.core.loggers import LogsWriter
from qubx.core.metrics import split_cumulative_pnl
from qubx.pandaz.utils import scols


class InMemoryLogsWriter(LogsWriter):
    _portfolio: list[dict[str, Any]]
    _execs: list[dict[str, Any]]
    _signals: list[dict[str, Any]]
    _targets: list[dict[str, Any]]

    def __init__(self, account_id: str, strategy_id: str, run_id: str) -> None:
        super().__init__(account_id, strategy_id, run_id)
        self._portfolio = []
        self._execs = []
        self._signals = []
        self._targets = []

    def write_data(self, log_type: str, data: list[dict[str, Any]]):
        if len(data) > 0:
            match log_type:
                case "portfolio":
                    self._portfolio.extend(data)
                case "executions":
                    self._execs.extend(data)
                case "signals":
                    self._signals.extend(data)
                case "targets":
                    self._targets.extend(data)

    def get_portfolio(self, as_plain_dataframe=True) -> pd.DataFrame:
        try:
            pfl = pd.DataFrame.from_records(self._portfolio, index="timestamp")
            pfl.index = pd.DatetimeIndex(pfl.index)
            if as_plain_dataframe:
                # - convert to Qube presentation (TODO: temporary)
                pis = []
                # Use combination of exchange and symbol for unique grouping
                for s in set(pfl["exchange"] + ":" + pfl["symbol"]):
                    exchange, symbol = s.split(":", 1)  # Split only on first dot
                    pi = pfl[(pfl["exchange"] == exchange) & (pfl["symbol"] == symbol)]
                    pi = pi.drop(
                        columns=[
                            "symbol",
                            "exchange",
                            "market_type",
                            "realized_pnl_quoted",
                            "current_price",
                            "exchange_time",
                        ]
                    )
                    rename_dict = {
                        "pnl_quoted": "PnL",
                        "quantity": "Pos",
                        "avg_position_price": "Price",
                        "market_value_quoted": "Value",
                        "commissions_quoted": "Commissions",
                    }
                    # Only rename funding if column exists
                    if "cumulative_funding" in pi.columns:
                        rename_dict["cumulative_funding"] = "Funding"
                    pi = pi.rename(rename_dict, axis=1)
                    # We want to convert the value to just price * quantity
                    # in reality value of perps is just the unrealized pnl but
                    # it's not important after simulation for metric calculations
                    pi["Value"] = pi["Pos"] * pi["Price"] + pi["Value"]
                    pis.append(pi.rename(lambda x: s + "_" + x, axis=1))
                result_df = split_cumulative_pnl(scols(*pis))
                # Drop columns that contain only NaN values (e.g., funding columns for SPOT instruments)
                result_df = result_df.dropna(axis=1, how='all')
                return result_df
            return pfl
        except Exception as e:
            logger.error(f":: Error getting portfolio: {e} ::\n{self._portfolio}")
            return pd.DataFrame()

    def get_executions(self) -> pd.DataFrame:
        p = pd.DataFrame()
        if self._execs:
            p = pd.DataFrame.from_records(self._execs, index="timestamp")
            p.index = pd.DatetimeIndex(p.index)
        return p

    def get_signals(self) -> pd.DataFrame:
        p = pd.DataFrame()
        if self._signals:
            p = pd.DataFrame.from_records(self._signals, index="timestamp")
            p.index = pd.DatetimeIndex(p.index)
        return p

    def get_targets(self) -> pd.DataFrame:
        p = pd.DataFrame()
        if self._targets:
            p = pd.DataFrame.from_records(self._targets, index="timestamp")
            p.index = pd.DatetimeIndex(p.index)
        return p
