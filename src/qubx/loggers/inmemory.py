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

    def __init__(self, account_id: str, strategy_id: str, run_id: str) -> None:
        super().__init__(account_id, strategy_id, run_id)
        self._portfolio = []
        self._execs = []
        self._signals = []

    def write_data(self, log_type: str, data: list[dict[str, Any]]):
        if len(data) > 0:
            if log_type == "portfolio":
                self._portfolio.extend(data)
            elif log_type == "executions":
                self._execs.extend(data)
            elif log_type == "signals":
                self._signals.extend(data)

    def get_portfolio(self, as_plain_dataframe=True) -> pd.DataFrame:
        try:
            pfl = pd.DataFrame.from_records(self._portfolio, index="timestamp")
            pfl.index = pd.DatetimeIndex(pfl.index)
            if as_plain_dataframe:
                # - convert to Qube presentation (TODO: temporary)
                pis = []
                for s in set(pfl["symbol"]):
                    pi = pfl[pfl["symbol"] == s]
                    pi = pi.drop(columns=["symbol", "realized_pnl_quoted", "current_price", "exchange_time"])
                    pi = pi.rename(
                        {
                            "pnl_quoted": "PnL",
                            "quantity": "Pos",
                            "avg_position_price": "Price",
                            "market_value_quoted": "Value",
                            "commissions_quoted": "Commissions",
                        },
                        axis=1,
                    )
                    # We want to convert the value to just price * quantity
                    # in reality value of perps is just the unrealized pnl but
                    # it's not important after simulation for metric calculations
                    pi["Value"] = pi["Pos"] * pi["Price"] + pi["Value"]
                    pis.append(pi.rename(lambda x: s + "_" + x, axis=1))
                return split_cumulative_pnl(scols(*pis))
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
