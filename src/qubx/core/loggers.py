import csv
import os
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import (
    AssetBalance,
    Deal,
    Instrument,
    Position,
    TargetPosition,
)
from qubx.core.metrics import split_cumulative_pnl
from qubx.core.series import time_as_nsec
from qubx.core.utils import recognize_timeframe
from qubx.pandaz.utils import scols
from qubx.utils.misc import Stopwatch, makedirs
from qubx.utils.time import convert_tf_str_td64, floor_t64

_SW = Stopwatch()


class LogsWriter:
    account_id: str
    strategy_id: str
    run_id: str

    """
    Log writer interface with default implementation
    """

    def __init__(self, account_id: str, strategy_id: str, run_id: str) -> None:
        self.account_id = account_id
        self.strategy_id = strategy_id
        self.run_id = run_id

    def write_data(self, log_type: str, data: List[Dict[str, Any]]):
        pass

    def flush_data(self):
        pass

    def close(self):
        pass


class InMemoryLogsWriter(LogsWriter):
    _portfolio: List
    _execs: List
    _signals: List

    def __init__(self, account_id: str, strategy_id: str, run_id: str) -> None:
        super().__init__(account_id, strategy_id, run_id)
        self._portfolio = []
        self._execs = []
        self._signals = []

    def write_data(self, log_type: str, data: List[Dict[str, Any]]):
        if len(data) > 0:
            if log_type == "portfolio":
                self._portfolio.extend(data)
            elif log_type == "executions":
                self._execs.extend(data)
            elif log_type == "signals":
                self._signals.extend(data)

    def get_portfolio(self, as_plain_dataframe=True) -> pd.DataFrame:
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


class CsvFileLogsWriter(LogsWriter):
    """
    Simple CSV strategy log data writer. It does data writing in separate thread.
    """

    def __init__(self, account_id: str, strategy_id: str, run_id: str, log_folder="logs") -> None:
        super().__init__(account_id, strategy_id, run_id)

        path = makedirs(log_folder)
        # - it rewrites positions every time
        self._pos_file_path = f"{path}/{self.strategy_id}_{self.account_id}_positions.csv"
        self._balance_file_path = f"{path}/{self.strategy_id}_{self.account_id}_balance.csv"

        _pfl_path = f"{path}/{strategy_id}_{account_id}_portfolio.csv"
        _exe_path = f"{path}/{strategy_id}_{account_id}_executions.csv"
        _sig_path = f"{path}/{strategy_id}_{account_id}_signals.csv"
        self._hdr_pfl = not os.path.exists(_pfl_path)
        self._hdr_exe = not os.path.exists(_exe_path)
        self._hdr_sig = not os.path.exists(_sig_path)

        self._pfl_file_ = open(_pfl_path, "+a", newline="")
        self._execs_file_ = open(_exe_path, "+a", newline="")
        self._sig_file_ = open(_sig_path, "+a", newline="")
        self._pfl_writer = csv.writer(self._pfl_file_)
        self._exe_writer = csv.writer(self._execs_file_)
        self._sig_writer = csv.writer(self._sig_file_)
        self.pool = ThreadPool(3)

    @staticmethod
    def _header(d: dict) -> List[str]:
        return list(d.keys()) + ["run_id"]

    def _values(self, data: List[Dict[str, Any]]) -> List[List[str]]:
        # - attach run_id (last column)
        return [list((d | {"run_id": self.run_id}).values()) for d in data]

    def _do_write(self, log_type, data):
        match log_type:
            case "positions":
                with open(self._pos_file_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(self._header(data[0]))
                    w.writerows(self._values(data))

            case "portfolio":
                if self._hdr_pfl:
                    self._pfl_writer.writerow(self._header(data[0]))
                    self._hdr_pfl = False
                self._pfl_writer.writerows(self._values(data))
                self._pfl_file_.flush()

            case "executions":
                if self._hdr_exe:
                    self._exe_writer.writerow(self._header(data[0]))
                    self._hdr_exe = False
                self._exe_writer.writerows(self._values(data))
                self._execs_file_.flush()

            case "signals":
                if self._hdr_sig:
                    self._sig_writer.writerow(self._header(data[0]))
                    self._hdr_sig = False
                self._sig_writer.writerows(self._values(data))
                self._sig_file_.flush()

            case "balance":
                with open(self._balance_file_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(self._header(data[0]))
                    w.writerows(self._values(data))

    def write_data(self, log_type: str, data: List[Dict[str, Any]]):
        if len(data) > 0:
            self.pool.apply_async(self._do_write, (log_type, data))

    def flush_data(self):
        try:
            self._pfl_file_.flush()
            self._execs_file_.flush()
            self._sig_file_.flush()
        except Exception as e:
            logger.warning(f"Error flushing log writer: {str(e)}")

    def close(self):
        self._pfl_file_.close()
        self._execs_file_.close()
        self._sig_file_.close()
        self.pool.close()
        self.pool.join()


class _BaseIntervalDumper:
    """
    Basic functionality for all interval based dumpers
    """

    _last_log_time_ns: int
    _freq: np.timedelta64 | None

    def __init__(self, frequency: str | None) -> None:
        self._freq: np.timedelta64 | None = recognize_timeframe(frequency) if frequency else None
        self._last_log_time_ns = 0

    def store(self, timestamp: np.datetime64):
        _t_ns = time_as_nsec(timestamp)
        if self._freq:
            # Convert freq to nanoseconds for calculation
            _freq_ns = self._freq.astype("int64")
            _interval_start_time = int(_t_ns - (_t_ns % _freq_ns))
            if _t_ns - self._last_log_time_ns >= _freq_ns:
                self.dump(np.datetime64(_interval_start_time, "ns"), timestamp)
                self._last_log_time_ns = _interval_start_time
        else:
            self.dump(timestamp, timestamp)

    def dump(self, interval_start_time: np.datetime64, actual_timestamp: np.datetime64):
        raise NotImplementedError(
            f"dump(np.datetime64, np.datetime64) must be implemented in {self.__class__.__name__}"
        )


class PositionsDumper(_BaseIntervalDumper):
    """
    Positions dumper is designed to dump positions once per given interval to storage
    so we could check current situation.
    """

    positions: Dict[Instrument, Position]
    _writer: LogsWriter

    def __init__(
        self,
        writer: LogsWriter,
        interval: str,
    ) -> None:
        super().__init__(interval)
        self.positions = dict()
        self._writer = writer

    def attach_positions(self, *positions: Position) -> "PositionsDumper":
        for p in positions:
            self.positions[p.instrument] = p
        return self

    def dump(self, interval_start_time: np.datetime64, actual_timestamp: np.datetime64):
        data = []
        for i, p in self.positions.items():
            data.append(
                {
                    "timestamp": str(actual_timestamp),
                    "symbol": i.symbol,
                    "exchange": i.exchange,
                    "market_type": i.market_type,
                    "pnl_quoted": p.total_pnl(),
                    "quantity": p.quantity,
                    "notional": p.notional_value,
                    "realized_pnl_quoted": p.r_pnl,
                    "avg_position_price": p.position_avg_price if p.quantity != 0.0 else 0.0,
                    "current_price": p.last_update_price,
                    "market_value_quoted": p.market_value_funds,
                }
            )
        self._writer.write_data("positions", data)


class PortfolioLogger(PositionsDumper):
    """
    Portfolio logger - save portfolio records into storage
    """

    def __init__(self, writer: LogsWriter, interval: str) -> None:
        super().__init__(writer, interval)

    def dump(self, interval_start_time: np.datetime64, actual_timestamp: np.datetime64):
        data = []
        for i, p in self.positions.items():
            data.append(
                {
                    "timestamp": str(interval_start_time),
                    "symbol": i.symbol,
                    "exchange": i.exchange,
                    "market_type": i.market_type,
                    "pnl_quoted": p.total_pnl(),
                    "quantity": p.quantity,
                    "realized_pnl_quoted": p.r_pnl,
                    "avg_position_price": p.position_avg_price if p.quantity != 0.0 else 0.0,
                    "current_price": p.last_update_price,
                    "market_value_quoted": p.market_value_funds,
                    "exchange_time": str(actual_timestamp),
                    "commissions_quoted": p.commissions,
                }
            )
        self._writer.write_data("portfolio", data)

    def close(self):
        self._writer.flush_data()


class ExecutionsLogger(_BaseIntervalDumper):
    """
    Executions logger - save strategy executions into storage
    """

    _writer: LogsWriter
    _deals: List[Tuple[Instrument, Deal]]

    def __init__(self, writer: LogsWriter, max_records=10) -> None:
        super().__init__(None)  # no intervals
        self._writer = writer
        self._max_records = max_records
        self._deals: List[Tuple[Instrument, Deal]] = []

    def record_deals(self, instrument: Instrument, deals: List[Deal]):
        for d in deals:
            self._deals.append((instrument, d))
            l_time = d.time

        if len(self._deals) >= self._max_records:
            self.dump(l_time, l_time)

    def dump(self, interval_start_time: np.datetime64, actual_timestamp: np.datetime64):
        data = []
        for i, d in self._deals:
            data.append(
                {
                    "timestamp": d.time,
                    "symbol": i.symbol,
                    "exchange": i.exchange,
                    "market_type": i.market_type,
                    "side": "buy" if d.amount > 0 else "sell",
                    "filled_qty": d.amount,
                    "price": d.price,
                    "commissions": d.fee_amount,
                    "commissions_quoted": d.fee_currency,
                    "order_id": d.order_id,
                }
            )
        self._deals.clear()
        self._writer.write_data("executions", data)

    def store(self, timestamp: np.datetime64):
        pass

    def close(self):
        if self._deals:
            t = self._deals[-1][1].time
            self.dump(t, t)
        self._writer.flush_data()


class SignalsLogger(_BaseIntervalDumper):
    """
    Signals logger - save signals generated by strategy
    """

    _writer: LogsWriter
    _targets: List[TargetPosition]

    def __init__(self, writer: LogsWriter, max_records=100) -> None:
        super().__init__(None)
        self._writer = writer
        self._max_records = max_records
        self._targets = []

    def record_signals(self, signals: List[TargetPosition]):
        self._targets.extend(signals)

        if len(self._targets) >= self._max_records:
            self.dump(None, None)

    def dump(self, interval_start_time: np.datetime64 | None, actual_timestamp: np.datetime64 | None):
        data = []
        for s in self._targets:
            data.append(
                {
                    "timestamp": s.time,
                    "symbol": s.instrument.symbol,
                    "exchange": s.instrument.exchange,
                    "market_type": s.instrument.market_type,
                    "signal": s.signal.signal,
                    "target_position": s.target_position_size,
                    "reference_price": s.signal.reference_price,
                    "price": s.price,
                    "take": s.take,
                    "stop": s.stop,
                    "group": s.signal.group,
                    "comment": s.signal.comment,
                    "service": s.is_service,
                }
            )
        self._targets.clear()
        self._writer.write_data("signals", data)

    def store(self, timestamp: np.datetime64):
        pass

    def close(self):
        if self._targets:
            self.dump(None, None)
        self._writer.flush_data()


class BalanceLogger(_BaseIntervalDumper):
    """
    Balance logger - save balance information at regular intervals similar to positions
    """

    _writer: LogsWriter
    _balance: dict[str, AssetBalance]

    def __init__(self, writer: LogsWriter, interval: str) -> None:
        super().__init__(interval)
        self._writer = writer
        self._balance = {}

    def record_balance(self, timestamp: np.datetime64, balance: dict[str, AssetBalance]):
        if balance:
            self._balance = balance
            self.dump(timestamp, timestamp)

    def dump(self, interval_start_time: np.datetime64, actual_timestamp: np.datetime64):
        if self._balance:
            data = []
            for s, d in self._balance.items():
                data.append(
                    {
                        "timestamp": str(interval_start_time),
                        "currency": s,
                        "total": d.total,
                        "locked": d.locked,
                    }
                )
            self._writer.write_data("balance", data)

    def close(self):
        self._writer.flush_data()


class StrategyLogging:
    """
    Just combined loggers functionality
    """

    positions_dumper: PositionsDumper | None = None
    portfolio_logger: PortfolioLogger | None = None
    executions_logger: ExecutionsLogger | None = None
    balance_logger: BalanceLogger | None = None
    signals_logger: SignalsLogger | None = None
    heartbeat_freq: np.timedelta64 | None = None

    _last_heartbeat_ts: np.datetime64 | None = None

    def __init__(
        self,
        logs_writer: LogsWriter | None = None,
        positions_log_freq: str = "1Min",
        portfolio_log_freq: str = "5Min",
        num_exec_records_to_write=1,  # in live let's write every execution
        num_signals_records_to_write=1,
        heartbeat_freq: str | None = None,
    ) -> None:
        # - instantiate loggers
        if logs_writer:
            if positions_log_freq:
                # - store current positions
                self.positions_dumper = PositionsDumper(logs_writer, positions_log_freq)

            if portfolio_log_freq:
                # - store portfolio log
                self.portfolio_logger = PortfolioLogger(logs_writer, portfolio_log_freq)

            # - store executions
            if num_exec_records_to_write >= 1:
                self.executions_logger = ExecutionsLogger(logs_writer, num_exec_records_to_write)

            # - store signals
            if num_signals_records_to_write >= 1:
                self.signals_logger = SignalsLogger(logs_writer, num_signals_records_to_write)

            # - balance logger
            self.balance_logger = BalanceLogger(logs_writer, positions_log_freq)
        else:
            logger.warning("Log writer is not defined - strategy activity will not be saved !")

        self.logs_writer = logs_writer
        self.heartbeat_freq = convert_tf_str_td64(heartbeat_freq) if heartbeat_freq else None

    def initialize(
        self,
        timestamp: np.datetime64,
        positions: dict[Instrument, Position],
        balances: dict[str, AssetBalance],
    ) -> None:
        # - attach positions to loggers
        if self.positions_dumper:
            self.positions_dumper.attach_positions(*list(positions.values()))

        if self.portfolio_logger:
            self.portfolio_logger.attach_positions(*list(positions.values()))

        # - send balance on start
        if self.balance_logger:
            self.balance_logger.record_balance(timestamp, balances)

    def close(self):
        if self.portfolio_logger:
            self.portfolio_logger.close()

        if self.executions_logger:
            self.executions_logger.close()

        if self.signals_logger:
            self.signals_logger.close()

        if self.balance_logger:
            self.balance_logger.close()

        if self.logs_writer:
            self.logs_writer.close()

    @_SW.watch("loggers")
    def notify(self, timestamp: np.datetime64):
        # - notify position logger
        if self.positions_dumper:
            self.positions_dumper.store(timestamp)

        # - notify portfolio records logger
        if self.portfolio_logger:
            self.portfolio_logger.store(timestamp)

        # - notify balance logger
        if self.balance_logger:
            self.balance_logger.store(timestamp)

        # - log heartbeat
        self._log_heartbeat(timestamp)

    def save_deals(self, instrument: Instrument, deals: List[Deal]):
        if self.executions_logger:
            self.executions_logger.record_deals(instrument, deals)

    def save_signals_targets(self, targets: List[TargetPosition]):
        if self.signals_logger and targets:
            self.signals_logger.record_signals(targets)

    def _log_heartbeat(self, timestamp: np.datetime64):
        if not self.heartbeat_freq:
            return
        _floored_ts = floor_t64(timestamp, self.heartbeat_freq)
        if not self._last_heartbeat_ts or _floored_ts - self._last_heartbeat_ts >= self.heartbeat_freq:
            self._last_heartbeat_ts = _floored_ts
            logger.info(f"Heartbeat at {_floored_ts.astype('datetime64[s]')}")
