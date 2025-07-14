from typing import Any

import numpy as np

from qubx import logger
from qubx.core.basics import (
    AssetBalance,
    Deal,
    Instrument,
    MarketType,
    Position,
    Signal,
    TargetPosition,
)
from qubx.core.series import time_as_nsec
from qubx.core.utils import recognize_timeframe
from qubx.utils.misc import Stopwatch
from qubx.utils.time import convert_tf_str_td64, floor_t64

_SW = Stopwatch()


class LogsWriter:
    """
    Log writer interface with default implementation
    """

    account_id: str
    strategy_id: str
    run_id: str

    def __init__(self, account_id: str, strategy_id: str, run_id: str) -> None:
        self.account_id = account_id
        self.strategy_id = strategy_id
        self.run_id = run_id

    def write_data(self, log_type: str, data: list[dict[str, Any]]):
        pass

    def flush_data(self):
        pass

    def close(self):
        pass


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

    positions: dict[Instrument, Position]
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
            data_dict = {
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
            # Only add funding for SWAP instruments
            if i.market_type == MarketType.SWAP:
                data_dict["cumulative_funding"] = p.cumulative_funding
            data.append(data_dict)
        self._writer.write_data("portfolio", data)

    def close(self):
        self._writer.flush_data()


class ExecutionsLogger(_BaseIntervalDumper):
    """
    Executions logger - save strategy executions into storage
    """

    _writer: LogsWriter
    _deals: list[tuple[Instrument, Deal]]

    def __init__(self, writer: LogsWriter, max_records=10) -> None:
        super().__init__(None)  # no intervals
        self._writer = writer
        self._max_records = max_records
        self._deals = []

    def record_deals(self, instrument: Instrument, deals: list[Deal]):
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


class SignalsAndTargetsLogger(_BaseIntervalDumper):
    """
    Signals and targets logger - save signals generated by strategy
    """

    _writer: LogsWriter
    _targets: list[TargetPosition]
    _signals: list[Signal]

    def __init__(self, writer: LogsWriter, max_records=100) -> None:
        super().__init__(None)
        self._writer = writer
        self._max_records = max_records
        self._targets = []
        self._signals = []

    def record_targets(self, signals: list[TargetPosition]):
        self._targets.extend(signals)

        if len(self._targets) >= self._max_records:
            self.dump(None, None)

    def record_signals(self, signals: list[Signal]):
        self._signals.extend(signals)

        if len(self._signals) >= self._max_records:
            self.dump(None, None)

    def dump(self, interval_start_time: np.datetime64 | None, actual_timestamp: np.datetime64 | None):
        t_data = [
            {
                "timestamp": t.time,
                "symbol": t.instrument.symbol,
                "exchange": t.instrument.exchange,
                "market_type": t.instrument.market_type,
                "target_position": t.target_position_size,
                "entry_price": t.entry_price,
                "take_price": t.take_price,
                "stop_price": t.stop_price,
            }
            for t in self._targets
        ]

        s_data = [
            {
                "timestamp": s.time,
                "symbol": s.instrument.symbol,
                "exchange": s.instrument.exchange,
                "market_type": s.instrument.market_type,
                "signal": s.signal,
                "reference_price": s.reference_price,
                "price": s.price,
                "take": s.take,
                "stop": s.stop,
                "group": s.group,
                "comment": s.comment,
                "service": s.is_service,
            }
            for s in self._signals
        ]
        self._writer.write_data("targets", t_data)
        self._targets.clear()

        self._writer.write_data("signals", s_data)
        self._signals.clear()

    def store(self, timestamp: np.datetime64):
        pass

    def close(self):
        if self._targets or self._signals:
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
    signals_logger: SignalsAndTargetsLogger | None = None
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
                self.signals_logger = SignalsAndTargetsLogger(logs_writer, num_signals_records_to_write)

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

    def save_deals(self, instrument: Instrument, deals: list[Deal]):
        if self.executions_logger:
            self.executions_logger.record_deals(instrument, deals)

    def save_targets(self, targets: list[TargetPosition]):
        if self.signals_logger and targets:
            self.signals_logger.record_targets(targets)

    def save_signals(self, signals: list[Signal]):
        if self.signals_logger and signals:
            self.signals_logger.record_signals(signals)

    def _log_heartbeat(self, timestamp: np.datetime64):
        if not self.heartbeat_freq:
            return
        _floored_ts = floor_t64(timestamp, self.heartbeat_freq)
        if not self._last_heartbeat_ts or _floored_ts - self._last_heartbeat_ts >= self.heartbeat_freq:
            self._last_heartbeat_ts = _floored_ts
            logger.info(f"Heartbeat at {_floored_ts.astype('datetime64[s]')}")
