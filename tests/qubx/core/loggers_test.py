import shutil
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd
import tabulate

from qubx.core.basics import AssetBalance, Deal, Position
from qubx.core.loggers import (
    BalanceLogger,
    ExecutionsLogger,
    LogsWriter,
    PositionsDumper,
    StrategyLogging,
)
from qubx.core.lookups import lookup
from qubx.loggers.csv import CsvFileLogsWriter

_DT = lambda seconds: (pd.Timestamp("2022-01-01") + pd.to_timedelta(seconds, unit="s")).to_datetime64()


class ConsolePositionsWriter(LogsWriter):
    """
    Simple positions - just writes current positions to the standard output as funcy table
    """

    def _dump_positions(self, data: List[Dict[str, Any]]):
        table = defaultdict(list)
        total_pnl, total_rpnl, total_mkv = 0, 0, 0

        for r in data:
            table["Symbol"].append(r["symbol"])
            table["Time"].append(r["timestamp"])
            table["Quantity"].append(r["quantity"])
            table["AvgPrice"].append(r["avg_position_price"])
            table["LastPrice"].append(r["current_price"])
            table["PnL"].append(r["pnl_quoted"])
            table["RealPnL"].append(r["realized_pnl_quoted"])
            table["MarketValue"].append(r["market_value_quoted"])
            total_pnl += r["pnl_quoted"]
            total_rpnl += r["realized_pnl_quoted"]
            total_mkv += r["market_value_quoted"]

        table["Symbol"].append("TOTAL")
        table["PnL"].append(total_pnl)
        table["RealPnL"].append(total_rpnl)
        table["MarketValue"].append(total_mkv)

        # - write to database table here
        print(f" ::: Strategy {self.strategy_id} @ {self.account_id} :::")
        print(
            tabulate.tabulate(
                table,
                ["Symbol", "Time", "Quantity", "AvgPrice", "LastPrice", "PnL", "RealPnL", "MarketValue"],
                tablefmt="rounded_grid",
            )
        )

    def _dump_balance(self, data: List[Dict[str, Any]]):
        table = defaultdict(list)

        for r in data:
            table["Currency"].append(r["currency"])
            table["Time"].append(r["timestamp"])
            table["Total"].append(r["total"])
            table["Locked"].append(r["locked"])

        print(f" ::: Balance {self.strategy_id} @ {self.account_id} :::")
        print(
            tabulate.tabulate(
                table,
                ["Currency", "Time", "Total", "Locked"],
                tablefmt="rounded_grid",
            )
        )

    def write_data(self, log_type: str, data: List[Dict[str, Any]]):
        match log_type:
            case "positions":
                self._dump_positions(data)

            case "portfolio":
                pass

            case "executions":
                for d in data:
                    print(f" --- DEAL: {d}")

            case "balance":
                self._dump_balance(data)


class TestPortfolioLoggers:
    def test_positions_dumper(self):
        # - initialize positions: this will be done in StrategyContext
        positions = [
            Position(lookup.find_symbol("BINANCE", s))  # type: ignore
            for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        ]
        positions[0].change_position_by(_DT(0), 0.05, 63000)
        positions[1].change_position_by(_DT(0), 0.5, 3200)
        positions[2].change_position_by(_DT(0), 10, 56)

        writer = ConsolePositionsWriter("Account1", "Strategy1", "test-run-id-0")
        # - create dumper and attach positions
        console_dumper = PositionsDumper(writer, "1Sec").attach_positions(*positions)  # dumps positions once per 1 sec

        # - create executions logger
        execs_logger = ExecutionsLogger(writer, 1)

        # - emulating updates from strategy (this will be done in StategyContext)
        for _ in range(30):
            t = pd.Timestamp("now").asm8
            for p in positions:
                # - selling 10% of position every tick
                p.change_position_by(t, -p.quantity * 0.1, p.last_update_price + 10)
            # - this method will be called inside the platform !
            console_dumper.store(t)
            time.sleep(0.25)

        # - emulate deals
        instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instrument is not None
        execs_logger.record_deals(
            instrument,
            [
                Deal("1", "111", _DT(0), 0.1, 4444, False),
                Deal("2", "222", _DT(1), 0.1, 5555, False),
                Deal("2", "222", _DT(2), 0.2, 6666, False),
            ],
        )
        execs_logger.close()

    def test_csv_writer(self):
        # Create a unique temporary directory for this test
        test_dir = tempfile.mkdtemp(prefix="qubx_test_")
        writer = CsvFileLogsWriter("Account1", "Strategy1", "test-run-id-0", log_folder=test_dir)

        # - create executions logger
        execs_logger = ExecutionsLogger(writer, 10)
        instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instrument is not None
        execs_logger.record_deals(
            instrument,
            [
                Deal("11", "111", _DT(0), 0.1, 9999, False),
                Deal("22", "222", _DT(1), 0.1, 8888, False),
                Deal("33", "222", _DT(2), 0.2, 7777, False),
            ],
        )
        execs_logger.close()

        # Cleanup: close writer and remove temporary directory
        writer.close()
        shutil.rmtree(test_dir, ignore_errors=True)

    def test_dump_instrument_writes_single_row(self):
        btc = lookup.find_symbol("BINANCE", "BTCUSDT")
        eth = lookup.find_symbol("BINANCE", "ETHUSDT")
        assert btc is not None and eth is not None
        pos_btc = Position(btc)
        pos_eth = Position(eth)
        pos_btc.change_position_by(_DT(0), 0.05, 63000)
        pos_eth.change_position_by(_DT(0), 0.5, 3200)

        captured: list[tuple[str, list[dict]]] = []

        class _CapturingWriter(LogsWriter):
            def write_data(self, log_type: str, data: list[dict]):
                captured.append((log_type, data))

        dumper = PositionsDumper(_CapturingWriter("acc", "strat", "run"), "1Sec").attach_positions(pos_btc, pos_eth)

        dumper.dump_instrument(btc, _DT(1))
        assert len(captured) == 1
        log_type, data = captured[0]
        assert log_type == "positions"
        assert len(data) == 1
        assert data[0]["symbol"] == "BTCUSDT"
        assert data[0]["quantity"] == pos_btc.quantity

        # Unknown instrument is a no-op.
        unknown = lookup.find_symbol("BINANCE", "SOLUSDT")
        assert unknown is not None
        dumper.dump_instrument(unknown, _DT(2))
        assert len(captured) == 1

    def test_save_deals_triggers_on_change_dump(self):
        btc = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert btc is not None
        pos = Position(btc)
        pos.change_position_by(_DT(0), 0.1, 50000)

        captured: list[tuple[str, list[dict]]] = []

        class _CapturingWriter(LogsWriter):
            def write_data(self, log_type: str, data: list[dict]):
                captured.append((log_type, data))

        writer = _CapturingWriter("acc", "strat", "run")
        logging = StrategyLogging(
            logs_writer=writer,
            positions_log_freq="1Min",
            portfolio_log_freq=None,
            positions_log_on_change=True,
        )
        assert logging.positions_dumper is not None
        logging.positions_dumper.attach_positions(pos)

        logging.save_deals(btc, [Deal("1", "o1", _DT(5), 0.1, 50050, False)])

        position_writes = [d for t, d in captured if t == "positions"]
        assert len(position_writes) == 1
        assert position_writes[0][0]["symbol"] == "BTCUSDT"

    def test_save_deals_respects_flag_off(self):
        btc = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert btc is not None
        pos = Position(btc)
        pos.change_position_by(_DT(0), 0.1, 50000)

        captured: list[tuple[str, list[dict]]] = []

        class _CapturingWriter(LogsWriter):
            def write_data(self, log_type: str, data: list[dict]):
                captured.append((log_type, data))

        logging = StrategyLogging(
            logs_writer=_CapturingWriter("acc", "strat", "run"),
            positions_log_freq="1Min",
            portfolio_log_freq=None,
            positions_log_on_change=False,
        )
        assert logging.positions_dumper is not None
        logging.positions_dumper.attach_positions(pos)

        logging.save_deals(btc, [Deal("1", "o1", _DT(5), 0.1, 50050, False)])

        assert not any(t == "positions" for t, _ in captured)

    def test_save_deals_defaults_to_off(self):
        btc = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert btc is not None
        pos = Position(btc)
        pos.change_position_by(_DT(0), 0.1, 50000)

        captured: list[tuple[str, list[dict]]] = []

        class _CapturingWriter(LogsWriter):
            def write_data(self, log_type: str, data: list[dict]):
                captured.append((log_type, data))

        # Omit positions_log_on_change to exercise the default (False: unchanged behavior).
        logging = StrategyLogging(
            logs_writer=_CapturingWriter("acc", "strat", "run"),
            positions_log_freq="1Min",
            portfolio_log_freq=None,
        )
        assert logging.positions_dumper is not None
        logging.positions_dumper.attach_positions(pos)

        logging.save_deals(btc, [Deal("1", "o1", _DT(5), 0.1, 50050, False)])

        assert not any(t == "positions" for t, _ in captured)

    def test_strategy_logging_disables_loggers_when_none(self):
        captured: list[tuple[str, list[dict]]] = []

        class _CapturingWriter(LogsWriter):
            def write_data(self, log_type: str, data: list[dict]):
                captured.append((log_type, data))

        logging = StrategyLogging(
            logs_writer=_CapturingWriter("acc", "strat", "run"),
            positions_log_freq=None,
            portfolio_log_freq=None,
        )
        assert logging.positions_dumper is None
        assert logging.portfolio_logger is None
        assert logging.balance_logger is None

    def test_balance_logger(self):
        writer = ConsolePositionsWriter("Account1", "Strategy1", "test-run-id-0")

        # Create balance logger with 1sec interval
        balance_logger = BalanceLogger(writer, "1Sec")

        # Create test balances as list
        balances = [
            AssetBalance(exchange="TEST", currency="USDT", total=1000.0, locked=100.0, free=900.0),
            AssetBalance(exchange="TEST", currency="BTC", total=0.5, locked=0.0, free=0.5),
            AssetBalance(exchange="TEST", currency="ETH", total=5.0, locked=1.0, free=4.0),
        ]

        # Record initial balance
        t = _DT(0)
        balance_logger.record_balance(t, balances)

        # Simulate balance changes over time
        for i in range(5):
            t = _DT(i + 1)
            # Update some balances
            balances = [
                AssetBalance(
                    exchange="TEST", currency="USDT", total=balances[0].total - 50.0, locked=balances[0].locked, free=balances[0].total - 50.0 - balances[0].locked
                ),
                AssetBalance(
                    exchange="TEST", currency="BTC", total=balances[1].total + 0.01, locked=balances[1].locked, free=balances[1].total + 0.01 - balances[1].locked
                ),
                balances[2],  # ETH unchanged
            ]

            # Store at current time
            balance_logger.store(t)
            time.sleep(0.5)

        balance_logger.close()
