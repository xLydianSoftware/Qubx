import time
from collections import defaultdict
from typing import Any, Dict, List, Union

import pandas as pd
import tabulate

from qubx.core.basics import ZERO_COSTS, Deal, Position, dt_64
from qubx.core.loggers import CsvFileLogsWriter, ExecutionsLogger, LogsWriter, PositionsDumper
from qubx.core.lookups import lookup

_DT = lambda seconds: (pd.Timestamp("2022-01-01") + pd.to_timedelta(seconds, unit="s")).to_datetime64()


class ConsolePositionsWriter(LogsWriter):
    """
    Simple positions - just writes current positions to the standard output as funcy table
    """

    def _dump_positions(self, data: List[Dict[str, Any]]):
        table = defaultdict(list)
        total_pnl, total_rpnl, total_mkv = 0, 0, 0

        for r in data:
            table["Symbol"].append(r["instrument_id"])
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

    def write_data(self, log_type: str, data: List[Dict[str, Any]]):
        match log_type:
            case "positions":
                self._dump_positions(data)

            case "portfolio":
                pass

            case "executions":
                for d in data:
                    print(f" --- DEAL: {d}")


class TestPortfolioLoggers:
    def test_positions_dumper(self):
        # - initialize positions: this will be done in StrategyContext
        positions = [
            Position(lookup.find_symbol("BINANCE", s))
            for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # type: ignore
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
                Deal(1, "111", _DT(0), 0.1, 4444, False),
                Deal(2, "222", _DT(1), 0.1, 5555, False),
                Deal(2, "222", _DT(2), 0.2, 6666, False),
            ],
        )
        execs_logger.close()

    def test_csv_writer(self):
        writer = CsvFileLogsWriter("Account1", "Strategy1", "test-run-id-0")

        # - create executions logger
        execs_logger = ExecutionsLogger(writer, 10)
        instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instrument is not None
        execs_logger.record_deals(
            instrument,
            [
                Deal(11, "111", _DT(0), 0.1, 9999, False),
                Deal(22, "222", _DT(1), 0.1, 8888, False),
                Deal(33, "222", _DT(2), 0.2, 7777, False),
            ],
        )
        execs_logger.close()
