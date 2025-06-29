import csv
import os
from multiprocessing.pool import ThreadPool
from typing import Any

from qubx import logger
from qubx.core.loggers import LogsWriter
from qubx.utils.misc import makedirs


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
        _tgt_path = f"{path}/{strategy_id}_{account_id}_targets.csv"

        self._hdr_pfl = not os.path.exists(_pfl_path)
        self._hdr_exe = not os.path.exists(_exe_path)
        self._hdr_sig = not os.path.exists(_sig_path)
        self._hdr_tgt = not os.path.exists(_tgt_path)

        self._pfl_file_ = open(_pfl_path, "+a", newline="")
        self._execs_file_ = open(_exe_path, "+a", newline="")
        self._sig_file_ = open(_sig_path, "+a", newline="")
        self._tgt_file_ = open(_tgt_path, "+a", newline="")

        self._pfl_writer = csv.writer(self._pfl_file_)
        self._exe_writer = csv.writer(self._execs_file_)
        self._sig_writer = csv.writer(self._sig_file_)
        self._tgt_writer = csv.writer(self._tgt_file_)

        self.pool = ThreadPool(4)

    @staticmethod
    def _header(d: dict) -> list[str]:
        return list(d.keys()) + ["run_id"]

    def _values(self, data: list[dict[str, Any]]) -> list[list[str]]:
        # - attach run_id (last column)
        return [list((d | {"run_id": self.run_id}).values()) for d in data]

    def _do_write(self, log_type: str, data: list[dict[str, Any]]):
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

            case "targets":
                if self._hdr_tgt:
                    self._tgt_writer.writerow(self._header(data[0]))
                    self._hdr_tgt = False
                self._tgt_writer.writerows(self._values(data))
                self._tgt_file_.flush()

            case "balance":
                with open(self._balance_file_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(self._header(data[0]))
                    w.writerows(self._values(data))

    def write_data(self, log_type: str, data: list[dict[str, Any]]):
        if len(data) > 0:
            self.pool.apply_async(self._do_write, (log_type, data))

    def flush_data(self):
        try:
            self._pfl_file_.flush()
            self._execs_file_.flush()
            self._sig_file_.flush()
            self._tgt_file_.flush()
        except Exception as e:
            logger.warning(f"Error flushing log writer: {str(e)}")

    def close(self):
        self._pfl_file_.close()
        self._execs_file_.close()
        self._sig_file_.close()
        self._tgt_file_.close()
        self.pool.close()
        self.pool.join()
