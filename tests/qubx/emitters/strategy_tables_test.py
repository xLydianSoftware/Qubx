"""Tests for strategy-owned tables: ensure_table + emit_record (spec §2)."""

import csv as csv_mod
import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import qubx.emitters.questdb as qdb_mod
from qubx.core.interfaces import DEFAULT_TABLE_TTL, IMetricEmitter
from qubx.emitters.composite import CompositeMetricEmitter
from qubx.emitters.csv import CSVMetricEmitter
from qubx.emitters.questdb import QuestDBMetricEmitter, ttl_hours


class TestInterfaceDefaults:
    def test_ensure_table_is_noop(self):
        IMetricEmitter().ensure_table("t", {"a": "DOUBLE"})  # must not raise

    def test_emit_record_is_noop(self):
        IMetricEmitter().emit_record("t", {"a": 1.0})  # must not raise

    def test_ensure_table_accepts_max_ttl(self):
        # retention is part of the interface: a caller declaring a table through the abstract
        # type (or any no-op backend) must be able to pass it without a TypeError
        IMetricEmitter().ensure_table("t", {"a": "DOUBLE"}, max_ttl="30 days")
        IMetricEmitter().ensure_table("t", {"a": "DOUBLE"}, max_ttl=None)


class TestCompositeForwarding:
    def test_forwards_to_children(self):
        child_a, child_b = MagicMock(spec=IMetricEmitter), MagicMock(spec=IMetricEmitter)
        comp = CompositeMetricEmitter([child_a, child_b])
        comp.ensure_table(
            "frab.trades", {"net_pnl": "DOUBLE"}, symbol_columns=("pair",), dedup_keys=("timestamp", "trade_id")
        )
        comp.emit_record("frab.trades", {"net_pnl": 1.5}, symbol_columns=("pair",))
        for child in (child_a, child_b):
            child.ensure_table.assert_called_once_with(
                "frab.trades",
                {"net_pnl": "DOUBLE"},
                symbol_columns=("pair",),
                dedup_keys=("timestamp", "trade_id"),
                partition_by="DAY",
                max_ttl=DEFAULT_TABLE_TTL,
            )
            child.emit_record.assert_called_once_with(
                "frab.trades", {"net_pnl": 1.5}, symbol_columns=("pair",), timestamp=None
            )

    @pytest.mark.parametrize("max_ttl", ["30 days", None])
    def test_forwards_max_ttl(self, max_ttl):
        # None is a real value ("leave retention alone"), not "unspecified": it must reach the
        # children as-is instead of being replaced by the default
        child = MagicMock(spec=IMetricEmitter)
        CompositeMetricEmitter([child]).ensure_table("frab.pairs", {"ev": "DOUBLE"}, max_ttl=max_ttl)
        assert child.ensure_table.call_args.kwargs["max_ttl"] == max_ttl

    def test_child_error_is_isolated(self):
        bad, good = MagicMock(spec=IMetricEmitter), MagicMock(spec=IMetricEmitter)
        bad.emit_record.side_effect = RuntimeError("boom")
        comp = CompositeMetricEmitter([bad, good])
        comp.emit_record("t", {"a": 1.0})  # must not raise
        good.emit_record.assert_called_once()


class FakeSender:
    def __init__(self):
        self.rows: list[tuple] = []

    def row(self, table, symbols=None, columns=None, at=None):
        self.rows.append((table, symbols, columns, at))

    def flush(self):
        pass

    def close(self):
        pass


class SyncWorker:
    """Runs submitted work inline — stands in for BoundedWorker so tests can assert
    synchronously without waiting on a background thread."""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)

    def stop(self, **kwargs):
        pass


@pytest.fixture
def emitter(monkeypatch):
    ddl = MagicMock()
    monkeypatch.setattr(qdb_mod, "QuestDBClient", MagicMock(return_value=ddl))
    # Sender is a Cython extension type (immutable); patch the module-level name instead
    # of Sender.from_conf directly (setattr on the class raises TypeError: immutable type).
    monkeypatch.setattr(qdb_mod, "Sender", MagicMock(from_conf=MagicMock(side_effect=RuntimeError("no net"))))
    em = QuestDBMetricEmitter(host="qdb", tags={"strategy": "bot-1", "run_id": "r-1", "environment": "dev"})
    em._sender = FakeSender()
    em._worker = SyncWorker()
    ddl.reset_mock()  # drop the signals/deals DDL calls from __init__
    em._ddl_client_for_test = ddl
    return em


def _create_sql(emitter) -> str:
    """
    The CREATE statement, not the ALTER ... SET TTL that follows it.
    """
    for call in emitter._ddl_client_for_test.execute.call_args_list:
        if call[0][0].lstrip().upper().startswith("CREATE"):
            return call[0][0]
    raise AssertionError("no CREATE statement was executed")


class TestEnsureTable:
    def test_generates_create_table_with_scope_columns_and_dedup(self, emitter):
        emitter.ensure_table(
            "frab.trades",
            {"trade_id": "STRING", "entry_time": "TIMESTAMP", "net_pnl": "DOUBLE"},
            symbol_columns=("pair", "asset"),
            dedup_keys=("timestamp", "trade_id"),
        )
        ddl_sql = _create_sql(emitter)
        assert 'CREATE TABLE IF NOT EXISTS "frab.trades"' in ddl_sql
        assert '"timestamp" TIMESTAMP' in ddl_sql
        # scope columns injected: strategy/environment are SYMBOL_TAGS -> SYMBOL, run_id STRING, is_live BOOLEAN
        assert '"strategy" SYMBOL' in ddl_sql
        assert '"environment" SYMBOL' in ddl_sql
        assert '"run_id" STRING' in ddl_sql
        assert '"is_live" BOOLEAN' in ddl_sql
        assert '"pair" SYMBOL' in ddl_sql
        assert '"net_pnl" DOUBLE' in ddl_sql
        assert '"entry_time" TIMESTAMP' in ddl_sql
        assert 'TIMESTAMP(timestamp) PARTITION BY DAY WAL DEDUP UPSERT KEYS("timestamp", "trade_id")' in ddl_sql
        assert emitter._declared_symbols["frab.trades"] >= {"pair", "asset", "strategy", "environment"}

    def test_rejects_unknown_type(self, emitter):
        emitter.ensure_table("t", {"x": "JSONB"})  # must not raise (logged), and no DDL executed
        emitter._ddl_client_for_test.execute.assert_not_called()

    def test_symbol_columns_do_not_override_reserved_scope_types(self, emitter):
        emitter.ensure_table(
            "frab.reserved",
            {"net_pnl": "DOUBLE"},
            symbol_columns=("run_id", "pair"),
        )
        ddl_sql = _create_sql(emitter)
        # run_id is a reserved scope column (STRING) -> symbol_columns must not flip it to SYMBOL
        assert '"run_id" STRING' in ddl_sql
        assert '"pair" SYMBOL' in ddl_sql

    def test_rejects_bare_string_symbol_columns(self, emitter):
        # a bare string iterates as characters ("p", "a", "i", "r", ...) -> would mint bogus
        # single-letter SYMBOL columns; must be rejected instead, not silently misinterpreted.
        emitter.ensure_table("t", {}, symbol_columns="pair")  # must not raise (logged), no DDL executed
        emitter._ddl_client_for_test.execute.assert_not_called()


def _ttl_frame(value: int, unit: str) -> pd.DataFrame:
    return pd.DataFrame([{"ttlValue": value, "ttlUnit": unit}])


class TestEnsureTableRetention:
    @staticmethod
    def _ttl_statements(emitter) -> list[str]:
        return [
            call[0][0]
            for call in emitter._ddl_client_for_test.execute.call_args_list
            if "SET TTL" in call[0][0].upper()
        ]

    def test_default_applies_interface_ttl_to_a_table_without_retention(self, emitter):
        emitter._ddl_client_for_test.query.return_value = _ttl_frame(0, "HOUR")
        emitter.ensure_table("frab.trades", {"net_pnl": "DOUBLE"})
        assert self._ttl_statements(emitter) == [f'ALTER TABLE "frab.trades" SET TTL {DEFAULT_TABLE_TTL}']

    def test_explicit_ttl_is_applied_to_a_new_table(self, emitter):
        emitter._ddl_client_for_test.query.return_value = _ttl_frame(0, "HOUR")
        emitter.ensure_table("frab.pairs", {"ev": "DOUBLE"}, max_ttl="30 days")
        assert self._ttl_statements(emitter) == ['ALTER TABLE "frab.pairs" SET TTL 30 days']

    def test_shorter_existing_retention_is_kept(self, emitter):
        # the platform tightened dev to 10 days; the strategy's 30-day cap must not undo it
        emitter._ddl_client_for_test.query.return_value = _ttl_frame(10, "DAY")
        emitter.ensure_table("frab.pairs", {"ev": "DOUBLE"}, max_ttl="30 days")
        assert self._ttl_statements(emitter) == []

    def test_equal_existing_retention_is_kept(self, emitter):
        emitter._ddl_client_for_test.query.return_value = _ttl_frame(30, "DAY")
        emitter.ensure_table("frab.pairs", {"ev": "DOUBLE"}, max_ttl="30 days")
        assert self._ttl_statements(emitter) == []

    def test_longer_existing_retention_is_tightened(self, emitter):
        emitter._ddl_client_for_test.query.return_value = _ttl_frame(52, "WEEK")
        emitter.ensure_table("frab.pairs", {"ev": "DOUBLE"}, max_ttl="30 days")
        assert self._ttl_statements(emitter) == ['ALTER TABLE "frab.pairs" SET TTL 30 days']

    def test_unreadable_retention_applies_the_cap(self, emitter):
        # fail open toward the strategy's bound: a read error must not leave a table unbounded
        emitter._ddl_client_for_test.query.side_effect = RuntimeError("tables() unavailable")
        emitter.ensure_table("frab.pairs", {"ev": "DOUBLE"}, max_ttl="30 days")
        assert self._ttl_statements(emitter) == ['ALTER TABLE "frab.pairs" SET TTL 30 days']

    def test_unparseable_cap_is_applied_verbatim(self, emitter):
        # QuestDB is the authority on syntax; an unparseable spec skips the comparison, not the ALTER
        emitter._ddl_client_for_test.query.return_value = _ttl_frame(10, "DAY")
        emitter.ensure_table("frab.pairs", {"ev": "DOUBLE"}, max_ttl="3 fortnights")
        assert self._ttl_statements(emitter) == ['ALTER TABLE "frab.pairs" SET TTL 3 fortnights']

    def test_none_leaves_retention_alone(self, emitter):
        emitter.ensure_table("exec.fills", {"qty": "DOUBLE"}, max_ttl=None)
        assert _create_sql(emitter)  # table still declared
        assert self._ttl_statements(emitter) == []
        emitter._ddl_client_for_test.query.assert_not_called()


class TestEmitRecord:
    def test_scope_tags_overwrite_and_symbols_split(self, emitter):
        emitter.ensure_table("frab.trades", {"net_pnl": "DOUBLE", "trade_id": "STRING"}, symbol_columns=("pair",))
        emitter.emit_record(
            "frab.trades",
            {"pair": "SOL:BIN:HPL", "net_pnl": 1.5, "trade_id": "SOL:123", "strategy": "SPOOFED"},
            timestamp=datetime.datetime(2026, 7, 24, 12, 0, 0),
        )
        table, symbols, columns, at = emitter._sender.rows[0]
        assert table == "frab.trades"
        assert symbols["strategy"] == "bot-1"  # injected wins over caller "SPOOFED"
        assert symbols["pair"] == "SOL:BIN:HPL"
        assert columns["net_pnl"] == 1.5
        assert columns["trade_id"] == "SOL:123"
        assert "is_live" not in columns  # no context set -> is_live is not injected at all
        assert at == datetime.datetime(2026, 7, 24, 12, 0, 0)

    def test_none_values_skipped_and_datetime_columns_converted(self, emitter):
        emitter.emit_record("frab.trades", {"ev": None, "entry_time": datetime.datetime(2026, 7, 24)})
        _, _, columns, _ = emitter._sender.rows[0]
        assert "ev" not in columns
        assert isinstance(columns["entry_time"], datetime.datetime)

    def test_undeclared_table_uses_symbol_columns_arg(self, emitter):
        emitter.emit_record("frab.decisions", {"pair": "X", "ev": 1.0}, symbol_columns=("pair",))
        _, symbols, columns, _ = emitter._sender.rows[0]
        assert symbols["pair"] == "X"
        assert columns["ev"] == 1.0

    def test_errors_never_raise(self, emitter):
        emitter._sender = None
        emitter.emit_record("t", {"a": 1.0})  # early return, no raise

    def test_numpy_scalar_columns_pass_through_natively(self, emitter):
        emitter.emit_record("frab.trades", {"score": np.float64(1.5)})
        _, _, columns, _ = emitter._sender.rows[0]
        assert columns["score"] == 1.5
        assert isinstance(columns["score"], float) and not isinstance(columns["score"], np.floating)

    def test_numpy_datetime64_column_converted_to_datetime(self, emitter):
        emitter.emit_record("frab.trades", {"entry_time": np.datetime64("2026-07-24T12:00:00")})
        _, _, columns, _ = emitter._sender.rows[0]
        assert isinstance(columns["entry_time"], datetime.datetime)
        assert not isinstance(columns["entry_time"], np.datetime64)

    def test_warns_once_per_table_for_undeclared_keys(self, emitter, monkeypatch):
        warning = MagicMock()
        monkeypatch.setattr(qdb_mod.logger, "warning", warning)
        emitter.ensure_table("frab.trades", {"net_pnl": "DOUBLE"})

        emitter.emit_record("frab.trades", {"net_pnl": 1.0, "surprise": "x"})
        emitter.emit_record("frab.trades", {"net_pnl": 2.0, "surprise": "y"})

        assert len(emitter._sender.rows) == 2  # row is still written both times
        assert emitter._sender.rows[0][2]["net_pnl"] == 1.0
        assert emitter._sender.rows[1][2]["net_pnl"] == 2.0
        warning.assert_called_once()  # but the undeclared-column warning fires only once

    def test_undeclared_table_warns_once_across_multiple_emits(self, emitter, monkeypatch):
        warning = MagicMock()
        monkeypatch.setattr(qdb_mod.logger, "warning", warning)

        emitter.emit_record("frab.undeclared", {"ev": 1.0})
        emitter.emit_record("frab.undeclared", {"ev": 2.0})

        assert len(emitter._sender.rows) == 2  # rows are still written both times
        assert emitter._sender.rows[0][2]["ev"] == 1.0
        assert emitter._sender.rows[1][2]["ev"] == 2.0
        warning.assert_called_once()  # but the "not declared via ensure_table" warning fires only once
        msg = warning.call_args[0][0]
        assert "frab.undeclared" in msg
        assert "not declared via ensure_table" in msg

    def test_rejects_bare_string_symbol_columns(self, emitter):
        emitter.emit_record("t", {"a": 1.0}, symbol_columns="pair")  # must not raise (logged), no row emitted
        assert emitter._sender.rows == []

    def test_record_timestamp_key_is_dropped(self, emitter):
        emitter.emit_record("frab.trades", {"net_pnl": 1.0, "timestamp": datetime.datetime(2020, 1, 1)})
        _, _, columns, _ = emitter._sender.rows[0]
        assert "timestamp" not in columns
        assert columns["net_pnl"] == 1.0

    def test_unsupported_timestamp_type_is_caught_and_logged(self, emitter):
        emitter.emit_record("t", {"a": 1.0}, timestamp=object())  # must not raise (error logged)
        assert emitter._sender.rows == []  # nothing was queued/emitted


class TestCSVRecords:
    def _emitter(self, tmp_path):
        return CSVMetricEmitter(file_path=str(tmp_path / "metrics.csv"), tags={"strategy": "LoeTest"})

    def test_ensure_table_writes_a_header_file_per_table(self, tmp_path):
        em = self._emitter(tmp_path)
        em.ensure_table("loe.execution", {"price": "DOUBLE"}, symbol_columns=("kind", "symbol"))

        path = tmp_path / "loe.execution.csv"
        header = path.read_text().splitlines()[0].split(",")
        assert header[0] == "timestamp"
        assert {"kind", "symbol", "price", "strategy", "run_id", "is_live"} <= set(header)

    def test_emit_record_appends_one_row_per_call(self, tmp_path):
        em = self._emitter(tmp_path)
        em.ensure_table("loe.execution", {"price": "DOUBLE"}, symbol_columns=("kind", "symbol"))
        em.emit_record(
            "loe.execution",
            {"kind": "FILL", "symbol": "SLPUSDT", "price": 0.1001},
            timestamp=np.datetime64("2026-07-01T00:00:00", "ns"),
        )
        em.emit_record("loe.execution", {"kind": "PHASE_END", "symbol": "SLPUSDT"})

        rows = list(csv_mod.DictReader((tmp_path / "loe.execution.csv").read_text().splitlines()))
        assert [r["kind"] for r in rows] == ["FILL", "PHASE_END"]
        assert rows[0]["price"] == "0.1001"
        assert rows[0]["timestamp"].startswith("2026-07-01T00:00:00")
        assert rows[0]["strategy"] == "LoeTest"
        # a column the row does not carry stays blank rather than shifting the line
        assert rows[1]["price"] == ""

    def test_ensure_table_accepts_max_ttl_and_ignores_it(self, tmp_path):
        # a caller that declares a table without knowing which emitter it holds passes max_ttl;
        # a TypeError here would make it skip declaring, and the first row's columns would then
        # become the schema for every later row
        em = self._emitter(tmp_path)
        em.ensure_table("loe.execution", {"price": "DOUBLE"}, symbol_columns=("kind",), max_ttl="90 days")

        header = (tmp_path / "loe.execution.csv").read_text().splitlines()[0].split(",")
        assert {"kind", "price"} <= set(header)

    def test_a_declared_table_keeps_columns_absent_from_the_first_row(self, tmp_path):
        em = self._emitter(tmp_path)
        em.ensure_table("loe.execution", {"price": "DOUBLE", "filled_qty": "DOUBLE"}, max_ttl=None)
        em.emit_record("loe.execution", {"price": 0.1})  # first row lacks filled_qty
        em.emit_record("loe.execution", {"filled_qty": 100.0})

        rows = list(csv_mod.DictReader((tmp_path / "loe.execution.csv").read_text().splitlines()))
        assert rows[1]["filled_qty"] == "100.0"

    def test_emit_record_without_ensure_table_still_writes(self, tmp_path):
        em = self._emitter(tmp_path)
        em.emit_record("adhoc.table", {"a": 1.0})

        rows = list(csv_mod.DictReader((tmp_path / "adhoc.table.csv").read_text().splitlines()))
        assert rows[0]["a"] == "1.0"

    def test_rejects_bare_string_symbol_columns(self, tmp_path):
        # a bare string iterates as characters -> would mint bogus single-letter columns in the header
        em = self._emitter(tmp_path)
        em.ensure_table("t", {}, symbol_columns="pair")  # must not raise (logged), no file written
        assert not (tmp_path / "t.csv").exists()

    def test_a_write_failure_does_not_raise(self, tmp_path):
        em = self._emitter(tmp_path)
        em.ensure_table("loe.execution", {"price": "DOUBLE"})
        em._record_path = lambda table: tmp_path / "no_such_dir" / "x" / "y.csv"
        em.emit_record("loe.execution", {"price": 1.0})  # must not raise


class TestTtlHours:
    @pytest.mark.parametrize(
        "spec,hours",
        [
            ("30 days", 720.0),
            ("52 weeks", 8736.0),
            ("4 hours", 4.0),
            ("1 day", 24.0),
            ("6 months", 4320.0),
            ("1 year", 8760.0),
            ("30d", 720.0),
            ("4h", 4.0),
            ("2w", 336.0),
            ("6M", 4320.0),
            ("1y", 8760.0),
            ("30 DAYS", 720.0),
            ("  30 days ", 720.0),
        ],
    )
    def test_parses_questdb_ttl_specs(self, spec, hours):
        assert ttl_hours(spec) == hours

    @pytest.mark.parametrize("spec", ["", "days", "30", "30 minutes", "30m", "-1 day", "1.5 days"])
    def test_rejects_unparseable_specs(self, spec):
        with pytest.raises(ValueError):
            ttl_hours(spec)
