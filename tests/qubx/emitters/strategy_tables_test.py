"""Tests for strategy-owned tables: ensure_table + emit_record (spec §2)."""

import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

import qubx.emitters.questdb as qdb_mod
from qubx.core.interfaces import IMetricEmitter
from qubx.emitters.composite import CompositeMetricEmitter
from qubx.emitters.questdb import QuestDBMetricEmitter


class TestInterfaceDefaults:
    def test_ensure_table_is_noop(self):
        IMetricEmitter().ensure_table("t", {"a": "DOUBLE"})  # must not raise

    def test_emit_record_is_noop(self):
        IMetricEmitter().emit_record("t", {"a": 1.0})  # must not raise


class TestCompositeForwarding:
    def test_forwards_to_children(self):
        child_a, child_b = MagicMock(spec=IMetricEmitter), MagicMock(spec=IMetricEmitter)
        comp = CompositeMetricEmitter([child_a, child_b])
        comp.ensure_table("frab.trades", {"net_pnl": "DOUBLE"}, symbol_columns=("pair",), dedup_keys=("timestamp", "trade_id"))
        comp.emit_record("frab.trades", {"net_pnl": 1.5}, symbol_columns=("pair",))
        for child in (child_a, child_b):
            child.ensure_table.assert_called_once_with(
                "frab.trades", {"net_pnl": "DOUBLE"}, symbol_columns=("pair",), dedup_keys=("timestamp", "trade_id"), partition_by="DAY"
            )
            child.emit_record.assert_called_once_with("frab.trades", {"net_pnl": 1.5}, symbol_columns=("pair",), timestamp=None)

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


class SyncExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)

    def shutdown(self, **kwargs):
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
    em._executor = SyncExecutor()
    ddl.reset_mock()  # drop the signals/deals DDL calls from __init__
    em._ddl_client_for_test = ddl
    return em


class TestEnsureTable:
    def test_generates_create_table_with_scope_columns_and_dedup(self, emitter):
        emitter.ensure_table(
            "frab.trades",
            {"trade_id": "STRING", "entry_time": "TIMESTAMP", "net_pnl": "DOUBLE"},
            symbol_columns=("pair", "asset"),
            dedup_keys=("timestamp", "trade_id"),
        )
        ddl_sql = emitter._ddl_client_for_test.execute.call_args[0][0]
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
        ddl_sql = emitter._ddl_client_for_test.execute.call_args[0][0]
        # run_id is a reserved scope column (STRING) -> symbol_columns must not flip it to SYMBOL
        assert '"run_id" STRING' in ddl_sql
        assert '"pair" SYMBOL' in ddl_sql

    def test_rejects_bare_string_symbol_columns(self, emitter):
        # a bare string iterates as characters ("p", "a", "i", "r", ...) -> would mint bogus
        # single-letter SYMBOL columns; must be rejected instead, not silently misinterpreted.
        emitter.ensure_table("t", {}, symbol_columns="pair")  # must not raise (logged), no DDL executed
        emitter._ddl_client_for_test.execute.assert_not_called()


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
