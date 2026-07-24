"""Tests for strategy-owned tables: ensure_table + emit_record (spec §2)."""

from unittest.mock import MagicMock

from qubx.core.interfaces import IMetricEmitter
from qubx.emitters.composite import CompositeMetricEmitter


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
