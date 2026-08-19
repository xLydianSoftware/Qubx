"""Tests for QuestDBMetricEmitter: BoundedWorker + ILP Sender timeouts (spec: 2026-08-19-safe-state-persistence)."""

from unittest.mock import MagicMock, patch

import qubx.emitters.questdb as qdb_mod
from qubx.emitters.questdb import QuestDBMetricEmitter
from qubx.utils.threading import BoundedWorker


def _make_emitter(**kwargs) -> QuestDBMetricEmitter:
    """Construct a QuestDBMetricEmitter with QuestDBClient/Sender mocked out (no real network calls).

    Mirrors the fixtures in tests/qubx/emitters/metric_emitters_test.py: QuestDBClient is used
    directly by _ensure_signals_table_exists/_ensure_deals_table_exists (bypassing the Sender
    mock), and would otherwise attempt real DNS resolution against host="qdb.local".
    """
    mock_client = MagicMock()
    mock_client.return_value.execute.return_value = None
    mock_sender = MagicMock()
    mock_sender.from_conf.return_value = mock_sender
    mock_sender.establish.return_value = None
    with patch.object(qdb_mod, "QuestDBClient", mock_client), patch.object(qdb_mod, "Sender", mock_sender):
        return QuestDBMetricEmitter(host="qdb.local", port=9000, **kwargs)


def test_conn_string_has_bounded_timeouts():
    em = _make_emitter()
    assert "request_timeout=5000;" in em._conn_str
    assert "retry_timeout=5000;" in em._conn_str
    em._worker.stop()


def test_uses_bounded_worker():
    em = _make_emitter(max_queue=123)
    assert isinstance(em._worker, BoundedWorker)
    assert em._worker._maxlen == 123
    em._worker.stop()
