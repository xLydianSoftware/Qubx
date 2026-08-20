"""
The QuestDB emitter must not widen its tables.

Any tag it does not declare would otherwise become a permanent column under ILP
auto-create, for every bot writing to the same server.
"""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from qubx.emitters.questdb import QuestDBMetricEmitter


@pytest.fixture
def emitter():
    with patch("qubx.emitters.questdb.Sender"), patch("qubx.emitters.questdb.QuestDBClient"):
        em = QuestDBMetricEmitter(tags={"strategy": "s", "environment": "prod"})
    em._sender = MagicMock()
    return em


def _row(emitter) -> tuple[str, dict, dict]:
    call = emitter._sender.row.call_args
    return call.args[0], call.kwargs["symbols"], call.kwargs["columns"]


def test_undeclared_tags_go_to_custom(emitter):
    emitter._emit_to_questdb(
        "spread", 1.5, {"strategy": "s", "type": "stats", "pair": "ADA:A:B", "lookback": "30d"}, None
    )

    table, symbols, columns = _row(emitter)
    assert table == "qubx.metrics"
    assert symbols == {"strategy": "s", "type": "stats"}
    assert columns["custom"] == '{"lookback": "30d", "pair": "ADA:A:B"}'
    assert "pair" not in columns and "lookback" not in columns


def test_custom_is_absent_when_every_tag_is_declared(emitter):
    emitter._emit_to_questdb("total_capital", 1.0, {"strategy": "s", "type": "stats"}, None)

    _, _, columns = _row(emitter)
    assert "custom" not in columns


def test_health_rows_go_to_the_health_table(emitter):
    emitter._emit_to_questdb(
        "context_degradation", 1.0, {"type": "health", "reason": "QUEUE_LAG", "scope": "BINANCE.UM"}, None
    )

    table, symbols, columns = _row(emitter)
    assert table == "qubx.health"
    assert symbols["reason"] == "QUEUE_LAG"
    assert symbols["scope"] == "BINANCE.UM"
    assert "custom" not in columns  # - the health table does not declare it


def test_rate_limit_rows_go_to_their_own_table(emitter):
    emitter._emit_to_questdb(
        "rate_limit.utilization",
        0.42,
        {"type": "rate_limit", "exchange": "binance.pm", "pool": "orders", "scope": "account"},
        None,
    )

    table, symbols, _ = _row(emitter)
    assert table == "qubx.rate_limits"
    assert symbols["pool"] == "orders"
    assert symbols["scope"] == "account"


def test_pool_and_scope_never_reach_the_metrics_table(emitter):
    # - they are rate-limit tags; on qubx.metrics they would be NULL on every other row
    assert "pool" not in QuestDBMetricEmitter.METRICS_COLUMNS
    assert "scope" not in QuestDBMetricEmitter.METRICS_COLUMNS
    assert "event_type" not in QuestDBMetricEmitter.METRICS_COLUMNS


def test_signals_and_deals_do_not_declare_custom():
    assert "custom" not in QuestDBMetricEmitter.SIGNALS_COLUMNS
    assert "custom" not in QuestDBMetricEmitter.DEALS_COLUMNS
    assert "custom" in QuestDBMetricEmitter.METRICS_COLUMNS


def test_timestamp_is_the_designated_column_of_every_reserved_table():
    for schema in (
        QuestDBMetricEmitter.METRICS_COLUMNS,
        QuestDBMetricEmitter.SIGNALS_COLUMNS,
        QuestDBMetricEmitter.DEALS_COLUMNS,
        QuestDBMetricEmitter.HEALTH_COLUMNS,
        QuestDBMetricEmitter.RATE_LIMITS_COLUMNS,
    ):
        assert next(iter(schema)) == "timestamp"


def test_unknown_type_falls_back_to_the_metrics_table(emitter):
    emitter._emit_to_questdb("proba", 0.7, {"type": "prediction", "predictor": "m1"}, None)

    table, _, columns = _row(emitter)
    assert table == "qubx.metrics"
    assert columns["custom"] == '{"predictor": "m1"}'


def test_timestamp_is_passed_through(emitter):
    ts = datetime.datetime(2026, 8, 20, 1, 11, 0)
    emitter._emit_to_questdb("total_capital", 1.0, {"type": "stats"}, ts)

    assert emitter._sender.row.call_args.kwargs["at"] == ts


def test_ensure_table_refuses_a_reserved_name(emitter):
    with patch("qubx.emitters.questdb.QuestDBClient") as client:
        emitter.ensure_table("qubx.metrics", columns={"v": "DOUBLE"})

    client.assert_not_called()


def test_emit_record_refuses_a_reserved_name(emitter):
    emitter.emit_record("qubx.health", {"v": 1.0})

    emitter._sender.row.assert_not_called()


def test_a_strategy_table_is_still_accepted(emitter):
    with patch("qubx.emitters.questdb.QuestDBClient") as client:
        emitter.ensure_table("loe.execution", columns={"v": "DOUBLE"})

    client.assert_called_once()
    assert "loe.execution" in emitter._declared_columns
