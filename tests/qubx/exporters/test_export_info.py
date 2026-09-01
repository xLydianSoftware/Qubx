"""get_export_info() — the exporter's self-description consumed by the runtime-info blob."""

from qubx.core.interfaces import ITradeDataExport
from qubx.exporters.composite import CompositeExporter
from qubx.exporters.redis_streams import RedisStreamsExporter

# redis.from_url is lazy (no connection until a command runs), so constructing
# the exporter against a dead URL is fine — same approach as the bounded-worker tests.
DEAD_REDIS = "redis://localhost:6399/0"


def test_default_is_empty():
    assert ITradeDataExport().get_export_info() == {}


def test_redis_exporter_reports_only_enabled_streams():
    exporter = RedisStreamsExporter(
        redis_url=DEAD_REDIS,
        strategy_name="binance.factors-buf",
        export_position_changes=True,
        position_changes_stream="strategy:binance.factors-buf:position_changes",
    )
    try:
        assert exporter.get_export_info() == {
            "position_changes": ["strategy:binance.factors-buf:position_changes"]
        }
    finally:
        exporter.stop()


def test_redis_exporter_defaults_streams_from_strategy_name():
    exporter = RedisStreamsExporter(
        redis_url=DEAD_REDIS,
        strategy_name="mybot",
        export_signals=True,
        export_targets=True,
    )
    try:
        assert exporter.get_export_info() == {
            "signals": ["strategy:mybot:signals"],
            "targets": ["strategy:mybot:targets"],
        }
    finally:
        exporter.stop()


def test_composite_concatenates_children():
    a = RedisStreamsExporter(
        redis_url=DEAD_REDIS, strategy_name="a", export_position_changes=True
    )
    b = RedisStreamsExporter(
        redis_url=DEAD_REDIS, strategy_name="b", export_position_changes=True, export_signals=True
    )
    composite = CompositeExporter([a, b])
    try:
        assert composite.get_export_info() == {
            "position_changes": [
                "strategy:a:position_changes",
                "strategy:b:position_changes",
            ],
            "signals": ["strategy:b:signals"],
        }
    finally:
        composite.stop()
