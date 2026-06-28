"""Tests for QUBX_DEBUG_AREAS — per-area DEBUG gating without flipping the global level."""

from qubx import QubxLogConfig, _area_matches, connector_logger, debug_enabled, logger


def test_area_matches_hierarchy():
    # parent enables dotted children; a specific child enables only itself
    assert _area_matches("connector.ccxt", {"connector"})
    assert _area_matches("connector.ccxt", {"connector.ccxt"})
    assert _area_matches("connector.binance.um", {"connector.binance"})
    assert not _area_matches("connector.ccxt", {"connector.hyperliquid"})
    assert not _area_matches("connectorx", {"connector"})  # not a dotted child
    assert _area_matches("account_manager", {"account_manager", "connector"})
    assert not _area_matches("account_manager", {"connector"})
    assert not _area_matches(None, {"connector"})
    assert not _area_matches("connector", set())


def test_get_debug_areas_parses_env(monkeypatch):
    monkeypatch.setenv("QUBX_DEBUG_AREAS", " account_manager , connector ,")
    assert QubxLogConfig.get_debug_areas() == {"account_manager", "connector"}
    monkeypatch.setenv("QUBX_DEBUG_AREAS", "")
    assert QubxLogConfig.get_debug_areas() == set()


def test_setup_logger_drives_debug_enabled(monkeypatch):
    monkeypatch.setenv("QUBX_DEBUG_AREAS", "connector")
    try:
        QubxLogConfig.setup_logger("INFO")
        assert debug_enabled("connector")
        assert debug_enabled("connector.ccxt")  # child of an enabled parent
        assert not debug_enabled("account_manager")
    finally:
        monkeypatch.delenv("QUBX_DEBUG_AREAS", raising=False)
        QubxLogConfig.setup_logger("INFO")
        assert not debug_enabled("connector")


def test_connector_logger_tags_lowercased_area():
    records = []
    sink_id = logger.add(
        lambda m: records.append(m.record),
        level="DEBUG",
        filter=lambda r: str(r["extra"].get("area", "")).startswith("connector"),
    )
    try:
        connector_logger("Hyperliquid").debug("hi")
        assert any(r["extra"].get("area") == "connector.hyperliquid" for r in records)
    finally:
        logger.remove(sink_id)
