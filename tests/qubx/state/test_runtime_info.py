"""build_runtime_info — the static blob written once to state:{bot_id}:info."""

import json
import sys

from qubx.core.interfaces import ITradeDataExport
from qubx.state.runtime_info import (
    RUNTIME_INFO_KEY,
    RUNTIME_INFO_SCHEMA_VERSION,
    build_runtime_info,
)


class _FakeStrategy:
    pass


class _FakeExporter(ITradeDataExport):
    def get_export_info(self) -> dict[str, list[str]]:
        return {"position_changes": ["strategy:x:position_changes"]}


class _BrokenExporter(ITradeDataExport):
    def get_export_info(self) -> dict[str, list[str]]:
        raise RuntimeError("boom")


def test_blob_shape_and_versions():
    info = build_runtime_info(_FakeStrategy(), _FakeExporter(), "2026-09-01T09:00:00")

    assert info["schema_version"] == RUNTIME_INFO_SCHEMA_VERSION
    assert info["started_at"] == "2026-09-01T09:00:00"
    assert info["python"] == ".".join(str(v) for v in sys.version_info[:3])
    # qubx is installed in the test env, so both the top-level field and the
    # packages dict must carry a real version string.
    assert info["qubx"] == info["packages"]["qubx"]
    assert info["qubx"][0].isdigit() or info["qubx"] == "Dev"
    assert info["exports"] == {"position_changes": ["strategy:x:position_changes"]}
    # The strategy class is recorded fully qualified; the test strategy lives in
    # this test module, which belongs to no installed distribution -> version None.
    assert info["strategy"]["class"].endswith("._FakeStrategy")
    assert info["strategy"]["version"] is None


def test_blob_is_json_serializable():
    # SafeStatePersistence.save does an eager json.dumps; a non-serializable
    # value would silently lose the blob at runtime.
    json.dumps(build_runtime_info(_FakeStrategy(), _FakeExporter(), "t"))


def test_no_exporter_and_broken_exporter_degrade_to_empty_exports():
    assert build_runtime_info(_FakeStrategy(), None, "t")["exports"] == {}
    assert build_runtime_info(_FakeStrategy(), _BrokenExporter(), "t")["exports"] == {}


def test_key_constant():
    assert RUNTIME_INFO_KEY == "info"
