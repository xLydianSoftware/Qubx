import subprocess
import sys

import pytest

from qubx.data.registry import StorageRegistry
from qubx.data.storages._missing_extra import register_missing_extra


@pytest.fixture
def clean_registry():
    yield
    StorageRegistry._storages.pop("fake_lake", None)


def test_a_missing_extra_storage_names_the_extra_on_use(clean_registry):
    error = ModuleNotFoundError("No module named 'fakepkg'", name="fakepkg")
    register_missing_extra("fake_lake", "qubx[fake]", error)
    with pytest.raises(ImportError, match=r"qubx\[fake\]") as info:
        StorageRegistry.get("fake_lake::acct")
    assert info.value.__cause__ is error


def test_a_real_storage_is_never_replaced_by_the_placeholder(clean_registry):
    @StorageRegistry.register("fake_lake")
    class Real:
        def __init__(self, *args, **kwargs):
            pass

    register_missing_extra("fake_lake", "qubx[fake]", ModuleNotFoundError("x", name="fakepkg"))
    assert StorageRegistry.get_class("fake_lake") is Real


def test_iceberg_is_registered_when_pyiceberg_is_installed():
    pytest.importorskip("pyiceberg")
    import qubx.data  # noqa: F401
    from qubx.data.storages.iceberg import IcebergLakeStorage

    assert StorageRegistry.get_class("iceberg") is IcebergLakeStorage


_BLOCKER = """
import sys


class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {name!r}:
            raise {error}("blocked by the test", name={name!r})


sys.meta_path.insert(0, Block())
"""

_PROBE = """
import qubx.data
from qubx.data.registry import StorageRegistry

try:
    StorageRegistry.get("iceberg::x")
except ImportError as e:
    print("PLACEHOLDER:", e)
"""


def _import_qubx_data_blocking(name: str, error: str) -> subprocess.CompletedProcess:
    code = _BLOCKER.format(name=name, error=error) + _PROBE
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize(
    "name, error",
    [
        ("pyiceberg", "ModuleNotFoundError"),
        # - a pyiceberg other than the pinned one, where the private symbol the reader imports has moved
        ("pyiceberg.io.pyarrow", "ImportError"),
    ],
)
def test_a_missing_or_mismatched_pyiceberg_leaves_qubx_data_importable(name, error):
    run = _import_qubx_data_blocking(name, error)
    assert run.returncode == 0, run.stderr
    assert "PLACEHOLDER:" in run.stdout
    assert "qubx[iceberg]" in run.stdout
    assert "pyiceberg==0.12.0" in run.stdout


def test_a_non_pyiceberg_import_error_in_the_iceberg_storage_still_propagates():
    run = _import_qubx_data_blocking("duckdb", "ImportError")
    assert run.returncode != 0
    assert "storages/iceberg.py" in run.stderr
    assert "blocked by the test" in run.stderr
