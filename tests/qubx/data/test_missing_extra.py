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
