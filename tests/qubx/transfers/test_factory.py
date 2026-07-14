import pytest

from qubx.core.interfaces import ITransferManager
from qubx.transfers import (
    TRANSFER_MANAGER_REGISTRY,
    XChangesTransferService,
    create_transfer_manager,
    register_transfer_manager,
)

_PARAMS = {"base_url": "http://svc/api", "provider": "XLYDIAN", "user": "bohdan"}


class _StubTransferManager(ITransferManager):
    def __init__(self, tag: str = "", is_simulation=None):
        self.tag = tag
        self.is_simulation = is_simulation


class _NoGuardStub(ITransferManager):
    def __init__(self, tag: str = ""):
        self.tag = tag


class TestCreateTransferManager:
    def test_creates_registered_type(self):
        assert isinstance(create_transfer_manager("xchanges", dict(_PARAMS)), XChangesTransferService)

    def test_type_match_is_case_insensitive(self):
        assert isinstance(create_transfer_manager("XChanges", dict(_PARAMS)), XChangesTransferService)

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown transfer manager type: bogus.*Available types.*xchanges"):
            create_transfer_manager("bogus")

    def test_bad_parameter_name_raises_type_error(self):
        # no silent param filtering: a mistyped kwarg must fail startup, not drop a safety cap
        with pytest.raises(TypeError):
            create_transfer_manager("xchanges", dict(_PARAMS, bogus_param=1))

    def test_is_simulation_injected_when_constructor_accepts(self):
        guard = lambda: True
        svc = create_transfer_manager("xchanges", dict(_PARAMS), is_simulation=guard)
        assert isinstance(svc, XChangesTransferService)
        with pytest.raises(RuntimeError, match="simulation"):
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0)


class TestRegisterTransferManager:
    def test_register_lowercases_key_and_builds(self):
        register_transfer_manager("StubXfer", _StubTransferManager)
        try:
            assert TRANSFER_MANAGER_REGISTRY["stubxfer"] is _StubTransferManager
            guard = lambda: True
            tm = create_transfer_manager("STUBXFER", {"tag": "x"}, is_simulation=guard)
            assert isinstance(tm, _StubTransferManager)
            assert tm.tag == "x"
            assert tm.is_simulation is guard
        finally:
            TRANSFER_MANAGER_REGISTRY.pop("stubxfer", None)

    def test_registered_type_without_guard_param_skips_injection(self):
        register_transfer_manager("noguard", _NoGuardStub)
        try:
            tm = create_transfer_manager("noguard", is_simulation=lambda: True)
            assert isinstance(tm, _NoGuardStub)
        finally:
            TRANSFER_MANAGER_REGISTRY.pop("noguard", None)
