"""Soft-default contract for IAccountViewer per-instrument settings.

BasicAccountProcessor must return None (or float('inf') for the notional cap)
for the four per-instrument-settings getters, with no dependency on
undeclared subclass attributes. Live processors override these.
"""

from unittest.mock import MagicMock

from qubx.core.account import BasicAccountProcessor


def _bare_processor() -> BasicAccountProcessor:
    """Construct a minimal BasicAccountProcessor by bypassing __init__.

    We're testing the *getter contract*, not init wiring, so it's fine to
    sidestep the constructor and avoid pulling in TCC / health monitor stubs.
    """
    proc = BasicAccountProcessor.__new__(BasicAccountProcessor)
    return proc


def test_get_instrument_leverage_default_is_none():
    proc = _bare_processor()
    assert proc.get_instrument_leverage(MagicMock()) is None


def test_get_max_instrument_leverage_default_is_none():
    proc = _bare_processor()
    assert proc.get_max_instrument_leverage(MagicMock()) is None


def test_get_max_instrument_notional_default_is_inf():
    proc = _bare_processor()
    assert proc.get_max_instrument_notional(MagicMock()) == float("inf")


def test_get_margin_mode_default_is_none():
    proc = _bare_processor()
    assert proc.get_margin_mode(MagicMock()) is None


def test_no_undeclared_cache_attribute_required():
    """The base class must not depend on subclass-defined private caches.

    Regression guard: previously the base read `getattr(self, "_*_cache", None)`
    against attributes only subclasses defined.  Verify a subclass that
    deliberately omits all caches still gets the soft defaults.
    """
    class _NoCache(BasicAccountProcessor):  # type: ignore[misc]
        pass

    proc = _NoCache.__new__(_NoCache)
    instr = MagicMock()
    assert proc.get_instrument_leverage(instr) is None
    assert proc.get_max_instrument_leverage(instr) is None
    assert proc.get_max_instrument_notional(instr) == float("inf")
    assert proc.get_margin_mode(instr) is None
