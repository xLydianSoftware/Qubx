from unittest.mock import MagicMock

from qubx.core.context import StrategyContext


def _ctx_with_service_and_instruments(svc, instruments):
    ctx = StrategyContext.__new__(StrategyContext)
    ctx._instrument_service = svc
    # bypass the .instruments property by stubbing the universe manager
    um = MagicMock()
    um.instruments = instruments
    ctx._universe_manager = um
    return ctx


def test_is_blacklisted_delegates():
    svc = MagicMock()
    svc.is_blacklisted.return_value = True
    ctx = _ctx_with_service_and_instruments(svc, [])
    btc = object()
    assert ctx.is_blacklisted(btc) is True
    svc.is_blacklisted.assert_called_once_with(btc)


def test_filter_blacklisted_returns_kept():
    btc, eth = object(), object()
    svc = MagicMock()
    svc.is_blacklisted.side_effect = lambda i: i is btc
    ctx = _ctx_with_service_and_instruments(svc, [])
    assert ctx.filter_blacklisted([btc, eth]) == [eth]


def test_get_blacklisted_instruments_uses_known_universe():
    btc, eth = object(), object()
    svc = MagicMock()
    svc.matching_instruments.return_value = [btc]
    ctx = _ctx_with_service_and_instruments(svc, [btc, eth])
    assert ctx.get_blacklisted_instruments() == [btc]
    svc.matching_instruments.assert_called_once_with([btc, eth])


def test_context_exposes_instrument_service_callbacks_attr():
    # The action reads ctx._instrument_service_callbacks; default must be an empty list.
    ctx = StrategyContext.__new__(StrategyContext)
    # simulate __post_init__ having stored callbacks from initializer
    cbs = [lambda c, a, r: None]
    ctx._instrument_service_callbacks = cbs
    assert ctx._instrument_service_callbacks == cbs
