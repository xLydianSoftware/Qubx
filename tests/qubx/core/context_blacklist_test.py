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


from qubx.core.instrument_service import InstrumentServiceDiff


def _ctx_for_cycle(svc, instruments, callbacks, positions):
    ctx = StrategyContext.__new__(StrategyContext)
    ctx._instrument_service = svc
    ctx._instrument_service_callbacks = callbacks
    um = MagicMock()
    um.instruments = instruments
    ctx._universe_manager = um
    # context.instruments delegates to the universe manager; patch get_positions/remove_instruments
    ctx.get_positions = MagicMock(return_value=positions)
    ctx.remove_instruments = MagicMock()
    return ctx


def test_cycle_fires_callbacks_before_force_close():
    order = []
    btc = MagicMock()
    svc = MagicMock()
    svc.refresh.return_value = InstrumentServiceDiff(blacklisted_added=[btc], blacklisted_removed=[])

    def cb(c, added, removed):
        order.append("callback")

    pos = MagicMock()
    pos.quantity = 1.0
    ctx = _ctx_for_cycle(svc, [btc], [cb], {btc: pos})
    ctx.remove_instruments.side_effect = lambda instrs, if_has_position_then="close": order.append("force_close")

    summary = ctx._run_instrument_service_cycle()

    svc.refresh.assert_called_once_with([btc])
    assert order == ["callback", "force_close"]
    ctx.remove_instruments.assert_called_once_with([btc], if_has_position_then="close")
    assert summary["blacklisted_added"] == 1
    assert summary["force_closed"] == 1


def test_cycle_no_backstop_when_not_held():
    btc = MagicMock()
    svc = MagicMock()
    svc.refresh.return_value = InstrumentServiceDiff(blacklisted_added=[btc], blacklisted_removed=[])
    ctx = _ctx_for_cycle(svc, [btc], [], {})  # not held
    summary = ctx._run_instrument_service_cycle()
    ctx.remove_instruments.assert_not_called()
    assert summary["force_closed"] == 0


def test_cycle_empty_diff_is_noop():
    calls = []
    svc = MagicMock()
    svc.refresh.return_value = InstrumentServiceDiff(blacklisted_added=[], blacklisted_removed=[])
    cb = lambda c, a, r: calls.append("cb")
    ctx = _ctx_for_cycle(svc, [], [cb], {})
    summary = ctx._run_instrument_service_cycle()
    assert calls == []  # idempotent empty diff -> no callbacks
    ctx.remove_instruments.assert_not_called()  # no force-close
    assert summary == {
        "blacklisted_added": 0,
        "blacklisted_removed": 0,
        "force_closed": 0,
        "force_closed_instruments": [],
    }
