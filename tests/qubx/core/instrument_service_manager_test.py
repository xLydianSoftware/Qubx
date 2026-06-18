from unittest.mock import MagicMock

from qubx.core.instrument_service import HttpInstrumentService, InstrumentServiceDiff, NullInstrumentService
from qubx.core.mixins.instrument_service import InstrumentServiceManager


def _mgr(service, instruments=None, positions=None, callbacks=None):
    ctx = MagicMock()
    ctx.instruments = instruments or []
    ctx.get_positions.return_value = positions or {}
    m = InstrumentServiceManager(ctx, service)
    if callbacks is not None:
        m.set_callbacks(callbacks)
    return m, ctx


def test_read_helpers_delegate_to_service():
    btc, eth = object(), object()
    svc = MagicMock()
    svc.is_blacklisted.side_effect = lambda i: i is btc
    svc.matching_instruments.return_value = [btc]
    m, ctx = _mgr(svc, instruments=[btc, eth])
    assert m.is_blacklisted(btc) is True
    assert m.filter_blacklisted([btc, eth]) == [eth]
    assert m.get_blacklisted_instruments() == [btc]
    svc.matching_instruments.assert_called_once_with([btc, eth])


def test_run_cycle_fires_callbacks_before_force_close():
    order = []
    btc = MagicMock()
    pos = MagicMock(); pos.quantity = 1.0
    svc = MagicMock()
    svc.refresh.return_value = InstrumentServiceDiff(blacklisted_added=[btc], blacklisted_removed=[])
    m, ctx = _mgr(svc, instruments=[btc], positions={btc: pos}, callbacks=[lambda c, a, r: order.append("callback")])
    ctx.remove_instruments.side_effect = lambda instrs, if_has_position_then="close": order.append("force_close")
    summary = m.run_cycle()
    svc.refresh.assert_called_once_with([btc])
    assert order == ["callback", "force_close"]
    ctx.remove_instruments.assert_called_once_with([btc], if_has_position_then="close")
    assert summary["blacklisted_added"] == 1
    assert summary["force_closed"] == 1


def test_run_cycle_no_backstop_when_not_held():
    btc = MagicMock()
    svc = MagicMock()
    svc.refresh.return_value = InstrumentServiceDiff(blacklisted_added=[btc], blacklisted_removed=[])
    m, ctx = _mgr(svc, instruments=[btc], positions={}, callbacks=[])
    summary = m.run_cycle()
    ctx.remove_instruments.assert_not_called()
    assert summary["force_closed"] == 0


def test_run_cycle_empty_diff_is_noop():
    calls = []
    svc = MagicMock()
    svc.refresh.return_value = InstrumentServiceDiff(blacklisted_added=[], blacklisted_removed=[])
    m, ctx = _mgr(svc, callbacks=[lambda c, a, r: calls.append("cb")])
    summary = m.run_cycle()
    assert calls == []
    ctx.remove_instruments.assert_not_called()
    assert summary == {"blacklisted_added": 0, "blacklisted_removed": 0, "force_closed": 0, "force_closed_instruments": []}


def test_run_cycle_fires_callbacks_on_entries_change_with_empty_universe_diff():
    # Removing an off-universe blacklist entry yields no added/removed (the instrument is no
    # longer in the universe) but entries_changed is True -> the re-fit callback must fire so
    # the strategy re-selects and can bring the un-blacklisted instrument back.
    calls = []
    svc = MagicMock()
    svc.refresh.return_value = InstrumentServiceDiff(
        blacklisted_added=[], blacklisted_removed=[], entries_changed=True
    )
    svc.is_blacklisted.return_value = False
    m, ctx = _mgr(svc, positions={}, callbacks=[lambda c, a, r: calls.append((list(a), list(r)))])
    m.run_cycle()
    assert calls == [([], [])]
    ctx.remove_instruments.assert_not_called()


def test_run_cycle_accepts_scheduler_ctx_arg():
    # the scheduler dispatches scheduled methods as method(ctx); run_cycle must accept it
    svc = MagicMock()
    svc.refresh.return_value = InstrumentServiceDiff(blacklisted_added=[], blacklisted_removed=[])
    m, ctx = _mgr(svc)
    result = m.run_cycle(ctx)
    assert result["blacklisted_added"] == 0


def test_start_registers_startup_oneshot_but_not_poll_when_non_null():
    svc = MagicMock(spec=HttpInstrumentService)
    m, ctx = _mgr(svc)
    m.start()
    ctx.delay.assert_called_once_with("1s", m.run_cycle)
    ctx.schedule.assert_not_called()


def test_start_noop_with_null_service():
    m, ctx = _mgr(NullInstrumentService())
    m.start()
    ctx.delay.assert_not_called()
    ctx.schedule.assert_not_called()


def test_set_callbacks_preserves_order():
    m, ctx = _mgr(MagicMock())
    a = lambda c, ad, rm: None
    b = lambda c, ad, rm: None
    m.set_callbacks([a, b])
    assert m._callbacks == [a, b]


def test_enforce_at_fit_refreshes_and_force_closes_without_callbacks():
    ondo = MagicMock()
    pos = MagicMock(); pos.quantity = -891.4
    svc = MagicMock()
    svc.refresh.return_value = InstrumentServiceDiff(blacklisted_added=[], blacklisted_removed=[])
    svc.is_blacklisted.side_effect = lambda i: i is ondo
    calls = []
    m, ctx = _mgr(svc, instruments=[ondo], positions={ondo: pos},
                  callbacks=[lambda c, a, r: calls.append("cb")])
    assert m.enforce_at_fit() is None
    svc.refresh.assert_called_once_with([ondo])
    assert calls == []  # no change callbacks at fit time
    ctx.remove_instruments.assert_called_once_with([ondo], if_has_position_then="close")


def test_enforce_at_fit_noop_with_null_service():
    m, ctx = _mgr(NullInstrumentService())
    assert m.enforce_at_fit() is None
    ctx.remove_instruments.assert_not_called()


def test_run_cycle_force_closes_all_held_blacklisted_not_just_delta():
    # WLFI regression: an already-blacklisted holding (absent from the change delta)
    # must still be force-closed.
    ondo, wlfi, btc = MagicMock(), MagicMock(), MagicMock()
    pos = MagicMock(); pos.quantity = -34767.0
    btc_pos = MagicMock(); btc_pos.quantity = 1.0
    svc = MagicMock()
    # Only ONDO is newly-added this cycle; WLFI was blacklisted earlier.
    svc.refresh.return_value = InstrumentServiceDiff(blacklisted_added=[ondo], blacklisted_removed=[])
    svc.is_blacklisted.side_effect = lambda i: i in (ondo, wlfi)
    m, ctx = _mgr(svc, instruments=[wlfi, btc], positions={wlfi: pos, btc: btc_pos})
    summary = m.run_cycle()
    ctx.remove_instruments.assert_called_once_with([wlfi], if_has_position_then="close")
    assert summary["force_closed"] == 1
    assert summary["force_closed_instruments"] == [str(wlfi)]
