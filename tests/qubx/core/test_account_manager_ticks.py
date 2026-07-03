from unittest.mock import MagicMock

import numpy as np

from qubx import logger
from qubx.core.account_manager import AccountManager, AccountManagerConfig
from qubx.core.basics import Order, OrderChange, OrderOrigin, OrderStatus
from qubx.core.mixins.processing import ProcessingManager
from tests.qubx.core.conftest import make_pm


class _T:
    def __init__(self):
        self.t = np.datetime64("2026-05-28T00:00:00")

    def time(self):
        return self.t

    def adv(self, ms):
        self.t = self.t + np.timedelta64(ms, "ms")


def _real_pm(am: AccountManager) -> ProcessingManager:
    # A REAL ProcessingManager wired to the real AM — the give-up path must exercise the
    # genuine process_event -> apply -> _safe_call dispatch (error isolation included),
    # not a mock that would skip the apply.
    return make_pm(_account_manager=am)


def _am(connectors, cfg=None):
    am = AccountManager(
        connectors=connectors,
        base_currencies={ex: "USDT" for ex in connectors},
        time=_T(),
        cfg=cfg or AccountManagerConfig(missing_order_wait_ms=5_000, missing_order_retries=3),
        account_id="test",
    )
    # assigned post-construction (not via set_processing_manager): the half-object PM
    # has no scheduler, and these tests drive the ticks directly.
    am._pm = _real_pm(am)
    return am


def _mk_order(cid, status, vid):
    return Order(
        client_order_id=cid,
        venue_order_id=vid,
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=None,
        submitted_at=np.datetime64("2026-05-28T00:00:00"),
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        status=status,
        time_in_force="gtc",
    )


def test_liveness_tick_isolates_raising_ws_check():
    # Same isolation rule for the liveness loop: a raising is_ws_ready on one connector
    # must not skip the health check of the rest.
    bad, good = MagicMock(), MagicMock()
    bad.is_ws_ready.side_effect = RuntimeError("boom")
    good.is_ws_ready.return_value = False
    am = _am({"binance": bad, "kraken": good}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick(None)  # must not raise
    am._time.adv(6_000)
    am._on_liveness_tick(None)
    good.reconnect.assert_called_once()


def test_liveness_tick_forces_reconnect_after_threshold():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick(None)
    conn.reconnect.assert_not_called()
    am._time.adv(6_000)
    am._on_liveness_tick(None)
    conn.reconnect.assert_called_once()


def test_liveness_tick_resets_when_ws_recovers():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick(None)
    # WS recovers before threshold -> unready timer cleared
    conn.is_ws_ready.return_value = True
    am._time.adv(3_000)
    am._on_liveness_tick(None)
    assert "binance" not in am._liveness_unready_since
    conn.reconnect.assert_not_called()


def test_liveness_tick_retries_when_reconnect_fails():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.return_value = False
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick(None)
    am._time.adv(6_000)
    am._on_liveness_tick(None)
    # Failed reconnect keeps the timestamp -> the very next tick retries without
    # waiting out the full threshold again.
    assert "binance" in am._liveness_unready_since
    am._time.adv(1_000)
    am._on_liveness_tick(None)
    assert conn.reconnect.call_count == 2


def test_liveness_tick_retries_when_reconnect_raises():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.side_effect = RuntimeError("boom")
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick(None)
    am._time.adv(6_000)
    am._on_liveness_tick(None)
    assert "binance" in am._liveness_unready_since
    am._time.adv(1_000)
    am._on_liveness_tick(None)
    assert conn.reconnect.call_count == 2


def test_liveness_tick_clears_timestamp_on_successful_reconnect():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.return_value = True
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick(None)
    am._time.adv(6_000)
    am._on_liveness_tick(None)
    assert "binance" not in am._liveness_unready_since


def test_init_registers_reconcile_and_liveness_ticks_via_pm_schedule():
    pm = MagicMock()
    conn = MagicMock()
    am = AccountManager(pm=pm, connectors={"binance": conn}, base_currencies={"binance": "USDT"}, time=_T())
    # one reconcile heartbeat (drives the Reconciler) + one liveness tick
    assert pm.schedule.call_count == 2
    scheduled = {call.args[1] for call in pm.schedule.call_args_list}
    assert am._on_reconcile_tick in scheduled
    assert am._on_liveness_tick in scheduled


def test_init_skips_disabled_ticks():
    pm = MagicMock()
    conn = MagicMock()
    cfg = AccountManagerConfig(
        reconcile_tick_interval_ms=0,
        liveness_check_interval_ms=5_000,
    )
    AccountManager(pm=pm, connectors={"binance": conn}, base_currencies={"binance": "USDT"}, time=_T(), cfg=cfg)
    assert pm.schedule.call_count == 1


def test_am_holds_no_strategy_reference():
    # I2 regression: the AM must never hold a strategy — all callbacks route through the PM.
    am = AccountManager(
        pm=MagicMock(), connectors={"binance": MagicMock()}, base_currencies={"binance": "USDT"}, time=_T()
    )
    assert not hasattr(am, "_strategy")
    assert not hasattr(am, "_ctx")
