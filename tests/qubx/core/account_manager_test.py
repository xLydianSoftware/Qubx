"""
Unit tests for the new-design account-management state machine (Q1).

The state machine is the bug-magnet of the redesign, so it is tested heavily here,
in-process, with no I/O. Transitions are derived from the sequence flows in the
`account-management` excalidraw (submit / cancel / update / stuck-order recovery).
"""

import dataclasses

import numpy as np
import pytest
from croniter import croniter

from qubx.core.account_manager.connector import ConnectorCapabilities
from qubx.core.account_manager.events import (
    AccountMessage,
    AccountSnapshotEvent,
    ChannelMessage,
    MarketDataMessage,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.account_manager.manager import AccountManager
from qubx.core.account_manager.scheduling import _ms_to_cron
from qubx.core.account_manager.state import AccountState, ManagedOrder  # noqa: F401
from qubx.core.account_manager.state_machine import (
    PENDING_STATES,
    TERMINAL_STATES,
    IllegalOrderTransition,
    OrderState,
    is_pending,
    is_terminal,
    transition,
)
from qubx.core.basics import Deal, dt_64
from qubx.core.lookups import lookup


class FakeConnector:
    """Test double implementing the IConnector protocol; records calls for assertions."""

    def __init__(self, exchange: str = "BINANCE.UM", capabilities: ConnectorCapabilities | None = None):
        self.exchange = exchange
        self.capabilities = capabilities or ConnectorCapabilities()
        self.calls: list[tuple] = []

    def connect(self) -> None:
        self.calls.append(("connect",))

    def submit_order(self, request) -> None:
        self.calls.append(("submit_order", request))

    def cancel_order(self, client_order_id: str, venue_order_id: str | None = None) -> None:
        self.calls.append(("cancel_order", client_order_id, venue_order_id))

    def update_order(
        self,
        client_order_id: str,
        venue_order_id: str | None = None,
        price: float | None = None,
        quantity: float | None = None,
    ) -> None:
        self.calls.append(("update_order", client_order_id, venue_order_id, price, quantity))

    def request_order_status(self, client_order_id: str, venue_order_id: str | None = None) -> None:
        self.calls.append(("request_order_status", client_order_id, venue_order_id))

    def request_snapshot(self) -> None:
        self.calls.append(("request_snapshot",))


class MutableClock:
    """Controllable time provider (nanosecond datetime64) for sweep/age tests."""

    def __init__(self, t0: str = "2026-01-01T00:00:00"):
        self._t = np.datetime64(t0, "ns")

    def time(self) -> dt_64:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t = self._t + np.timedelta64(int(seconds * 1e9), "ns")


class TestTerminalStates:
    def test_filled_canceled_rejected_are_terminal(self):
        assert is_terminal(OrderState.FILLED)
        assert is_terminal(OrderState.CANCELED)
        assert is_terminal(OrderState.REJECTED)

    def test_live_states_are_not_terminal(self):
        for s in (
            OrderState.SUBMITTED,
            OrderState.ACCEPTED,
            OrderState.PARTIALLY_FILLED,
            OrderState.PENDING_CANCEL,
            OrderState.PENDING_UPDATE,
        ):
            assert not is_terminal(s)

    def test_terminal_states_set_matches(self):
        assert TERMINAL_STATES == {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED}


class TestPendingStates:
    def test_pending_cancel_and_update_are_pending(self):
        assert is_pending(OrderState.PENDING_CANCEL)
        assert is_pending(OrderState.PENDING_UPDATE)
        assert PENDING_STATES == {OrderState.PENDING_CANCEL, OrderState.PENDING_UPDATE}

    def test_non_pending_states_are_not_pending(self):
        assert not is_pending(OrderState.SUBMITTED)
        assert not is_pending(OrderState.ACCEPTED)
        assert not is_pending(OrderState.FILLED)


LIVE_STATES = [
    OrderState.SUBMITTED,
    OrderState.ACCEPTED,
    OrderState.PARTIALLY_FILLED,
    OrderState.PENDING_CANCEL,
    OrderState.PENDING_UPDATE,
    OrderState.STALE,
]

LEGAL_TRANSITIONS = [
    # submit lifecycle
    (OrderState.SUBMITTED, OrderState.ACCEPTED),
    (OrderState.SUBMITTED, OrderState.PENDING_CANCEL),
    (OrderState.SUBMITTED, OrderState.PENDING_UPDATE),  # update an unacked order (symmetry with cancel)
    (OrderState.SUBMITTED, OrderState.STALE),  # sweep quarantine: never confirmed
    # stale quarantine -> resolved by a late authoritative signal
    (OrderState.STALE, OrderState.ACCEPTED),
    (OrderState.STALE, OrderState.PARTIALLY_FILLED),
    # accepted lifecycle
    (OrderState.ACCEPTED, OrderState.PENDING_CANCEL),
    (OrderState.ACCEPTED, OrderState.PENDING_UPDATE),
    (OrderState.ACCEPTED, OrderState.PARTIALLY_FILLED),
    # partial fills
    (OrderState.PARTIALLY_FILLED, OrderState.PARTIALLY_FILLED),
    (OrderState.PARTIALLY_FILLED, OrderState.PENDING_CANCEL),
    (OrderState.PARTIALLY_FILLED, OrderState.PENDING_UPDATE),
    # pending cancel -> revert (to pre-pending status)
    (OrderState.PENDING_CANCEL, OrderState.SUBMITTED),
    (OrderState.PENDING_CANCEL, OrderState.ACCEPTED),
    (OrderState.PENDING_CANCEL, OrderState.PARTIALLY_FILLED),
    # pending update -> confirm or revert
    (OrderState.PENDING_UPDATE, OrderState.ACCEPTED),
    (OrderState.PENDING_UPDATE, OrderState.PARTIALLY_FILLED),
    (OrderState.PENDING_UPDATE, OrderState.SUBMITTED),  # revert when update was issued pre-ack
]

ILLEGAL_TRANSITIONS = [
    # nothing leaves a terminal state — not even to another terminal
    (OrderState.FILLED, OrderState.ACCEPTED),
    (OrderState.CANCELED, OrderState.ACCEPTED),
    (OrderState.REJECTED, OrderState.SUBMITTED),
    (OrderState.FILLED, OrderState.CANCELED),
    (OrderState.CANCELED, OrderState.REJECTED),
    # cannot un-accept a live order
    (OrderState.ACCEPTED, OrderState.SUBMITTED),
]


class TestLegalTransitions:
    def test_allowed(self):
        for frm, to in LEGAL_TRANSITIONS:
            assert transition(frm, to) == to, f"{frm.value} -> {to.value} should be legal"


class TestVenueAuthoritativeTerminalization:
    """A venue can unilaterally terminalize any live order, in any order, from any live state."""

    def test_any_live_state_can_reach_any_terminal(self):
        for s in LIVE_STATES:
            for t in TERMINAL_STATES:
                assert transition(s, t) == t, f"{s.value} -> {t.value} should be legal (venue-driven)"

    def test_terminal_states_never_transition(self):
        for s in TERMINAL_STATES:
            for t in OrderState:
                with pytest.raises(IllegalOrderTransition):
                    transition(s, t)


class TestIllegalTransitions:
    def test_raises(self):
        for frm, to in ILLEGAL_TRANSITIONS:
            with pytest.raises(IllegalOrderTransition):
                transition(frm, to)

    def test_exception_carries_from_and_to(self):
        with pytest.raises(IllegalOrderTransition) as ei:
            transition(OrderState.FILLED, OrderState.ACCEPTED)
        assert ei.value.frm == OrderState.FILLED
        assert ei.value.to == OrderState.ACCEPTED


class TestMsToCron:
    def test_two_second_tick(self):
        assert _ms_to_cron(2000) == "* * * * * */2"

    def test_five_second_tick(self):
        assert _ms_to_cron(5000) == "* * * * * */5"

    def test_one_second_tick(self):
        assert _ms_to_cron(1000) == "* * * * * */1"

    def test_one_minute_drops_seconds_field(self):
        assert _ms_to_cron(60000) == "* * * * *"

    def test_output_is_croniter_parseable_with_expected_period(self):
        for ms in (1000, 2000, 5000, 30000, 60000):
            expr = _ms_to_cron(ms)
            it = croniter(expr, 0.0)
            a, b = it.get_next(float), it.get_next(float)
            assert (b - a) == ms / 1000.0, f"{expr} period mismatch for {ms}ms"

    def test_non_positive_raises(self):
        with pytest.raises(ValueError):
            _ms_to_cron(0)
        with pytest.raises(ValueError):
            _ms_to_cron(-1000)

    def test_sub_second_or_fractional_second_raises(self):
        with pytest.raises(ValueError):
            _ms_to_cron(500)
        with pytest.raises(ValueError):
            _ms_to_cron(2500)

    def test_multi_minute_unsupported_raises(self):
        # this helper targets the sub-minute in-flight tick; use interval_to_cron above that
        with pytest.raises(ValueError):
            _ms_to_cron(120000)


def _deal(trade_id: str = "t1", order_id: str = "o1") -> Deal:
    return Deal(
        id=trade_id,
        order_id=order_id,
        time=np.datetime64("2026-01-01T00:00:00", "ns"),
        amount=1.0,
        price=100.0,
        aggressive=True,
    )


ORDER_EVENTS = [
    OrderAcceptedEvent(client_id="c1", venue_id="v1"),
    OrderRejectedEvent(client_id="c1", reason="insufficient margin"),
    OrderFilledEvent(client_id="c1", fill=_deal()),
    OrderCanceledEvent(client_id="c1"),
    OrderCancelRejectedEvent(client_id="c1", reason="order not found"),
    OrderUpdatedEvent(client_id="c1", venue_id="v1", price=101.0, quantity=2.0),
    OrderUpdateRejectedEvent(client_id="c1", reason="price out of band"),
]


class TestEventHierarchy:
    def test_order_events_are_account_messages(self):
        for ev in ORDER_EVENTS:
            assert isinstance(ev, AccountMessage)
            assert isinstance(ev, ChannelMessage)

    def test_snapshot_is_account_message(self):
        snap = AccountSnapshotEvent(orders=[], positions=[], balances=[])
        assert isinstance(snap, AccountMessage)
        assert isinstance(snap, ChannelMessage)

    def test_market_data_is_channel_message_but_not_account_message(self):
        assert issubclass(MarketDataMessage, ChannelMessage)
        assert not issubclass(MarketDataMessage, AccountMessage)

    def test_every_order_event_carries_client_id(self):
        for ev in ORDER_EVENTS:
            assert ev.client_id == "c1"

    def test_accepted_and_updated_carry_venue_id(self):
        assert OrderAcceptedEvent(client_id="c1", venue_id="v1").venue_id == "v1"
        assert OrderUpdatedEvent(client_id="c1", venue_id="v1", price=1.0, quantity=1.0).venue_id == "v1"

    def test_rejection_events_carry_reason(self):
        assert OrderRejectedEvent(client_id="c1", reason="x").reason == "x"
        assert OrderCancelRejectedEvent(client_id="c1", reason="y").reason == "y"
        assert OrderUpdateRejectedEvent(client_id="c1", reason="z").reason == "z"

    def test_fill_event_carries_deal_with_trade_id(self):
        ev = OrderFilledEvent(client_id="c1", fill=_deal(trade_id="trade-42"))
        assert ev.fill.id == "trade-42"

    def test_events_are_frozen(self):
        ev = OrderAcceptedEvent(client_id="c1", venue_id="v1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.client_id = "c2"  # type: ignore[misc]


class TestConnectorContract:
    def test_capabilities_defaults(self):
        from qubx.core.account_manager.connector import ConnectorCapabilities

        caps = ConnectorCapabilities()
        assert caps.modify_pattern in ("in_place", "replace")
        assert isinstance(caps.cancel_by_cloid_supported, bool)
        assert isinstance(caps.native_snapshot_push, bool)
        assert isinstance(caps.synthesized_trade_id_supported, bool)

    def test_modify_pattern_replace_is_settable(self):
        from qubx.core.account_manager.connector import ConnectorCapabilities

        caps = ConnectorCapabilities(modify_pattern="replace", cancel_by_cloid_supported=True)
        assert caps.modify_pattern == "replace"
        assert caps.cancel_by_cloid_supported is True

    def test_fake_connector_satisfies_protocol(self):
        from qubx.core.account_manager.connector import IConnector

        assert isinstance(FakeConnector(), IConnector)

    def test_object_missing_methods_does_not_satisfy_protocol(self):
        from qubx.core.account_manager.connector import IConnector

        assert not isinstance(object(), IConnector)


# --------------------------------------------------------------------------------------
# AccountState / AccountManager lifecycle
# --------------------------------------------------------------------------------------

INSTR = lookup.find_symbol("BINANCE.UM", "BTCUSDT")


def mk_order(cid="c1", qty=1.0, price=100.0, side="BUY", otype="LIMIT") -> "ManagedOrder":
    return ManagedOrder(
        client_id=cid,
        instrument=INSTR,
        side=side,
        quantity=qty,
        price=price,
        order_type=otype,
    )


def mk_manager(clock=None, connector=None, **kw):
    clock = clock or MutableClock()
    conn = connector or FakeConnector(exchange=INSTR.exchange)
    am = AccountManager(connectors={conn.exchange: conn}, time_provider=clock, **kw)
    return am, clock, conn


def fill(cid="c1", trade_id="t1", amount=1.0, price=100.0) -> OrderFilledEvent:
    d = Deal(
        id=trade_id,
        order_id=cid,
        time=np.datetime64("2026-01-01T00:00:00", "ns"),
        amount=amount,
        price=price,
        aggressive=True,
    )
    return OrderFilledEvent(client_id=cid, fill=d)


class TestAddOrder:
    def test_add_order_registers_in_active_and_inflight(self):
        am, clock, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        o = am.get_order("c1")
        assert o is not None
        assert o.status == OrderState.SUBMITTED
        assert "c1" in am.state.active_orders
        assert "c1" in am.state._inflight_index
        assert o.last_updated_at == clock.time()

    def test_get_unknown_order_returns_none(self):
        am, _, _ = mk_manager()
        assert am.get_order("nope") is None


class TestAccepted:
    def test_submitted_to_accepted_sets_and_indexes_venue_id(self):
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        o = am.get_order("c1")
        assert o.status == OrderState.ACCEPTED
        assert o.venue_id == "v1"
        assert am.state._venue_id_index["v1"] == "c1"
        # accepted is no longer awaiting a venue verdict
        assert "c1" not in am.state._inflight_index

    def test_accepted_resets_retry_count(self):
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.get_order("c1").retry_count = 3
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        assert am.get_order("c1").retry_count == 0

    def test_accept_for_unknown_order_is_ignored(self):
        am, _, _ = mk_manager()
        am.apply(OrderAcceptedEvent(client_id="ghost", venue_id="v9"))  # no raise
        assert am.get_order("ghost") is None


class TestRejected:
    def test_submitted_to_rejected_is_terminal_and_deindexed(self):
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderRejectedEvent(client_id="c1", reason="insufficient margin"))
        o = am.get_order("c1")
        assert o.status == OrderState.REJECTED
        assert "c1" not in am.state._inflight_index
        assert "c1" in am.state._pending_evict_index


class TestCancel:
    def test_transition_to_pending_cancel_captures_pre_pending(self):
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.transition_order("c1", OrderState.PENDING_CANCEL)
        o = am.get_order("c1")
        assert o.status == OrderState.PENDING_CANCEL
        assert o.pre_pending_status == OrderState.ACCEPTED
        assert "c1" in am.state._inflight_index  # awaiting cancel verdict

    def test_cancel_ack_terminalizes_and_clears_pre_pending(self):
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.transition_order("c1", OrderState.PENDING_CANCEL)
        am.apply(OrderCanceledEvent(client_id="c1"))
        o = am.get_order("c1")
        assert o.status == OrderState.CANCELED
        assert o.pre_pending_status is None
        assert "c1" not in am.state._inflight_index
        assert "c1" in am.state._pending_evict_index

    def test_cancel_rejected_reverts_to_pre_pending_status(self):
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        # cancel issued before the order was acknowledged (still SUBMITTED)
        am.transition_order("c1", OrderState.PENDING_CANCEL)
        assert am.get_order("c1").pre_pending_status == OrderState.SUBMITTED
        am.apply(OrderCancelRejectedEvent(client_id="c1", reason="order not found"))
        o = am.get_order("c1")
        assert o.status == OrderState.SUBMITTED
        assert o.pre_pending_status is None
        assert "c1" in am.state._inflight_index


class TestUpdate:
    def test_update_confirmed_applies_new_params_and_clears_pre_pending(self):
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1", price=100.0, qty=1.0))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.transition_order("c1", OrderState.PENDING_UPDATE)
        assert am.get_order("c1").pre_pending_status == OrderState.ACCEPTED
        am.apply(OrderUpdatedEvent(client_id="c1", venue_id="v1", price=101.0, quantity=2.0))
        o = am.get_order("c1")
        assert o.status == OrderState.ACCEPTED
        assert o.price == 101.0
        assert o.quantity == 2.0
        assert o.pre_pending_status is None

    def test_update_reindexes_venue_id_on_replace(self):
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.transition_order("c1", OrderState.PENDING_UPDATE)
        am.apply(OrderUpdatedEvent(client_id="c1", venue_id="v2", price=101.0, quantity=1.0))
        o = am.get_order("c1")
        assert o.venue_id == "v2"
        assert am.state._venue_id_index.get("v2") == "c1"
        assert "v1" not in am.state._venue_id_index

    def test_update_rejected_reverts_to_pre_pending_status(self):
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.transition_order("c1", OrderState.PENDING_UPDATE)
        am.apply(OrderUpdateRejectedEvent(client_id="c1", reason="price out of band"))
        o = am.get_order("c1")
        assert o.status == OrderState.ACCEPTED
        assert o.pre_pending_status is None


class TestFills:
    def test_partial_then_full_fill(self):
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1", qty=2.0))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.apply(fill(cid="c1", trade_id="t1", amount=1.0))
        o = am.get_order("c1")
        assert o.status == OrderState.PARTIALLY_FILLED
        assert o.filled_quantity == 1.0
        am.apply(fill(cid="c1", trade_id="t2", amount=1.0))
        o = am.get_order("c1")
        assert o.status == OrderState.FILLED
        assert o.filled_quantity == 2.0
        assert "c1" not in am.state._inflight_index
        assert "c1" in am.state._pending_evict_index

    def test_duplicate_fill_is_deduped_by_trade_id(self):
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1", qty=2.0))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.apply(fill(cid="c1", trade_id="t1", amount=1.0))
        am.apply(fill(cid="c1", trade_id="t1", amount=1.0))  # same trade id replayed
        o = am.get_order("c1")
        assert o.filled_quantity == 1.0
        assert o.status == OrderState.PARTIALLY_FILLED


class TestStuckOrderSweep:
    def test_young_inflight_order_is_not_swept(self):
        am, clock, conn = mk_manager(min_inflight_age_sec=5.0)
        am.add_order(mk_order(cid="c1"))
        clock.advance(3.0)  # younger than min age
        am.on_inflight_tick()
        assert ("request_order_status", "c1", None) not in conn.calls
        assert am.get_order("c1").retry_count == 0

    def test_stuck_order_triggers_status_query_and_increments_retry(self):
        am, clock, conn = mk_manager(min_inflight_age_sec=5.0)
        am.add_order(mk_order(cid="c1"))
        clock.advance(6.0)
        am.on_inflight_tick()
        assert ("request_order_status", "c1", None) in conn.calls
        assert am.get_order("c1").retry_count == 1

    def test_status_query_uses_known_venue_id(self):
        am, clock, conn = mk_manager(min_inflight_age_sec=5.0)
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        # re-enter in-flight via a pending cancel that the venue never answers
        am.transition_order("c1", OrderState.PENDING_CANCEL)
        clock.advance(6.0)
        am.on_inflight_tick()
        assert ("request_order_status", "c1", "v1") in conn.calls

    def test_order_quarantined_after_max_retries(self):
        am, clock, conn = mk_manager(min_inflight_age_sec=5.0, max_inflight_retries=3)
        am.add_order(mk_order(cid="c1"))
        clock.advance(6.0)
        for _ in range(3):  # three status queries
            am.on_inflight_tick()
        assert am.get_order("c1").retry_count == 3
        assert conn.calls.count(("request_order_status", "c1", None)) == 3
        am.on_inflight_tick()  # fourth tick: give up -> quarantine (not auto-reject)
        o = am.get_order("c1")
        assert o.status == OrderState.STALE
        assert "c1" not in am.state._inflight_index

    def test_venue_ack_during_recovery_resets_and_stops_sweeping(self):
        am, clock, conn = mk_manager(min_inflight_age_sec=5.0)
        am.add_order(mk_order(cid="c1"))
        clock.advance(6.0)
        am.on_inflight_tick()
        assert am.get_order("c1").retry_count == 1
        # status query comes back 'open' -> synthesized OrderAcceptedEvent
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        o = am.get_order("c1")
        assert o.status == OrderState.ACCEPTED
        assert o.retry_count == 0
        assert "c1" not in am.state._inflight_index


class TestSnapshotReconcile:
    def test_adopts_unknown_order_from_snapshot(self):
        am, _, _ = mk_manager()
        snap_order = mk_order(cid="c9")
        snap_order.status = OrderState.ACCEPTED
        snap_order.venue_id = "v9"
        am.apply(AccountSnapshotEvent(orders=[snap_order]))
        o = am.get_order("c9")
        assert o is not None
        assert am.state._venue_id_index.get("v9") == "c9"

    def test_grace_window_blocks_resurrecting_evicted_order(self):
        am, clock, _ = mk_manager(evict_grace_sec=30.0)
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderRejectedEvent(client_id="c1", reason="x"))  # terminal -> evict index
        clock.advance(60.0)
        am.evict_terminal()  # past grace: drop the order object, keep the tombstone
        assert am.get_order("c1") is None
        assert "c1" in am.state._pending_evict_index
        # a late snapshot still listing it must not resurrect it
        stale = mk_order(cid="c1")
        stale.status = OrderState.ACCEPTED
        am.apply(AccountSnapshotEvent(orders=[stale]))
        assert am.get_order("c1") is None

    def test_freshness_keeps_locally_newer_state(self):
        am, clock, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        clock.advance(10.0)
        local_ts = am.get_order("c1").last_updated_at
        # snapshot stamped BEFORE the local update -> stale, must not override
        stale = mk_order(cid="c1")
        stale.status = OrderState.CANCELED
        snap_ts = local_ts - np.timedelta64(5, "s")
        am.apply(AccountSnapshotEvent(orders=[stale], timestamp=snap_ts))
        assert am.get_order("c1").status == OrderState.ACCEPTED


class TestSimulationAccountManager:
    def test_base_manager_registers_inflight_tick(self):
        am, _, _ = mk_manager()
        assert am.should_register_inflight_tick is True

    def test_base_manager_tick_cron_is_two_seconds(self):
        am, _, _ = mk_manager()
        assert am.inflight_tick_cron() == "* * * * * */2"

    def test_simulation_manager_skips_tick_by_default(self):
        from qubx.core.account_manager.manager import SimulationAccountManager

        sim = SimulationAccountManager(time_provider=MutableClock())
        assert isinstance(sim, AccountManager)
        assert sim.should_register_inflight_tick is False

    def test_simulation_manager_tick_is_opt_in(self):
        from qubx.core.account_manager.manager import SimulationAccountManager

        sim = SimulationAccountManager(time_provider=MutableClock(), register_inflight_tick=True)
        assert sim.should_register_inflight_tick is True

    def test_simulation_manager_can_still_sweep_when_driven_explicitly(self):
        # the Q3 stuck-order-recovery test arms the sweep and drives it deterministically
        from qubx.core.account_manager.manager import SimulationAccountManager

        clock = MutableClock()
        conn = FakeConnector(exchange=INSTR.exchange)
        sim = SimulationAccountManager(
            connectors={conn.exchange: conn},
            time_provider=clock,
            register_inflight_tick=True,
            min_inflight_age_sec=5.0,
        )
        sim.add_order(mk_order(cid="c1"))
        clock.advance(6.0)
        sim.on_inflight_tick()
        assert ("request_order_status", "c1", None) in conn.calls

    def test_simulation_manager_shares_lifecycle_behaviour(self):
        from qubx.core.account_manager.manager import SimulationAccountManager

        sim = SimulationAccountManager(time_provider=MutableClock())
        sim.add_order(mk_order(cid="c1"))
        sim.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        assert sim.get_order("c1").status == OrderState.ACCEPTED


class TestVenueDrivenOrderings:
    """Regressions for venue-driven event orderings the original tests never exercised."""

    def test_venue_initiated_cancel_on_accepted(self):  # H3
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.apply(OrderCanceledEvent(client_id="c1"))  # GTD expiry / liquidation / admin cancel
        o = am.get_order("c1")
        assert o.status == OrderState.CANCELED
        assert "c1" not in am.state._inflight_index
        assert "c1" in am.state._pending_evict_index

    def test_venue_initiated_cancel_on_submitted(self):  # H3
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderCanceledEvent(client_id="c1"))
        assert am.get_order("c1").status == OrderState.CANCELED

    def test_late_reject_on_partially_filled(self):  # H4
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1", qty=2.0))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.apply(fill(cid="c1", trade_id="t1", amount=1.0))
        am.apply(OrderRejectedEvent(client_id="c1", reason="late venue reject"))
        assert am.get_order("c1").status == OrderState.REJECTED

    def test_fill_while_pending_cancel_still_applies(self):  # race fill (non-terminal)
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1", qty=2.0))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.transition_order("c1", OrderState.PENDING_CANCEL)
        am.apply(fill(cid="c1", trade_id="t1", amount=2.0))
        assert am.get_order("c1").status == OrderState.FILLED

    def test_fill_after_terminal_is_ignored(self):  # H2
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1", qty=2.0))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.transition_order("c1", OrderState.PENDING_CANCEL)
        am.apply(OrderCanceledEvent(client_id="c1"))  # terminal
        am.apply(fill(cid="c1", trade_id="t9", amount=1.0))  # late race fill
        o = am.get_order("c1")
        assert o.status == OrderState.CANCELED
        assert o.filled_quantity == 0.0

    def test_update_on_unacked_submitted_order(self):  # M2
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.transition_order("c1", OrderState.PENDING_UPDATE)
        assert am.get_order("c1").status == OrderState.PENDING_UPDATE
        assert am.get_order("c1").pre_pending_status == OrderState.SUBMITTED
        am.apply(OrderUpdateRejectedEvent(client_id="c1", reason="x"))
        assert am.get_order("c1").status == OrderState.SUBMITTED


class TestStuckPendingGiveUp:
    def test_stuck_submitted_quarantines_to_stale(self):  # R1: never-acked submit -> STALE, not REJECTED
        am, clock, _ = mk_manager(min_inflight_age_sec=5.0, max_inflight_retries=2)
        am.add_order(mk_order(cid="c1"))
        clock.advance(6.0)
        for _ in range(3):
            am.on_inflight_tick()
        o = am.get_order("c1")
        assert o.status == OrderState.STALE  # quarantined, not auto-rejected
        assert "c1" not in am.state._inflight_index  # stops hammering the venue
        assert "c1" not in am.state._pending_evict_index  # not terminal

    def test_late_ack_resurrects_stale_order(self):  # R1: a slow ack must recover the order
        am, clock, _ = mk_manager(min_inflight_age_sec=5.0, max_inflight_retries=2)
        am.add_order(mk_order(cid="c1"))
        clock.advance(6.0)
        for _ in range(3):
            am.on_inflight_tick()
        assert am.get_order("c1").status == OrderState.STALE
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))  # the ack was just slow
        assert am.get_order("c1").status == OrderState.ACCEPTED

    def test_stale_order_terminalized_by_explicit_venue_reject(self):  # only explicit reject -> REJECTED
        am, clock, _ = mk_manager(min_inflight_age_sec=5.0, max_inflight_retries=2)
        am.add_order(mk_order(cid="c1"))
        clock.advance(6.0)
        for _ in range(3):
            am.on_inflight_tick()
        am.apply(OrderRejectedEvent(client_id="c1", reason="venue says not found"))
        assert am.get_order("c1").status == OrderState.REJECTED

    def test_stale_order_resolved_by_late_fill(self):
        am, clock, _ = mk_manager(min_inflight_age_sec=5.0, max_inflight_retries=2)
        am.add_order(mk_order(cid="c1", qty=1.0))
        clock.advance(6.0)
        for _ in range(3):
            am.on_inflight_tick()
        assert am.get_order("c1").status == OrderState.STALE
        am.apply(fill(cid="c1", trade_id="t1", amount=1.0))
        assert am.get_order("c1").status == OrderState.FILLED

    def test_stuck_pending_cancel_reverts_not_rejects(self):  # H1
        am, clock, _ = mk_manager(min_inflight_age_sec=5.0, max_inflight_retries=2)
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.transition_order("c1", OrderState.PENDING_CANCEL)
        clock.advance(6.0)
        for _ in range(4):  # exhaust retries then give up
            am.on_inflight_tick()
        o = am.get_order("c1")
        # cancel never acked, but the underlying order is presumed still live -> revert, not reject
        assert o.status == OrderState.ACCEPTED
        assert "c1" not in am.state._inflight_index

    def test_open_status_on_pending_cancel_does_not_livelock(self):  # M1
        am, clock, _ = mk_manager(min_inflight_age_sec=5.0, max_inflight_retries=3)
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.transition_order("c1", OrderState.PENDING_CANCEL)
        clock.advance(6.0)
        for _ in range(6):  # each query returns 'open' (synthesized accept) -> must still converge
            am.on_inflight_tick()
            am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        assert am.get_order("c1").status == OrderState.ACCEPTED  # converged, not stuck forever


class TestRobustnessFixes:
    def test_add_order_preserves_explicit_epoch_created_at(self):  # L1
        am, _, _ = mk_manager()
        o = mk_order(cid="c1")
        o.created_at = np.datetime64(0, "ns")  # epoch is falsy under `or`
        am.add_order(o)
        assert am.get_order("c1").created_at == np.datetime64(0, "ns")

    def test_tombstones_purged_after_ttl(self):  # M3
        am, clock, _ = mk_manager(evict_grace_sec=30.0, tombstone_ttl_sec=300.0)
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderRejectedEvent(client_id="c1", reason="x"))
        clock.advance(60.0)
        am.evict_terminal()
        assert "c1" in am.state._pending_evict_index  # tombstone retained within ttl
        clock.advance(400.0)
        am.evict_terminal()
        assert "c1" not in am.state._pending_evict_index  # purged after ttl

    def test_apply_does_not_propagate_handler_errors(self):  # M4
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))

        def boom(ev):
            raise IllegalOrderTransition(OrderState.FILLED, OrderState.ACCEPTED)

        am._dispatch[OrderAcceptedEvent] = boom
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))  # must not raise

    def test_apply_swallows_non_transition_exceptions(self):  # R2: net must catch any handler error
        am, _, _ = mk_manager()

        def boom(ev):
            raise AttributeError("simulated handler bug")

        am._dispatch[OrderAcceptedEvent] = boom
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))  # must not raise

    def test_unknown_event_type_is_ignored(self):  # L5
        am, _, _ = mk_manager()
        am.apply(MarketDataMessage())  # not an account event -> ignored, no raise

    def test_stray_update_while_pending_cancel_is_ignored(self):  # L4 + R6
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1", price=100.0, qty=1.0))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.transition_order("c1", OrderState.PENDING_CANCEL)
        am.apply(OrderUpdatedEvent(client_id="c1", venue_id="v2", price=999.0, quantity=5.0))
        o = am.get_order("c1")
        assert o.status == OrderState.PENDING_CANCEL
        assert o.pre_pending_status == OrderState.ACCEPTED  # cancel's revert state untouched
        assert o.price == 100.0 and o.quantity == 1.0  # R6: params NOT mutated by stray update
        assert o.venue_id == "v1"

    def test_update_reducing_qty_to_filled_finalizes(self):  # R7
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1", qty=2.0))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        am.apply(fill(cid="c1", trade_id="t1", amount=1.0))  # filled 1 of 2
        am.transition_order("c1", OrderState.PENDING_UPDATE)
        am.apply(OrderUpdatedEvent(client_id="c1", venue_id="v1", quantity=1.0))  # shrink to filled size
        assert am.get_order("c1").status == OrderState.FILLED

    def test_add_order_overwrite_cleans_stale_venue_index(self):  # R5
        am, _, _ = mk_manager()
        am.add_order(mk_order(cid="c1"))
        am.apply(OrderAcceptedEvent(client_id="c1", venue_id="v1"))
        assert am.state._venue_id_index.get("v1") == "c1"
        am.add_order(mk_order(cid="c1"))  # re-submit same cid (new order, no venue id)
        assert "v1" not in am.state._venue_id_index

    def test_snapshot_adopts_submitted_order_into_inflight(self):  # L2
        am, _, _ = mk_manager()
        snap = mk_order(cid="c9")
        snap.status = OrderState.SUBMITTED
        am.apply(AccountSnapshotEvent(orders=[snap]))
        assert "c9" in am.state._inflight_index

    def test_snapshot_with_wrong_shaped_order_is_skipped_not_crash(self):  # R3
        from qubx.core.basics import Order

        am, _, _ = mk_manager()
        core_order = Order(
            id="x",
            type="LIMIT",
            instrument=INSTR,
            time=np.datetime64("2026-01-01", "ns"),
            quantity=1.0,
            price=100.0,
            side="BUY",
            status="OPEN",
            time_in_force="gtc",
            client_id="c1",
        )
        am.apply(AccountSnapshotEvent(orders=[core_order]))  # must not raise
        assert am.get_order("c1") is None  # not adopted (wrong type)
