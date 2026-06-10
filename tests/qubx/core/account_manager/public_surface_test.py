"""The package re-exports the account event surface from qubx.core.events (see
__init__.py); this test is the package-path consumer keeping that surface honest."""

from qubx.core import events
from qubx.core.account_manager import (
    AccountMessage,
    AccountSnapshot,
    AccountSnapshotEvent,
    BalanceUpdateEvent,
    DealEvent,
    FundingPaymentEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
    PositionUpdateEvent,
)


def test_account_event_surface_is_reexported_via_package_path():
    assert AccountMessage is events.AccountMessage
    assert AccountSnapshot is events.AccountSnapshot
    assert AccountSnapshotEvent is events.AccountSnapshotEvent
    assert BalanceUpdateEvent is events.BalanceUpdateEvent
    assert DealEvent is events.DealEvent
    assert FundingPaymentEvent is events.FundingPaymentEvent
    assert OrderAcceptedEvent is events.OrderAcceptedEvent
    assert OrderCanceledEvent is events.OrderCanceledEvent
    assert OrderCancelRejectedEvent is events.OrderCancelRejectedEvent
    assert OrderEvent is events.OrderEvent
    assert OrderExpiredEvent is events.OrderExpiredEvent
    assert OrderFilledEvent is events.OrderFilledEvent
    assert OrderPartiallyFilledEvent is events.OrderPartiallyFilledEvent
    assert OrderRejectedEvent is events.OrderRejectedEvent
    assert OrderUpdatedEvent is events.OrderUpdatedEvent
    assert OrderUpdateRejectedEvent is events.OrderUpdateRejectedEvent
    assert PositionUpdateEvent is events.PositionUpdateEvent
