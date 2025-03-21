"""
Error types that are sent through the event channel.
"""

from dataclasses import dataclass

from qubx.core.basics import Instrument, dt_64


@dataclass
class BaseErrorEvent:
    timestamp: dt_64
    message: str


def create_error_event(error: BaseErrorEvent) -> tuple[None, str, BaseErrorEvent, bool]:
    return None, "error", error, False


@dataclass
class OrderCreationError(BaseErrorEvent):
    instrument: Instrument
    amount: float
    price: float | None
    order_type: str
    side: str


@dataclass
class OrderCancellationError(BaseErrorEvent):
    instrument: Instrument
    order_id: str
