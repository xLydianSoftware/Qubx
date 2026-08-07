"""
Error types that are sent through the event channel.
"""

from dataclasses import dataclass
from enum import Enum

from qubx.core.basics import Instrument, dt_64


class ErrorLevel(Enum):
    LOW = 1  # continue trading
    MEDIUM = 2  # send notifications and continue trading
    HIGH = 3  # send notification and cancel orders and close positions
    CRITICAL = 4  # send notification and shutdown strategy


@dataclass
class BaseErrorEvent:
    timestamp: dt_64
    message: str
    level: ErrorLevel
    error: Exception | None

    def __str__(self):
        return f"[{self.level}] : {self.timestamp} : {self.message} / {self.error}"


def create_error_event(error: BaseErrorEvent) -> tuple[None, str, BaseErrorEvent, bool]:
    return None, "error", error, False


@dataclass
class OrderCreationError(BaseErrorEvent):
    instrument: Instrument
    amount: float
    price: float | None
    order_type: str
    side: str
    client_id: str | None = None

    def __str__(self):
        cid = f", client_id={self.client_id}" if self.client_id else ""
        return f"[{self.level}] : {self.timestamp} : {self.message} / {self.error} ||| Order creation error for {self.order_type} {self.side} {self.instrument} {self.amount}{cid}"


@dataclass
class OrderCancellationError(BaseErrorEvent):
    instrument: Instrument
    order_id: str

    def __str__(self):
        return f"[{self.level}] : {self.timestamp} : {self.message} / {self.error} ||| Order cancellation error for {self.order_id} {self.instrument}"


@dataclass
class VenueOperationError(BaseErrorEvent):
    """A venue write that failed after its caller had already been released.

    Some writes cannot block their caller — ``set_instrument_leverage`` is issued from
    the ProcessorThread and sends the request off-thread — so a refusal has no return
    value to travel back on. Without this it would be visible only in the connector's log.

    ``operation`` is the connector method that failed; ``instrument`` is None for an
    account-wide call.
    """

    operation: str
    instrument: Instrument | None = None

    def __str__(self):
        where = self.instrument if self.instrument is not None else "account"
        return f"[{self.level}] : {self.timestamp} : {self.message} / {self.error} ||| {self.operation} failed for {where}"
