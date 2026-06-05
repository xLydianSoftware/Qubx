from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qubx.core.basics import OrderStatus


class BaseError(Exception):
    pass


class ExchangeError(BaseError):
    pass


class BadRequest(ExchangeError):
    pass


class SymbolNotFound(ExchangeError):
    pass


class InvalidOrder(ExchangeError):
    pass


class InvalidOrderParameters(ExchangeError):
    pass


class InvalidOrderSize(InvalidOrderParameters):
    pass


class OrderNotFound(InvalidOrder):
    pass


class InvalidOrderTransition(Exception):
    """Raised when an order is moved between states the lifecycle state machine forbids.

    OrderStatus annotations are forward-ref strings (never evaluated) to avoid a circular
    import: qubx.core.basics imports from this module at top level, so this module must not
    import OrderStatus for real.
    """

    def __init__(self, client_order_id: str, current: "OrderStatus", attempted: "OrderStatus"):
        super().__init__(f"Order {client_order_id}: illegal transition {current.value} → {attempted.value}")
        self.client_order_id = client_order_id
        self.current = current
        self.attempted = attempted


class NotSupported(ExchangeError):
    pass


class QueueTimeout(BaseError):
    pass


class StrategyExceededMaxNumberOfRuntimeFailuresError(Exception):
    pass


class SimulationError(Exception):
    pass


class SimulationConfigError(Exception):
    pass


class WarmupValidationError(Exception):
    pass
