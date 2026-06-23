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


class NotSupported(ExchangeError):
    pass


class QueueTimeout(BaseError):
    pass


class InvalidOrderTransition(Exception):
    # OrderStatus annotations are forward-ref strings (never evaluated) to avoid a
    # circular import: qubx.core.basics imports from this module at top level.
    def __init__(self, client_order_id: str, current: "OrderStatus", attempted: "OrderStatus"):  # noqa: F821
        super().__init__(f"Order {client_order_id}: illegal transition {current.value} → {attempted.value}")
        self.client_order_id = client_order_id
        self.current = current
        self.attempted = attempted


class OrderAlreadyTerminal(Exception):
    pass


class ReadOnlyConnector(Exception):
    pass


class StrategyExceededMaxNumberOfRuntimeFailuresError(Exception):
    pass


class SimulationError(Exception):
    pass


class SimulationConfigError(Exception):
    pass


class WarmupValidationError(Exception):
    pass
