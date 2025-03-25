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


class OrderNotFound(InvalidOrder):
    pass


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
