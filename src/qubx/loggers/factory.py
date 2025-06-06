import inspect
from typing import Type

from qubx.core.loggers import LogsWriter
from qubx.loggers.csv import CsvFileLogsWriter
from qubx.loggers.inmemory import InMemoryLogsWriter
from qubx.loggers.mongo import MongoDBLogsWriter

# Registry of logs writer types
LOGS_WRITER_REGISTRY: dict[str, Type[LogsWriter]] = {
    "CsvFileLogsWriter": CsvFileLogsWriter,
    "MongoDBLogsWriter": MongoDBLogsWriter,
    "InMemoryLogsWriter": InMemoryLogsWriter,
}


def create_logs_writer(log_writer_type: str, parameters: dict | None = None) -> LogsWriter:
    """
    Create a logs writer based on configuration.

    Args:
        log_wirter_type: The type of logs writer to create.
        parameters: Parameters to pass to the logs writer constructor.

    Returns:
        An instance of the specified logs writer.

    Raises:
        ValueError: If the specified logs writer type is not registered.
    """
    if log_writer_type not in LOGS_WRITER_REGISTRY:
        raise ValueError(
            f"Unknown logs writer type: {log_writer_type}. Available types: {', '.join(LOGS_WRITER_REGISTRY.keys())}"
        )

    logs_writer_class = LOGS_WRITER_REGISTRY[log_writer_type]
    params = parameters.copy() if parameters else {}

    sig = inspect.signature(logs_writer_class)
    accepted_params = set(sig.parameters.keys())
    filtered_params = {k: v for k, v in params.items() if k in accepted_params}

    return logs_writer_class(**filtered_params)


def register_logs_writer(log_writer_type: str, logs_witer_class: Type[LogsWriter]) -> None:
    """
    Register a new logs writer type.

    Args:
        log_writer_type: The name of the logs writer type.
        logs_witer_class: The logs writer class to register.
    """
    LOGS_WRITER_REGISTRY[log_writer_type] = logs_witer_class
