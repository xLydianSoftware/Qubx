from qubx.data.registry import storage
from qubx.data.storage import IReader, IStorage


@storage("nostorage")
@storage("stub")
class NoConfiguredStorage(IStorage):
    """
    No storage - just empty storage that raises on any read attempt.
    Useful as a stub when no aux data is needed.
    """

    def __init__(self, reason: str):
        self._reason = reason

    def get_exchanges(self) -> list[str]:
        raise ValueError(self._reason)

    def get_market_types(self, exchange: str) -> list[str]:
        raise ValueError(self._reason)

    def get_reader(self, exchange: str, market: str) -> IReader:
        raise ValueError(self._reason)
