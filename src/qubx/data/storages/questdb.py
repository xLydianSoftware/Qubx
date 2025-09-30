from collections import defaultdict
from collections.abc import Iterator
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg as pg
import pyarrow as pa

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.storage import IReader, IStorage, RawData
from qubx.data.storages.utils import find_time_col_idx, recognize_t
from qubx.utils.time import handle_start_stop


def _retry(fn):
    @wraps(fn)
    def wrapper(*args, **kw):
        cls = args[0]
        for x in range(cls._reconnect_tries):
            try:
                return fn(*args, **kw)
            except (pg.InterfaceError, pg.OperationalError, AttributeError):
                logger.debug("Database Connection [InterfaceError or OperationalError]")
                # time.sleep(cls._reconnect_idle)
                cls._connect()

    return wrapper


class QuestDBReader(IReader):
    pass


class QuestDBStorage(IStorage):
    _connection: pg.connection.Connection | None
    _reconnect_tries = 5
    _reconnect_idle = 0.1  # wait seconds before retying
    _host: str
    _user: str
    _password: str
    _port: int

    def __init__(self, host="localhost", user="admin", password="quest", port=8812) -> None:
        self._connection = None
        self._user = user
        self._password = password
        self._host = host
        self._port = port
        self._connect()

    def _connect(self):
        self._connection = pg.connect(self.connection_url, autocommit=True)
        logger.debug(f"Connected to QuestDB at {self._host}:{self._port}")

    @property
    def connection_url(self):
        return " ".join(
            [f"user={self._user}", f"password={self._password}", f"host={self._host}", f"port={self._port}"]
        )

    def __getstate__(self):
        if self._connection:
            self._connection.close()
            self._connection = None
        state = self.__dict__.copy()
        return state

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.debug(f"Disconnected from QuestDB at {self._host}:{self._port}")

    @_retry
    def execute(self, query: str) -> tuple[list, list]:
        _cursor = self._connection.cursor()  # type: ignore
        _cursor.execute(query)  # type: ignore
        names = [d.name for d in _cursor.description]  # type: ignore
        records = _cursor.fetchall()
        return names, records

    def __del__(self):
        try:
            if self._connection is not None:
                logger.debug("Closing connection")
                self._connection.close()
        except:  # noqa: E722
            pass

    def get_exchanges(self) -> list[str]: ...

    def get_market_types(self, exchange: str) -> list[str]: ...

    def get_reader(self, exchange: str, market: str) -> IReader: ...
