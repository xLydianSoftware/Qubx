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
from qubx.data.storage import IReader, IStorage, RawData, RawMultiData
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
                cls._connect()

    return wrapper


class PGConnectionHelper:
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


_HACKED_RECODING = {
    ("binance", "umswap"): ("binance.um", "swap"),
    ("binance", "margin"): ("binance.pm", "swap"),
    # ("coingecko", "fundamental"): ("coingecko", "coins"),
}


class QuestDBReader(IReader):
    def __init__(self, exchange: str, market: str, pgc: PGConnectionHelper) -> None:
        self.exchange = exchange
        self.market = market
        self.pgc = pgc

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str] | dict[DataType, list[str]]:
        # match dtype
        ...


class QuestDBStorage(IStorage, PGConnectionHelper):
    pgc: PGConnectionHelper

    def __init__(self, host="localhost", user="admin", password="quest", port=8812) -> None:
        self.pgc = PGConnectionHelper(host, user, password, port)

    @property
    def connection_url(self):
        return self.pgc.connection_url

    def __getstate__(self):
        return self.pgc.__getstate__()

    def close(self):
        self.pgc.close()

    def __del__(self):
        self.pgc.__del__()

    def get_exchanges(self) -> list[str]:
        return list(self._get_exch_info().keys())

    def _get_exch_info(self) -> dict[str, dict[str, list[str]]]:
        tables = self.pgc.execute("select table_name as name from tables()")[1]
        exch_mkt_dtype: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for (t,) in tables:
            ss = t.split(".")
            if len(ss) > 1:
                exch, mkt, typ = ss[0], ss[1], (ss[2] if len(ss) > 2 else None)
                exch, mkt = _HACKED_RECODING.get((exch, mkt), (exch, mkt))
                exch_mkt_dtype[exch.upper()][mkt.upper()].append(typ or mkt.upper())
        return exch_mkt_dtype  # todo: convert to normal dict

    def get_market_types(self, exchange: str) -> list[str]:
        exc_i = self._get_exch_info()
        return list(exc_i.get(exchange.upper(), dict()).keys())

    def get_reader(self, exchange: str, market: str) -> IReader:
        e_info = self._get_exch_info()

        if exchange in e_info and market in e_info[exchange]:
            return QuestDBReader(exchange, market, self.pgc)

        raise ValueError(f"Can't provide data reader for exchange {exchange} and market type {market}")
