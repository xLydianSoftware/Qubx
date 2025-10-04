import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg as pg
import pyarrow as pa

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.storage import IReader, IStorage, RawData, RawMultiData, Transformable
from qubx.data.storages.utils import find_time_col_idx, recognize_t
from qubx.utils.time import handle_start_stop, timedelta_to_str


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


@dataclass
class xLTableMetaInfo:
    """
    Table meta info container and decoder
        TODO: we need to fix tables naming in DB to drop any special processing code
    """

    exchange: str
    market_type: str
    dtype: DataType
    data_timeframe: str | None
    table_name: str
    alias_for_record_type: str | None = None
    data_ids: list[str] | None = None  # data IDs available in this meta

    _TABLES_FIX = {
        ("binance", "umswap"): ("binance.um", "swap"),
        ("binance", "umfutures"): ("binance.um", "future"),
        ("binance", "cmswap"): ("binance.cm", "swap"),
        ("binance", "cmfutures"): ("binance.cm", "future"),
        ("binance", "margin"): ("binance.pm", "swap"),
    }

    _DTYPE_FIX = {
        "candles": "ohlc",
        "orderbooks": "orderbook",
        "liquidations": "liquidation",
        # "interest_rates": "record",
    }

    _d_pattern = r"^(.+?)(?:_(\d+(?:min|[mhdw])))?$"

    @staticmethod
    def decode_table_metadata(table_name: str) -> "xLTableMetaInfo | None":
        """
        Decode table name and try to extract metadata:
            binance.umswap.candles_1m -> BINANCE.UM, SWAP, OHLC[1min]
        """
        ss = table_name.split(".")
        if len(ss) > 1:
            exch, mkt, data_type = ss[0], ss[1], (ss[2] if len(ss) > 2 else ss[1])
            exch, mkt = xLTableMetaInfo._TABLES_FIX.get((exch, mkt), (exch, mkt))

            # - consider it as valid data if we can recognize structure
            if sg := re.match(xLTableMetaInfo._d_pattern, data_type):
                data_type, tframe = sg.groups()
                f_data_type = xLTableMetaInfo._DTYPE_FIX.get(data_type, data_type)
                r_dt, _ = DataType.from_str(f_data_type)
                r_dt = DataType.RECORD if r_dt == DataType.NONE else r_dt
                _alias = data_type if r_dt == DataType.RECORD else None
                return xLTableMetaInfo(exch.upper(), mkt.upper(), r_dt, tframe, table_name, _alias)

        return None


# fmt: off
_ext_frames = pd.to_timedelta(
    [
        "1s", "2s", "3s", "5s", "10s", "15s", "30s",
        "1Min", "2Min", "3Min", "5Min", "10Min", "15Min", "30Min",
        "1h", "2h", "3h", "4h", "6h", "8h", "12h",
        "1d", "1w",
    ]
)
# fmt: on


class QuestDBReader(IReader):
    # Info about datatypes for every data_id for this exchange and market type
    #   - {symbol -> { dtype -> xLTableMetaInfo}
    _symbols_lookup: dict[str, dict[DataType, xLTableMetaInfo]]

    def __init__(
        self,
        exchange: str,
        market: str,
        available_data: list[xLTableMetaInfo],
        pgc: PGConnectionHelper,
        synthetic_ohlc_timeframes_types: bool,
    ) -> None:
        self.exchange = exchange
        self.market = market
        self.synthetic_ohlc_timeframes_types = synthetic_ohlc_timeframes_types
        self.pgc = pgc

        # - build lookup
        self._symbols_lookup = self._create_symbols_lookup(available_data)

    @staticmethod
    def _convert_time_delta_to_qdb_resample_format(c_tf: str) -> str:
        if c_tf:
            _t = re.match(r"(\d+)(\w+)", c_tf)
            if _t and len(_t.groups()) > 1:
                c_tf = f"{_t[1]}{_t[2][0].lower()}"
        return c_tf

    def _create_symbols_lookup(
        self, available_data: list[xLTableMetaInfo]
    ) -> dict[str, dict[DataType, xLTableMetaInfo]]:
        _lookup = defaultdict(dict)

        for x in available_data:
            symbols_info = self.pgc.execute(f"select distinct(symbol) from {x.table_name}")[1]
            symbols = []
            for s in symbols_info:
                symbols.append(symb := s[0])
                if x.dtype == DataType.OHLC:
                    if self.synthetic_ohlc_timeframes_types and x.data_timeframe:
                        # - for making life bit easy let's generate all possible frames we can contruct from available base
                        for f in _ext_frames[_ext_frames.searchsorted(pd.Timedelta(x.data_timeframe)) :]:
                            _lookup[symb][f"ohlc({timedelta_to_str(f)})"] = x
                    else:
                        _lookup[symb][DataType.OHLC[x.data_timeframe]] = x
                else:
                    _lookup[symb][x.alias_for_record_type if x.alias_for_record_type else x.dtype] = x

            # - attach symbols from this table
            if x.data_ids is None:
                x.data_ids = sorted(symbols)

        return _lookup

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        d_ids = set()
        for s, di in self._symbols_lookup.items():
            for dt, x in di.items():
                if x.dtype == dtype or dtype == DataType.ALL or str(dtype) == x.alias_for_record_type:
                    d_ids.add(s)
        return list(sorted(d_ids))

    def get_data_types(self, data_id: str) -> list[DataType]:
        return list(self._symbols_lookup.get(data_id, {}).keys())

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
        chunksize=0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        selected = []
        if isinstance(data_id, (list, tuple)):
            for di in data_id:
                if dtype in self._symbols_lookup.get(di, {}).keys():
                    selected.append(di)
        else:
            pass
            # if data_id in

        print(selected)

    def _read_ohlc(self, data_ids: list[str], data_meta: xLTableMetaInfo):
        query = f"""
            select 
                timestamp, 
                upper(symbol)               as symbol,
                first(open)                 as open, 
                max(high)                   as high,
                min(low)                    as low,
                last(close)                 as close,
                sum(volume)                 as volume,
                sum(quote_volume)           as quote_volume,
                sum(count)                  as count,
                sum(taker_buy_volume)       as taker_buy_volume,
                sum(taker_buy_quote_volume) as taker_buy_quote_volume
            from "{data_meta.table_name}" {where_clause} {_rsmpl};
        """
        ...


class QuestDBStorage(IStorage):
    """
    QuestDB storage implementation
    """

    pgc: PGConnectionHelper

    def __init__(
        self, host="localhost", user="admin", password="quest", port=8812, synthetic_ohlc_timeframes_types: bool = True
    ) -> None:
        self.pgc = PGConnectionHelper(host, user, password, port)
        self.synthetic_ohlc_timeframes_types = synthetic_ohlc_timeframes_types

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
        return list(self._read_database_meta_structure().keys())

    def _read_database_meta_structure(self) -> dict[str, dict[str, list[xLTableMetaInfo]]]:
        """
        Read DB structure and build tables mapping:
            exchange -> {market_type -> [xLTableMetaInfo] }
        """
        dbm: dict[str, dict[str, list[xLTableMetaInfo]]] = defaultdict(lambda: defaultdict(list))
        tables = self.pgc.execute("select table_name as name from tables()")[1]
        for t in tables:
            if x := xLTableMetaInfo.decode_table_metadata(t[0]):
                dbm[x.exchange][x.market_type].append(x)
        # - for sanity convert default dict to standard dict
        return {s0: {s1: v1 for s1, v1 in v0.items()} for s0, v0 in dbm.items()}

    def get_market_types(self, exchange: str) -> list[str]:
        exc_i = self._read_database_meta_structure()
        return list(exc_i.get(exchange.upper(), dict()).keys())

    def get_reader(self, exchange: str, market: str) -> IReader:
        e_info = self._read_database_meta_structure()

        if exchange in e_info and market in e_info[exchange]:
            return QuestDBReader(
                exchange,
                market,
                e_info[exchange][market],
                self.pgc,
                synthetic_ohlc_timeframes_types=self.synthetic_ohlc_timeframes_types,
            )

        raise ValueError(f"Can't provide data reader for exchange '{exchange}' and market type '{market}'")
