import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import wraps

import numpy as np
import pandas as pd
import psycopg as pg

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.containers import RawData, RawMultiData
from qubx.data.registry import storage
from qubx.data.storage import IReader, IStorage, Transformable
from qubx.data.storages.utils import calculate_time_windows_for_chunking, find_column_index_in_list
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

    def __setstate__(self, state):
        self.__dict__.update(state)
        # - don't connect immediately - let _retry decorator handle lazy connection on first use

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.debug(f"Disconnected from QuestDB at {self._host}:{self._port}")

    @_retry
    def execute(self, query: str) -> tuple[list, list]:
        with self._connection.cursor() as _cursor:  # type: ignore
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
        "liquidations": "aggregated_liquidations",
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
            exch, mkt = xLTableMetaInfo._TABLES_FIX.get((exch.lower(), mkt.lower()), (exch.lower(), mkt.lower()))

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
    """
    TODO: create docstring here
    """

    # Info about datatypes and symbols for fast access
    #   | Symbols lookup: {symbol   -> { dtype -> xLTableMetaInfo}
    #   | Dtypes lookup:  {DataType -> ([symbols], xLTableMetaInfo)
    _symbols_lookup: dict[str, dict[DataType, xLTableMetaInfo]]
    _dtype_lookup: dict[str, tuple[set[str], xLTableMetaInfo]]
    _external_sql_builders: dict[str, Callable[[set[str], list[str], str], str]]

    def __init__(
        self,
        exchange: str,
        market: str,
        available_data: list[xLTableMetaInfo],
        pgc: PGConnectionHelper,
        synthetic_ohlc_timeframes_types: bool,
        min_symbols_for_all_data_request: int,
        symbol_column_name="symbol",
    ) -> None:
        self.exchange = exchange
        self.market = market
        self.synthetic_ohlc_timeframes_types = synthetic_ohlc_timeframes_types
        self.min_symbols_for_all_data_request = min_symbols_for_all_data_request
        self.symbol_column_name = symbol_column_name
        self.pgc = pgc
        self._external_sql_builders = {}

        # - build lookup
        self._build_lookups(available_data)

    @staticmethod
    def _convert_time_delta_to_qdb_resample_format(c_tf: str) -> tuple[str, str]:
        """
        Convert standard timedelta format into quest db: "15Min" -> (15, "m) etc
        Returns tuple (number, units)
        """
        if c_tf:
            _t = re.match(r"(\d+)(\w+)", c_tf)
            if _t and len(_t.groups()) > 1:
                number = _t[1]
                units = _t[2][0].lower()
                return number, units
        return c_tf, ""

    def _build_lookups(self, available_data: list[xLTableMetaInfo]):
        """
        TODO: create docstring here
        """
        _symbs_lookup = defaultdict(dict)
        _dtype_lookup = dict()

        for x in available_data:
            # - collect symbols for every table in this reader
            symbols_info = self.pgc.execute(f"select distinct({self.symbol_column_name}) from {x.table_name}")[1]
            symbols, dtypes = [], []
            for s in symbols_info:
                symbols.append(symb := s[0])

                if x.dtype == DataType.OHLC or x.dtype == DataType.AGGREGATED_LIQUIDATIONS:
                    _type_ctor = DataType.OHLC if x.dtype == DataType.OHLC else DataType.AGGREGATED_LIQUIDATIONS
                    if self.synthetic_ohlc_timeframes_types and x.data_timeframe:
                        # - for making life bit easy let's generate all possible frames we can contruct from available base
                        for f in _ext_frames[_ext_frames.searchsorted(pd.Timedelta(x.data_timeframe)) :]:
                            _symbs_lookup[symb][dt := _type_ctor[timedelta_to_str(f)]] = x
                            dtypes.append(dt)
                    else:
                        _symbs_lookup[symb][dt := _type_ctor[x.data_timeframe]] = x
                        dtypes.append(dt)
                else:
                    _symbs_lookup[symb][dt := (x.alias_for_record_type if x.alias_for_record_type else x.dtype)] = x
                    dtypes.append(dt)

            # - for every dtype store symbols and reference to table meta info
            for d in dtypes:
                _dtype_lookup[d] = (set(symbols), x)

        self._symbols_lookup = dict(_symbs_lookup)
        self._dtype_lookup = _dtype_lookup

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        d_ids = set()
        for s, di in self._symbols_lookup.items():
            for dt, x in di.items():
                if x.dtype == dtype or dtype == DataType.ALL or str(dtype) == x.alias_for_record_type:
                    d_ids.add(s)
        return list(sorted(d_ids))

    def get_data_types(self, data_id: str) -> list[DataType]:
        return list(self._symbols_lookup.get(data_id, {}).keys())

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[np.datetime64, np.datetime64]:
        (storage_symbols, xtable) = self._dtype_lookup.get(str(dtype).lower(), (set(), None))
        if xtable is None or not storage_symbols:
            raise ValueError(f"Can't find table for {dtype} data !")
        if data_id not in storage_symbols:
            raise ValueError(f"{xtable.table_name} doesn't contain data for {data_id} !")

        _query = f"""(SELECT timestamp FROM "{xtable.table_name}" WHERE symbol='{data_id}' ORDER BY timestamp ASC LIMIT 1)
                        UNION
                   (SELECT timestamp FROM "{xtable.table_name}" WHERE symbol='{data_id}' ORDER BY timestamp DESC LIMIT 1)
                """
        _r = self.pgc.execute(_query)[1]
        _sr = sorted([np.datetime64(_r[0][0]), np.datetime64(_r[1][0])])
        return _sr[0], _sr[1]

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
        chunksize=0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        # - effectively cut additional info in orderbook (like DataType.ORDERBOOK[0, 10] - just for 10 levels)
        match dtype:
            case DataType.ORDERBOOK:
                x_type = str(DataType.ORDERBOOK).lower()

            case DataType.FUNDAMENTAL:
                x_type = str(DataType.FUNDAMENTAL).lower()

            case _:
                x_type = str(dtype).lower()

        # - get metainfo
        (storage_symbols, xtable) = self._dtype_lookup.get(x_type, (set(), None))
        if xtable is None or not storage_symbols:
            raise ValueError(f"Can't find table for {dtype} data !")

        # - handle symbols
        req_symbols = set(data_id if isinstance(data_id, (list, tuple, set)) else [data_id])
        req_symbols = storage_symbols & req_symbols

        # - check if it has any timeframe for resampling
        _, dt_params = DataType.from_str(dtype)
        resample = dt_params.get("timeframe", "")

        # - handle start / stop
        _start, _stop = handle_start_stop(start, stop)

        # - in case when we want to read by chunks
        if chunksize > 0:

            def _iter_efficient_chunks() -> Iterator[Transformable]:
                time_windows = calculate_time_windows_for_chunking(_start, _stop, resample, chunksize)

                for window_start, window_end in time_windows:
                    yield self._read_data_block(xtable, data_id, req_symbols, dtype, window_start, window_end, resample)

            return _iter_efficient_chunks()

        else:
            return self._read_data_block(xtable, data_id, req_symbols, dtype, _start, _stop, resample)

    def _read_data_block(
        self,
        xtable: xLTableMetaInfo,
        data_id: str | list[str],
        symbols: set[str],
        dtype: DataType | str,
        start: str | pd.Timestamp | None,
        stop: str | pd.Timestamp | None,
        resample: str,
    ) -> Transformable:
        """
        Read data from xtable for given set ot symbols from start to stop
        """
        conditions = []

        if start:
            conditions.append(f"timestamp >= '{start}'")

        if stop:
            conditions.append(f"timestamp < '{stop}'")

        # - get SQL query
        _query = self._prepare_sql_for_dtype(dtype, symbols, xtable, conditions, resample)

        # - process retrieved data
        columns, records = self.pgc.execute(_query)
        data_id_col_idx = find_column_index_in_list(columns, "symbol", "asset")

        # - split received data by symbol
        splitted_records = defaultdict(list)
        for r in records:
            # - keep only requested symbols
            if (symbol := r[data_id_col_idx]) in symbols:
                splitted_records[symbol].append(r)

        # - when requested single symbol just returns single RawData
        if isinstance(data_id, str):
            return RawData(data_id, columns, dtype, splitted_records.get(data_id, []))  # type: ignore

        return RawMultiData([RawData(k, columns, dtype, sr) for k, sr in splitted_records.items()])  # type: ignore

    def _name_in_set(self, name: str, symbols: set[str]) -> str:
        QUOTIFY = lambda ws: map(lambda x: f"'{x.upper()}'", ws)

        # - if requested not too many - just select only them
        if len(symbols) < self.min_symbols_for_all_data_request:
            return f"{name} in ({', '.join(QUOTIFY(symbols))})"

        return ""

    def add_external_builder(self, dtype: str, fn: Callable[[set[str], list[str], str], str]):
        self._external_sql_builders[dtype] = fn

    def _prepare_sql_for_dtype(
        self, dtype: str, symbols: set[str], xtable: xLTableMetaInfo, conditions: list[str], resample: str
    ) -> str:
        match dtype:
            case DataType.OHLC:
                r = """
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
                    from "{table}" {where} {resample};
                """

                # - select symbols
                conditions.append(self._name_in_set("symbol", symbols))

            case DataType.ORDERBOOK:
                r = """SELECT * FROM "{table}" {where} ORDER BY timestamp ASC"""

                # - process orderbook parameters
                _, params = DataType.from_str(dtype)
                if "depth" in params and params.get("depth") is not None:
                    max_depth = max(int(params["depth"]), 1)
                    conditions.append(f"abs(level) <= {max_depth}")

                # - select assets
                conditions.append(self._name_in_set("symbol", symbols))

            case DataType.FUNDAMENTAL:
                r = """
                    select timestamp, asset, metric, last(value) as value
                    from "{table}" {where} {resample};
                """

                # - check if we want to get specific metrics only
                _, params = DataType.from_str(dtype)
                if "fields" in params and (fields := params.get("fields")) is not None:
                    QUOTIFY = lambda ws: map(lambda x: f"'{x}'", ws)
                    conditions.append(f"metric in ({', '.join(QUOTIFY(fields))})")

                # - select assets
                conditions.append(self._name_in_set("asset", symbols))

            case DataType.FUNDING_PAYMENT:
                r = """
                    SELECT timestamp, symbol, funding_rate, funding_interval_hours
                    FROM "{table}" {where} {resample} ORDER BY timestamp ASC;
                """

                # - select assets
                conditions.append(self._name_in_set("symbol", symbols))

            case DataType.FUNDING_RATE:
                r = """
                    SELECT timestamp, symbol, funding_rate as rate, funding_interval_hours as interval, next_funding_time, mark_price, index_price
                    FROM "{table}" {where} {resample} ORDER BY timestamp ASC;
                """

                # - select assets
                conditions.append(self._name_in_set("symbol", symbols))

            case DataType.LIQUIDATION:
                # - TODO: we need to clarify table's structure
                r = """
                    SELECT timestamp, symbol, quantity, price, side
                    FROM "{table}" {where} ORDER BY timestamp ASC;
                """
                # - select assets
                conditions.append(self._name_in_set("symbol", symbols))

            case DataType.AGGREGATED_LIQUIDATIONS:
                r = (
                    """
                    select
                        {shift} timestamp,
                        upper(symbol)               as symbol,
                        avg(avg_buy_price)          as avg_buy_price,
                        sum(buy_amount)             as buy_amount,
                        sum(buy_count)              as buy_count,
                        sum(buy_notional)           as buy_notional,
                        last(last_buy_price)        as last_buy_price,
                        avg(avg_sell_price)         as avg_sell_price,
                        sum(sell_amount)            as sell_amount,
                        sum(sell_count)             as sell_count,
                        sum(sell_notional)          as sell_notional,
                        last(last_sell_price)       as last_sell_price
                    from "{table}" {where} {resample};
                """
                    if resample
                    else """
                    select
                        timestamp,
                        upper(symbol)               as symbol,
                        avg_buy_price               as avg_buy_price,
                        buy_amount                  as buy_amount,
                        buy_count                   as buy_count,
                        buy_notional                as buy_notional,
                        last_buy_price              as last_buy_price,
                        avg_sell_price              as avg_sell_price,
                        sell_amount                 as sell_amount,
                        sell_count                  as sell_count,
                        sell_notional               as sell_notional,
                        last_sell_price             as last_sell_price
                    from "{table}" {where};
                """
                )

                # - select assets
                conditions.append(self._name_in_set("symbol", symbols))

            case _:
                if dtype in self._external_sql_builders:
                    r = self._external_sql_builders.get(dtype)(symbols, conditions, resample)
                else:
                    r = """SELECT * FROM "{table}" {where};"""

                    # - select assets
                    conditions.append(self._name_in_set("symbol", symbols))

        COMBINE = lambda cs: " and ".join(filter(lambda x: x, cs)) if cs else ""
        where = COMBINE(conditions)
        shift = ""
        if resample:
            n, u = self._convert_time_delta_to_qdb_resample_format(resample)
            # - if request need to be timestamped at right bound of interval when resampling
            shift = f"dateadd('{u}', {n}, timestamp)"
            resample = f"SAMPLE by {n}{u} FILL(NONE)"

        return r.format(
            table=xtable.table_name, where="" if not where else f"where {where}", resample=resample, shift=shift
        )


@storage("qdb")
@storage("questdb")
class QuestDBStorage(IStorage):
    """
    QuestDB storage implementation
    """

    pgc: PGConnectionHelper

    def __init__(
        self,
        host="localhost",
        user="admin",
        password="quest",
        port=8812,
        min_symbols_for_all_data_request: int = 50,
        synthetic_ohlc_timeframes_types: bool = True,
    ) -> None:
        self.pgc = PGConnectionHelper(host, user, password, port)
        self.min_symbols_for_all_data_request = min_symbols_for_all_data_request
        self.synthetic_ohlc_timeframes_types = synthetic_ohlc_timeframes_types

    @property
    def connection_url(self):
        return self.pgc.connection_url

    def __getstate__(self):
        # - close connection before pickling
        state = self.__dict__.copy()
        state["pgc"] = self.pgc.__getstate__()
        return state

    def __setstate__(self, state):
        # - restore state and recreate connection
        self.__dict__.update(state)
        # - recreate PGConnectionHelper with stored connection params
        pgc_state = state["pgc"]
        self.pgc = PGConnectionHelper.__new__(PGConnectionHelper)
        self.pgc.__setstate__(pgc_state)

    def close(self):
        self.pgc.close()

    def __del__(self):
        if hasattr(self, "pgc"):
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
                min_symbols_for_all_data_request=self.min_symbols_for_all_data_request,
                # - for fundamental data we want to search in assets not in symbols
                symbol_column_name="asset" if market.lower() == "fundamental" else "symbol",
            )

        raise ValueError(f"Can't provide data reader for exchange '{exchange}' and market type '{market}'")
