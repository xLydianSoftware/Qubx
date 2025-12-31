#
# New experimental data reading interface. We need to deprecate old DataReader approach after this new one will be finished and approved
#
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import (
    AggregatedLiquidations,
    DataType,
    FundingPayment,
    FundingRate,
    Liquidation,
    OpenInterest,
    TimestampedDict,
)
from qubx.core.interfaces import Timestamped
from qubx.core.series import OHLCV, Bar, GenericSeries, Quote, Trade
from qubx.data.storage import IDataTransformer
from qubx.data.storages.utils import build_snapshots, find_column_index_in_list
from qubx.pandaz.utils import scols, srows
from qubx.utils.time import infer_series_frequency


class PandasFrame(IDataTransformer):
    """
    Transform data to pandas dataframe
    """

    def __init__(self, id_in_index: bool = False) -> None:
        self._dataid_in_index = id_in_index

    def process_data(
        self, data_id: str, dtype: DataType, raw_data: Iterable[np.ndarray], names: list[str], index: int
    ) -> pd.DataFrame:
        t_name = names[index]
        if self._dataid_in_index:
            df = pd.DataFrame(raw_data, columns=names)
            t_values = pd.to_datetime(df[t_name].values)
            df = df.assign(**{t_name: t_values, "symbol": data_id}).set_index([t_name, "symbol"])
        else:
            df = pd.DataFrame(raw_data, columns=names).set_index(t_name)
            df.index = pd.DatetimeIndex(df.index)

        return df

    def combine_data(self, transformed: dict[str, pd.DataFrame]) -> Any:
        if transformed:
            if self._dataid_in_index:
                return srows(*transformed.values())
            return scols(*transformed.values(), keys=transformed.keys())
        return pd.DataFrame()


class OHLCVSeries(IDataTransformer):
    """
    Transform data to qubx OHLCV series. For that input data must have at least open, high, and close columns
    """

    def __init__(self, timestamp_units="ns", max_length=np.inf) -> None:
        self.timestamp_units = timestamp_units
        self.max_length = max_length

    def _get_volume_block_indexes(
        self, column_names: list[str]
    ) -> tuple[int | None, int | None, int | None, int | None, int | None]:
        def _safe_find_col(*args):
            try:
                return find_column_index_in_list(column_names, *args)
            except Exception:
                return None

        _volume_idx = _safe_find_col("volume", "vol")
        _b_volume_idx = _safe_find_col("bought_volume", "taker_buy_volume", "taker_bought_volume")
        _volume_quote_idx = _safe_find_col("volume_quote", "quote_volume")
        _b_volume_quote_idx = _safe_find_col(
            "bought_volume_quote", "taker_buy_quote_volume", "taker_bought_quote_volume"
        )
        _trade_count_idx = _safe_find_col("trade_count", "count")
        return _volume_idx, _b_volume_idx, _volume_quote_idx, _b_volume_quote_idx, _trade_count_idx

    def _time(self, t: float | np.int64, timestamp_units: str) -> int:
        t = int(t) if isinstance(t, float) or isinstance(t, np.int64) else t  # type: ignore
        if timestamp_units == "ns":
            return np.datetime64(t, "ns").item()
        return np.datetime64(t, timestamp_units).astype("datetime64[ns]").item()

    def process_data(
        self, data_id: str, dtype: DataType, raw_data: list[np.ndarray], names: list[str], index: int
    ) -> OHLCV:
        _volume_idx = None
        _b_volume_idx = None
        try:
            _close_idx = find_column_index_in_list(names, "close")
            _open_idx = find_column_index_in_list(names, "open")
            _high_idx = find_column_index_in_list(names, "high")
            _low_idx = find_column_index_in_list(names, "low")
            _volume_idx, _b_volume_idx, _volume_quote_idx, _b_volume_quote_idx, _trade_count_idx = (
                self._get_volume_block_indexes(names)
            )

        except Exception as e:
            raise ValueError(f"Can't find columns in data: {e}")

        ts = [t[index] for t in raw_data[:100]]
        timeframe = pd.Timedelta(infer_series_frequency(ts)).asm8.item()
        ohlc = OHLCV(data_id, timeframe, max_series_length=self.max_length)

        for d in raw_data:
            ohlc.update_by_bar(
                self._time(d[index], self.timestamp_units),
                open=d[_open_idx],
                high=d[_high_idx],
                low=d[_low_idx],
                close=d[_close_idx],
                vol_incr=d[_volume_idx] if _volume_idx else 0,
                b_vol_incr=d[_b_volume_idx] if _b_volume_idx else 0,
                volume_quote=d[_volume_quote_idx] if _volume_quote_idx else 0,  # type: ignore
                bought_volume_quote=d[_b_volume_quote_idx] if _b_volume_quote_idx else 0,  # type: ignore
                trade_count=d[_trade_count_idx] if _trade_count_idx else 0,  # type: ignore
            )

        return ohlc


class TypedRecords(IDataTransformer):
    """
    Transform data to list of qubx timestamped objects (Quote, Trade, etc).
    Type of generated object depends on dtype from RawData container.
    If dtype is not supported/recognized it returns list of TimestampedDict.
    TODO:
        - probably we need to add customization of how it recognize type's constructor in more flexible way
    """

    def __init__(self, timestamp_units="ns") -> None:
        self.timestamp_units = timestamp_units

    @staticmethod
    def _time(t, timestamp_units: str) -> int:
        t = int(t) if isinstance(t, float) or isinstance(t, np.int64) else t  # type: ignore
        if timestamp_units == "ns":
            return np.datetime64(t, "ns").item()
        return np.datetime64(t, timestamp_units).astype("datetime64[ns]").item()

    def _convert_to_type(
        self, data: np.ndarray, dtype: DataType, scheme: dict[str, int], names: list[str]
    ) -> Timestamped:
        init_args = {
            a_name: self._time(data[a_idx], self.timestamp_units)
            if self.timestamp_units and (a_name.startswith("time") or "_time" in a_name)
            else data[a_idx]
            for a_name, a_idx in scheme.items()
        }

        match dtype:
            case DataType.TRADE:
                return Trade(**init_args)

            case DataType.QUOTE:
                return Quote(**init_args)

            case DataType.OHLC:
                return Bar(**init_args)

            # - old orderbook data (deprecated)
            case DataType.ORDERBOOK:
                raise ValueError("It shouldn't reach this code !")

            case DataType.LIQUIDATION:
                return Liquidation(**init_args)  # type: ignore

            case DataType.AGGREGATED_LIQUIDATIONS:
                return AggregatedLiquidations(**init_args)  # type: ignore

            case DataType.FUNDING_RATE:
                return FundingRate(**init_args)  # type: ignore

            case DataType.FUNDING_PAYMENT:
                return FundingPayment(**init_args)  # type: ignore

            case DataType.OPEN_INTEREST:
                return OpenInterest(**init_args)  # type: ignore

            case DataType.OHLC_QUOTES:
                raise NotImplementedError("OHLC_QUOTES processing is not yet implemented ")

            case DataType.OHLC_TRADES:
                raise NotImplementedError("OHLC_TRADES processing is not yet implemented ")

        # - if nothing is found just returns timestamped dictionary
        dict_data = init_args | {"data": {n: data[k] for k, n in enumerate(names) if not n.startswith("time")}}
        return TimestampedDict(**dict_data)  # type: ignore

    def _recognize_type_ctor_scheme(self, dtype: DataType, names: list[str], t_index: int) -> dict[str, int]:
        """
        Recognize what need to be used from the data to construct appropriate Qubx object.
        Returns dict with names of constructor argument and index of column in input data array.
        """

        def _column_index_for(
            ctor_argument: str, presented_names: list[str], possible_names: list[str], mandatory: bool = False
        ):
            try:
                return {ctor_argument: find_column_index_in_list(presented_names, *possible_names)}
            except:
                if mandatory:
                    raise ValueError(
                        f"Can't find possible aliases for '{ctor_argument}' in provided list '{names}' for creating '{dtype}' !"
                    )
                else:
                    return {}

        ctor_args = {"time": t_index}

        # fmt: off
        match dtype:
            case DataType.TRADE:
                ctor_args |= (
                    _column_index_for("price", names, ["price"], mandatory=True)
                    | _column_index_for("size", names, ["size", "amount", "qty", "quantity"], mandatory=True)
                    | _column_index_for("side", names, ["side", "is_buyer_maker"])
                    | _column_index_for("trade_id", names, ["id"])
                )

            case DataType.QUOTE:
                ctor_args |= (
                    _column_index_for("bid", names, ["bid"], mandatory=True)
                    | _column_index_for("ask", names, ["ask"], mandatory=True)
                    | _column_index_for("bid_size", names, ["bidvol", "bid_vol", "bidsize", "bid_size"], mandatory=True)
                    | _column_index_for("ask_size", names, ["askvol", "ask_vol", "asksize", "ask_size"], mandatory=True)
                )

            case DataType.OHLC | DataType.OHLC_QUOTES | DataType.OHLC_TRADES:
                ctor_args |= (
                    _column_index_for("open", names, ["open"], mandatory=True)
                    | _column_index_for("high", names, ["high"], mandatory=True)
                    | _column_index_for("low", names, ["low"], mandatory=True)
                    | _column_index_for("close", names, ["close"], mandatory=True)
                    | _column_index_for("volume", names, ["volume", "vol"], mandatory=dtype == DataType.OHLC_TRADES) # for trades it needs volume !
                    | _column_index_for("bought_volume", names, ["bought_volume", "taker_buy_volume", "taker_bought_volume"])
                    | _column_index_for("volume_quote", names, ["volume_quote", "quote_volume"])
                    | _column_index_for("bought_volume_quote", names, ["bought_volume_quote", "taker_buy_quote_volume", "taker_bought_quote_volume"],)
                    | _column_index_for("trade_count", names, ["trade_count", "count"])
                )

            case DataType.ORDERBOOK:
                # ctor_args |= (
                #     _column_index_for("top_bid", names, ["top_bid"], mandatory=True)
                #     | _column_index_for("top_ask", names, ["top_ask"], mandatory=True)
                #     | _column_index_for("tick_size", names, ["tick_size"], mandatory=True)
                #     | _column_index_for("tick_size", names, ["tick_size"], mandatory=True)
                # )
                ctor_args |= (
                    _column_index_for("level", names, ["level"], mandatory=True)
                    | _column_index_for("price", names, ["price"], mandatory=True)
                    | _column_index_for("size", names, ["size"], mandatory=True)
                )

            case DataType.LIQUIDATION:
                ctor_args |= (
                    _column_index_for("quantity", names, ["quantity"], mandatory=True)
                    | _column_index_for("price", names, ["price"], mandatory=True)
                    | _column_index_for("side", names, ["side"], mandatory=True)
                )

            case DataType.AGGREGATED_LIQUIDATIONS:
                ctor_args |= (
                    _column_index_for("avg_buy_price", names, ["avg_buy_price"], mandatory=True)
                    | _column_index_for("last_buy_price", names, ["last_buy_price"], mandatory=True)
                    | _column_index_for("buy_amount", names, ["buy_amount"], mandatory=True)
                    | _column_index_for("buy_count", names, ["buy_count"], mandatory=True)
                    | _column_index_for("buy_notional", names, ["buy_notional"], mandatory=True)
                    | _column_index_for("avg_sell_price", names, ["avg_sell_price"], mandatory=True)
                    | _column_index_for("last_sell_price", names, ["last_sell_price"], mandatory=True)
                    | _column_index_for("sell_amount", names, ["sell_amount"], mandatory=True)
                    | _column_index_for("sell_count", names, ["sell_count"], mandatory=True)
                    | _column_index_for("sell_notional", names, ["sell_notional"], mandatory=True)
                )

            case DataType.FUNDING_RATE:
                ctor_args |= (
                    _column_index_for("rate", names, ["rate"], mandatory=True)
                    | _column_index_for("interval", names, ["interval"], mandatory=True)
                    | _column_index_for("next_funding_time", names, ["next_funding_time"], mandatory=True)
                    | _column_index_for("mark_price", names, ["mark_price"])
                    | _column_index_for("index_price", names, ["index_price"])
                )

            case DataType.FUNDING_PAYMENT:
                ctor_args |= (
                    _column_index_for("funding_rate", names, ["funding_rate"], mandatory=True)
                    | _column_index_for("funding_interval_hours", names, ["funding_interval_hours"], mandatory=True)
                )

            case DataType.OPEN_INTEREST:
                ctor_args |= (
                    _column_index_for("symbol", names, ["symbol"], mandatory=True)
                    | _column_index_for("open_interest", names, ["open_interest"], mandatory=True)
                    | _column_index_for("open_interest_usd", names, ["open_interest_usd"], mandatory=True)
                )
        # fmt: on

        return ctor_args

    def process_data(
        self, data_id: str, dtype: DataType, raw_data: Iterable[np.ndarray], names: list[str], index: int
    ) -> list[Timestamped]:
        scheme = self._recognize_type_ctor_scheme(dtype, names, index)

        # - special case for sequental orderbook updates
        if dtype == DataType.ORDERBOOK:
            return build_snapshots(raw_data, scheme["level"], scheme["price"], scheme["size"], index)  # type: ignore

        return [self._convert_to_type(d, dtype, scheme, names) for d in raw_data]


class TypedGenericSeries(TypedRecords):
    """
    Transform data to GenericSeries of qubx timestamped objects (Quote, Trade, etc).
    Type of generated object depends on dtype from RawData container.
    If dtype is not supported/recognized it returns list of TimestampedDict.
    """

    def __init__(self, timeframe=None, timestamp_units="ns", max_length=np.inf) -> None:
        self.timestamp_units = timestamp_units
        self.max_length = max_length
        self.timeframe = timeframe

    def process_data(
        self, data_id: str, dtype: DataType, raw_data: list[np.ndarray], names: list[str], index: int
    ) -> GenericSeries:
        timeframe = self.timeframe
        if not timeframe:
            # - some data may contains many records with same timestamps
            ts = list(sorted(set([t[index] for t in raw_data[:1000]])))
            timeframe = pd.Timedelta(infer_series_frequency(ts)).asm8.item()

        gens = GenericSeries(data_id, timeframe, max_series_length=self.max_length)
        scheme = self._recognize_type_ctor_scheme(dtype, names, index)

        # - special case for sequental orderbook updates
        if dtype == DataType.ORDERBOOK:
            for s in build_snapshots(raw_data, scheme["level"], scheme["price"], scheme["size"], index):  # type: ignore:
                gens.update(s)
        else:
            for d in raw_data:
                gens.update(self._convert_to_type(d, dtype, scheme, names))

        return gens


class TickSeries(IDataTransformer):
    """
    Transform OHLC bars into simulates ticks (Quotes or Trades)
    """

    @staticmethod
    def timedelta_to_numpy(x: str) -> int:
        return pd.Timedelta(x).to_numpy().item()

    D1, H1 = timedelta_to_numpy("1D"), timedelta_to_numpy("1h")
    MS1 = 1_000_000
    S1 = 1000 * MS1
    M1 = 60 * S1

    DEFAULT_DAILY_SESSION = (timedelta_to_numpy("00:00:00.100"), timedelta_to_numpy("23:59:59.900"))
    STOCK_DAILY_SESSION = (timedelta_to_numpy("9:30:00.100"), timedelta_to_numpy("15:59:59.900"))
    CME_FUTURES_DAILY_SESSION = (timedelta_to_numpy("8:30:00.100"), timedelta_to_numpy("15:14:59.900"))

    def __init__(
        self,
        trades: bool = False,  # if we also wants 'trades'
        default_bid_size=1e9,  # default bid/ask is big
        default_ask_size=1e9,  # default bid/ask is big
        daily_session_start_end=DEFAULT_DAILY_SESSION,
        timestamp_units="ns",
        spread=0.0,
        open_close_time_shift_secs=1.0,
        quotes=True,
    ) -> None:
        self._d_session_start = daily_session_start_end[0]
        self._d_session_end = daily_session_start_end[1]
        self._timestamp_units = timestamp_units
        self._open_close_time_shift_secs = open_close_time_shift_secs  # type: ignore
        self._trades = trades
        self._quotes = quotes
        self._bid_size = default_bid_size
        self._ask_size = default_ask_size
        self._s2 = spread / 2.0

    def _detect_emulation_timestamps(self, time_index: int, rows_data: list[list]):
        ts = [t[time_index] for t in rows_data[:100]]
        try:
            self._freq = infer_series_frequency(ts)
        except ValueError:
            logger.warning("Can't determine frequency for incoming data")
            return

        # - timestamps when we emit simulated quotes
        dt = self._freq.astype("timedelta64[ns]").item()
        dt10 = dt // 10

        # - adjust open-close time shift to avoid overlapping timestamps
        if self._open_close_time_shift_secs * self.S1 >= (dt // 2 - dt10):
            self._open_close_time_shift_secs = (dt // 2 - 2 * dt10) // self.S1

        if dt < self.D1:
            self._t_start = self._open_close_time_shift_secs * self.S1
            self._t_mid1 = dt // 2 - dt10
            self._t_mid2 = dt // 2 + dt10
            self._t_end = dt - self._open_close_time_shift_secs * self.S1
        else:
            self._t_start = self._d_session_start + self._open_close_time_shift_secs * self.S1
            self._t_mid1 = dt // 2 - self.H1
            self._t_mid2 = dt // 2 + self.H1
            self._t_end = self._d_session_end - self._open_close_time_shift_secs * self.S1

    def process_data(
        self, data_id: str, dtype: DataType, raw_data: list[np.ndarray], names: list[str], index: int
    ) -> TypedRecords:
        if len(raw_data) < 2:
            raise ValueError("Input data must contain at least two records for ticks simulation !")

        try:
            _close_idx = find_column_index_in_list(names, "close")
            _open_idx = find_column_index_in_list(names, "open")
            _high_idx = find_column_index_in_list(names, "high")
            _low_idx = find_column_index_in_list(names, "low")
        except:
            raise ValueError(
                f"Incoming data must be presented as OHLC bars and contains open, high, low, close fields, passed '{names}' !"
            )

        # - for trades we need volumes
        _volume_idx = -1
        if self._trades:
            _volume_idx = find_column_index_in_list(names, "vol", "volume")

        # - detect parameters for transformation
        self._detect_emulation_timestamps(index, raw_data)
        s2 = self._s2

        buffer = []
        for data in raw_data:
            ti = TypedRecords._time(data[index], self._timestamp_units)
            o = data[_open_idx]
            h = data[_high_idx]
            l = data[_low_idx]
            c = data[_close_idx]
            rv = data[_volume_idx] if _volume_idx >= 0 else 0
            rv = rv / (h - l) if h > l else rv

            # - opening quote
            if self._quotes:
                buffer.append(Quote(ti + self._t_start, o - s2, o + s2, self._bid_size, self._ask_size))

            if c >= o:
                if self._trades:
                    buffer.append(Trade(ti + self._t_start, o - s2, rv * (o - l)))  # sell 1

                if self._quotes:
                    buffer.append(Quote(ti + self._t_mid1, l - s2, l + s2, self._bid_size, self._ask_size))

                if self._trades:
                    buffer.append(Trade(ti + self._t_mid1, l + s2, rv * (c - o)))  # buy 1

                if self._quotes:
                    buffer.append(Quote(ti + self._t_mid2, h - s2, h + s2, self._bid_size, self._ask_size))

                if self._trades:
                    buffer.append(Trade(ti + self._t_mid2, h - s2, rv * (h - c)))  # sell 2
            else:
                if self._trades:
                    buffer.append(Trade(ti + self._t_start, o + s2, rv * (h - o)))  # buy 1

                if self._quotes:
                    buffer.append(Quote(ti + self._t_mid1, h - s2, h + s2, self._bid_size, self._ask_size))

                if self._trades:
                    buffer.append(Trade(ti + self._t_mid1, h - s2, rv * (o - c)))  # sell 1

                if self._quotes:
                    buffer.append(Quote(ti + self._t_mid2, l - s2, l + s2, self._bid_size, self._ask_size))

                if self._trades:
                    buffer.append(Trade(ti + self._t_mid2, l + s2, rv * (c - l)))  # buy 2

            # - closing quote
            if self._quotes:
                buffer.append(Quote(ti + self._t_end, c - s2, c + s2, self._bid_size, self._ask_size))
        return buffer
