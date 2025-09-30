from collections.abc import Iterable

import numpy as np
import pandas as pd

from qubx.core.basics import DataType
from qubx.core.interfaces import Timestamped
from qubx.core.series import OHLCV, Bar, OrderBook, Quote, Trade, TradeArray
from qubx.data.storage import IDataTransformer
from qubx.data.storages.utils import find_column_index_in_list, find_time_col_idx, recognize_t
from qubx.utils.time import infer_series_frequency


class PandasFrame(IDataTransformer):
    def __init__(self, id_in_index: bool = False) -> None:
        self._dataid_in_index = id_in_index

    def process_data(
        self, data_id: str, dtype: DataType, raw_data: Iterable[np.ndarray], names: list[str], index: int
    ) -> pd.DataFrame:
        if self._dataid_in_index:
            df = pd.DataFrame(raw_data, columns=names)
            df = df.assign(symbol=data_id).set_index([names[index], "symbol"])
        else:
            df = pd.DataFrame(raw_data, columns=names).set_index(names[index])

        return df


class OHLCVSeries(IDataTransformer):
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
                d[_open_idx],
                d[_high_idx],
                d[_low_idx],
                d[_close_idx],
                d[_volume_idx] if _volume_idx else 0,
                d[_b_volume_idx] if _b_volume_idx else 0,
            )

        return ohlc


class TimestampedList(IDataTransformer):
    def _convert_to_type(self, data: np.ndarray, dtype: DataType, scheme: list[int]) -> Timestamped:
        init_args = [(data[i] if i >= 0 else np.nan) for i in scheme]

        match dtype:
            case DataType.TRADE:
                return Trade(*init_args)

            case DataType.QUOTE:
                return Quote(*init_args)

            case DataType.OHLC:
                return Bar(*init_args)

            case DataType.ORDERBOOK:
                return OrderBook(*init_args)

    def _build_scheme(self, dtype: DataType, names: list[str], t_index: int) -> list[int]:
        match dtype:
            case DataType.TRADE:
                _price_idx = find_column_index_in_list(names, "price")
                _size_idx = find_column_index_in_list(names, "size", "amount", "qty", "quantity")
                _side_idx = find_column_index_in_list(names, "side", "is_buyer_maker")  # ???
                _id_idx = find_column_index_in_list(names, "id")
                return [t_index, _price_idx, _size_idx, _side_idx, _id_idx]

            case DataType.QUOTE:
                pass

            case DataType.OHLC:
                pass

            case DataType.ORDERBOOK:
                pass

        pass

    def process_data(
        self, data_id: str, dtype: DataType, raw_data: Iterable[np.ndarray], names: list[str], index: int
    ) -> list[Timestamped]:
        # _n_idx = dict(zip(names, range(len(names))))
        scheme = self._build_scheme(dtype, names, index)
        return [self._convert_to_type(d, dtype, scheme) for d in raw_data]
