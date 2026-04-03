from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

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
from qubx.core.series import OHLCV, Bar, ColumnarSeries, GenericSeries, Quote, Trade
from qubx.data.storage import IDataTransformer, IRawContainer
from qubx.data.storages.utils import build_snapshots, find_column_index_in_list
from qubx.pandaz.utils import scols, srows
from qubx.utils.time import convert_times_to_ns, infer_series_frequency, to_timedelta


def _extract_column(data: pa.RecordBatch, field_idx: int | None, default_dtype: type = np.float64) -> np.ndarray:
    """
    Extract column as numpy array. Converts object dtype to target dtype.
    For integer dtypes, NaN values are filled with 0.
    For string/object target dtype, no conversion is performed.
    """
    if field_idx is None:
        return np.array([])

    arr = data.column(field_idx).to_numpy(zero_copy_only=False)

    # - for string/object dtype, return as-is
    if default_dtype is object or default_dtype is str:
        return arr

    # - convert object dtype (None -> NaN for float, or fill with 0 for int)
    if arr.dtype == object:
        arr = arr.astype(np.float64)

    # - for integer target dtype, fill NaN with 0 first
    if np.issubdtype(default_dtype, np.integer):
        arr = np.nan_to_num(arr, nan=0).astype(default_dtype)
    elif arr.dtype != default_dtype:
        arr = arr.astype(default_dtype)

    return arr


class PandasFrame(IDataTransformer):
    """
    Transform data to pandas dataframe
    """

    def __init__(self, id_in_index: bool = False) -> None:
        self._dataid_in_index = id_in_index

    def process_data(self, raw_data: IRawContainer) -> pd.DataFrame:
        t_name = raw_data.names[raw_data.index]
        if self._dataid_in_index:
            df = raw_data.data.to_pandas()
            t_values = pd.to_datetime(df[t_name].values)
            # - if data already has a symbol column, use existing values instead of overwriting with data_id
            if "symbol" in df.columns:
                df = df.assign(**{t_name: t_values}).set_index([t_name, "symbol"])
            else:
                df = df.assign(**{t_name: t_values, "symbol": raw_data.data_id}).set_index([t_name, "symbol"])
        else:
            df = raw_data.data.to_pandas().set_index(t_name)
            df.index = pd.DatetimeIndex(df.index)

        return df

    def combine_data(self, transformed: dict[str, Any]) -> Any:
        if not transformed:
            return pd.DataFrame()

        first = next(iter(transformed.values()))

        if isinstance(first, IRawContainer):
            # - raw containers passed directly from RawMultiData.transform;
            #   dispatch to fast Arrow path or per-symbol conversion
            if self._dataid_in_index:
                return self._fast_arrow_combine(transformed)

            # - for id_in_index=False: need per-symbol DataFrames for pivot logic
            transformed = {k: self.process_data(raw) for k, raw in transformed.items()}

        # - transformed is now dict[str, pd.DataFrame]
        if self._dataid_in_index:
            return srows(*transformed.values())

        # - long-format data (e.g. FUNDAMENTAL) produces per-symbol frames with
        #   non-unique timestamp indices; pivot them to wide before column-concat
        if any(not df.index.is_unique for df in transformed.values()):
            transformed = {k: self._pivot_to_wide(df) for k, df in transformed.items()}

        return scols(*transformed.values(), keys=transformed.keys())

    def _fast_arrow_combine(self, raws: dict[str, IRawContainer]) -> pd.DataFrame:
        """
        Bulk Arrow concat path for id_in_index=True.

        Concatenates all per-symbol Arrow batches at the Arrow level (near zero-copy),
        then does a single to_pandas() + set_index().  For N symbols this is ~10-15x
        faster than N individual to_pandas() calls followed by pd.concat(N DataFrames).
        """
        sample = next(iter(raws.values()))
        t_name = sample.names[sample.index]
        has_symbol_col = "symbol" in sample.names

        tables = []
        for data_id, raw in raws.items():
            tbl = pa.Table.from_batches([raw.data])
            if not has_symbol_col:
                sym_col = pa.array([data_id] * len(tbl), type=pa.large_string())
                tbl = tbl.append_column(pa.field("symbol", pa.large_string()), sym_col)
            tables.append(tbl)

        tbl = pa.concat_tables(tables, promote_options="permissive")
        df = tbl.to_pandas()
        df[t_name] = pd.to_datetime(df[t_name])
        return df.set_index([t_name, "symbol"]).sort_index()

    @staticmethod
    def _pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a long-format DataFrame with a non-unique DatetimeIndex to wide format.

        Uses pivot_table (not unstack) so duplicate (timestamp, category) pairs from
        data corrections or re-ingestion are handled gracefully via last-value aggregation.

        Candidate pivot columns are tried in order: non-numeric first (categorical
        discriminators are almost always strings), then numeric.  Constant-value columns
        (e.g. an 'asset' column that always equals the symbol name) are excluded first
        so they don't shadow the real discriminating column.

        If only one value-column remains after the pivot the top-level of the resulting
        MultiIndex columns is collapsed (e.g. ``(value, market_cap)`` → ``market_cap``).

        Raises:
            ValueError: when the frame has no non-constant columns to pivot on.
        """
        # - exclude constant columns (same value on every row) — they don't discriminate
        df = df[[c for c in df.columns if df[c].nunique() > 1]]
        if df.empty or df.columns.empty:
            raise ValueError(
                "DataFrame has a non-unique timestamp index but no varying columns to pivot on. "
                "Use to_pd(True) for row-concat with a (timestamp, symbol) MultiIndex instead."
            )

        # - prefer non-numeric columns as pivot keys (categorical discriminators first)
        non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        # - reset index so the time column becomes a regular column for pivot_table;
        #   avoids issues with unnamed DatetimeIndex being ignored by pivot_table
        t_col = df.index.name or "index"
        df_reset = df.reset_index()

        for col in non_numeric + numeric:
            # - value columns: everything except the time column and the pivot key
            values = [c for c in df_reset.columns if c not in (t_col, col)]
            # - pivot_table groups by t_col, uses col as column labels,
            #   and resolves duplicate (timestamp, col) pairs via last-value aggregation
            df_wide = df_reset.pivot_table(index=t_col, columns=col, values=values if values else None, aggfunc="last")
            df_wide.index = pd.DatetimeIndex(df_wide.index)
            df_wide.index.name = t_col
            # - collapse redundant top level when only one value column is left
            if df_wide.columns.nlevels > 1 and df_wide.columns.get_level_values(0).nunique() == 1:
                df_wide.columns = df_wide.columns.droplevel(0)
                df_wide.columns.name = None
            return df_wide

        raise ValueError(
            "DataFrame has a non-unique timestamp index and no column to pivot on. "
            "Use to_pd(True) for row-concat with a (timestamp, symbol) MultiIndex instead."
        )


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

    def process_data(self, raw_data: IRawContainer) -> OHLCV:
        index = raw_data.index
        names = raw_data.names

        try:
            _open_idx = find_column_index_in_list(names, "open")
            _high_idx = find_column_index_in_list(names, "high")
            _low_idx = find_column_index_in_list(names, "low")
            _close_idx = find_column_index_in_list(names, "close")
            _volume_idx, _b_volume_idx, _volume_quote_idx, _b_volume_quote_idx, _trade_count_idx = (
                self._get_volume_block_indexes(names)
            )
        except Exception as e:
            raise ValueError(
                f"Can't find one of required mandatory columns (open, high, low, close) in provided data: {e}"
            )

        _data = raw_data.data

        # - extract time column and convert to nanoseconds int64
        times = convert_times_to_ns(_data.column(index).to_numpy(zero_copy_only=False), self.timestamp_units)

        # - infer timeframe from first 100 timestamps
        timeframe = to_timedelta(infer_series_frequency(pd.DatetimeIndex(times[:100]))).asm8.item()
        ohlc = OHLCV(raw_data.data_id, timeframe, max_series_length=self.max_length)

        # - use vectorized append_data (Cython)
        ohlc.append_data(  # type: ignore
            times,
            _extract_column(_data, _open_idx),
            _extract_column(_data, _high_idx),
            _extract_column(_data, _low_idx),
            _extract_column(_data, _close_idx),
            _extract_column(_data, _volume_idx),
            _extract_column(_data, _b_volume_idx),
            _extract_column(_data, _volume_quote_idx),
            _extract_column(_data, _b_volume_quote_idx),
            _extract_column(_data, _trade_count_idx, np.int64),
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

    def _build_typed_object(self, row: dict[str, Any], dtype: DataType, names: list[str]) -> Timestamped:
        """
        Build typed object from row data dict.
        Time fields are already converted to nanoseconds int64.
        """
        match dtype:
            case DataType.TRADE:
                return Trade(**row)

            case DataType.QUOTE:
                return Quote(**row)

            case DataType.OHLC:
                # - helper to handle NaN (which is truthy, so `or` doesn't work)
                def _v(key, default=0.0):
                    v = row.get(key)
                    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v

                return Bar(
                    time=row["time"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=_v("volume"),
                    bought_volume=_v("bought_volume"),
                    volume_quote=_v("volume_quote"),
                    bought_volume_quote=_v("bought_volume_quote"),
                    trade_count=_v("trade_count", 0),
                )

            # - old orderbook data (deprecated)
            case DataType.ORDERBOOK:
                raise ValueError("It shouldn't reach this code !")

            case DataType.LIQUIDATION:
                return Liquidation(**row)  # type: ignore

            case DataType.AGGREGATED_LIQUIDATIONS:
                return AggregatedLiquidations(**row)  # type: ignore

            case DataType.FUNDING_RATE:
                return FundingRate(**row)  # type: ignore

            case DataType.FUNDING_PAYMENT:
                return FundingPayment(
                    time=int(row["time"]),
                    funding_rate=float(row["funding_rate"]),
                    funding_interval_hours=int(row["funding_interval_hours"]),
                )

            case DataType.OPEN_INTEREST:
                return OpenInterest(**row)  # type: ignore

            case DataType.OHLC_QUOTES:
                raise NotImplementedError("OHLC_QUOTES processing is not yet implemented ")

            case DataType.OHLC_TRADES:
                raise NotImplementedError("OHLC_TRADES processing is not yet implemented ")

        # - if nothing is found just returns timestamped dictionary
        data_fields = {n: row.get(n) for n in names if not n.startswith("time") and n in row}
        return TimestampedDict(time=int(row["time"]), data=data_fields)

    def _recognize_type_ctor_scheme(
        self, dtype: DataType, names: list[str], t_index: int
    ) -> dict[str, tuple[int, type]]:
        """
        Recognize what need to be used from the data to construct appropriate Qubx object.
        Returns dict with names of constructor argument and (column_index, numpy_dtype) tuple.
        """

        def _column_index_dtype_for(
            ctor_argument: str,
            presented_names: list[str],
            possible_names: list[str],
            mandatory: bool = False,
            col_dtype: type = np.float64,
        ):
            try:
                return {ctor_argument: (find_column_index_in_list(presented_names, *possible_names), col_dtype)}
            except:
                if mandatory:
                    raise ValueError(
                        f"Can't find possible aliases for '{ctor_argument}' in provided list '{names}' for creating '{dtype}' !"
                    )
                else:
                    return {}

        ctor_args: dict[str, tuple[int, type]] = {"time": (t_index, np.int64)}

        # fmt: off
        match dtype:
            case DataType.TRADE:
                ctor_args |= (
                    _column_index_dtype_for("price", names, ["price"], mandatory=True)
                    | _column_index_dtype_for("size", names, ["size", "amount", "qty", "quantity"], mandatory=True)
                    | _column_index_dtype_for("side", names, ["side", "is_buyer_maker"])
                    | _column_index_dtype_for("trade_id", names, ["id"], col_dtype=np.int64)
                )

            case DataType.QUOTE:
                ctor_args |= (
                    _column_index_dtype_for("bid", names, ["bid", "bid_price"], mandatory=True)
                    | _column_index_dtype_for("ask", names, ["ask", "ask_price"], mandatory=True)
                    | _column_index_dtype_for("bid_size", names, ["bidvol", "bid_vol", "bidsize", "bid_size", "bid_amount"], mandatory=True)
                    | _column_index_dtype_for("ask_size", names, ["askvol", "ask_vol", "asksize", "ask_size", "ask_amount"], mandatory=True)
                )

            case DataType.OHLC | DataType.OHLC_QUOTES | DataType.OHLC_TRADES:
                ctor_args |= (
                    _column_index_dtype_for("open", names, ["open"], mandatory=True)
                    | _column_index_dtype_for("high", names, ["high"], mandatory=True)
                    | _column_index_dtype_for("low", names, ["low"], mandatory=True)
                    | _column_index_dtype_for("close", names, ["close"], mandatory=True)
                    | _column_index_dtype_for("volume", names, ["volume", "vol"], mandatory=dtype == DataType.OHLC_TRADES) # for trades it needs volume !
                    | _column_index_dtype_for("bought_volume", names, ["bought_volume", "taker_buy_volume", "taker_bought_volume"])
                    | _column_index_dtype_for("volume_quote", names, ["volume_quote", "quote_volume"])
                    | _column_index_dtype_for("bought_volume_quote", names, ["bought_volume_quote", "taker_buy_quote_volume", "taker_bought_quote_volume"],)
                    | _column_index_dtype_for("trade_count", names, ["trade_count", "count"], col_dtype=np.int64)
                )

            case DataType.ORDERBOOK:
                ctor_args |= (
                    _column_index_dtype_for("level", names, ["level"], mandatory=True)
                    | _column_index_dtype_for("price", names, ["price"], mandatory=True)
                    | _column_index_dtype_for("size", names, ["size"], mandatory=True)
                )

            case DataType.LIQUIDATION:
                ctor_args |= (
                    _column_index_dtype_for("quantity", names, ["quantity"], mandatory=True)
                    | _column_index_dtype_for("price", names, ["price"], mandatory=True)
                    | _column_index_dtype_for("side", names, ["side"], mandatory=True)
                )

            case DataType.AGGREGATED_LIQUIDATIONS:
                ctor_args |= (
                    _column_index_dtype_for("avg_buy_price", names, ["avg_buy_price"], mandatory=True)
                    | _column_index_dtype_for("last_buy_price", names, ["last_buy_price"], mandatory=True)
                    | _column_index_dtype_for("buy_amount", names, ["buy_amount"], mandatory=True)
                    | _column_index_dtype_for("buy_count", names, ["buy_count"], mandatory=True, col_dtype=np.int64)
                    | _column_index_dtype_for("buy_notional", names, ["buy_notional"], mandatory=True)
                    | _column_index_dtype_for("avg_sell_price", names, ["avg_sell_price"], mandatory=True)
                    | _column_index_dtype_for("last_sell_price", names, ["last_sell_price"], mandatory=True)
                    | _column_index_dtype_for("sell_amount", names, ["sell_amount"], mandatory=True)
                    | _column_index_dtype_for("sell_count", names, ["sell_count"], mandatory=True, col_dtype=np.int64)
                    | _column_index_dtype_for("sell_notional", names, ["sell_notional"], mandatory=True)
                )

            case DataType.FUNDING_RATE:
                ctor_args |= (
                    _column_index_dtype_for("rate", names, ["rate"], mandatory=True)
                    | _column_index_dtype_for("interval", names, ["interval"], mandatory=True)
                    | _column_index_dtype_for("next_funding_time", names, ["next_funding_time"], mandatory=True)
                    | _column_index_dtype_for("mark_price", names, ["mark_price"])
                    | _column_index_dtype_for("index_price", names, ["index_price"])
                )

            case DataType.FUNDING_PAYMENT:
                ctor_args |= (
                    _column_index_dtype_for("funding_rate", names, ["funding_rate"], mandatory=True)
                    | _column_index_dtype_for("funding_interval_hours", names, ["funding_interval_hours"], mandatory=True)
                )

            case DataType.OPEN_INTEREST:
                ctor_args |= (
                    _column_index_dtype_for("symbol", names, ["symbol"], mandatory=True, col_dtype=str)
                    | _column_index_dtype_for("open_interest", names, ["open_interest"], mandatory=True)
                    | _column_index_dtype_for("open_interest_usd", names, ["open_interest_usd"], mandatory=True)
                )

            case _:
                # - for unknown types (RECORD etc), include all columns
                for i, name in enumerate(names):
                    if i != t_index:
                        ctor_args[name] = (i, np.float64)
        # fmt: on

        return ctor_args

    def process_data(self, raw_data: IRawContainer) -> list[Timestamped]:
        dtype = raw_data.dtype
        names = raw_data.names
        index = raw_data.index
        _data = raw_data.data

        scheme = self._recognize_type_ctor_scheme(dtype, names, index)

        # - special case for sequential orderbook updates
        if dtype == DataType.ORDERBOOK:
            times = convert_times_to_ns(_data.column(index).to_numpy(zero_copy_only=False), self.timestamp_units)
            levels = _data.column(scheme["level"][0]).to_numpy(zero_copy_only=False)
            prices = _data.column(scheme["price"][0]).to_numpy(zero_copy_only=False)
            sizes = _data.column(scheme["size"][0]).to_numpy(zero_copy_only=False)
            return build_snapshots(times, levels, prices, sizes)  # type: ignore

        # - extract columns as numpy arrays with proper dtype handling
        columns = {}
        for col_name, (col_idx, col_dtype) in scheme.items():
            if col_idx == index:
                # - time column: convert to nanoseconds int64
                arr = _data.column(col_idx).to_numpy(zero_copy_only=False)
                columns[col_name] = convert_times_to_ns(arr, self.timestamp_units)
            else:
                columns[col_name] = _extract_column(_data, col_idx, col_dtype)

        # - build typed objects row by row
        num_rows = _data.num_rows
        return [self._build_typed_object({k: v[i] for k, v in columns.items()}, dtype, names) for i in range(num_rows)]


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

    def process_data(self, raw_data: IRawContainer) -> GenericSeries:
        dtype = raw_data.dtype
        names = raw_data.names
        index = raw_data.index
        _data = raw_data.data

        # - infer timeframe if not provided
        timeframe = self.timeframe
        if not timeframe:
            times = _data.column(index).to_numpy(zero_copy_only=False)
            times = convert_times_to_ns(times, self.timestamp_units)
            ts = list(sorted(set(times[:1000].tolist())))
            timeframe = to_timedelta(infer_series_frequency(ts)).asm8.item()

        gens = GenericSeries(raw_data.data_id, timeframe, max_series_length=self.max_length)
        scheme = self._recognize_type_ctor_scheme(dtype, names, index)

        # - special case for sequential orderbook updates
        if dtype == DataType.ORDERBOOK:
            times = convert_times_to_ns(_data.column(index).to_numpy(zero_copy_only=False), self.timestamp_units)
            levels = _data.column(scheme["level"][0]).to_numpy(zero_copy_only=False)
            prices = _data.column(scheme["price"][0]).to_numpy(zero_copy_only=False)
            sizes = _data.column(scheme["size"][0]).to_numpy(zero_copy_only=False)
            for s in build_snapshots(times, levels, prices, sizes):
                gens.update(s)
        else:
            # - extract columns as numpy arrays with proper dtype handling
            columns = {}
            for col_name, (col_idx, col_dtype) in scheme.items():
                if col_idx == index:
                    # - time column: convert to nanoseconds int64
                    arr = _data.column(col_idx).to_numpy(zero_copy_only=False)
                    columns[col_name] = convert_times_to_ns(arr, self.timestamp_units)
                else:
                    columns[col_name] = _extract_column(_data, col_idx, col_dtype)

            # - build and add typed objects row by row
            for i in range(_data.num_rows):
                row = {k: v[i] for k, v in columns.items()}
                gens.update(self._build_typed_object(row, dtype, names))

        return gens


class ColumnarSeriesTransformer(IDataTransformer):
    """
    Transform data to ColumnarSeries with individual TimeSeries for each column.

    Unlike TypedGenericSeries which stores complete objects, ColumnarSeries decomposes
    data into separate TimeSeries for each column. This allows:
    - Direct column access via attribute (e.g., series.buy_volume)
    - Attaching indicators to individual columns
    - Proper indicator update propagation when new data arrives

    Example:
        transformer = ColumnarSeriesTransformer()
        cs = raw_data.transform(transformer)

        # Access columns as TimeSeries
        ratio = cs.taker_buy_sell_ratio  # Returns TimeSeries

        # Attach indicators
        sma = ta.Sma.wrap(cs.taker_buy_sell_ratio, 20)

        # Indicators update automatically when series is updated
    """

    def __init__(self, timeframe=None, timestamp_units="ns", max_length=np.inf) -> None:
        self.timestamp_units = timestamp_units
        self.max_length = max_length
        self.timeframe = timeframe

    def process_data(self, raw_data: IRawContainer) -> ColumnarSeries:
        names = raw_data.names
        index = raw_data.index
        _data = raw_data.data

        # Get timestamp column name and value columns
        ts_col_name = names[index]
        column_names = [n for n in names if n != ts_col_name]

        # Get times as int64 nanoseconds
        times = _data.column(index).to_numpy(zero_copy_only=False)
        times = convert_times_to_ns(times, self.timestamp_units)

        # Infer timeframe if not provided
        timeframe = self.timeframe
        if not timeframe:
            ts_list = list(sorted(set(times[:1000].tolist())))
            timeframe = to_timedelta(infer_series_frequency(ts_list)).asm8.item()

        # Create ColumnarSeries with known columns
        series = ColumnarSeries(raw_data.data_id, timeframe, column_names, max_series_length=self.max_length)

        # Extract all value columns as numpy arrays
        columns = {}
        for col in column_names:
            columns[col] = _extract_column(_data, names.index(col), np.float64)

        # Feed data row by row (triggers child TimeSeries updates)
        for i in range(len(times)):
            row = {col: float(columns[col][i]) for col in column_names}
            obj = TimestampedDict(time=times[i], data=row)
            series.update(obj)

        return series
