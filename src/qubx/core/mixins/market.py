from collections import defaultdict
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import SW, DataType, Instrument, ITimeProvider, dt_64, td_64
from qubx.core.exceptions import SymbolNotFound
from qubx.core.helpers import extract_price
from qubx.core.interfaces import IDataProvider, IMarketDataCache, IMarketManager, IUniverseManager
from qubx.core.lookups import lookup
from qubx.core.series import OHLCV, Bar, GenericSeries, OrderBook, Quote, Trade, time_as_nsec
from qubx.data.storage import IReader, IStorage
from qubx.utils.time import (
    convert_tf_str_td64,
    floor_t64,
    infer_series_frequency,
    timedelta_to_str,
    to_timedelta,
    to_timestamp,
)

from .utils import EXCHANGE_MAPPINGS

INVERSE_EXCHANGE_MAPPINGS = {mapping: exchange for exchange, mapping in EXCHANGE_MAPPINGS.items()}

# - MIN_TIMEFRAMES_GAP_TO_REQUEST_PROVIDER > 1 prevents not necessary requests to dataprovider
#   if current time is very close to bar's end
MIN_TIMEFRAMES_GAP_TO_REQUEST_PROVIDER = 1.5


class CachedMarketDataHolder(IMarketDataCache):
    """
    Collected cached data updates from market
    """

    _last_bar: dict[Instrument, Bar | None]
    _ohlcvs: dict[Instrument, dict[np.timedelta64, OHLCV]]
    _updates: dict[Instrument, Bar | Quote | Trade]
    _generic_series: dict[Instrument, dict[str, GenericSeries]]
    _max_series_length: int
    _per_type_lengths: dict[str, int]
    _logged_caps: set[str]

    # Appends and the *_snapshot reads (FitContext on the StrategyFitThread)
    # serialize on this lock so a threaded fit never sees a torn series. An
    # uncontended RLock costs the same as a no-op context manager, so it is
    # taken unconditionally — simulation just never contends on it.
    _series_lock: RLock

    def __init__(
        self,
        default_timeframe: str | None = None,
        max_buffer_size: int = 10_000,
        per_type_lengths: dict[str, int] | None = None,
    ) -> None:
        self._ohlcvs = dict()
        self._last_bar = defaultdict(lambda: None)
        self._updates = dict()
        self._generic_series = defaultdict(dict)
        self._max_series_length = max_buffer_size
        self._per_type_lengths = dict(per_type_lengths or {})
        self._logged_caps = set()
        self._series_lock = RLock()
        if default_timeframe:
            self.update_default_timeframe(default_timeframe)

    def update_default_timeframe(self, default_timeframe: str):
        self.default_timeframe = convert_tf_str_td64(default_timeframe)

    def _resolve_series_length(self, event_type: str) -> int:
        """
        Resolve the retention cap for a subscription/event type by its base dtype
        name (e.g. "orderbook(0,1)" -> "orderbook"), falling back to the default
        buffer size when the base name has no configured cap.
        """
        try:
            base = DataType.from_str(event_type)[0].value
            if base == DataType.NONE.value:
                base = event_type.split("(")[0]
        except Exception:
            base = event_type.split("(")[0]

        n = self._per_type_lengths.get(base, self._max_series_length)
        if base in self._per_type_lengths and base not in self._logged_caps:
            self._logged_caps.add(base)
            logger.info(f"market cache retention: {base}={n}")
        return n

    def _resolve_ohlc_length(self, max_size: float | int) -> float | int:
        # OHLC series default to unbounded (today's behavior, relied on by long
        # backtests/warmups reading full history) unless an explicit "ohlc" cap
        # is configured -- unlike generic series, which have always defaulted
        # to max_buffer_size. Only overrides the caller's own default sentinel.
        if max_size != np.inf:
            return max_size
        if DataType.OHLC.value in self._per_type_lengths:
            return self._resolve_series_length(DataType.OHLC.value)
        return max_size

    def init_ohlcv(self, instrument: Instrument, max_size=np.inf):
        if instrument not in self._ohlcvs:
            _length = self._resolve_ohlc_length(max_size)
            self._ohlcvs[instrument] = {
                self.default_timeframe: OHLCV(instrument.symbol, self.default_timeframe, _length),
            }
        # - Clear updates to prevent stale data when re-adding instruments
        self._updates.pop(instrument, None)
        self._last_bar.pop(instrument, None)

    def remove(self, instrument: Instrument) -> None:
        with self._series_lock:
            self._ohlcvs.pop(instrument, None)
            self._last_bar.pop(instrument, None)
            self._updates.pop(instrument, None)
            self._generic_series.pop(instrument, None)

    def set_state_from(self, other: "IMarketDataCache", instruments: list[Instrument] | None = None) -> None:
        """
        Set the internal state of this CachedMarketDataHolder to the state of another instance.

        WARNING: This is a shallow copy of the internal state dictionaries.

        Args:
            other: Another IMarketDataCache instance to copy state from
            instruments: If provided, only transfer state for these instruments
        """
        if not isinstance(other, CachedMarketDataHolder):
            raise TypeError(f"Expected CachedMarketDataHolder, got {type(other).__name__}")

        self.default_timeframe = other.default_timeframe

        if instruments is not None:
            # - only transfer state for specified instruments
            _instrument_set = set(instruments)
            self._ohlcvs = {k: v for k, v in other._ohlcvs.items() if k in _instrument_set}
            self._updates = {k: v for k, v in other._updates.items() if k in _instrument_set}
            self._generic_series = defaultdict(
                dict,
                {k: v for k, v in other._generic_series.items() if k in _instrument_set},
            )
        else:
            self._ohlcvs = other._ohlcvs
            self._updates = other._updates
            self._generic_series = other._generic_series

        self._last_bar = defaultdict(lambda: None)  # - reset the last bar

    @SW.watch("CachedMarketDataHolder")
    def get_ohlcv(
        self,
        instrument: Instrument,
        timeframe: str | td_64 | None = None,
        max_size: float | int = np.inf,
    ) -> OHLCV:
        return self._get_ohlcv_series(instrument, timeframe, max_size)

    def get_ohlcv_snapshot(
        self,
        instrument: Instrument,
        timeframe: str | td_64 | None = None,
        max_size: float | int = np.inf,
    ) -> OHLCV:
        """FitContext entrypoint (StrategyFitThread): build/read the series under the
        lock and hand out a clone, so the caller's iteration can't race ProcessorThread
        appends. The clone has no indicators attached — do not attach live indicators
        to it."""
        with self._series_lock:
            return self._get_ohlcv_series(instrument, timeframe, max_size).clone()

    def _get_ohlcv_series(
        self, instrument: Instrument, timeframe: str | td_64 | None = None, max_size: float | int = np.inf
    ) -> OHLCV:
        with self._series_lock:
            return self._get_ohlcv_series_unlocked(instrument, timeframe, max_size)

    def _get_ohlcv_series_unlocked(
        self, instrument: Instrument, timeframe: str | td_64 | None = None, max_size: float | int = np.inf
    ) -> OHLCV:
        # Locked wrapper above: this can lazily CREATE a series (dict insert + resample
        # from the live basis), and it is reachable from the fit thread via internal
        # real-ctx reads (e.g. get_min_size → quote → cache fallback) — an unlocked
        # create would race the ProcessorThread's locked iteration of the same dicts.
        if timeframe is None:
            tf = self.default_timeframe
        elif isinstance(timeframe, str):
            tf = convert_tf_str_td64(timeframe)
        else:  # td_64
            tf = timeframe

        if instrument not in self._ohlcvs:
            self._ohlcvs[instrument] = {}

        if tf not in self._ohlcvs[instrument]:
            # - check requested timeframe
            _length = self._resolve_ohlc_length(max_size)
            new_ohlc = OHLCV(instrument.symbol, tf, _length)
            if tf < self.default_timeframe:
                logger.warning(
                    f"[{instrument.symbol}] Request for timeframe {timeframe} that is smaller then minimal {self.default_timeframe}"
                )
            else:
                # - first try to resample from smaller frame
                if basis := self._ohlcvs[instrument].get(self.default_timeframe):
                    for b in basis[::-1]:
                        new_ohlc.update_by_bar(
                            b.time,
                            b.open,
                            b.high,
                            b.low,
                            b.close,
                            b.volume,
                            b.bought_volume,
                            b.volume_quote,
                            b.bought_volume_quote,
                            b.trade_count,
                        )

            self._ohlcvs[instrument][tf] = new_ohlc

        return self._ohlcvs[instrument][tf]

    def get_data(self, instrument: Instrument, event_type: str) -> GenericSeries:
        """
        Get (or lazily create) a GenericSeries for the given instrument and event type.

        The series stores every incoming update individually (tick resolution, 1ns timeframe),
        so all events are preserved in arrival order. Attached IndicatorGeneric instances
        receive an update call for every single event, enabling streaming computations
        without manual accumulation code in on_market_data.

        Args:
            instrument: The instrument to get data for
            event_type: The subscription/event type (e.g. DataType.TRADE, DataType.QUOTE)

        Returns:
            GenericSeries updated on every incoming event for this instrument/type
        """
        _instr_series = self._generic_series[instrument]
        if event_type not in _instr_series:
            # - use 1 (= 1ns in internal nanosecond units) so every timestamped
            #   event becomes its own item with no timeframe bucketing
            _instr_series[event_type] = GenericSeries(
                f"{instrument.symbol}.{event_type}",
                1,
                self._resolve_series_length(event_type),
            )
        return _instr_series[event_type]

    def get_data_snapshot(self, instrument: Instrument, event_type: str) -> GenericSeries:
        """
        FitContext entrypoint (StrategyFitThread): build/read the series under the
        lock and hand out a clone, so the caller's iteration can't race ProcessorThread
        appends. The clone has no indicators attached — do not attach live indicators
        to it.
        """
        with self._series_lock:
            return self.get_data(instrument, event_type).clone()

    def update(self, instrument: Instrument, event_type: str, data: Any, update_ohlc: bool = False) -> None:
        # - update GenericSeries for non-OHLC data (supports indicator attachment)
        if event_type != DataType.OHLC:
            # - same lock as the OHLCV paths: get_data can lazily insert, and the append
            #   must not tear a concurrent get_data_snapshot on the fit thread
            with self._series_lock:
                _series = self.get_data(instrument, event_type)
                if not (_series.times and time_as_nsec(data.time) < _series.times[0]):
                    _series.update(data)

        if not update_ohlc:
            return

        match event_type:
            case DataType.OHLC:
                self.update_by_bar(instrument, data)
            case DataType.QUOTE:
                self.update_by_quote(instrument, data)
            case DataType.TRADE:
                self.update_by_trade(instrument, data)
            case DataType.ORDERBOOK:
                assert isinstance(data, OrderBook)
                self.update_by_quote(instrument, data.to_quote())
            case _:
                pass

    @SW.watch("CachedMarketDataHolder")
    def update_by_bars(self, instrument: Instrument, timeframe: str | td_64, bars: list[Bar]) -> OHLCV:
        """
        Update or create OHLCV series with the provided historical bars.

        This method:
        1. Creates a new OHLCV series if one doesn't exist for the instrument/timeframe
        2. Updates an existing OHLCV series with the new bars using the OHLCV.update_by_bars method
           which handles:
           - Adding older bars to the back of the series
           - Skipping bars that are already present
           - Adding newer bars to the front
        """
        with self._series_lock:
            return self._update_by_bars(instrument, timeframe, bars)

    def _update_by_bars(self, instrument: Instrument, timeframe: str | td_64, bars: list[Bar]) -> OHLCV:
        if instrument not in self._ohlcvs:
            self._ohlcvs[instrument] = {}

        tf = convert_tf_str_td64(timeframe) if isinstance(timeframe, str) else timeframe

        # Get existing OHLCV or create a new one
        if tf in self._ohlcvs[instrument]:
            ohlc = self._ohlcvs[instrument][tf]
            # Update the existing OHLCV with the new bars
            ohlc.update_by_bars(bars)
        else:
            # Create a new OHLCV and add the bars
            ohlc = OHLCV(instrument.symbol, tf)
            ohlc.update_by_bars(bars)
            self._ohlcvs[instrument][tf] = ohlc

        # - update last update for this instrument
        if bars:
            self._updates[instrument] = bars[-1]  # - use the last bar as the last update
            # - sync _last_bar to the newest fetched bar so the guard in update_by_bar
            # - correctly skips stale WebSocket updates that arrive after a historical refetch
            # - advanced the series beyond the current _last_bar position
            _cur_last = self._last_bar[instrument]
            _newest = bars[-1]
            if _cur_last is None or _newest.time > _cur_last.time:
                self._last_bar[instrument] = _newest

        return ohlc

    @SW.watch("CachedMarketDataHolder")
    def update_by_bar(self, instrument: Instrument, bar: Bar):
        with self._series_lock:
            self._update_by_bar(instrument, bar)

    def _update_by_bar(self, instrument: Instrument, bar: Bar):
        self._updates[instrument] = bar

        _last_bar = self._last_bar[instrument]
        v_tot_inc = bar.volume
        v_buy_inc = bar.bought_volume
        v_quote_inc = bar.volume_quote
        v_quote_buy_inc = bar.bought_volume_quote
        v_trade_count_inc = bar.trade_count

        if _last_bar is not None:
            if _last_bar.time == bar.time:  # just current bar updated
                v_tot_inc -= _last_bar.volume
                v_buy_inc -= _last_bar.bought_volume
                v_quote_inc -= _last_bar.volume_quote
                v_quote_buy_inc -= _last_bar.bought_volume_quote
                v_trade_count_inc -= _last_bar.trade_count

            if _last_bar.time > bar.time:  # update is too late - skip it
                return

        if instrument in self._ohlcvs:
            self._last_bar[instrument] = bar
            for ser in self._ohlcvs[instrument].values():
                ser.update_by_bar(
                    bar.time,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    v_tot_inc,
                    v_buy_inc,
                    v_quote_inc,
                    v_quote_buy_inc,
                    v_trade_count_inc,
                )

    @SW.watch("CachedMarketDataHolder")
    def update_by_quote(self, instrument: Instrument, quote: Quote):
        with self._series_lock:
            self._updates[instrument] = quote
            series = self._ohlcvs.get(instrument)
            if series:
                for ser in series.values():
                    if len(ser) > 0 and ser[0].time > quote.time:
                        continue
                    ser.update(quote.time, quote.mid_price(), 0)

    @SW.watch("CachedMarketDataHolder")
    def update_by_trade(self, instrument: Instrument, trade: Trade):
        with self._series_lock:
            self._update_by_trade(instrument, trade)

    def _update_by_trade(self, instrument: Instrument, trade: Trade):
        self._updates[instrument] = trade
        series = self._ohlcvs.get(instrument)
        if series:
            total_vol = trade.size
            bought_vol = total_vol if trade.side == 1 else 0.0
            volume_quote = trade.price * trade.size
            bought_volume_quote = volume_quote if trade.side == 1 else 0.0
            for ser in series.values():
                if len(ser) > 0:
                    current_bar_start = floor_t64(np.datetime64(ser[0].time, "ns"), np.timedelta64(ser.timeframe, "ns"))
                    trade_bar_start = floor_t64(np.datetime64(trade.time, "ns"), np.timedelta64(ser.timeframe, "ns"))
                    if trade_bar_start < current_bar_start:
                        # Trade belongs to a previous bar - skip it
                        continue
                ser.update(
                    trade.time,
                    trade.price,
                    volume=total_vol,
                    bvolume=bought_vol,
                    volume_quote=volume_quote,
                    bought_volume_quote=bought_volume_quote,
                    trade_count=1,
                )

    def finalize_ohlc_for_instruments(self, time: dt_64, instruments: list[Instrument]):
        """
        Finalize all OHLCV series at the given time for the given instruments.
        FIXME: (2025-06-17) This is part of urgent live fix and must be removed in future !!!.
        """
        for instrument in instruments:
            # - use most recent update
            if (_u := self._updates.get(instrument)) is not None:
                _px = extract_price(_u)

                # Floor the timestamp to the bar start time for each timeframe
                # This ensures proper consolidation in the cached data holder
                if instrument in self._ohlcvs:
                    for timeframe_ns, _ in self._ohlcvs[instrument].items():
                        # Convert timeframe_ns to timedelta64[ns] and use datetime64 for floor_t64
                        timeframe_td = np.timedelta64(timeframe_ns, "ns")
                        floored_time = floor_t64(time, timeframe_td)
                        floored_time_ns = time_as_nsec(floored_time)
                        self.update_by_bar(
                            instrument, Bar(floored_time_ns, _px, _px, _px, _px, volume=0, bought_volume=0)
                        )


class MarketManager(IMarketManager):
    _time_provider: ITimeProvider
    _cache: CachedMarketDataHolder
    _data_providers: list[IDataProvider]
    _universe_manager: IUniverseManager
    _aux_data_storage: IStorage
    _exchange_to_data_provider: dict[str, IDataProvider]
    _aux_readers: dict[tuple[str, str], IReader]

    def __init__(
        self,
        time_provider: ITimeProvider,
        data_providers: list[IDataProvider],
        universe_manager: IUniverseManager,
        aux_data_storage: IStorage,
        max_buffer_size: int = 10_000,
        per_type_lengths: dict[str, int] | None = None,
    ):
        self._time_provider = time_provider
        self._cache = CachedMarketDataHolder(max_buffer_size=max_buffer_size, per_type_lengths=per_type_lengths)
        self._data_providers = data_providers
        self._universe_manager = universe_manager
        self._aux_data_storage = aux_data_storage
        self._exchange_to_data_provider = {data_provider.exchange(): data_provider for data_provider in data_providers}
        self._aux_readers = dict()

    def get_market_data_cache(self) -> IMarketDataCache:
        return self._cache

    def time(self) -> dt_64:
        return self._time_provider.time()

    def ohlc(
        self,
        instrument: Instrument,
        timeframe: str | td_64 | None = None,
        length: int | None = None,
    ) -> OHLCV:
        return self._ohlc(instrument, timeframe, length, snapshot=False)

    def _ohlc(
        self,
        instrument: Instrument,
        timeframe: str | td_64 | None,
        length: int | None,
        snapshot: bool,
    ) -> OHLCV:
        # snapshot=True (FitContext on the StrategyFitThread): all cache touches go
        # through the holder's locked snapshot paths and the returned series is a clone.
        if timeframe is None:
            timeframe = timedelta_to_str(self._cache.default_timeframe)
        elif isinstance(timeframe, td_64):
            timeframe = timedelta_to_str(timeframe)
        elif isinstance(timeframe, (int, np.int64)):  # type: ignore
            timeframe = timedelta_to_str(timeframe)

        rc = (
            self._cache.get_ohlcv_snapshot(instrument, timeframe)
            if snapshot
            else self._cache.get_ohlcv(instrument, timeframe)
        )
        _data_provider = self._get_data_provider(instrument.exchange)

        # - check if we need to fetch more data
        # TODO: - we need to review strategy when we can request data from provider !
        # - we could do it only when requested bars bigger than we have now
        # - if we see gap in recent data - it's probably issue in realtime data feeds etc
        _need_history_request = False
        if (_l_rc := len(rc)) > 0:
            _last_bar_time = rc[0].time

            # - temporary fix:
            _min_delta_ns = MIN_TIMEFRAMES_GAP_TO_REQUEST_PROVIDER * to_timedelta(timeframe).asm8.item()
            _time_now = _data_provider.time_provider.time().item()

            # - if need to do request
            if (_time_now - _last_bar_time > _min_delta_ns) or (length and _l_rc < length):
                _need_history_request = True

        else:
            _need_history_request = True

        # - send request for historical data
        if _need_history_request and length is not None:
            bars = _data_provider.get_ohlc(instrument, timeframe, length)
            if snapshot:
                # - merge into the PRIVATE clone: the fit thread never writes shared
                #   series content, so ProcessorThread-side live reads can't tear against
                #   a fit-time fetch. Instruments the fit actually adds get their history
                #   into the shared cache through the subscription warmup at the commit.
                rc.update_by_bars(bars)
            else:
                rc = self._cache.update_by_bars(instrument, timeframe, bars)
        return rc

    def ohlc_pd(
        self,
        instrument: Instrument,
        timeframe: str | td_64 | None = None,
        length: int | None = None,
        consolidated: bool = True,
    ) -> pd.DataFrame:
        return self._ohlc_pd(instrument, timeframe, length, consolidated, snapshot=False)

    def _ohlc_pd(
        self,
        instrument: Instrument,
        timeframe: str | td_64 | None,
        length: int | None,
        consolidated: bool,
        snapshot: bool,
    ) -> pd.DataFrame:
        if snapshot:
            series = self._ohlc(instrument, timeframe, length, snapshot=True)
        else:
            # - route via the public ohlc so overrides/mocks of it stay effective
            series = self.ohlc(instrument, timeframe, length)
        # Pass length directly to pd() - this avoids creating full DataFrame first
        ohlc = series.pd(length=length)

        if consolidated and not timeframe:
            timeframe = infer_series_frequency(ohlc[:20])

        if consolidated and timeframe:
            _time = to_timestamp(self._time_provider.time())
            _timedelta = to_timedelta(timeframe)
            if len(ohlc) > 0:  # Check if DataFrame is not empty
                _last_bar_time = ohlc.index[-1]
                if _last_bar_time + _timedelta > _time:
                    ohlc = ohlc.iloc[:-1]

        # No more redundant tail() operation needed since length was already applied
        return ohlc

    def quote(self, instrument: Instrument) -> Quote | None:
        return self._quote(instrument, snapshot=False)

    def _quote(self, instrument: Instrument, snapshot: bool) -> Quote | None:
        # snapshot=True: provider quotes are plain reads; only the no-quote OHLC
        # fallback touches the cache, via the locked snapshot path.
        _data_provider = self._get_data_provider(instrument.exchange)
        quote = _data_provider.get_quote(instrument)
        if quote is None:
            ohlcv = self._cache.get_ohlcv_snapshot(instrument) if snapshot else self._cache.get_ohlcv(instrument)
            if len(ohlcv) > 0:
                last_bar = ohlcv[0]
                quote = Quote(
                    last_bar.time,
                    last_bar.close - instrument.tick_size / 2,
                    last_bar.close + instrument.tick_size / 2,
                    0,
                    0,
                )
        return quote

    def get_cached_market_data(self, instrument: Instrument, sub_type: str) -> GenericSeries:
        return self._get_cached_market_data(instrument, sub_type, snapshot=False)

    def _get_cached_market_data(self, instrument: Instrument, sub_type: str, snapshot: bool) -> GenericSeries:
        # snapshot=True (FitContext on the StrategyFitThread): locked clone, detached from
        # the series the ProcessorThread appends to. No history fetch on either path —
        # unlike _ohlc, the non-snapshot read is a plain cache read too.
        return (
            self._cache.get_data_snapshot(instrument, sub_type)
            if snapshot
            else self._cache.get_data(instrument, sub_type)
        )

    def get_aux_reader(self, exchange: str, mtype: str) -> IReader:
        _rd_key = (exchange.upper(), mtype.upper())
        if _rd_key not in self._aux_readers:
            self._aux_readers[_rd_key] = self._aux_data_storage.get_reader(exchange, mtype)
        return self._aux_readers[_rd_key]

    def get_aux_data_storage(self) -> IStorage:
        return self._aux_data_storage

    def get_instruments(self) -> list[Instrument]:
        return self._universe_manager.instruments

    def query_instrument(self, symbol: str, exchange: str | None = None) -> Instrument:
        _e, _mt, _s = Instrument.parse_notation(symbol)

        # - use parsed exchange or fallback to provided/default
        if _e is not None:
            exchange = _e
        if exchange is None:
            exchange = self.exchanges()[0]

        instrument = lookup.find_symbol(exchange, _s, market_type=_mt)
        if instrument is None:
            if exchange in INVERSE_EXCHANGE_MAPPINGS:
                instrument = lookup.find_symbol(INVERSE_EXCHANGE_MAPPINGS[exchange], _s, market_type=_mt)
            if instrument is None:
                raise SymbolNotFound(f"Symbol not found: {_s} on {exchange}")
        return instrument

    def exchanges(self) -> list[str]:
        """
        What exchanges are supported by the market manager.
        Theoretically it can manage multiple exchanges.
        """
        return list(self._exchange_to_data_provider.keys())

    def update_base_subscription(self, sub_type: str):
        """
        Set base subscription for market data
        """
        _, params = DataType.from_str(sub_type)
        __default_timeframe = params.get("timeframe", "1sec")
        self._cache.update_default_timeframe(__default_timeframe)

    def _get_data_provider(self, exchange: str) -> IDataProvider:
        if exchange in self._exchange_to_data_provider:
            return self._exchange_to_data_provider[exchange]
        if exchange in EXCHANGE_MAPPINGS and EXCHANGE_MAPPINGS[exchange] in self._exchange_to_data_provider:
            return self._exchange_to_data_provider[EXCHANGE_MAPPINGS[exchange]]
        raise ValueError(f"Data provider for exchange {exchange} not found")

    def is_instrument_listed(self, instrument: Instrument) -> bool:
        try:
            dp = self._get_data_provider(instrument.exchange)
        except ValueError:
            return True  # no provider => can't tell => fail-open
        return dp.is_instrument_listed(instrument)
