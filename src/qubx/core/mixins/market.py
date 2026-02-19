from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import SW, DataType, Instrument, ITimeProvider, dt_64, td_64
from qubx.core.exceptions import SymbolNotFound
from qubx.core.helpers import extract_price
from qubx.core.interfaces import IDataProvider, IMarketDataCache, IMarketManager, IUniverseManager
from qubx.core.lookups import lookup
from qubx.core.series import OHLCV, Bar, OrderBook, Quote, Trade, time_as_nsec
from qubx.data.storage import IReader, IStorage
from qubx.utils.time import (
    convert_tf_str_td64,
    floor_t64,
    infer_series_frequency,
    timedelta_to_str,
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

    _instr_to_sub_to_buffer: dict[Instrument, dict[str, deque]]

    def __init__(self, default_timeframe: str | None = None, max_buffer_size: int = 10_000) -> None:
        self._ohlcvs = dict()
        self._last_bar = defaultdict(lambda: None)
        self._updates = dict()
        self._instr_to_sub_to_buffer = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_buffer_size)))
        if default_timeframe:
            self.update_default_timeframe(default_timeframe)

    def update_default_timeframe(self, default_timeframe: str):
        self.default_timeframe = convert_tf_str_td64(default_timeframe)

    def init_ohlcv(self, instrument: Instrument, max_size=np.inf):
        if instrument not in self._ohlcvs:
            self._ohlcvs[instrument] = {
                self.default_timeframe: OHLCV(instrument.symbol, self.default_timeframe, max_size),
            }
        # - Clear updates to prevent stale data when re-adding instruments
        self._updates.pop(instrument, None)
        self._last_bar.pop(instrument, None)

    def remove(self, instrument: Instrument) -> None:
        self._ohlcvs.pop(instrument, None)
        self._last_bar.pop(instrument, None)
        self._updates.pop(instrument, None)
        self._instr_to_sub_to_buffer.pop(instrument, None)

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
            self._instr_to_sub_to_buffer = defaultdict(
                lambda: defaultdict(lambda: deque(maxlen=10_000)),
                {k: v for k, v in other._instr_to_sub_to_buffer.items() if k in _instrument_set},
            )
        else:
            self._ohlcvs = other._ohlcvs
            self._updates = other._updates
            self._instr_to_sub_to_buffer = other._instr_to_sub_to_buffer

        self._last_bar = defaultdict(lambda: None)  # - reset the last bar

    @SW.watch("CachedMarketDataHolder")
    def get_ohlcv(
        self, instrument: Instrument, timeframe: str | td_64 | None = None, max_size: float | int = np.inf
    ) -> OHLCV:
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
            new_ohlc = OHLCV(instrument.symbol, tf, max_size)
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

    def get_data(self, instrument: Instrument, event_type: str) -> list[Any]:
        return list(self._instr_to_sub_to_buffer[instrument][event_type])

    def update(self, instrument: Instrument, event_type: str, data: Any, update_ohlc: bool = False) -> None:
        # - store data in buffer if it's not OHLC
        if event_type != DataType.OHLC:
            self._instr_to_sub_to_buffer[instrument][event_type].append(data)

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

        # Update the last update for this instrument
        if bars:
            self._updates[instrument] = bars[-1]  # Use the last bar as the last update

        return ohlc

    @SW.watch("CachedMarketDataHolder")
    def update_by_bar(self, instrument: Instrument, bar: Bar):
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
        self._updates[instrument] = quote
        series = self._ohlcvs.get(instrument)
        if series:
            for ser in series.values():
                if len(ser) > 0 and ser[0].time > quote.time:
                    continue
                ser.update(quote.time, quote.mid_price(), 0)

    @SW.watch("CachedMarketDataHolder")
    def update_by_trade(self, instrument: Instrument, trade: Trade):
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
    ):
        self._time_provider = time_provider
        self._cache = CachedMarketDataHolder()
        self._data_providers = data_providers
        self._universe_manager = universe_manager
        self._aux_data_storage = aux_data_storage
        self._exchange_to_data_provider = {data_provider.exchange(): data_provider for data_provider in data_providers}
        self._aux_readers = dict()

    def get_market_data_cache(self) -> IMarketDataCache:
        return self._cache

    def time(self) -> dt_64:
        return self._time_provider.time()

    def ohlc(self, instrument: Instrument, timeframe: str | td_64 | None = None, length: int | None = None) -> OHLCV:
        if timeframe is None:
            timeframe = timedelta_to_str(self._cache.default_timeframe)
        elif isinstance(timeframe, td_64):
            timeframe = timedelta_to_str(timeframe)
        elif isinstance(timeframe, (int, np.int64)):  # type: ignore
            timeframe = timedelta_to_str(timeframe)

        rc = self._cache.get_ohlcv(instrument, timeframe)
        _data_provider = self._get_data_provider(instrument.exchange)

        # - check if we need to fetch more data
        # TODO: - we need to review strategy when we can request data from provider !
        # - we could do it only when requested bars bigger than we have now
        # - if we see gap in recent data - it's probably issue in realtime data feeds etc
        _need_history_request = False
        if (_l_rc := len(rc)) > 0:
            _last_bar_time = rc[0].time

            # - temporary fix:
            _min_delta_ns = MIN_TIMEFRAMES_GAP_TO_REQUEST_PROVIDER * pd.Timedelta(timeframe).asm8.item()
            _time_now = _data_provider.time_provider.time().item()

            # - if need to do request
            if (_time_now - _last_bar_time > _min_delta_ns) or (length and _l_rc < length):
                _need_history_request = True

        else:
            _need_history_request = True

        # - send request for historical data
        if _need_history_request and length is not None:
            bars = _data_provider.get_ohlc(instrument, timeframe, length)
            rc = self._cache.update_by_bars(instrument, timeframe, bars)
        return rc

    def ohlc_pd(
        self,
        instrument: Instrument,
        timeframe: str | td_64 | None = None,
        length: int | None = None,
        consolidated: bool = True,
    ) -> pd.DataFrame:
        # Pass length directly to pd() - this avoids creating full DataFrame first
        ohlc = self.ohlc(instrument, timeframe, length).pd(length=length)

        if consolidated and not timeframe:
            timeframe = infer_series_frequency(ohlc[:20])

        if consolidated and timeframe:
            _time = pd.Timestamp(self._time_provider.time())
            _timedelta = pd.Timedelta(timeframe)
            if len(ohlc) > 0:  # Check if DataFrame is not empty
                _last_bar_time = ohlc.index[-1]
                if _last_bar_time + _timedelta > _time:
                    ohlc = ohlc.iloc[:-1]

        # No more redundant tail() operation needed since length was already applied
        return ohlc

    def quote(self, instrument: Instrument) -> Quote | None:
        _data_provider = self._get_data_provider(instrument.exchange)
        quote = _data_provider.get_quote(instrument)
        if quote is None:
            ohlcv = self._cache.get_ohlcv(instrument)
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

    def get_cached_market_data(self, instrument: Instrument, sub_type: str) -> list[Any]:
        return self._cache.get_data(instrument, sub_type)

    def get_aux_reader(self, exchange: str, mtype: str) -> IReader:
        _rd_key = (exchange.upper(), mtype.upper())
        if _rd_key not in self._aux_readers:
            self._aux_readers[_rd_key] = self._aux_data_storage.get_reader(exchange, mtype)
        return self._aux_readers[_rd_key]

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
