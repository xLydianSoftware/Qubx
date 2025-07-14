from typing import Any

import numpy as np
import pandas as pd

from qubx.core.basics import Instrument, ITimeProvider, dt_64, td_64
from qubx.core.exceptions import SymbolNotFound
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.interfaces import (
    IDataProvider,
    IMarketManager,
    IUniverseManager,
)
from qubx.core.lookups import lookup
from qubx.core.series import OHLCV, Quote
from qubx.data.readers import DataReader
from qubx.utils.time import infer_series_frequency, timedelta_to_str


class MarketManager(IMarketManager):
    _time_provider: ITimeProvider
    _cache: CachedMarketDataHolder
    _data_providers: list[IDataProvider]
    _universe_manager: IUniverseManager
    _aux_data_provider: DataReader | None
    _exchange_to_data_provider: dict[str, IDataProvider]

    def __init__(
        self,
        time_provider: ITimeProvider,
        cache: CachedMarketDataHolder,
        data_providers: list[IDataProvider],
        universe_manager: IUniverseManager,
        aux_data_provider: DataReader | None = None,
    ):
        self._time_provider = time_provider
        self._cache = cache
        self._data_providers = data_providers
        self._universe_manager = universe_manager
        self._aux_data_provider = aux_data_provider
        self._exchange_to_data_provider = {data_provider.exchange(): data_provider for data_provider in data_providers}

    def time(self) -> dt_64:
        return self._time_provider.time()

    def ohlc(
        self,
        instrument: Instrument,
        timeframe: str | td_64 | None = None,
        length: int | None = None,
    ) -> OHLCV:
        if timeframe is None:
            timeframe = timedelta_to_str(self._cache.default_timeframe)
        elif isinstance(timeframe, td_64):
            timeframe = timedelta_to_str(timeframe)
        elif isinstance(timeframe, (int, np.int64)):  # type: ignore
            timeframe = timedelta_to_str(timeframe)

        rc = self._cache.get_ohlcv(instrument, timeframe)
        _data_provider = self._exchange_to_data_provider[instrument.exchange]

        # - check if we need to fetch more data
        _need_history_request = False
        if (_l_rc := len(rc)) > 0:
            _last_bar_time = rc[0].time
            _timeframe_ns = pd.Timedelta(timeframe).asm8.item()

            # - check if we need to fetch more data
            if (_last_bar_time + _timeframe_ns < _data_provider.time_provider.time().item()) or (
                length and _l_rc < length
            ):
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
        _data_provider = self._exchange_to_data_provider[instrument.exchange]
        return _data_provider.get_quote(instrument)

    def get_data(self, instrument: Instrument, sub_type: str) -> list[Any]:
        return self._cache.get_data(instrument, sub_type)

    def get_aux_data(self, data_id: str, **parameters) -> pd.DataFrame | None:
        return self._aux_data_provider.get_aux_data(data_id, **parameters) if self._aux_data_provider else None

    def get_instruments(self) -> list[Instrument]:
        return self._universe_manager.instruments

    def query_instrument(self, symbol: str, exchange: str | None = None) -> Instrument:
        if exchange is None:
            parts = symbol.split(":")
            if len(parts) == 2:
                exchange, symbol = parts
            else:
                exchange = self.exchanges()[0]

        instrument = lookup.find_symbol(exchange, symbol)
        if instrument is None:
            raise SymbolNotFound(f"Symbol not found: {symbol} on {exchange}")
        return instrument

    def exchanges(self) -> list[str]:
        """
        What exchanges are supported by the market manager.
        Theoretically it can manage multiple exchanges.
        """
        return list(self._exchange_to_data_provider.keys())
