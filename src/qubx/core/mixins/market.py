from typing import Any

import pandas as pd

from qubx.core.basics import Instrument, ITimeProvider, dt_64
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
from qubx.utils.time import timedelta_to_str


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
        timeframe: str | None = None,
        length: int | None = None,
    ) -> OHLCV:
        timeframe = timeframe or timedelta_to_str(self._cache.default_timeframe)
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
