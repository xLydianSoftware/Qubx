from typing import Any, Iterator

import pandas as pd

from qubx import logger
from qubx.backtester.sentinels import NoDataContinue
from qubx.core.basics import DataType, Instrument, MarketType, Timestamped
from qubx.core.exceptions import SimulationError
from qubx.data.composite import IteratedDataStreamsSlicer
from qubx.data.readers import (
    AsDict,
    AsFundingPayments,
    AsOrderBook,
    AsQuotes,
    AsTrades,
    DataReader,
    DataTransformer,
    RestoredBarsFromOHLC,
    RestoreQuotesFromOHLC,
    RestoreTradesFromOHLC,
)


class DataFetcher:
    _fetcher_id: str
    _reader: DataReader
    _requested_data_type: str
    _producing_data_type: str
    _params: dict[str, object]
    _specs: list[str]

    _transformer: DataTransformer
    _timeframe: str | None = None
    _warmup_period: pd.Timedelta | None = None
    _chunksize: int = 5000

    def __init__(
        self,
        fetcher_id: str,
        reader: DataReader,
        subtype: str,
        params: dict[str, Any],
        warmup_period: pd.Timedelta | None = None,
        chunksize: int = 5000,
        open_close_time_indent_secs=1.0,  # open/close shift may depends on simulation
    ) -> None:
        self._fetcher_id = fetcher_id
        self._params = params
        self._reader = reader

        match subtype:
            case DataType.OHLC_QUOTES:
                # - requested restore quotes from OHLC
                self._transformer = RestoreQuotesFromOHLC(open_close_time_shift_secs=open_close_time_indent_secs)
                self._requested_data_type = "ohlc"
                self._producing_data_type = "quote"
                if "timeframe" in params:
                    self._timeframe = params.get("timeframe", "1Min")

            case DataType.OHLC_TRADES:
                # - requested restore trades from OHLC
                self._transformer = RestoreTradesFromOHLC(open_close_time_shift_secs=open_close_time_indent_secs)
                self._requested_data_type = "ohlc"
                self._producing_data_type = "trade"
                if "timeframe" in params:
                    self._timeframe = params.get("timeframe", "1Min")

            case DataType.OHLC:
                # - requested restore bars from OHLC
                self._transformer = RestoredBarsFromOHLC(open_close_time_shift_secs=open_close_time_indent_secs)
                self._requested_data_type = "ohlc"
                self._producing_data_type = "ohlc"
                if "timeframe" in params:
                    self._timeframe = params.get("timeframe", "1Min")

            case DataType.TRADE:
                self._requested_data_type = "trade"
                self._producing_data_type = "trade"
                self._transformer = AsTrades()

            case DataType.QUOTE:
                # self._requested_data_type = "orderbook"
                self._requested_data_type = "quote"
                self._producing_data_type = "quote"  # ???
                self._transformer = AsQuotes()

            case DataType.ORDERBOOK:
                self._requested_data_type = "orderbook"
                self._producing_data_type = "orderbook"
                self._transformer = AsOrderBook()

            case DataType.FUNDING_PAYMENT:
                self._requested_data_type = "funding_payment"
                self._producing_data_type = "funding_payment"
                self._transformer = AsFundingPayments()

            case _:
                self._requested_data_type = subtype
                self._producing_data_type = subtype
                self._transformer = AsDict()

        self._warmup_period = warmup_period
        self._warmed = {}
        self._specs = []
        self._chunksize = chunksize

    @staticmethod
    def _make_request_id(instrument: Instrument) -> str:
        return f"{instrument.exchange}:{instrument.symbol}"

    def attach_instrument(self, instrument: Instrument) -> str:
        _data_id = self._make_request_id(instrument)

        if _data_id not in self._specs:
            self._specs.append(_data_id)
            self._warmed[_data_id] = False

        return self._fetcher_id + "." + _data_id

    def remove_instrument(self, instrument: Instrument) -> str:
        _data_id = self._make_request_id(instrument)

        if _data_id in self._specs:
            self._specs.remove(_data_id)
            del self._warmed[_data_id]

        return self._fetcher_id + "." + _data_id

    def has_instrument(self, instrument: Instrument) -> bool:
        return self._make_request_id(instrument) in self._specs

    def get_instruments_indices(self) -> list[str]:
        return [self._fetcher_id + "." + i for i in self._specs]

    def get_instrument_index(self, instrument: Instrument) -> str:
        return self._fetcher_id + "." + self._make_request_id(instrument)

    def load(
        self, start: str | pd.Timestamp, end: str | pd.Timestamp, to_load: list[Instrument] | None
    ) -> dict[str, Iterator]:
        """
        Loads data for specified instruments within a given time range.

        Parameters:
            - start (str | pd.Timestamp): The start time for data loading, can be a string or a pandas Timestamp.
            - end (str | pd.Timestamp): The end time for data loading, can be a string or a pandas Timestamp.
            - to_load (list[Instrument] | None): A list of instruments to load data for. If None, data for all subscribed instruments is loaded.

        Returns:
            - dict[str, Iterator]: A dictionary where keys are instrument identifiers and values are iterators over the loaded data.
        """
        _requests = self._specs if not to_load else set(self._make_request_id(i) for i in to_load)
        _r_iters = {}

        for _r in _requests:  # - TODO: replace this loop with multi-instrument request after DataReader refactoring
            if _r in self._specs:
                _start = pd.Timestamp(start)
                if self._warmup_period and not self._warmed.get(_r):
                    _start -= self._warmup_period
                    self._warmed[_r] = True

                _args = dict(
                    data_id=_r,
                    start=_start,
                    stop=end,
                    transform=self._transformer,
                    data_type=self._requested_data_type,
                    chunksize=self._chunksize,
                )

                if self._timeframe:
                    _args["timeframe"] = self._timeframe

                # get arguments from self._reader.read
                _reader_args = set(self._reader.read.__code__.co_varnames[: self._reader.read.__code__.co_argcount])
                # match _args with self._params
                _params = {k: v for k, v in self._params.items() if k in _reader_args}
                _args = {**_args, **_params}

                try:
                    _r_iters[self._fetcher_id + "." + _r] = self._reader.read(**_args)  # type: ignore
                except Exception as e:
                    logger.error(f">>> (DataFetcher::load) - failed to load <g>'{self._fetcher_id}'</g> data: {e}")
            else:
                raise IndexError(
                    f"Instrument {_r} is not subscribed for this data {self._requested_data_type} in {self._fetcher_id} !"
                )

        return _r_iters

    def __repr__(self) -> str:
        return f"{self._requested_data_type}({self._params}) (-{self._warmup_period if self._warmup_period else '--'}) [{','.join(self._specs)}] :-> {self._transformer.__class__.__name__}"


class IterableSimulationData(Iterator):
    """
    This class is a crucial component for backtesting system.
    It provides a flexible and efficient way to simulate market data feeds for strategy testing.

    Key Features:
        - Supports multiple data types (OHLC, trades, quotes) and instruments.
        - Allows for dynamic addition and removal of instruments during simulation.
        - Handles warmup periods for data preloading.
        - Manages historical and current data distinction during iteration.
        - Utilizes a data slicer (IteratedDataStreamsSlicer) to merge and order data from multiple sources.

    TODO:
        1. think how to provide initial "market quote" for each instrument
        2. optimization for historical data (return bunch of history instead of each bar in next(...))
    """

    _readers: dict[str, DataReader]
    _subtyped_fetchers: dict[str, DataFetcher]
    _warmups: dict[str, pd.Timedelta]
    _instruments: dict[str, tuple[Instrument, DataFetcher, DataType]]
    _open_close_time_indent_secs: int | float

    _slicer_ctrl: IteratedDataStreamsSlicer | None = None
    _slicing_iterator: Iterator | None = None
    _start: pd.Timestamp | None = None
    _stop: pd.Timestamp | None = None
    _current_time: int | None = None

    def __init__(
        self,
        readers: dict[str, DataReader],
        open_close_time_indent_secs=1,  # open/close ticks shift
    ):
        self._readers = dict(readers)
        self._instruments = {}
        self._subtyped_fetchers = {}
        self._warmups = {}
        self._open_close_time_indent_secs = open_close_time_indent_secs

    def set_typed_reader(self, data_type: str, reader: DataReader):
        self._readers[data_type] = reader
        if _fetcher := self._subtyped_fetchers.get(data_type):
            _fetcher._reader = reader

    def set_warmup_period(self, subscription: str, warmup_period: str | None = None):
        if warmup_period:
            _access_key, _, _ = self._parse_subscription_spec(subscription)
            self._warmups[_access_key] = pd.Timedelta(warmup_period)

    def _parse_subscription_spec(self, subscription: str) -> tuple[str, str, dict[str, object]]:
        _subtype, _params = DataType.from_str(subscription)
        match _subtype:
            case DataType.OHLC | DataType.OHLC_QUOTES:
                _timeframe = _params.get("timeframe", "1Min")
                _access_key = f"{_subtype}.{_timeframe}"
            case DataType.TRADE | DataType.QUOTE | DataType.ORDERBOOK:
                _access_key = f"{_subtype}"
            case _:
                # - any arbitrary data type is passed as is
                _params = {}
                _subtype = subscription
                _access_key = f"{_subtype}"
        return _access_key, _subtype, _params

    def _filter_instruments_for_subscription(self, data_type: str, instruments: list[Instrument]) -> list[Instrument]:
        """
        Filter instruments based on subscription type requirements.

        For funding payment subscriptions, only SWAP instruments are supported since
        funding payments are specific to perpetual swap contracts.

        Args:
            data_type: The data type being subscribed to
            instruments: List of instruments to filter

        Returns:
            Filtered list of instruments appropriate for the subscription type
        """
        # Only funding payments require special filtering
        if data_type == DataType.FUNDING_PAYMENT:
            original_count = len(instruments)
            filtered_instruments = [i for i in instruments if i.market_type == MarketType.SWAP]
            filtered_count = len(filtered_instruments)

            # Log if instruments were filtered out (debug info)
            if filtered_count < original_count:
                logger.debug(
                    f"Filtered {original_count - filtered_count} non-SWAP instruments from funding payment subscription"
                )

            return filtered_instruments

        # For all other subscription types, return instruments unchanged
        return instruments

    def add_instruments_for_subscription(self, subscription: str, instruments: list[Instrument] | Instrument):
        instruments = instruments if isinstance(instruments, list) else [instruments]
        _subt_key, _data_type, _params = self._parse_subscription_spec(subscription)

        # Filter instruments based on subscription type requirements
        instruments = self._filter_instruments_for_subscription(_data_type, instruments)

        # If no instruments remain after filtering, skip subscription
        if not instruments:
            return

        fetcher = self._subtyped_fetchers.get(_subt_key)
        if not fetcher:
            _reader = self._readers.get(_data_type)

            if _reader is None:
                raise SimulationError(f"No reader configured for data type: {_data_type}")

            self._subtyped_fetchers[_subt_key] = (
                fetcher := DataFetcher(
                    _subt_key,
                    _reader,
                    _data_type,
                    _params,
                    warmup_period=self._warmups.get(_subt_key),
                    open_close_time_indent_secs=self._open_close_time_indent_secs,
                )
            )

        _instrs_to_preload = []
        for i in instruments:
            if not fetcher.has_instrument(i):
                idx = fetcher.attach_instrument(i)
                self._instruments[idx] = (i, fetcher, subscription)  # type: ignore
                _instrs_to_preload.append(i)

        if self.is_running and _instrs_to_preload:
            self._slicer_ctrl += fetcher.load(
                pd.Timestamp(self._current_time, unit="ns"),  # type: ignore
                self._stop,  # type: ignore
                _instrs_to_preload,
            )

    def peek_historical_data(self, instrument: Instrument, subscription: str) -> list[Timestamped]:
        """
        Retrieves historical data for a specified instrument and subscription type up to the current simulation time.

        Parameters:
            - instrument (Instrument): instrument for which historical data is requested.
            - subscription (str): type of data subscription (e.g., OHLC, trades, quotes) for the instrument.

        Returns:
            - list[Timestamped]: A list of historical data elements for the specified instrument and subscription type
            that occurred before the current simulation time. If the simulation is not running, returns an empty list.

        Raises:
            SimulationError: If the instrument does not have the specified subscription in the simulation data provider.
        """
        if not self.has_subscription(instrument, subscription):
            raise SimulationError(
                f"Instrument: {instrument} has no subscription: {subscription} in this simulation data provider"
            )

        if not self.is_running:
            return []

        _subt_key, _, _ = self._parse_subscription_spec(subscription)
        _i_key = self._subtyped_fetchers[_subt_key].get_instrument_index(instrument)

        assert self._slicer_ctrl is not None and self._current_time is not None

        # fetch historical data for current time
        return self._slicer_ctrl.fetch_before_time(_i_key, self._current_time)

    def get_instruments_for_subscription(self, subscription: str) -> list[Instrument]:
        if subscription == DataType.ALL:
            return list((i for i, *_ in self._instruments.values()))

        _subt_key, _, _ = self._parse_subscription_spec(subscription)
        if (fetcher := self._subtyped_fetchers.get(_subt_key)) is not None:
            return [self._instruments[k][0] for k in fetcher.get_instruments_indices()]

        return []

    def get_subscriptions_for_instrument(self, instrument: Instrument | None) -> list[str]:
        r = []
        for i, f, s in self._instruments.values():
            if instrument is not None:
                if i == instrument:
                    r.append(s)
            else:
                r.append(s)
        return list(set(r))

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        for i, f, s in self._instruments.values():
            if i == instrument and s == subscription_type:
                return True
        return False

    def remove_instruments_from_subscription(self, subscription: str, instruments: list[Instrument] | Instrument):
        def _remove_from_fetcher(_subt_key: str, instruments: list[Instrument]):
            fetcher = self._subtyped_fetchers.get(_subt_key)
            if not fetcher:
                logger.warning(f"No configured data fetcher for '{_subt_key}' subscription !")
                return

            _keys_to_remove = []
            for i in instruments:
                # - try to remove from data fetcher
                if idx := fetcher.remove_instrument(i):
                    if idx in self._instruments:
                        self._instruments.pop(idx)
                        _keys_to_remove.append(idx)

            # print("REMOVING FROM:", _keys_to_remove)
            if self.is_running and _keys_to_remove:
                self._slicer_ctrl.remove(_keys_to_remove)  # type: ignore

        instruments = instruments if isinstance(instruments, list) else [instruments]

        # - if we want to remove instruments from all subscriptions
        if subscription == DataType.ALL:
            _f_keys = list(self._subtyped_fetchers.keys())
            for s in _f_keys:
                _remove_from_fetcher(s, instruments)
            return

        _subt_key, _, _ = self._parse_subscription_spec(subscription)
        _remove_from_fetcher(_subt_key, instruments)

    @property
    def is_running(self) -> bool:
        return self._current_time is not None

    def create_iterable(self, start: str | pd.Timestamp, stop: str | pd.Timestamp) -> Iterator:
        self._start = pd.Timestamp(start)
        self._stop = pd.Timestamp(stop)
        self._current_time = None
        self._slicer_ctrl = IteratedDataStreamsSlicer()
        return self

    def __iter__(self) -> Iterator:
        assert self._start is not None
        self._current_time = int(pd.Timestamp(self._start).timestamp() * 1e9)
        _ct_timestap = pd.Timestamp(self._current_time, unit="ns")

        for f in self._subtyped_fetchers.values():
            logger.debug(
                f"  [<c>IteratedDataStreamsSlicer</c>] :: Preloading initial data for {f._fetcher_id} {self._start} : {self._stop} ..."
            )
            self._slicer_ctrl += f.load(_ct_timestap, self._stop, None)  # type: ignore

        self._slicing_iterator = iter(self._slicer_ctrl)
        return self

    def __next__(self) -> tuple[Instrument, str, Timestamped, bool]:  # type: ignore
        try:
            while data := next(self._slicing_iterator):  # type: ignore
                k, t, v = data

                # Check if we've reached or exceeded the stop time
                # It's commented out because we expect data readers to stop on their own
                # if self._stop is not None and t > self._stop.value:
                #     raise StopIteration
                
                # Handle NoDataContinue sentinel
                if isinstance(v, NoDataContinue):
                    # Return the sentinel as the event - the runner will detect it with isinstance
                    return None, "", v, False

                instr, fetcher, subt = self._instruments[k]
                data_type = fetcher._producing_data_type
                _is_historical = False
                if t < self._current_time:  # type: ignore
                    _is_historical = True
                else:
                    # only update the current time if the event is not historical
                    self._current_time = t

                return instr, data_type, v, _is_historical
        except StopIteration as e:  # noqa: F841
            raise StopIteration
