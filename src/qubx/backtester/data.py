from collections import defaultdict

import pandas as pd

from qubx import logger
from qubx.backtester.simulated_data import IterableSimulationData
from qubx.core.basics import (
    CtrlChannel,
    DataType,
    Instrument,
    TimestampedDict,
)
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IDataProvider
from qubx.core.series import Bar, Quote, time_as_nsec
from qubx.data.readers import AsDict, DataReader
from qubx.utils.time import infer_series_frequency

from .account import SimulatedAccountProcessor
from .utils import SimulatedTimeProvider


class SimulatedDataProvider(IDataProvider):
    time_provider: SimulatedTimeProvider
    channel: CtrlChannel

    _scheduler: BasicScheduler
    _account: SimulatedAccountProcessor
    _last_quotes: dict[Instrument, Quote | None]
    _readers: dict[str, DataReader]
    _data_source: IterableSimulationData
    _open_close_time_indent_ns: int

    def __init__(
        self,
        exchange_id: str,
        channel: CtrlChannel,
        scheduler: BasicScheduler,
        time_provider: SimulatedTimeProvider,
        account: SimulatedAccountProcessor,
        readers: dict[str, DataReader],
        data_source: IterableSimulationData,
        open_close_time_indent_secs=1,
    ):
        self.channel = channel
        self.time_provider = time_provider
        self._exchange_id = exchange_id
        self._scheduler = scheduler
        self._account = account
        self._readers = readers

        # - simulation data source
        self._data_source = data_source
        self._open_close_time_indent_ns = open_close_time_indent_secs * 1_000_000_000  # convert seconds to nanoseconds

        # - create exchange's instance
        self._last_quotes = defaultdict(lambda: None)

        logger.info(f"{self.__class__.__name__}.{exchange_id} is initialized")

    @property
    def is_simulation(self) -> bool:
        return True

    def subscribe(self, subscription_type: str, instruments: set[Instrument], reset: bool) -> None:
        _new_instr = [i for i in instruments if not self.has_subscription(i, subscription_type)]
        self._data_source.add_instruments_for_subscription(subscription_type, list(instruments))

        # - provide historical data and last quote for subscribed instruments
        for i in _new_instr:
            # Check if the instrument was actually subscribed (not filtered out)
            if not self.has_subscription(i, subscription_type):
                continue
                
            h_data = self._data_source.peek_historical_data(i, subscription_type)
            if h_data:
                # _s_type = DataType.from_str(subscription_type)[0]
                last_update = h_data[-1]
                if last_quote := self._account._exchange.emulate_quote_from_data(i, last_update.time, last_update):  # type: ignore
                    # - send historical data to the channel
                    self.channel.send((i, subscription_type, h_data, True))

                    # - set last quote
                    self._last_quotes[i] = last_quote

                    # - also need to pass this quote to OME !
                    self._account.process_market_data(last_quote.time, i, last_quote)  # type: ignore

                    logger.debug(f" | subscribed {subscription_type} {i} -> {last_quote}")

    def unsubscribe(self, subscription_type: str, instruments: set[Instrument] | Instrument | None = None) -> None:
        # logger.debug(f" | unsubscribe: {subscription_type} -> {instruments}")
        if instruments is not None:
            self._data_source.remove_instruments_from_subscription(
                subscription_type, [instruments] if isinstance(instruments, Instrument) else list(instruments)
            )

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        return self._data_source.has_subscription(instrument, subscription_type)

    def get_subscriptions(self, instrument: Instrument) -> list[str]:
        _s_lst = self._data_source.get_subscriptions_for_instrument(instrument)
        # logger.debug(f" | get_subscriptions {instrument} -> {_s_lst}")
        return _s_lst

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
        _in_lst = self._data_source.get_instruments_for_subscription(subscription_type or DataType.ALL)
        # logger.debug(f" | get_subscribed_instruments {subscription_type} -> {_in_lst}")
        return _in_lst

    def warmup(self, configs: dict[tuple[str, Instrument], str]) -> None:
        for si, warm_period in configs.items():
            logger.debug(f" | Warming up {si} -> {warm_period}")
            self._data_source.set_warmup_period(si[0], warm_period)

    def get_ohlc(self, instrument: Instrument, timeframe: str, nbarsback: int) -> list[Bar]:
        _reader = self._readers.get(DataType.OHLC)
        if _reader is None:
            logger.error(f"Reader for {DataType.OHLC} data not configured")
            return []

        start = pd.Timestamp(self.time_provider.time())
        end = start - nbarsback * (_timeframe := pd.Timedelta(timeframe))
        _spec = f"{instrument.exchange}:{instrument.symbol}"
        return self._convert_records_to_bars(
            _reader.read(data_id=_spec, start=start, stop=end, timeframe=timeframe, transform=AsDict()),  # type: ignore
            time_as_nsec(self.time_provider.time()),
            _timeframe.asm8.item(),
        )

    def get_quote(self, instrument: Instrument) -> Quote | None:
        return self._last_quotes[instrument]

    def close(self):
        pass

    def _convert_records_to_bars(
        self, records: list[TimestampedDict], cut_time_ns: int, timeframe_ns: int
    ) -> list[Bar]:
        """
        Convert records to bars and we need to cut last bar up to the cut_time_ns
        """
        bars = []

        # - if no records, return empty list to avoid exception from infer_series_frequency
        if not records or records is None:
            return bars

        if len(records) > 1:
            _data_tf = infer_series_frequency([r.time for r in records[:50]])
            timeframe_ns = _data_tf.item()

        for r in records:
            _b_ts_0 = r.time
            _b_ts_1 = _b_ts_0 + timeframe_ns - self._open_close_time_indent_ns

            if _b_ts_0 <= cut_time_ns and cut_time_ns < _b_ts_1:
                break

            bars.append(
                Bar(
                    _b_ts_0,
                    r.data["open"],
                    r.data["high"],
                    r.data["low"],
                    r.data["close"],
                    r.data.get("volume", 0),
                    r.data.get("bought_volume", 0),
                )
            )

        return bars

    def exchange(self) -> str:
        return self._exchange_id.upper()
