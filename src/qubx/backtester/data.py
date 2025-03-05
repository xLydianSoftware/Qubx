from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from qubx import logger
from qubx.backtester.simulated_data import IterableSimulationData
from qubx.core.basics import (
    CtrlChannel,
    DataType,
    Instrument,
    TimestampedDict,
)
from qubx.core.exceptions import SimulationError
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
    _last_quotes: Dict[Instrument, Optional[Quote]]
    _readers: dict[str, DataReader]
    _scheduler: BasicScheduler
    _pregenerated_signals: dict[Instrument, pd.Series | pd.DataFrame]
    _to_process: dict[Instrument, list]
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
        open_close_time_indent_secs=1,
    ):
        self.channel = channel
        self.time_provider = time_provider
        self._exchange_id = exchange_id
        self._scheduler = scheduler
        self._account = account
        self._readers = readers

        # - create exchange's instance
        self._last_quotes = defaultdict(lambda: None)

        # - pregenerated signals storage
        self._pregenerated_signals = dict()
        self._to_process = {}

        # - simulation data source
        self._data_source = IterableSimulationData(
            self._readers, open_close_time_indent_secs=open_close_time_indent_secs
        )
        self._open_close_time_indent_ns = open_close_time_indent_secs * 1_000_000_000  # convert seconds to nanoseconds

        logger.info(f"{self.__class__.__name__}.{exchange_id} is initialized")

    def run(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        silent: bool = False,
    ) -> None:
        logger.info(f"{self.__class__.__name__} ::: Simulation started at {start} :::")

        if self._pregenerated_signals:
            self._prepare_generated_signals(start, end)
            _run = self._process_generated_signals
        else:
            _run = self._process_strategy

        start, end = pd.Timestamp(start), pd.Timestamp(end)
        total_duration = end - start
        update_delta = total_duration / 100
        prev_dt = pd.Timestamp(start)

        # - date iteration
        qiter = self._data_source.create_iterable(start, end)
        if silent:
            for instrument, data_type, event, is_hist in qiter:
                if not _run(instrument, data_type, event, is_hist):
                    break
        else:
            _p = 0
            with tqdm(total=100, desc="Simulating", unit="%", leave=False) as pbar:
                for instrument, data_type, event, is_hist in qiter:
                    if not _run(instrument, data_type, event, is_hist):
                        break
                    dt = pd.Timestamp(event.time)
                    # update only if date has changed
                    if dt - prev_dt > update_delta:
                        _p += 1
                        pbar.n = _p
                        pbar.refresh()
                        prev_dt = dt
                pbar.n = 100
                pbar.refresh()

        logger.info(f"{self.__class__.__name__} ::: Simulation finished at {end} :::")

    def set_generated_signals(self, signals: pd.Series | pd.DataFrame):
        logger.debug(
            f"[<y>{self.__class__.__name__}</y>] :: Using pre-generated signals:\n {str(signals.count()).strip('ndtype: int64')}"
        )
        # - sanity check
        signals.index = pd.DatetimeIndex(signals.index)

        if isinstance(signals, pd.Series):
            self._pregenerated_signals[str(signals.name)] = signals  # type: ignore

        elif isinstance(signals, pd.DataFrame):
            for col in signals.columns:
                self._pregenerated_signals[col] = signals[col]  # type: ignore
        else:
            raise ValueError("Invalid signals or strategy configuration")

    @property
    def is_simulation(self) -> bool:
        return True

    def subscribe(self, subscription_type: str, instruments: set[Instrument], reset: bool) -> None:
        _new_instr = [i for i in instruments if not self.has_subscription(i, subscription_type)]
        self._data_source.add_instruments_for_subscription(subscription_type, list(instruments))

        # - provide historical data and last quote for subscribed instruments
        for i in _new_instr:
            h_data = self._data_source.peek_historical_data(i, subscription_type)
            if h_data:
                # _s_type = DataType.from_str(subscription_type)[0]
                last_update = h_data[-1]
                if last_quote := self._account.emulate_quote_from_data(i, last_update.time, last_update):  # type: ignore
                    # - send historical data to the channel
                    self.channel.send((i, subscription_type, h_data, True))

                    # - set last quote
                    self._last_quotes[i] = last_quote

                    # - also need to pass this quote to OME !
                    self._account._process_new_data(i, last_quote)

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
            _reader.read(data_id=_spec, start=start, stop=end, transform=AsDict()),  # type: ignore
            time_as_nsec(self.time_provider.time()),
            _timeframe.asm8.item(),
        )

    def get_quote(self, instrument: Instrument) -> Quote | None:
        return self._last_quotes[instrument]

    def close(self):
        pass

    def _prepare_generated_signals(self, start: str | pd.Timestamp, end: str | pd.Timestamp):
        for s, v in self._pregenerated_signals.items():
            _s_inst = None

            for i in self.get_subscribed_instruments():
                # - we can process series with variable id's if we can find some similar instrument
                if s == i.symbol or s == str(i) or s == f"{i.exchange}:{i.symbol}" or str(s) == str(i):
                    _start, _end = pd.Timestamp(start), pd.Timestamp(end)
                    _start_idx, _end_idx = v.index.get_indexer([_start, _end], method="ffill")
                    sel = v.iloc[max(_start_idx, 0) : _end_idx + 1]

                    # TODO: check if data has exec_price - it means we have deals
                    self._to_process[i] = list(zip(sel.index, sel.values))
                    _s_inst = i
                    break

            if _s_inst is None:
                logger.error(f"Can't find instrument for pregenerated signals with id '{s}'")
                raise SimulationError(f"Can't find instrument for pregenerated signals with id '{s}'")

    def _convert_records_to_bars(
        self, records: list[TimestampedDict], cut_time_ns: int, timeframe_ns: int
    ) -> list[Bar]:
        """
        Convert records to bars and we need to cut last bar up to the cut_time_ns
        """
        bars = []

        # - if no records, return empty list to avoid exception from infer_series_frequency
        if not records:
            return bars

        _data_tf = infer_series_frequency([r.time for r in records[:50]])
        timeframe_ns = _data_tf.item()

        if records is not None:
            for r in records:
                # _b_ts_0 = np.datetime64(r.time, "ns").item()
                _b_ts_0 = r.time
                _b_ts_1 = _b_ts_0 + timeframe_ns - self._open_close_time_indent_ns

                if _b_ts_0 <= cut_time_ns and cut_time_ns < _b_ts_1:
                    break

                bars.append(
                    Bar(
                        _b_ts_0, r.data["open"], r.data["high"], r.data["low"], r.data["close"], r.data.get("volume", 0)
                    )
                )

        return bars

    def _process_generated_signals(self, instrument: Instrument, data_type: str, data: Any, is_hist: bool) -> bool:
        cc = self.channel
        t = np.datetime64(data.time, "ns")

        if not is_hist:
            # - signals for this instrument
            sigs = self._to_process[instrument]

            while sigs and t >= (_signal_time := sigs[0][0].as_unit("ns").asm8):
                self.time_provider.set_time(_signal_time)
                cc.send((instrument, "event", {"order": sigs[0][1]}, False))
                sigs.pop(0)

            if q := self._account.emulate_quote_from_data(instrument, t, data):
                self._last_quotes[instrument] = q

        self.time_provider.set_time(t)
        cc.send((instrument, data_type, data, is_hist))

        return cc.control.is_set()

    def _process_strategy(self, instrument: Instrument, data_type: str, data: Any, is_hist: bool) -> bool:
        cc = self.channel
        t = np.datetime64(data.time, "ns")

        if not is_hist:
            if t >= (_next_exp_time := self._scheduler.next_expected_event_time()):
                # - we use exact event's time
                self.time_provider.set_time(_next_exp_time)
                self._scheduler.check_and_run_tasks()

            if q := self._account.emulate_quote_from_data(instrument, t, data):
                self._last_quotes[instrument] = q

        self.time_provider.set_time(t)
        cc.send((instrument, data_type, data, is_hist))

        return cc.control.is_set()

    def exchange(self) -> str:
        return self._exchange_id.upper()
