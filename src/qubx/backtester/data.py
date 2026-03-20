from collections import defaultdict
from typing import TypeVar

import pandas as pd

from qubx import logger
from qubx.backtester.simulated_data import SimulatedDataIterator
from qubx.core.basics import (
    CtrlChannel,
    DataType,
    Instrument,
)
from qubx.core.interfaces import IDataProvider
from qubx.core.series import Bar, Quote

from .account import SimulatedAccountProcessor
from .utils import SimulatedTimeProvider

T = TypeVar("T")


class SimulatedDataProvider(IDataProvider):
    """
    Per-exchange data provider for backtesting simulation.

    Thin wrapper around SimulatedDataIterator (shared across all exchanges in a simulation).
    Handles exchange-specific concerns: subscribe/unsubscribe lifecycle, last-quote tracking,
    and account notifications (OME). One instance per exchange in the simulation.

    Data flow for market data subscriptions:
        subscribe() → SimulatedDataIterator.add_instruments_for_subscription()
                     → peek historical data → emulate last quote → notify account OME

    Data flow for historical OHLC lookback (ctx.ohlc()):
        get_ohlc(instrument, timeframe, nbarsback)
          → uses time_provider.time() as current simulated time
          → delegates to SimulatedDataIterator.get_ohlc() which reads from IStorage,
            transforms via TypedRecords, and applies open_close_time_indent cut
            to exclude bars that haven't "closed" yet at the simulated time
    """

    time_provider: SimulatedTimeProvider
    channel: CtrlChannel

    _account: SimulatedAccountProcessor
    _last_quotes: dict[Instrument, Quote | None]
    _data_source: SimulatedDataIterator

    def __init__(
        self,
        exchange_id: str,
        channel: CtrlChannel,
        time_provider: SimulatedTimeProvider,
        account: SimulatedAccountProcessor,
        data_source: SimulatedDataIterator,
    ):
        self.channel = channel
        self.time_provider = time_provider
        self._exchange_id = exchange_id
        self._account = account

        # - simulation data source
        self._data_source = data_source

        # - create last quote holder
        self._last_quotes = defaultdict(lambda: None)

        logger.info(f"{self.__class__.__name__}.{exchange_id} is initialized")

    @property
    def is_simulation(self) -> bool:
        return True

    def is_connected(self) -> bool:
        """
        Check if the data provider is currently connected to the exchange.

        For simulated data provider, always returns True since data is loaded from files.

        Returns:
            bool: Always True for simulated data
        """
        return True

    def subscribe(self, subscription_type: str, instruments: set[Instrument], reset: bool) -> None:
        _new_instr = [i for i in instruments if not self.has_subscription(i, subscription_type)]

        self._data_source.add_instruments_for_subscription(subscription_type, list(instruments))

        # - provide historical data and last quote for subscribed instruments
        for i in _new_instr:
            # - check if the instrument was actually subscribed (not filtered out)
            if not self.has_subscription(i, subscription_type):
                continue

            # - notify simulating exchange that instrument is subscribed
            self._account._exchange.on_subscribe(i)

            # - we need to clear last quote as it can be staled
            self._last_quotes.pop(i, None)

            # - try to peek most recent market data
            h_data = self._data_source.peek_historical_data(i, subscription_type)
            if h_data:
                self.channel.send((i, subscription_type, h_data, True))

                last_update = h_data[-1]
                if last_quote := self._account._exchange.emulate_quote_from_data(i, last_update.time, last_update):  # type: ignore
                    self._last_quotes[i] = last_quote
                    self._account.process_market_data(last_quote.time, i, last_quote)  # type: ignore

                logger.debug(f" | subscribed {subscription_type} {i} -> {len(h_data)} records")

    def unsubscribe(self, subscription_type: str, instruments: set[Instrument] | Instrument | None = None) -> None:
        # logger.debug(f" | unsubscribe: {subscription_type} -> {instruments}")
        if instruments is not None:
            _instruments = [instruments] if isinstance(instruments, Instrument) else list(instruments)
            self._data_source.remove_instruments_from_subscription(subscription_type, _instruments)

            # - Clear last quotes for unsubscribed instruments
            for instr in _instruments:
                # - clear last quote
                self._last_quotes.pop(instr, None)

                # - Notify simulating exchange that instrument is unsubscribed
                self._account._exchange.on_unsubscribe(instr)

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
        start = pd.Timestamp(self.time_provider.time())
        end = start - nbarsback * pd.Timedelta(timeframe)
        return self._data_source.get_ohlc(instrument, timeframe, start, end)

    def get_quote(self, instrument: Instrument) -> Quote | None:
        return self._last_quotes[instrument]

    def close(self):
        pass

    def exchange(self) -> str:
        return self._exchange_id.upper()
