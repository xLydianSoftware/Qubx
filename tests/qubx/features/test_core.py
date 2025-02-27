import gzip
import json
from functools import partial

import numpy as np
import pandas as pd
import pytest
import pytest_mock

from qubx.connectors.ccxt.utils import ccxt_convert_orderbook
from qubx.core.basics import Instrument
from qubx.core.interfaces import IBroker, IMarketManager, IStrategyContext, TriggerEvent
from qubx.core.lookups import lookup
from qubx.core.series import OHLCV, Bar, Quote, Trade
from qubx.data import AsOhlcvSeries, DataReader, loader
from qubx.data.helpers import InMemoryCachedReader, TimeGuardedWrapper
from qubx.features import (
    AtrFeatureProvider,
    FeatureManager,
    OrderbookImbalance,
    TradeVolumeImbalance,
)
from qubx.ta.indicators import atr

ns_to_dt_64 = lambda ns: np.datetime64(ns, "ns")


class WrappedTestingDataProvider(IMarketManager):
    """
    Wrapped data reader for testing purposes.
    Primary goal is to be used for quick test of portfolio models functionality in notebook
    """

    reader: DataReader
    _actual_time: str = "now"

    def __init__(self, reader: InMemoryCachedReader):
        self.reader = TimeGuardedWrapper(reader, self)

    def ohlc(self, instrument: Instrument, timeframe: str, length: int | None = None) -> OHLCV:
        if not length:
            return self.reader.read(
                f"{instrument.exchange}:{instrument.symbol}",
                timeframe=timeframe,
                transform=AsOhlcvSeries(),
            )  # type: ignore
        return self.reader.read(  # type: ignore
            instrument.symbol if isinstance(instrument, Instrument) else instrument,
            start=self.time() - length * pd.Timedelta(timeframe),  # type: ignore
            stop=self.time(),  # type: ignore
            transform=AsOhlcvSeries(timeframe=timeframe),
        )

    def quote(self, symbol: str) -> Quote | None:
        return None

    def travel_in_time(self, past_time: str) -> None:
        self._actual_time = past_time or "now"

    def time(self) -> np.datetime64:
        return pd.Timestamp(self._actual_time).asm8

    def get_aux_data(self, data_id: str, **parametes) -> pd.DataFrame | None:
        return self.reader.get_aux_data(data_id, **parametes)

    def get_instruments(self) -> list[Instrument]:
        return [x for s in self.reader.get_symbols("", "") if (x := lookup.find_symbol(*s.split(":"))) is not None]

    def get_instrument(self, symbol: str) -> Instrument | None:
        return lookup.find_symbol(*symbol.split(":"))


def __to_time_ns(time: pd.Timestamp):
    return time.as_unit("ns").asm8


class TestStrategyContext(IStrategyContext):
    """Mock implementation of IStrategyContext for testing"""

    def __init__(self):
        self.loader = None
        self.broker_provider = None
        self._subscriptions = {}
        self._instruments = []

    def set_start(self, start: pd.Timestamp | str):
        if not hasattr(self, "loader") or self.loader is None:
            return
        self.loader._start = pd.Timestamp(start)

    def set_stop(self, stop: pd.Timestamp | str):
        if not hasattr(self, "loader") or self.loader is None:
            return
        self.loader._stop = pd.Timestamp(stop)

    def get_instrument(self, symbol: str) -> Instrument | None:
        """Get instrument by symbol"""
        exchange, symbol = symbol.split(":")
        return lookup.find_symbol(exchange, symbol)

    def get_instruments(self) -> list[Instrument]:
        """Get all instruments"""
        return self._instruments

    def subscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None) -> None:
        """Subscribe to market data"""
        if isinstance(instruments, Instrument):
            instruments = [instruments]
        if instruments is None:
            instruments = self._instruments

        for instrument in instruments:
            if instrument not in self._subscriptions:
                self._subscriptions[instrument] = set()
            self._subscriptions[instrument].add(subscription_type)

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """Check if subscription exists"""
        return instrument in self._subscriptions and subscription_type in self._subscriptions[instrument]

    def ohlc(self, instrument: Instrument, timeframe: str, length: int | None = None) -> OHLCV:
        """Get OHLCV data"""
        # This will be mocked in tests, but we need to return a valid OHLCV object
        # to satisfy the type checker
        from qubx.core.series import OHLCV

        return OHLCV(name=f"{instrument.symbol}", timeframe=timeframe)

    # Implement other required methods with minimal functionality
    def time(self) -> np.datetime64:
        """Get current time"""
        return pd.Timestamp("now").asm8

    def start(self, blocking: bool = False):
        """Start strategy context"""
        pass

    def stop(self):
        """Stop strategy context"""
        pass

    def is_running(self) -> bool:
        """Check if running"""
        return True

    @property
    def is_simulation(self) -> bool:
        """Check if simulation"""
        return True

    @property
    def exchanges(self) -> list[str]:
        """Get exchanges"""
        return ["BINANCE.UM"]


@pytest.fixture
def test_broker(mocker: pytest_mock.MockFixture) -> IBroker:
    mock = mocker.Mock(IBroker)
    mock.get_ohlc = mocker.Mock(
        return_value=[
            Bar(
                time=__to_time_ns(pd.Timestamp("2021-01-01 00:00:00")),
                open=1.0,
                high=2.0,
                low=0.5,
                close=1.5,
                volume=1000.0,
            ),
            Bar(
                time=__to_time_ns(pd.Timestamp("2021-01-01 01:00:00")),
                open=1.5,
                high=2.5,
                low=1.0,
                close=2.0,
                volume=1500.0,
            ),
        ]
    )
    return mock


@pytest.fixture(scope="module")
def test_loader() -> DataReader:
    return loader(
        "BINANCE.UM",
        "1h",
        source="csv::tests/data/csv",
        fundamental_data=pd.read_csv("tests/data/csv/fundamental_data.csv.gz", parse_dates=["timestamp"]).set_index(
            ["timestamp", "symbol"]
        ),
        n_jobs=1,
    )


@pytest.fixture(scope="function")
def ctx(
    test_loader: InMemoryCachedReader,
    test_broker: IBroker,
    mocker: pytest_mock.MockFixture,
) -> IStrategyContext:
    mkt = WrappedTestingDataProvider(test_loader)
    ctx = mocker.Mock(TestStrategyContext)

    # Create a mock OHLCV object with at least 2 data points
    from qubx.core.series import OHLCV, TimeSeries

    mock_ohlcv = OHLCV(name="BTCUSDT", timeframe="1h")
    # Add two data points
    mock_ohlcv.update_by_bar(
        time=int(__to_time_ns(pd.Timestamp("2021-01-01 00:00:00"))),
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        vol_incr=1000.0,
    )
    mock_ohlcv.update_by_bar(
        time=int(__to_time_ns(pd.Timestamp("2021-01-01 01:00:00"))),
        open=1.5,
        high=2.5,
        low=1.0,
        close=2.0,
        vol_incr=1500.0,
    )

    # Mock the ohlc method to return our mock OHLCV object
    ctx.ohlc = mocker.Mock(return_value=mock_ohlcv)

    ctx.get_instrument.side_effect = mkt.get_instrument
    ctx.broker_provider = test_broker
    ctx.loader = test_loader
    ctx.set_start.side_effect = partial(TestStrategyContext.set_start, ctx)
    ctx.set_stop.side_effect = partial(TestStrategyContext.set_stop, ctx)
    return ctx


class TestFeaturesCore:
    def test_feature_manager(
        self,
        ctx: TestStrategyContext,
    ):
        # - initialize strategy context
        ctx.set_start("2021-01-01")
        ctx.set_stop("2021-03-01")

        s1 = ctx.get_instrument("BINANCE.UM:BTCUSDT")
        assert s1 is not None

        ctx.get_instruments.return_value = [s1]

        # - initialize feature manager
        manager = FeatureManager(reader=ctx.loader)
        _atr_params = dict(period=14, smoother="sma", percentage=True)
        manager += AtrFeatureProvider(timeframe="1h", **_atr_params)

        # - initialize the start of the strategy
        manager.on_start(ctx)

        # - check the ATR feature
        _atr = manager["BTCUSDT", "ATR(14,1h,sma,pct=True)"]
        _expected_atr = atr(ctx.ohlc(s1, "1h"), **_atr_params)  # type: ignore
        assert _atr == _expected_atr

        # - check all features
        _feats = manager["BTCUSDT", "ATR"]
        assert "ATR(14,1h,sma,pct=True)" in _feats

        _feats = manager["BTCUSDT"]
        assert "ATR(14,1h,sma,pct=True)" in _feats

    def test_trade_subscription(
        self,
        ctx: TestStrategyContext,
    ):
        trade_data = pd.read_csv("tests/data/csv/BTCUSDT.trades.csv.gz")
        trades = [
            Trade(
                time=row.time,
                price=row.price,
                size=row.size,
                taker=row.side,  # type: ignore
                trade_id=row.Index,  # type: ignore
            )
            for row in trade_data.itertuples()
        ]

        # - initialize strategy context
        ctx.set_start("2021-01-01")
        ctx.set_stop("2021-03-01")

        s1 = ctx.get_instrument("BINANCE.UM:BTCUSDT")
        assert s1 is not None

        ctx.get_instruments.return_value = [s1]
        # this is to make sure we call the subscribe on trades
        ctx.has_subscription.return_value = False

        # - initialize feature manager
        manager = FeatureManager(reader=ctx.loader)
        manager += TradeVolumeImbalance(trade_period="1Min", timeframe="1s")

        manager.on_start(ctx)

        # Check how many times ctx.subscribe was called
        subscribe_call_count = ctx.subscribe.call_count
        assert subscribe_call_count > 0

        for trade in trades:
            manager.on_event(ctx, TriggerEvent(ns_to_dt_64(trade.time), "trade", s1, trade))

        _tvi = manager["BTCUSDT", "TVI(1Min,tf=1s)"]
        assert _tvi is not None and len(_tvi) > 0

    def test_orderbook_subscription(self, ctx: TestStrategyContext):
        # setup context
        ctx.set_start("2021-01-01")
        ctx.set_stop("2021-03-01")
        s1 = ctx.get_instrument("BINANCE.UM:BTCUSDT")
        assert s1 is not None
        ctx.get_instruments.return_value = [s1]
        ctx.has_subscription.return_value = False

        orderbooks_path = "tests/data/csv/BTCUSDT.orderbooks.txt.gz"

        with gzip.open(orderbooks_path, "rt") as f:
            orderbooks = [json.loads(line) for line in f]

        obs = [ccxt_convert_orderbook(ob, s1) for ob in orderbooks]
        # Filter out None values
        obs = [ob for ob in obs if ob is not None]

        manager = FeatureManager(reader=ctx.loader)
        manager += OrderbookImbalance(depths=[1])

        manager.on_start(ctx)

        for ob in obs:
            manager.on_event(ctx, TriggerEvent(ns_to_dt_64(ob.time), "orderbook", s1, ob))

        _obi = manager["BTCUSDT", "OBI(1)"]
        assert _obi is not None and len(_obi) > 0

    def test_cached_feature_provider(self):
        pass
