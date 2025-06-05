import asyncio
import time
from collections import defaultdict
from typing import Callable

import pytest

from qubx import QubxLogConfig, logger
from qubx.core.basics import DataType, Instrument, MarketEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, Position
from qubx.utils.runner.configs import ExchangeConfig, LiveConfig, LoggingConfig
from qubx.utils.runner.runner import AccountConfigurationManager, StrategyConfig, run_strategy


async def wait(condition: Callable[[], bool] | None = None, timeout: int = 10, period: float = 1.0):
    start = time.time()
    if condition is None:
        await asyncio.sleep(timeout)
        return
    while time.time() - start < timeout:
        if condition():
            return
        await asyncio.sleep(period)
    raise TimeoutError("Timeout reached")


class DebugStrategy(IStrategy):
    _data_counter: int = 0
    _instr_to_dtype_to_count: dict[Instrument, dict[str, int]]

    def on_init(self, ctx):
        ctx.set_base_subscription(DataType.OHLC["1m"])
        ctx.set_warmup({DataType.OHLC["1m"]: "1h"})
        self._instr_to_dtype_to_count = defaultdict(lambda: defaultdict(int))

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent):
        if data.instrument is None:
            return
        self._instr_to_dtype_to_count[data.instrument][data.type] += 1

    def get_dtype_count(self, instr: Instrument, dtype: str) -> int:
        return self._instr_to_dtype_to_count[instr][dtype]


class UniverseChangeStrategy(IStrategy):
    def __init__(self):
        super().__init__()
        self._ohlc_counts = defaultdict(int)
        self._universe_changed = False
        self._new_instruments_added = []

    def on_init(self, ctx):
        ctx.set_base_subscription(DataType.OHLC["1m"])
        ctx.set_warmup({DataType.OHLC["1m"]: "1h"})

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent):
        if data.instrument is None:
            return
        if data.type == DataType.OHLC["1m"]:
            self._ohlc_counts[data.instrument] += 1
            logger.debug(f"OHLC data received for {data.instrument.symbol}: count={self._ohlc_counts[data.instrument]}")

    def on_universe_change(
        self, ctx: IStrategyContext, add_instruments: list[Instrument], rm_instruments: list[Instrument]
    ):
        logger.info(f"Universe changed: +{len(add_instruments)} instruments, -{len(rm_instruments)} instruments")
        if add_instruments:
            self._universe_changed = True
            self._new_instruments_added.extend(add_instruments)
            # Subscribe new instruments to OHLC data
            ctx.subscribe(DataType.OHLC["1m"], add_instruments)
            ctx.commit()

    def get_ohlc_count(self, instrument: Instrument) -> int:
        return self._ohlc_counts[instrument]

    def has_universe_changed(self) -> bool:
        return self._universe_changed

    def get_new_instruments(self) -> list[Instrument]:
        return self._new_instruments_added


class TestCcxtOhlcResubscription:
    @pytest.fixture(autouse=True)
    def setup(self):
        QubxLogConfig.set_log_level("DEBUG")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.skip(reason="Skip by default, run manually if needed")
    async def test_ohlc_resubscription_universe_change_binance(self):
        """Test that OHLC resubscription works when universe is changed for Binance spot"""
        await self._test_ohlc_resubscription_universe_change("BINANCE", ["BTCUSDT"], ["ETHUSDT"])

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.skip(reason="Skip by default, run manually if needed")
    async def test_ohlc_resubscription_universe_change_binance_um(self):
        """Test that OHLC resubscription works when universe is changed for Binance futures"""
        await self._test_ohlc_resubscription_universe_change(
            "BINANCE.UM", ["BTCUSDT"], ["ETHUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT"]
        )

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.skip(reason="Skip by default, run manually if needed")
    async def test_ohlc_resubscription_universe_change_binance_um_twice(self):
        """Test that OHLC resubscription works when universe is changed twice for Binance futures"""
        await self._test_ohlc_resubscription_universe_change_twice(
            "BINANCE.UM", ["BTCUSDT"], ["ETHUSDT", "XRPUSDT"], ["ADAUSDT", "SOLUSDT", "DOGEUSDT"]
        )

    async def _test_ohlc_resubscription_universe_change(
        self, exchange: str, initial_symbols: list[str], new_symbols: list[str], timeout: int = 120
    ):
        """
        Test that OHLC data is properly resubscribed when the universe is changed.

        This test:
        1. Starts with initial symbols
        2. Waits for OHLC data on initial symbols
        3. Adds new symbols to the universe
        4. Verifies that OHLC data is received for new symbols
        """
        ctx = run_strategy(
            config=StrategyConfig(
                name="OhlcResubscriptionTest",
                strategy=UniverseChangeStrategy,
                live=LiveConfig(
                    exchanges={
                        exchange: ExchangeConfig(
                            connector="ccxt",
                            universe=initial_symbols,
                        )
                    },
                    logging=LoggingConfig(
                        logger="InMemoryLogsWriter",
                        position_interval="10s",
                        portfolio_interval="1m",
                        heartbeat_interval="10m",
                    ),
                ),
            ),
            account_manager=AccountConfigurationManager(),
            paper=True,
            blocking=False,
        )

        # Get strategy instance
        strategy = ctx.strategy
        assert isinstance(strategy, UniverseChangeStrategy), "Strategy is not the expected type"

        # Wait for the strategy to be fitted
        await wait(lambda: ctx.is_fitted(), timeout=30)

        # Wait for initial OHLC data
        initial_instruments = [instr for instr in ctx.instruments if instr.symbol in initial_symbols]
        logger.info(
            f"Waiting for initial OHLC data for {len(initial_instruments)} instruments: {[i.symbol for i in initial_instruments]}"
        )

        def has_initial_ohlc_data():
            return all(strategy.get_ohlc_count(instr) > 0 for instr in initial_instruments)

        await wait(has_initial_ohlc_data, timeout=timeout)
        logger.info("Initial OHLC data received successfully")

        # Add new instruments to the universe
        from qubx.core.lookups import lookup

        new_instruments = []
        for symbol in new_symbols:
            instr = lookup.find_symbol(exchange, symbol)
            if instr:
                new_instruments.append(instr)

        logger.info(f"Adding {len(new_instruments)} new instruments to universe: {[i.symbol for i in new_instruments]}")
        current_universe = list(ctx.instruments)
        ctx.set_universe(current_universe + new_instruments)

        # Wait for universe change to be processed
        await wait(lambda: strategy.has_universe_changed(), timeout=30)
        logger.info("Universe change detected by strategy")

        # Wait for OHLC data on new instruments
        logger.info(f"Waiting for OHLC data on new instruments: {[i.symbol for i in new_instruments]}")

        def has_new_ohlc_data():
            added_instruments = strategy.get_new_instruments()
            return len(added_instruments) > 0 and all(strategy.get_ohlc_count(instr) > 0 for instr in added_instruments)

        await wait(has_new_ohlc_data, timeout=timeout)

        # Verify that we received OHLC data for new instruments
        added_instruments = strategy.get_new_instruments()
        assert len(added_instruments) > 0, "No instruments were added to the universe"

        for instr in added_instruments:
            count = strategy.get_ohlc_count(instr)
            logger.info(f"OHLC count for {instr.symbol}: {count}")
            assert count > 0, f"No OHLC data received for new instrument {instr.symbol}"

        logger.info("OHLC resubscription test completed successfully")

        # Stop the strategy
        ctx.stop()

    async def _test_ohlc_resubscription_universe_change_twice(
        self,
        exchange: str,
        initial_symbols: list[str],
        first_new_symbols: list[str],
        second_new_symbols: list[str],
        timeout: int = 120,
    ):
        """
        Test that OHLC data is properly resubscribed when the universe is changed twice.

        This test:
        1. Starts with initial symbols
        2. Waits for OHLC data on initial symbols
        3. Adds first batch of new symbols to the universe
        4. Verifies that OHLC data is received for first batch of new symbols
        5. Adds second batch of new symbols to the universe
        6. Verifies that OHLC data is received for second batch of new symbols
        """
        ctx = run_strategy(
            config=StrategyConfig(
                name="OhlcResubscriptionTwiceTest",
                strategy=UniverseChangeStrategy,
                live=LiveConfig(
                    exchanges={
                        exchange: ExchangeConfig(
                            connector="ccxt",
                            universe=initial_symbols,
                        )
                    },
                    logging=LoggingConfig(
                        logger="InMemoryLogsWriter",
                        position_interval="10s",
                        portfolio_interval="1m",
                        heartbeat_interval="10m",
                    ),
                ),
            ),
            account_manager=AccountConfigurationManager(),
            paper=True,
            blocking=False,
        )

        # Get strategy instance
        strategy = ctx.strategy
        assert isinstance(strategy, UniverseChangeStrategy), "Strategy is not the expected type"

        # Wait for the strategy to be fitted
        await wait(lambda: ctx.is_fitted(), timeout=30)

        # Wait for initial OHLC data
        initial_instruments = [instr for instr in ctx.instruments if instr.symbol in initial_symbols]
        logger.info(
            f"Waiting for initial OHLC data for {len(initial_instruments)} instruments: {[i.symbol for i in initial_instruments]}"
        )

        def has_initial_ohlc_data():
            return all(strategy.get_ohlc_count(instr) > 0 for instr in initial_instruments)

        await wait(has_initial_ohlc_data, timeout=timeout)
        logger.info("Initial OHLC data received successfully")

        # FIRST UNIVERSE CHANGE: Add first batch of new instruments
        from qubx.core.lookups import lookup

        first_new_instruments = []
        for symbol in first_new_symbols:
            instr = lookup.find_symbol(exchange, symbol)
            if instr:
                first_new_instruments.append(instr)

        logger.info(
            f"Adding first batch of {len(first_new_instruments)} new instruments to universe: {[i.symbol for i in first_new_instruments]}"
        )
        current_universe = list(ctx.instruments)
        ctx.set_universe(current_universe + first_new_instruments)

        # Wait for first universe change to be processed
        await wait(lambda: len(strategy.get_new_instruments()) >= len(first_new_instruments), timeout=30)
        logger.info("First universe change detected by strategy")

        # Wait for OHLC data on first batch of new instruments
        logger.info(
            f"Waiting for OHLC data on first batch of new instruments: {[i.symbol for i in first_new_instruments]}"
        )

        def has_first_new_ohlc_data():
            added_instruments = strategy.get_new_instruments()
            return len(added_instruments) >= len(first_new_instruments) and all(
                strategy.get_ohlc_count(instr) > 0 for instr in first_new_instruments
            )

        await wait(has_first_new_ohlc_data, timeout=timeout)
        logger.info("First batch of OHLC data received successfully")

        # SECOND UNIVERSE CHANGE: Add second batch of new instruments
        second_new_instruments = []
        for symbol in second_new_symbols:
            instr = lookup.find_symbol(exchange, symbol)
            if instr:
                second_new_instruments.append(instr)

        logger.info(
            f"Adding second batch of {len(second_new_instruments)} new instruments to universe: {[i.symbol for i in second_new_instruments]}"
        )
        current_universe = list(ctx.instruments)
        ctx.set_universe(current_universe + second_new_instruments)

        # Wait for second universe change to be processed
        total_expected_new_instruments = len(first_new_instruments) + len(second_new_instruments)
        await wait(lambda: len(strategy.get_new_instruments()) >= total_expected_new_instruments, timeout=30)
        logger.info("Second universe change detected by strategy")

        # Wait for OHLC data on second batch of new instruments
        logger.info(
            f"Waiting for OHLC data on second batch of new instruments: {[i.symbol for i in second_new_instruments]}"
        )

        def has_second_new_ohlc_data():
            added_instruments = strategy.get_new_instruments()
            return len(added_instruments) >= total_expected_new_instruments and all(
                strategy.get_ohlc_count(instr) > 0 for instr in second_new_instruments
            )

        await wait(has_second_new_ohlc_data, timeout=timeout)

        # Verify that we received OHLC data for all instruments
        added_instruments = strategy.get_new_instruments()
        assert len(added_instruments) >= total_expected_new_instruments, (
            f"Not all instruments were added to the universe. Expected: {total_expected_new_instruments}, Got: {len(added_instruments)}"
        )

        # Check first batch
        for instr in first_new_instruments:
            count = strategy.get_ohlc_count(instr)
            logger.info(f"OHLC count for first batch instrument {instr.symbol}: {count}")
            assert count > 0, f"No OHLC data received for first batch instrument {instr.symbol}"

        # Check second batch
        for instr in second_new_instruments:
            count = strategy.get_ohlc_count(instr)
            logger.info(f"OHLC count for second batch instrument {instr.symbol}: {count}")
            assert count > 0, f"No OHLC data received for second batch instrument {instr.symbol}"

        logger.info("Double OHLC resubscription test completed successfully")

        # Stop the strategy
        ctx.stop()


class TestCcxtDataProvider:
    @pytest.fixture(autouse=True)
    def setup(self):
        QubxLogConfig.set_log_level("DEBUG")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.skip(reason="Skip by default, run manually if needed")
    async def test_binance_reader(self):
        exchange = "BINANCE"
        await self._test_exchange_reading(exchange, ["BTCUSDT", "ETHUSDT"])

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.skip(reason="Skip by default, run manually if needed")
    async def test_binance_um_reader(self):
        exchange = "BINANCE.UM"
        await self._test_exchange_reading(exchange, ["BTCUSDT", "ETHUSDT"])

    async def _test_exchange_reading(self, exchange: str, symbols: list[str], timeout: int = 60):
        ctx = run_strategy(
            config=StrategyConfig(
                name="DataProviderTest",
                strategy=DebugStrategy,
                live=LiveConfig(
                    exchanges={
                        exchange: ExchangeConfig(
                            connector="ccxt",
                            universe=symbols,
                        )
                    },
                    logging=LoggingConfig(
                        logger="InMemoryLogsWriter",
                        position_interval="10s",
                        portfolio_interval="1m",
                        heartbeat_interval="10m",
                    ),
                ),
            ),
            account_manager=AccountConfigurationManager(),
            paper=True,
            blocking=False,
        )

        # Get strategy instance
        strategy = ctx.strategy
        assert isinstance(strategy, DebugStrategy), "Strategy is not the expected type"

        await wait(lambda: ctx.is_fitted(), timeout=30)

        ctx.subscribe(DataType.TRADE)
        ctx.subscribe(DataType.ORDERBOOK)
        ctx.commit()

        async def wait_for_instrument_data(instr: Instrument):
            async def check_counts():
                while True:
                    if (
                        strategy.get_dtype_count(instr, "trade") > 0
                        and strategy.get_dtype_count(instr, "orderbook") > 0
                    ):
                        return True
                    await asyncio.sleep(0.5)

            try:
                await asyncio.wait_for(check_counts(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for data on {instr}")
                return False

        results = await asyncio.gather(*(wait_for_instrument_data(instr) for instr in ctx.instruments))

        assert all(results), "Not all instruments received trade and orderbook data"

        # Stop the strategy
        ctx.stop()


class TestCcxtTrading:
    MIN_NOTIONAL = 100

    @pytest.fixture(autouse=True)
    def setup(self, exchange_credentials: dict[str, dict[str, str]]):
        QubxLogConfig.set_log_level("DEBUG")
        self._creds = exchange_credentials

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.skip(reason="Skip by default, run manually if needed")
    async def test_basic_binance(self):
        exchange = "BINANCE"
        await self._test_basic_exchange_functions(exchange, ["BTCUSDT"])

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.skip(reason="Skip by default, run manually if needed")
    async def test_basic_binance_um(self):
        exchange = "BINANCE.UM"
        await self._test_basic_exchange_functions(exchange, ["BTCUSDT"])

    async def _test_basic_exchange_functions(self, exchange: str, symbols: list[str]):
        # Convert credentials format
        account_manager = AccountConfigurationManager()
        # TODO: Add proper credential management for the new API

        ctx = run_strategy(
            config=StrategyConfig(
                name="TradingTest",
                strategy=DebugStrategy,
                live=LiveConfig(
                    exchanges={
                        exchange: ExchangeConfig(
                            connector="ccxt",
                            universe=symbols,
                        )
                    },
                    logging=LoggingConfig(
                        logger="InMemoryLogsWriter",
                        position_interval="10s",
                        portfolio_interval="1m",
                        heartbeat_interval="10m",
                    ),
                ),
            ),
            account_manager=account_manager,
            paper=False,  # This would require proper credentials setup
            blocking=False,
        )

        await wait(lambda: ctx.is_fitted(), timeout=30)
        await wait(timeout=5)

        i1 = ctx.instruments[0]
        pos = ctx.positions[i1]
        logger.info(f"Working with instrument {i1}")

        await self._close_open_positions(ctx, pos)

        logger.info(f"Position is {pos.quantity}")

        # 1. Enter market
        qty1 = pos.quantity
        amount = i1.min_size * 2
        price = ctx.ohlc(i1)[0].close
        if amount * price < self.MIN_NOTIONAL:
            amount = i1.round_size_up(self.MIN_NOTIONAL / price)

        logger.info(f"Entering market amount {amount} at price {price}")
        order1 = ctx.trade(i1, amount=amount)
        assert order1 is not None and order1.price is not None

        await wait(lambda pos=pos: not self._is_size_similar(pos.quantity, qty1, i1))

        # 2. Close position
        assert self._is_size_similar(pos.quantity, amount, i1)
        logger.info("Closing position")
        ctx.trade(i1, -pos.quantity)

        await wait(lambda pos=pos: not pos.is_open())

        # Stop strategy
        ctx.stop()

    async def _close_open_positions(self, ctx: IStrategyContext, pos: Position):
        if self._is_size_similar(pos.quantity, 0, pos.instrument):
            return
        logger.info(f"Found existing position quantity {pos.quantity}")
        ctx.trade(pos.instrument, -pos.quantity)
        await wait(lambda pos=pos: not pos.is_open())
        logger.info("Closed position")

    def _is_size_similar(self, a: float, b: float, i: Instrument) -> bool:
        return abs(a - b) < 2 * i.min_size
