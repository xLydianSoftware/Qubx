import asyncio
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pytest

from qubx import QubxLogConfig, logger
from qubx.core.basics import (
    FRAMEWORK_CID_PREFIX,
    DataType,
    Deal,
    Instrument,
    MarketEvent,
    Order,
    OrderChange,
    OrderStatus,
    Signal,
)
from qubx.core.interfaces import IStrategy, IStrategyContext, Position
from qubx.loggers.inmemory import InMemoryLogsWriter
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
    async def test_ohlc_resubscription_universe_change_binance(self):
        """Test that OHLC resubscription works when universe is changed for Binance spot"""
        await self._test_ohlc_resubscription_universe_change("BINANCE", ["BTCUSDT"], ["ETHUSDT"])

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_ohlc_resubscription_universe_change_binance_um(self):
        """Test that OHLC resubscription works when universe is changed for Binance futures"""
        await self._test_ohlc_resubscription_universe_change(
            "BINANCE.UM", ["BTCUSDT"], ["ETHUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT"]
        )

    @pytest.mark.asyncio
    @pytest.mark.e2e
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
    async def test_binance_reader(self):
        exchange = "BINANCE"
        await self._test_exchange_reading(exchange, ["BTCUSDT", "ETHUSDT"])

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_binance_um_reader(self):
        """Smoke test: paper-mode binance UM OHLC subscription end-to-end.

        Acts as the e2e regression test for the BinanceQVUSDM ``has[]``
        patch (see ``tests/qubx/connectors/ccxt/test_binance_has_restore.py``).
        Boots a paper-mode strategy via ``run_strategy()`` with the BINANCE.UM
        connector, subscribes to OHLC 1m for [BTCUSDT, ETHUSDT], and asserts
        data flows.

        Without the patch, this test fails at ccxt >= 4.5.52 because
        ``OhlcDataHandler.prepare_subscription`` raises ``NotSupported`` when
        the binanceusdm ``has[]`` flags are ``None`` (see ccxt PR #28493).
        """
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

    def _account_manager_with_credentials(self, exchange: str, tmp_path: Path) -> AccountConfigurationManager:
        """Build an AccountConfigurationManager from the exchange_credentials fixture, or skip.

        Live-trading tests need real API keys; the fixture reads them from the file passed
        via ``--env`` (default ``.env.integration``). Without keys for the exchange the test
        skips instead of failing against the venue.
        """
        creds = self._creds.get(exchange)
        if not creds or not creds.get("api_key") or not creds.get("secret"):
            pytest.skip(f"No {exchange} API credentials in --env file (.env.integration) — live trading test")
        account_config = tmp_path / "accounts.toml"
        account_config.write_text(
            "[[accounts]]\n"
            f'exchange = "{exchange}"\n'
            'name = "e2e"\n'
            f'api_key = "{creds["api_key"]}"\n'
            f'secret = "{creds["secret"]}"\n'
        )
        return AccountConfigurationManager(account_config=account_config)

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_basic_binance(self, tmp_path: Path):
        exchange = "BINANCE"
        await self._test_basic_exchange_functions(exchange, ["BTCUSDT"], tmp_path)

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_basic_binance_um(self, tmp_path: Path):
        exchange = "BINANCE.UM"
        await self._test_basic_exchange_functions(exchange, ["BTCUSDT"], tmp_path)

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_paper_trading_binance_um(self):
        """Paper-mode trading end-to-end (network, NO credentials).

        Boots paper mode against the PUBLIC BINANCE.UM OHLC feed, submits a market
        order via ctx.trade, and asserts it fills through the SimulatedConnector -> central
        SimulatedAccountManager path: the order reaches FILLED, the position opens, and the
        base-currency balance changes. Verifies the PR 7 commit-2 rewire under live market data.
        """
        exchange = "BINANCE.UM"
        symbol = "BTCUSDT"

        ctx = run_strategy(
            config=StrategyConfig(
                name="PaperTradingTest",
                strategy=DebugStrategy,
                live=LiveConfig(
                    exchanges={
                        exchange: ExchangeConfig(
                            connector="ccxt",
                            universe=[symbol],
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

        assert ctx.is_paper_trading, "expected paper trading mode"

        await wait(lambda: ctx.is_fitted(), timeout=30)
        # - wait until the live OHLC feed has primed a quote (the OME needs it to match)
        i1 = ctx.instruments[0]
        await wait(lambda: ctx.quote(i1) is not None, timeout=60)

        from qubx.core.basics import OrderStatus
        from qubx.loggers.inmemory import InMemoryLogsWriter

        pos = ctx.positions[i1]
        assert not pos.is_open()

        # - submit a market BUY; the SimulatedConnector OME matches against the live quote
        price = ctx.quote(i1).mid_price()
        amount = i1.round_size_up(self.MIN_NOTIONAL / price)
        order = ctx.trade(i1, amount=amount)
        assert order is not None

        # - fill flows back through the central AM: the position opens at the requested size
        await wait(lambda: ctx.positions[i1].is_open(), timeout=15)
        assert self._is_size_similar(ctx.positions[i1].quantity, amount, i1)

        # - the order reached FILLED (the SimulatedConnector emitted OrderFilledEvent; AM applied it).
        #   Channel dispatch is async on the data-loop thread, so allow it to settle.
        await wait(lambda: order.status == OrderStatus.FILLED, timeout=15)

        # - the fill was recorded as an execution by the logging path
        logs_writer = ctx._logging.logs_writer
        assert isinstance(logs_writer, InMemoryLogsWriter)
        assert len(logs_writer.get_executions()) > 0

        # - close out and stop
        ctx.trade(i1, -ctx.positions[i1].quantity)
        await wait(lambda: not ctx.positions[i1].is_open(), timeout=15)
        ctx.stop()

    async def _test_basic_exchange_functions(self, exchange: str, symbols: list[str], tmp_path: Path):
        account_manager = self._account_manager_with_credentials(exchange, tmp_path)

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
            paper=False,
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

        await wait(lambda pos=pos: not self._is_size_similar(pos.quantity, qty1, i1), timeout=30)

        # 2. Close position
        assert self._is_size_similar(pos.quantity, amount, i1)
        logger.info("Closing position")
        ctx.trade(i1, -pos.quantity)

        await wait(lambda pos=pos: not pos.is_open(), timeout=30)

        # Stop strategy
        ctx.stop()

    async def _close_open_positions(self, ctx: IStrategyContext, pos: Position):
        if self._is_size_similar(pos.quantity, 0, pos.instrument):
            return
        logger.info(f"Found existing position quantity {pos.quantity}")
        ctx.trade(pos.instrument, -pos.quantity)
        await wait(lambda pos=pos: not pos.is_open(), timeout=30)
        logger.info("Closed position")

    def _is_size_similar(self, a: float, b: float, i: Instrument) -> bool:
        return abs(a - b) < 2 * i.min_size


class SelfTradingStrategy(IStrategy):
    """Emits one BUY signal through the real signal path and records every account callback.

    Signal emission: ``on_market_data`` returns a ``Signal`` on the first event for which a
    quote is available (the declared return type of ``IStrategy.on_market_data``).
    ``ProcessingManager._run_strategy_pipeline`` collects returned signals and runs them
    through ``__process_signals``: the default ``PositionsTracker(FixedSizer(1.0,
    amount_in_quote=False))`` turns the signal value into a target of that many base units,
    and ``SimplePositionGatherer.alter_positions`` submits the order via ``ctx.trade`` —
    the full production tracker -> gatherer -> TradingManager chain.

    Callbacks run on the processing thread; ``list.append`` is atomic, so the test thread
    can poll the lists safely.
    """

    notional: float = 100.0

    def __init__(self):
        super().__init__()
        self.order_events: list[tuple[Order, OrderChange]] = []
        self.executions: list[tuple[Instrument, Deal]] = []
        self.position_changes: list[Position] = []
        self.entry_amount: float | None = None

    def on_init(self, ctx):
        ctx.set_base_subscription(DataType.OHLC["1m"])
        ctx.set_warmup({DataType.OHLC["1m"]: "1h"})

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent) -> Signal | None:
        if self.entry_amount is not None or data.instrument is None:
            return None
        quote = ctx.quote(data.instrument)
        if quote is None:
            return None
        self.entry_amount = data.instrument.round_size_up(self.notional / quote.mid_price())
        return data.instrument.signal(ctx, self.entry_amount)

    def on_order(self, ctx: IStrategyContext, order: Order, change: OrderChange) -> None:
        self.order_events.append((order, change))

    def on_execution(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal) -> None:
        self.executions.append((instrument, deal))

    def on_position_change(self, ctx: IStrategyContext, position: Position) -> None:
        self.position_changes.append(position)

    def changes_for(self, client_order_id: str) -> list[OrderChange]:
        return [change for order, change in self.order_events if order.client_order_id == client_order_id]


class TestCcxtPaperEventTrain:
    MIN_NOTIONAL = 100

    @pytest.fixture(autouse=True)
    def setup(self):
        QubxLogConfig.set_log_level("DEBUG")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_signal_driven_event_train_binance_um(self):
        """Paper-mode signal-driven event train end-to-end (network, NO credentials).

        Leg 1 — signal-driven entry: ``SelfTradingStrategy`` returns a BUY signal from
        ``on_market_data`` on the first usable quote (see its docstring for why this is the
        chosen production path), so the order is produced by the tracker -> position
        gatherer -> TradingManager chain rather than a test-thread ``ctx.trade``. Asserts
        the full train: framework cid (``qubx_`` prefix), on_order changes ending in FILLED
        (an instantly-crossing market order legitimately reports FILLED as its first applied
        change — the OME fills atomically and the reducer absorbs fill-before-accept by
        design), on_execution with matching instrument/amount, on_position_change, the
        opened position, and the logged execution in InMemoryLogsWriter.

        Leg 2 — limit + cancel lifecycle: from the test thread (sanctioned here) submits a
        deep resting limit BUY (~50% below mid), awaits on_order(ACCEPTED), cancels it via
        ``ctx.cancel_order`` and awaits on_order(CANCELED) + terminal status, asserting no
        position change and no execution was recorded for it.
        """
        exchange = "BINANCE.UM"
        symbol = "BTCUSDT"

        ctx = run_strategy(
            config=StrategyConfig(
                name="PaperEventTrainTest",
                strategy=SelfTradingStrategy,
                live=LiveConfig(
                    exchanges={
                        exchange: ExchangeConfig(
                            connector="ccxt",
                            universe=[symbol],
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

        assert ctx.is_paper_trading, "expected paper trading mode"
        strategy = ctx.strategy
        assert isinstance(strategy, SelfTradingStrategy), "Strategy is not the expected type"

        try:
            await wait(lambda: ctx.is_fitted(), timeout=30)
            i1 = ctx.instruments[0]
            # - quote priming off the live OHLC feed can take a while
            await wait(lambda: ctx.quote(i1) is not None, timeout=90)

            # ---------------- LEG 1: signal-driven entry ----------------
            await wait(lambda: strategy.entry_amount is not None, timeout=60)
            amount = strategy.entry_amount
            assert amount is not None and amount > 0

            # - the gatherer's order fills through the OME and the callbacks fire
            await wait(lambda: any(c == OrderChange.FILLED for _, c in strategy.order_events), timeout=30)
            await wait(lambda: ctx.positions[i1].is_open(), timeout=15)

            entry_cids = {o.client_order_id for o, _ in strategy.order_events}
            assert len(entry_cids) == 1, f"expected exactly one order in leg 1, saw {entry_cids}"
            entry_cid = next(iter(entry_cids))
            assert entry_cid.startswith(FRAMEWORK_CID_PREFIX), f"gatherer order cid {entry_cid} is not framework"

            # The simulated OME fills a crossing market order atomically, so no resting
            # ACCEPTED state ever exists at the venue and on_order may observe FILLED as the
            # first applied change (design.md: a FILLED arriving before its ACCEPTED is
            # absorbed by the reducer). ACCEPTED-first is pinned by the resting limit in leg 2.
            changes = strategy.changes_for(entry_cid)
            assert changes[-1] == OrderChange.FILLED, f"last change was {changes[-1]}, expected FILLED"
            if pre_fill := changes[:-1]:
                assert pre_fill[0] == OrderChange.ACCEPTED, f"unexpected train {changes}"
                assert all(c == OrderChange.PARTIALLY_FILLED for c in pre_fill[1:]), f"unexpected train {changes}"

            filled_order = next(o for o, c in strategy.order_events if c == OrderChange.FILLED)
            assert filled_order.status == OrderStatus.FILLED

            assert len(strategy.executions) >= 1, "on_execution was not called"
            assert {instr for instr, _ in strategy.executions} == {i1}
            filled_amount = sum(deal.amount for _, deal in strategy.executions)
            assert abs(filled_amount - amount) < 2 * i1.min_size

            assert any(p.instrument == i1 for p in strategy.position_changes), "on_position_change was not called"
            assert abs(ctx.positions[i1].quantity - amount) < 2 * i1.min_size
            assert sum(1 for p in ctx.positions.values() if p.is_open()) == 1

            logs_writer = ctx._logging.logs_writer
            assert isinstance(logs_writer, InMemoryLogsWriter)
            assert len(logs_writer.get_executions()) >= 1, "execution was not logged"

            # ---------------- LEG 2: limit + cancel lifecycle ----------------
            position_before = ctx.positions[i1].quantity
            executions_before = len(strategy.executions)
            logged_executions_before = len(logs_writer.get_executions())

            quote = ctx.quote(i1)
            assert quote is not None
            limit_price = quote.mid_price() * 0.5
            limit_amount = i1.round_size_up(self.MIN_NOTIONAL / limit_price)
            limit_order = ctx.trade(i1, amount=limit_amount, price=limit_price)
            cid = limit_order.client_order_id

            await wait(lambda: OrderChange.ACCEPTED in strategy.changes_for(cid), timeout=20)
            await wait(lambda: limit_order.status == OrderStatus.ACCEPTED, timeout=20)

            ctx.cancel_order(client_order_id=cid)
            await wait(lambda: OrderChange.CANCELED in strategy.changes_for(cid), timeout=20)
            await wait(lambda: limit_order.status == OrderStatus.CANCELED, timeout=20)
            assert limit_order.status.is_terminal

            assert strategy.changes_for(cid) == [OrderChange.ACCEPTED, OrderChange.CANCELED]
            assert ctx.positions[i1].quantity == position_before, "position changed on a canceled resting order"
            assert len(strategy.executions) == executions_before, "execution recorded for a canceled resting order"
            assert len(logs_writer.get_executions()) == logged_executions_before
        finally:
            try:
                open_positions = [p for p in ctx.positions.values() if p.is_open()]
                for pos in open_positions:
                    ctx.trade(pos.instrument, -pos.quantity)
                if open_positions:
                    await wait(lambda: not any(p.is_open() for p in ctx.positions.values()), timeout=15)
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}")
            finally:
                ctx.stop()
