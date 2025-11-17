from typing import Any, cast

import pytest

from qubx.core.basics import AssetBalance
from qubx.backtester.broker import SimulatedAccountProcessor, SimulatedBroker
from qubx.backtester.runner import SimulationRunner
from qubx.backtester.simulated_exchange import get_simulated_exchange
from qubx.backtester.utils import (
    SetupTypes,
    SimulatedCtrlChannel,
    SimulationSetup,
    find_instruments_and_exchanges,
    recognize_simulation_data_config,
)
from qubx.core.basics import DEFAULT_MAINTENANCE_MARGIN, ZERO_COSTS, DataType, Instrument
from qubx.core.interfaces import IBroker, IStrategy, IStrategyContext
from qubx.core.lookups import lookup
from qubx.core.mixins.trading import TradingManager
from qubx.data.readers import CsvStorageDataReader, DataReader
from qubx.loggers.inmemory import InMemoryLogsWriter
from qubx.pandaz.utils import *
from tests.qubx.core.utils_test import DummyTimeProvider


def balances_to_dict(balances: list[AssetBalance]) -> dict[str, AssetBalance]:
    """Helper to convert list of balances to dict for easier testing."""
    return {b.currency: b for b in balances}


class DummyStg(IStrategy):
    def on_init(self, ctx: IStrategyContext):
        ctx.set_base_subscription(DataType.OHLC["1h"])


def run_debug_sim(
    strategy_id: str,
    strategy: IStrategy,
    data_reader: DataReader,
    exchange: str,
    symbols: list[str | Instrument],
    commissions: str | None,
    start: str,
    stop: str,
    initial_capital: float,
    base_currency: str,
) -> tuple[IStrategyContext, InMemoryLogsWriter]:
    instruments, _ = find_instruments_and_exchanges(symbols, exchange)

    # Create a SimulationSetup
    setup = SimulationSetup(
        name=strategy_id,
        generator=strategy,
        setup_type=SetupTypes.STRATEGY,
        exchanges=[exchange],
        instruments=instruments,
        capital=float(initial_capital),
        base_currency=base_currency,
        commissions=commissions,
        accurate_stop_orders_execution=False,
        signal_timeframe="1h",
        tracker=None,
    )

    # Create a SimulationDataConfig
    data_config = recognize_simulation_data_config(data_reader, instruments)

    # Create and run the SimulationRunner
    runner = SimulationRunner(
        setup=setup,
        data_config=data_config,
        start=start,
        stop=stop,
        account_id=strategy_id,
        portfolio_log_freq="5Min",
    )

    # Run the simulation
    runner.run(silent=True, close_data_readers=True)

    # Return the context and logs_writer
    return runner.ctx, runner.logs_writer


class TestAccountProcessorStuff:
    INITIAL_CAPITAL = 100_000

    def get_instrument(self, exchange: str, symbol: str) -> Instrument:
        instr = lookup.find_symbol(exchange, symbol)
        assert instr is not None
        return instr

    @pytest.fixture
    def trading_manager(self, request) -> TradingManager:
        # Get the test function name to determine which exchange to use
        test_name = request.function.__name__

        # Set exchange based on test function
        if "spot" in test_name:
            exchange_name = "BINANCE"
        elif "swap" in test_name:
            exchange_name = "BINANCE.UM"
        else:
            exchange_name = "test"

        name = "test"
        channel = SimulatedCtrlChannel("data")
        exchange = get_simulated_exchange(exchange_name, DummyTimeProvider(), ZERO_COSTS)
        account = SimulatedAccountProcessor(
            account_id=name,
            channel=channel,
            exchange=exchange,
            base_currency="USDT",
            exchange_name=exchange_name,
            initial_capital=self.INITIAL_CAPITAL,
        )
        broker = SimulatedBroker(channel, account, exchange)

        class PrintCallback:
            def process_data(self, instrument: Instrument, d_type: str, data: Any, is_hist: bool):
                match d_type:
                    case "deals":
                        account.process_deals(instrument, data)
                    case "order":
                        account.process_order(data)

                print(data)

        channel.register(PrintCallback())

        # Create a mock context with the necessary methods
        from unittest.mock import Mock
        mock_context = Mock()
        mock_context.time = DummyTimeProvider().time

        # Mock the quote method to return a quote with mid_price
        mock_quote = Mock()
        mock_quote.mid_price = Mock(return_value=50000.0)
        mock_context.quote = Mock(return_value=mock_quote)

        # Create a mapping for the TradingManager's exchange_to_broker dictionary
        broker_map = {exchange_name: cast(IBroker, broker)}
        trading_manager = TradingManager(mock_context, [broker], account, name)
        # Manually set the exchange_to_broker map to ensure it has the correct keys
        trading_manager._exchange_to_broker = broker_map

        return trading_manager

    def test_spot_account_processor(self, trading_manager: TradingManager):
        account = trading_manager._account
        time_provider = trading_manager._context

        # - check initial state
        assert account.get_total_capital() == self.INITIAL_CAPITAL
        assert account.get_capital() == self.INITIAL_CAPITAL
        assert account.get_net_leverage() == 0
        assert account.get_gross_leverage() == 0
        balances = balances_to_dict(account.get_balances())
        assert balances["USDT"].free == self.INITIAL_CAPITAL
        assert balances["USDT"].locked == 0
        assert balances["USDT"].total == self.INITIAL_CAPITAL

        ##############################################
        # 1. Buy BTC on spot for half of the capital
        ##############################################
        i1 = self.get_instrument("BINANCE", "BTCUSDT")

        # - update instrument price
        account.update_position_price(
            time_provider.time(),
            i1,
            100_000.0,
        )

        # - execute trade for half of the initial capital
        o1 = trading_manager.trade(i1, 0.5)

        pos = account.positions[i1]
        assert pos.quantity == 0.5
        assert pos.market_value == pytest.approx(50_000)
        assert account.get_net_leverage() == pytest.approx(0.5)
        assert account.get_gross_leverage() == pytest.approx(0.5)
        assert account.get_capital() == pytest.approx(self.INITIAL_CAPITAL)
        assert account.get_total_capital() == pytest.approx(self.INITIAL_CAPITAL)
        balances = balances_to_dict(account.get_balances())
        assert balances["USDT"].free == pytest.approx(self.INITIAL_CAPITAL / 2)
        assert balances["BTC"].free == pytest.approx(0.5)

        ##############################################
        # 2. Test locking and unlocking of funds
        ##############################################
        o2 = trading_manager.trade(i1, 0.1, price=90_000)
        balances = balances_to_dict(account.get_balances())
        assert balances["USDT"].locked == pytest.approx(9_000)

        # Test that cancel_order returns success status
        cancel_success = trading_manager.cancel_order(o2.id)
        assert cancel_success is True, "Order cancellation should succeed"

        balances = balances_to_dict(account.get_balances())
        assert balances["USDT"].locked == pytest.approx(0)

        ##############################################
        # 3. Sell BTC on spot
        ##############################################
        # - update instrument price
        o2 = trading_manager.trade(i1, -0.5)

        assert account.get_net_leverage() == 0
        assert account.get_gross_leverage() == 0
        assert account.get_capital() == pytest.approx(self.INITIAL_CAPITAL)
        assert account.get_total_capital() == pytest.approx(self.INITIAL_CAPITAL)
        balances = balances_to_dict(account.get_balances())
        assert balances["USDT"].free == pytest.approx(self.INITIAL_CAPITAL)
        assert balances["BTC"].free == pytest.approx(0)

    def test_swap_account_processor(self, trading_manager: TradingManager):
        account = trading_manager._account
        time_provider = trading_manager._context

        i1 = self.get_instrument("BINANCE.UM", "BTCUSDT")

        account.update_position_price(
            time_provider.time(),
            i1,
            100_000.0,
        )

        # - execute trade for half of the initial capital
        o1 = trading_manager.trade(i1, 0.5)
        pos = account.positions[i1]

        # - check that market value of the position is close to 0 for swap
        assert pos.quantity == 0.5
        assert pos.market_value == pytest.approx(0, abs=1)

        # - check that USDT balance is actually left untouched
        balances_list = account.get_balances()
        assert len(balances_list) == 1
        balances = balances_to_dict(balances_list)
        assert balances["USDT"].free == pytest.approx(self.INITIAL_CAPITAL)

        # - check margin requirements
        # Since i1.maint_margin is 0, the default maintenance margin (5%) is used
        expected_margin = 50_000 * (i1.maint_margin or DEFAULT_MAINTENANCE_MARGIN)
        assert account.get_total_required_margin() == pytest.approx(expected_margin)

        # increase price 2x
        account.update_position_price(
            time_provider.time(),
            i1,
            200_000.0,
        )

        assert pos.market_value == pytest.approx(50_000, abs=1)
        expected_margin_2x = 100_000 * (i1.maint_margin or DEFAULT_MAINTENANCE_MARGIN)
        assert pos.maint_margin == pytest.approx(expected_margin_2x)

        # liquidate position
        o2 = trading_manager.trade(i1, -0.5)
        assert balances["USDT"].free == pytest.approx(self.INITIAL_CAPITAL + 50_000)
        assert pos.quantity == pytest.approx(0, abs=i1.min_size)
        assert pos.market_value == pytest.approx(0)

    def test_account_basics(self):
        initial_capital = 10_000

        ctx, _ = run_debug_sim(
            strategy_id="test0",
            strategy=DummyStg(),
            data_reader=CsvStorageDataReader("tests/data/csv"),
            exchange="BINANCE.UM",
            symbols=["BTCUSDT"],
            commissions=None,
            start="2024-01-01",
            stop="2024-01-02",
            initial_capital=initial_capital,
            base_currency="USDT",
        )

        # 1. Check account in the beginning
        assert 0 == ctx.get_net_leverage()
        assert 0 == ctx.get_gross_leverage()
        assert initial_capital == ctx.get_capital()
        assert initial_capital == ctx.get_total_capital()

        # 2. Execute a trade and check account
        leverage = 0.5
        instrument = ctx.instruments[0]
        quote = ctx.quote(instrument)
        assert quote is not None
        capital = ctx.get_total_capital()
        amount_in_base = capital * leverage
        amount = ctx.instruments[0].round_size_down(amount_in_base / quote.mid_price())
        leverage_adj = amount * quote.ask / capital
        ctx.trade(instrument, amount)

        # make the assertions work for floats
        assert leverage_adj == pytest.approx(ctx.get_net_leverage(), abs=0.01)
        assert leverage_adj == pytest.approx(ctx.get_gross_leverage(), abs=0.01)
        pos = ctx.get_position(instrument)
        assert initial_capital - pos.maint_margin == pytest.approx(ctx.get_capital(), abs=1)
        assert initial_capital == pytest.approx(ctx.get_total_capital(), abs=1)

        # 3. Exit trade and check account
        ctx.trade(instrument, -amount)

        # get tick size for BTCUSDT
        tick_size = ctx.instruments[0].tick_size
        trade_pnl = -tick_size / quote.ask * leverage_adj
        new_capital = initial_capital * (1 + trade_pnl)

        assert 0 == ctx.get_net_leverage()
        assert 0 == ctx.get_gross_leverage()
        assert new_capital == pytest.approx(ctx.get_capital(), abs=1)
        assert ctx.get_capital() == pytest.approx(ctx.get_total_capital(), abs=1)

    def test_commissions(self):
        initial_capital = 10_000

        ctx, logs_writer = run_debug_sim(
            strategy_id="test0",
            strategy=DummyStg(),
            data_reader=CsvStorageDataReader("tests/data/csv"),
            exchange="BINANCE.UM",
            symbols=["BTCUSDT"],
            commissions="vip0_usdt",
            start="2024-01-01",
            stop="2024-01-02",
            initial_capital=initial_capital,
            base_currency="USDT",
        )

        leverage = 0.5
        s = ctx.instruments[0]
        quote = ctx.quote(s)
        assert quote is not None
        capital = ctx.get_total_capital()
        amount_in_base = capital * leverage
        amount = ctx.instruments[0].round_size_down(amount_in_base / quote.mid_price())
        ctx.trade(s, amount)
        ctx.trade(s, -amount)

        execs = logs_writer.get_executions()
        commissions = execs.commissions
        assert not any(commissions.isna())
