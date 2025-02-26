from typing import Any

import pytest

from qubx.backtester.broker import SimulatedAccountProcessor
from qubx.backtester.simulator import (
    SimulatedBroker,
    SimulatedCtrlChannel,
    SimulatedDataProvider,
    find_instruments_and_exchanges,
    simulate,
)
from qubx.backtester.utils import SimulatedScheduler, SimulatedTimeProvider, recognize_simulation_data_config
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import DataType, Instrument, ITimeProvider, dt_64
from qubx.core.context import StrategyContext
from qubx.core.interfaces import IStrategy, IStrategyContext
from qubx.core.loggers import InMemoryLogsWriter, StrategyLogging
from qubx.core.lookups import lookup
from qubx.core.mixins.trading import TradingManager
from qubx.data.readers import CsvStorageDataReader, DataReader
from qubx.pandaz.utils import *
from tests.qubx.core.utils_test import DummyTimeProvider


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
    tcc = lookup.fees.find(exchange.lower(), commissions)
    assert tcc is not None
    time_provider = SimulatedTimeProvider(start)
    channel = SimulatedCtrlChannel("data")
    account = SimulatedAccountProcessor(
        account_id=strategy_id,
        channel=channel,
        base_currency=base_currency,
        initial_capital=initial_capital,
        time_provider=time_provider,
        tcc=tcc,
    )
    broker = SimulatedBroker(channel, account)
    scheduler = SimulatedScheduler(channel, lambda: time_provider.time().item())
    _data_cfg = recognize_simulation_data_config(data_reader, instruments, exchange)
    data_provider = SimulatedDataProvider(
        "dummy", channel, scheduler, time_provider, account, readers=_data_cfg.data_providers
    )
    logs_writer = InMemoryLogsWriter(strategy_id, strategy_id, "0")
    strategy_logging = StrategyLogging(logs_writer)
    ctx = StrategyContext(
        strategy=strategy,
        broker=broker,
        data_provider=data_provider,
        account=account,
        scheduler=scheduler,
        time_provider=time_provider,
        instruments=instruments,
        logging=strategy_logging,
    )
    ctx.start()
    data_provider.run(start, stop)
    return ctx, logs_writer


class TestAccountProcessorStuff:
    INITIAL_CAPITAL = 100_000

    def get_instrument(self, exchange: str, symbol: str) -> Instrument:
        instr = lookup.find_symbol(exchange, symbol)
        assert instr is not None
        return instr

    @pytest.fixture
    def trading_manager(self) -> TradingManager:
        name = "test"
        channel = SimulatedCtrlChannel("data")
        account = SimulatedAccountProcessor(
            account_id=name,
            channel=channel,
            base_currency="USDT",
            initial_capital=self.INITIAL_CAPITAL,
            time_provider=DummyTimeProvider(),
        )
        broker = SimulatedBroker(channel, account)

        class PrintCallback:
            def process_data(self, instrument: Instrument, d_type: str, data: Any, is_hist: bool):
                match d_type:
                    case "deals":
                        account.process_deals(instrument, data)
                    case "order":
                        account.process_order(data)

                print(data)

        channel.register(PrintCallback())

        return TradingManager(DummyTimeProvider(), broker, account, name)

    def test_spot_account_processor(self, trading_manager: TradingManager):
        account = trading_manager._account
        time_provider = trading_manager._time_provider

        # - check initial state
        assert account.get_total_capital() == self.INITIAL_CAPITAL
        assert account.get_capital() == self.INITIAL_CAPITAL
        assert account.get_net_leverage() == 0
        assert account.get_gross_leverage() == 0
        assert account.get_balances()["USDT"].free == self.INITIAL_CAPITAL
        assert account.get_balances()["USDT"].locked == 0
        assert account.get_balances()["USDT"].total == self.INITIAL_CAPITAL

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
        assert account.get_balances()["USDT"].free == pytest.approx(self.INITIAL_CAPITAL / 2)
        assert account.get_balances()["BTC"].free == pytest.approx(0.5)

        ##############################################
        # 2. Test locking and unlocking of funds
        ##############################################
        o2 = trading_manager.trade(i1, 0.1, price=90_000)
        assert account.get_balances()["USDT"].locked == pytest.approx(9_000)
        trading_manager.cancel_order(o2.id)
        assert account.get_balances()["USDT"].locked == pytest.approx(0)

        ##############################################
        # 3. Sell BTC on spot
        ##############################################
        # - update instrument price
        o2 = trading_manager.trade(i1, -0.5)

        assert account.get_net_leverage() == 0
        assert account.get_gross_leverage() == 0
        assert account.get_capital() == pytest.approx(self.INITIAL_CAPITAL)
        assert account.get_total_capital() == pytest.approx(self.INITIAL_CAPITAL)
        assert account.get_balances()["USDT"].free == pytest.approx(self.INITIAL_CAPITAL)
        assert account.get_balances()["BTC"].free == pytest.approx(0)

    def test_swap_account_processor(self, trading_manager: TradingManager):
        account = trading_manager._account
        time_provider = trading_manager._time_provider

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
        balances = account.get_balances()
        assert len(balances) == 1
        assert balances["USDT"].free == pytest.approx(self.INITIAL_CAPITAL)

        # - check margin requirements
        assert account.get_total_required_margin() == pytest.approx(50_000 * i1.maint_margin)

        # increase price 2x
        account.update_position_price(
            time_provider.time(),
            i1,
            200_000.0,
        )

        assert pos.market_value == pytest.approx(50_000, abs=1)
        assert pos.maint_margin == pytest.approx(100_000 * i1.maint_margin)

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
