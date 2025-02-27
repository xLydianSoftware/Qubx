import asyncio
import time
from collections import defaultdict
from typing import Callable

import pytest

from qubx import QubxLogConfig, logger
from qubx.core.basics import DataType, Instrument, MarketEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, Position


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

    def on_init(self, ctx: IStrategyContext):
        ctx.set_base_subscription(DataType.OHLC["1m"])
        ctx.set_warmup({DataType.OHLC["1m"]: "1h"})
        self._instr_to_dtype_to_count = defaultdict(lambda: defaultdict(int))

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent):
        if data.instrument is None:
            return
        self._instr_to_dtype_to_count[data.instrument][data.type] += 1

    def get_dtype_count(self, instr: Instrument, dtype: str) -> int:
        return self._instr_to_dtype_to_count[instr][dtype]


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
        from qubx.utils.runner import run_ccxt_trading

        ctx = run_ccxt_trading(
            strategy=(stg := DebugStrategy()),
            exchange_name=exchange,
            symbols=symbols,
            blocking=False,
            commissions="vip0_usdt",
            paper=True,
            use_testnet=True,
        )
        await wait(ctx.is_fitted)

        ctx.subscribe(DataType.TRADE)
        ctx.subscribe(DataType.ORDERBOOK)
        ctx.commit()

        async def wait_for_instrument_data(instr: Instrument):
            async def check_counts():
                while True:
                    if stg.get_dtype_count(instr, "trade") > 0 and stg.get_dtype_count(instr, "orderbook") > 0:
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
        # - temporary moved it here - otherwise it breaks the tests
        from qubx.utils.runner import run_ccxt_trading

        # - Start strategy
        ctx = run_ccxt_trading(
            strategy=(stg := DebugStrategy()),
            exchange_name=exchange,
            symbols=symbols,
            credentials=self._creds[exchange],
            blocking=False,
            use_testnet=True,
            commissions="vip0_usdt",
        )

        await wait(ctx.is_fitted)
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
        logger.info(f"Closing position of {pos.quantity} for {i1}")
        ctx.trade(i1, -pos.quantity)

        await wait(lambda pos=pos: not pos.is_open())

        # - Stop strategy
        ctx.stop()

    async def _close_open_positions(self, ctx: IStrategyContext, pos: Position):
        if self._is_size_similar(pos.quantity, 0, pos.instrument):
            return
        logger.info(f"Found existing position quantity {pos.quantity}")
        ctx.trade(pos.instrument, -pos.quantity)
        await wait(lambda pos=pos: not pos.is_open())
        logger.info(f"Closed position")

    def _is_size_similar(self, a: float, b: float, i: Instrument) -> bool:
        return abs(a - b) < 2 * i.min_size
