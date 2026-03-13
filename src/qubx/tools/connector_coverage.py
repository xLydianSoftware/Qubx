import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

import click

from qubx import logger
from qubx.core.basics import DataType, MarketEvent
from qubx.core.interfaces import IStrategy, IStrategyContext
from qubx.utils.runner.accounts import AccountConfigurationManager
from qubx.utils.runner.configs import ExchangeConfig, LiveConfig, LoggingConfig, StrategyConfig
from qubx.utils.runner.runner import run_strategy


class TestStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestResult:
    name: str
    category: str
    status: TestStatus
    duration: float = 0.0
    error: str | None = None


@dataclass
class CoverageReport:
    exchange: str
    symbols: list[str]
    results: list[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status in (TestStatus.FAIL, TestStatus.ERROR))

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIP)

    def to_dict(self) -> dict:
        return {
            "exchange": self.exchange,
            "symbols": self.symbols,
            "tests": [
                {
                    "name": r.name,
                    "category": r.category,
                    "status": r.status.value,
                    "duration": round(r.duration, 3),
                    "error": r.error,
                }
                for r in self.results
            ],
            "summary": {
                "total": len(self.results),
                "passed": self.passed,
                "failed": self.failed,
                "skipped": self.skipped,
            },
        }


class CoverageStrategy(IStrategy):
    """Lightweight strategy that counts received data per type."""

    _data_counts: dict[str, int]
    _warmup_done: bool

    def on_init(self, ctx: IStrategyContext):
        self._data_counts = defaultdict(int)
        self._warmup_done = False
        ctx.set_base_subscription(DataType.OHLC["1m"])
        ctx.set_warmup({DataType.OHLC["1m"]: "1h"})

    def on_fit(self, ctx: IStrategyContext):
        self._warmup_done = True

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent):
        if data.instrument is not None:
            self._data_counts[data.type] += 1

    def get_count(self, dtype: str) -> int:
        return self._data_counts.get(dtype, 0)


async def _wait(condition, timeout: float, poll: float = 0.5) -> bool:
    """Wait for condition to become True. Returns True if met, False if timed out."""
    start = time.time()
    while time.time() - start < timeout:
        if condition():
            return True
        await asyncio.sleep(poll)
    return False


class ConnectorCoverageRunner:
    def __init__(
        self,
        exchange: str,
        symbols: list[str],
        account_file: str | None = None,
        timeout: float = 30,
        skip_trading: bool = False,
    ):
        self.exchange = exchange
        self.symbols = symbols
        self.account_file = account_file
        self.timeout = timeout
        self.skip_trading = skip_trading
        self.report = CoverageReport(exchange=exchange, symbols=symbols)
        self._ctx: IStrategyContext | None = None
        self._strategy: CoverageStrategy | None = None

    def run(self) -> CoverageReport:
        """Run all tests synchronously."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._run_all())
        finally:
            loop.close()
        return self.report

    async def _run_all(self):
        # Start strategy context once, run all tests, then stop
        try:
            self._start_context()
        except Exception as e:
            # If we can't even start, fail instrument_lookup and skip the rest
            self.report.results.append(
                TestResult("instrument_lookup", "Public Data", TestStatus.ERROR, error=str(e))
            )
            return

        try:
            await self._run_public_data_tests()
            await self._run_account_tests()
            await self._run_trading_tests()
        finally:
            self._stop_context()

    def _start_context(self):
        from pathlib import Path

        account_manager = AccountConfigurationManager(
            account_config=Path(self.account_file) if self.account_file else None,
            search_qubx_dir=True,
        )

        has_creds = self.account_file is not None
        self._ctx = run_strategy(
            config=StrategyConfig(
                name="ConnectorCoverage",
                strategy=CoverageStrategy,
                live=LiveConfig(
                    exchanges={
                        self.exchange: ExchangeConfig(
                            connector="ccxt",
                            universe=self.symbols,
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
            paper=not has_creds,
            blocking=False,
        )
        self._strategy = self._ctx.strategy  # type: ignore

    def _stop_context(self):
        if self._ctx is not None:
            try:
                self._ctx.stop()
            except Exception:
                pass

    def _record(self, name: str, category: str, status: TestStatus, duration: float, error: str | None = None):
        self.report.results.append(TestResult(name, category, status, duration, error))

    async def _run_public_data_tests(self):
        assert self._ctx is not None and self._strategy is not None

        # instrument_lookup - just check that we have instruments
        t0 = time.time()
        try:
            ok = await _wait(lambda: len(self._ctx.instruments) > 0, self.timeout)  # type: ignore
            if ok:
                self._record("instrument_lookup", "Public Data", TestStatus.PASS, time.time() - t0)
            else:
                self._record(
                    "instrument_lookup", "Public Data", TestStatus.FAIL, time.time() - t0, "No instruments resolved"
                )
                return  # skip remaining if no instruments
        except Exception as e:
            self._record("instrument_lookup", "Public Data", TestStatus.ERROR, time.time() - t0, str(e))
            return

        # Wait for fitted (warmup done = ohlc_warmup test)
        t0 = time.time()
        try:
            ok = await _wait(lambda: self._ctx.is_fitted(), self.timeout)  # type: ignore
            if ok:
                self._record("ohlc_warmup", "Public Data", TestStatus.PASS, time.time() - t0)
            else:
                self._record(
                    "ohlc_warmup", "Public Data", TestStatus.FAIL, time.time() - t0, "Warmup did not complete"
                )
        except Exception as e:
            self._record("ohlc_warmup", "Public Data", TestStatus.ERROR, time.time() - t0, str(e))

        # ohlc_subscribe - wait for live OHLC data
        await self._test_data_subscription(
            "ohlc_subscribe", DataType.OHLC["1m"], already_subscribed=True
        )

        # Subscribe to additional data types and test them
        data_tests = [
            ("trade_subscribe", DataType.TRADE),
            ("orderbook_subscribe", DataType.ORDERBOOK),
            ("quote_subscribe", DataType.QUOTE),
        ]

        # Subscribe all at once
        for _, dtype in data_tests:
            try:
                self._ctx.subscribe(dtype)
            except Exception:
                pass

        # Futures-only subscriptions
        is_futures = any(
            tag in self.exchange.upper() for tag in (".UM", ".CM", ".F", "HYPERLIQUID")
        )
        futures_tests = [
            ("funding_rate_subscribe", DataType.FUNDING_RATE),
            ("open_interest_subscribe", DataType.OPEN_INTEREST),
        ]
        if is_futures:
            for _, dtype in futures_tests:
                try:
                    self._ctx.subscribe(dtype)
                except Exception:
                    pass

        try:
            self._ctx.commit()
        except Exception:
            pass

        # Now wait for each data type
        for name, dtype in data_tests:
            await self._test_data_subscription(name, dtype, already_subscribed=True)

        if is_futures:
            for name, dtype in futures_tests:
                await self._test_data_subscription(name, dtype, already_subscribed=True)
        else:
            for name, _ in futures_tests:
                self._record(name, "Public Data", TestStatus.SKIP, 0.0, "Not a futures exchange")

    async def _test_data_subscription(self, name: str, dtype: str, already_subscribed: bool = False):
        assert self._ctx is not None and self._strategy is not None
        t0 = time.time()
        try:
            initial_count = self._strategy.get_count(dtype)
            ok = await _wait(lambda: self._strategy.get_count(dtype) > initial_count, self.timeout)  # type: ignore
            if ok:
                self._record(name, "Public Data", TestStatus.PASS, time.time() - t0)
            else:
                self._record(name, "Public Data", TestStatus.FAIL, time.time() - t0, f"No {dtype} data received")
        except Exception as e:
            self._record(name, "Public Data", TestStatus.ERROR, time.time() - t0, str(e))

    async def _run_account_tests(self):
        if self.account_file is None:
            for name in ("fetch_balances", "fetch_positions", "fetch_open_orders"):
                self._record(name, "Account", TestStatus.SKIP, 0.0, "No credentials provided")
            return

        assert self._ctx is not None

        # fetch_balances
        t0 = time.time()
        try:
            balances = self._ctx.account.get_balances()
            if balances is not None:
                self._record("fetch_balances", "Account", TestStatus.PASS, time.time() - t0)
            else:
                self._record("fetch_balances", "Account", TestStatus.FAIL, time.time() - t0, "Returned None")
        except Exception as e:
            self._record("fetch_balances", "Account", TestStatus.ERROR, time.time() - t0, str(e))

        # fetch_positions
        t0 = time.time()
        try:
            positions = self._ctx.positions
            if positions is not None:
                self._record("fetch_positions", "Account", TestStatus.PASS, time.time() - t0)
            else:
                self._record("fetch_positions", "Account", TestStatus.FAIL, time.time() - t0, "Returned None")
        except Exception as e:
            self._record("fetch_positions", "Account", TestStatus.ERROR, time.time() - t0, str(e))

        # fetch_open_orders
        t0 = time.time()
        try:
            orders = self._ctx.account.get_orders()
            if orders is not None:
                self._record("fetch_open_orders", "Account", TestStatus.PASS, time.time() - t0)
            else:
                self._record("fetch_open_orders", "Account", TestStatus.FAIL, time.time() - t0, "Returned None")
        except Exception as e:
            self._record("fetch_open_orders", "Account", TestStatus.ERROR, time.time() - t0, str(e))

    async def _run_trading_tests(self):
        trading_tests = ("place_limit_order", "cancel_order", "update_order")
        if self.skip_trading or self.account_file is None:
            reason = "--skip-trading" if self.skip_trading else "No credentials provided"
            for name in trading_tests:
                self._record(name, "Trading", TestStatus.SKIP, 0.0, reason)
            return

        assert self._ctx is not None

        instr = self._ctx.instruments[0]
        # Get current price and place order far from market
        t0 = time.time()
        order_id = None
        try:
            ohlc = self._ctx.ohlc(instr)
            if len(ohlc) == 0:
                for name in trading_tests:
                    self._record(name, "Trading", TestStatus.ERROR, 0.0, "No OHLC data for price reference")
                return
            price = ohlc[0].close
            # Place limit buy at 50% below market
            safe_price = instr.round_price_down(price * 0.5)
            amount = instr.min_size * 2
            min_notional = 100
            if amount * safe_price < min_notional:
                amount = instr.round_size_up(min_notional / safe_price)

            order = self._ctx.trade(instr, amount, price=safe_price)
            if order is not None:
                order_id = order.id
                self._record("place_limit_order", "Trading", TestStatus.PASS, time.time() - t0)
            else:
                self._record("place_limit_order", "Trading", TestStatus.FAIL, time.time() - t0, "Order returned None")
        except Exception as e:
            self._record("place_limit_order", "Trading", TestStatus.ERROR, time.time() - t0, str(e))

        # cancel_order
        t0 = time.time()
        if order_id is not None:
            try:
                self._ctx.cancel_order(order_id=order_id)
                self._record("cancel_order", "Trading", TestStatus.PASS, time.time() - t0)
            except Exception as e:
                self._record("cancel_order", "Trading", TestStatus.ERROR, time.time() - t0, str(e))
        else:
            self._record("cancel_order", "Trading", TestStatus.SKIP, 0.0, "No order to cancel")

        # update_order - place a new order and try to update it
        t0 = time.time()
        try:
            ohlc = self._ctx.ohlc(instr)
            price = ohlc[0].close
            safe_price = instr.round_price_down(price * 0.5)
            amount = instr.min_size * 2
            min_notional = 100
            if amount * safe_price < min_notional:
                amount = instr.round_size_up(min_notional / safe_price)

            order = self._ctx.trade(instr, amount, price=safe_price)
            if order is not None:
                new_price = instr.round_price_down(price * 0.45)
                self._ctx.update_order(price=new_price, amount=amount, order_id=order.id)
                self._record("update_order", "Trading", TestStatus.PASS, time.time() - t0)
                # Clean up
                try:
                    self._ctx.cancel_order(order_id=order.id)
                except Exception:
                    pass
            else:
                self._record("update_order", "Trading", TestStatus.FAIL, time.time() - t0, "Could not place order")
        except Exception as e:
            self._record("update_order", "Trading", TestStatus.ERROR, time.time() - t0, str(e))


def format_console_report(report: CoverageReport) -> str:
    """Format report as colored console output."""
    lines: list[str] = []
    lines.append("")
    lines.append(click.style(f"Connector Coverage Report: {report.exchange}", fg="cyan", bold=True))
    lines.append(click.style("=" * 55, fg="cyan"))

    current_category = ""
    for r in report.results:
        if r.category != current_category:
            current_category = r.category
            lines.append(click.style(f"\n{current_category}", bold=True))

        status_colors = {
            TestStatus.PASS: "green",
            TestStatus.FAIL: "red",
            TestStatus.SKIP: "yellow",
            TestStatus.ERROR: "red",
        }
        color = status_colors.get(r.status, "white")
        status_str = click.style(f"{r.status:<7}", fg=color, bold=r.status in (TestStatus.FAIL, TestStatus.ERROR))

        duration_str = f"{r.duration:.1f}s" if r.duration > 0 else ""
        error_str = f"  ({r.error})" if r.error else ""

        lines.append(f"  {r.name:<26} {status_str} {duration_str:>6}{error_str}")

    # Summary
    lines.append("")
    summary_parts = []
    if report.passed:
        summary_parts.append(click.style(f"{report.passed} passed", fg="green"))
    if report.failed:
        summary_parts.append(click.style(f"{report.failed} failed", fg="red"))
    if report.skipped:
        summary_parts.append(click.style(f"{report.skipped} skipped", fg="yellow"))

    lines.append(f"Summary: {', '.join(summary_parts)} (total: {len(report.results)})")
    lines.append("")
    return "\n".join(lines)


def save_json_report(report: CoverageReport, path: str):
    """Save report as JSON file."""
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Report saved to {path}")
