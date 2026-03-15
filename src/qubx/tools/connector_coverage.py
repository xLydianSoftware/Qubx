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
            self._data_counts[(data.type, data.instrument.symbol)] += 1

    def get_count(self, dtype: str, symbol: str | None = None) -> int:
        if symbol is not None:
            return self._data_counts.get((dtype, symbol), 0)
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
        only_tests: list[str] | None = None,
    ):
        self.exchange = exchange
        self.symbols = symbols
        self.account_file = account_file
        self.timeout = timeout
        self.skip_trading = skip_trading
        self.only_tests = only_tests
        self.report = CoverageReport(exchange=exchange, symbols=symbols)
        self._ctx: IStrategyContext | None = None
        self._strategy: CoverageStrategy | None = None

    def _should_run(self, name: str) -> bool:
        return self.only_tests is None or name in self.only_tests

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
            # Always run public-data phase: it drives instrument warmup that trading tests depend on.
            # Individual tests inside are skipped if they don't match only_tests.
            await self._run_public_data_tests()
            await self._run_account_tests()
            await self._run_trading_tests()
            await self._run_subscription_management_tests()
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
                if self._should_run("instrument_lookup"):
                    self._record("instrument_lookup", "Public Data", TestStatus.PASS, time.time() - t0)
            else:
                if self._should_run("instrument_lookup"):
                    self._record(
                        "instrument_lookup", "Public Data", TestStatus.FAIL, time.time() - t0, "No instruments resolved"
                    )
                return  # skip remaining if no instruments
        except Exception as e:
            if self._should_run("instrument_lookup"):
                self._record("instrument_lookup", "Public Data", TestStatus.ERROR, time.time() - t0, str(e))
            return

        # Wait for fitted (warmup done = ohlc_warmup test)
        t0 = time.time()
        try:
            ok = await _wait(lambda: self._ctx.is_fitted(), self.timeout)  # type: ignore
            if self._should_run("ohlc_warmup"):
                if ok:
                    self._record("ohlc_warmup", "Public Data", TestStatus.PASS, time.time() - t0)
                else:
                    self._record(
                        "ohlc_warmup", "Public Data", TestStatus.FAIL, time.time() - t0, "Warmup did not complete"
                    )
        except Exception as e:
            if self._should_run("ohlc_warmup"):
                self._record("ohlc_warmup", "Public Data", TestStatus.ERROR, time.time() - t0, str(e))

        # Skip subscription tests when only specific tests are requested
        if self.only_tests is not None:
            return

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
        account_tests = ("fetch_balances", "fetch_positions", "fetch_open_orders")
        if self.account_file is None:
            for name in account_tests:
                if self._should_run(name):
                    self._record(name, "Account", TestStatus.SKIP, 0.0, "No credentials provided")
            return

        assert self._ctx is not None

        # fetch_balances
        if self._should_run("fetch_balances"):
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
        if self._should_run("fetch_positions"):
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
        if self._should_run("fetch_open_orders"):
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
        all_trading_tests = ("place_limit_order", "cancel_order", "update_order", "client_order_id_roundtrip")
        if self.skip_trading or self.account_file is None:
            reason = "--skip-trading" if self.skip_trading else "No credentials provided"
            for name in all_trading_tests:
                if self._should_run(name):
                    self._record(name, "Trading", TestStatus.SKIP, 0.0, reason)
            return

        assert self._ctx is not None

        instr = self._ctx.instruments[0]
        ohlc = self._ctx.ohlc(instr)
        if len(ohlc) == 0:
            for name in all_trading_tests:
                if self._should_run(name):
                    self._record(name, "Trading", TestStatus.ERROR, 0.0, "No OHLC data for price reference")
            return

        price = ohlc[0].close
        safe_price = instr.round_price_down(price * 0.5)
        amount = instr.min_size * 2
        min_notional = 100
        if amount * safe_price < min_notional:
            amount = instr.round_size_up(min_notional / safe_price)

        # place_limit_order
        order_id = None
        if self._should_run("place_limit_order"):
            t0 = time.time()
            try:
                order = self._ctx.trade(instr, amount, price=safe_price)
                if order is not None:
                    order_id = order.id
                    self._record("place_limit_order", "Trading", TestStatus.PASS, time.time() - t0)
                else:
                    self._record("place_limit_order", "Trading", TestStatus.FAIL, time.time() - t0, "Order returned None")
            except Exception as e:
                self._record("place_limit_order", "Trading", TestStatus.ERROR, time.time() - t0, str(e))

        # cancel_order
        if self._should_run("cancel_order"):
            t0 = time.time()
            if order_id is not None:
                try:
                    self._ctx.cancel_order(order_id=order_id)
                    self._record("cancel_order", "Trading", TestStatus.PASS, time.time() - t0)
                    order_id = None
                except Exception as e:
                    self._record("cancel_order", "Trading", TestStatus.ERROR, time.time() - t0, str(e))
            else:
                self._record("cancel_order", "Trading", TestStatus.SKIP, 0.0, "No order to cancel")

        # Clean up any lingering order from place_limit_order if cancel was skipped
        if order_id is not None:
            try:
                self._ctx.cancel_order(order_id=order_id)
            except Exception:
                pass

        # update_order - place a new order and try to update it
        if self._should_run("update_order"):
            t0 = time.time()
            update_order_id = None
            try:
                order = self._ctx.trade(instr, amount, price=safe_price)
                if order is not None:
                    update_order_id = order.id
                    new_price = instr.round_price_down(price * 0.45)
                    self._ctx.update_order(price=new_price, amount=amount, order_id=order.id)
                    self._record("update_order", "Trading", TestStatus.PASS, time.time() - t0)
                else:
                    self._record("update_order", "Trading", TestStatus.FAIL, time.time() - t0, "Could not place order")
            except Exception as e:
                self._record("update_order", "Trading", TestStatus.ERROR, time.time() - t0, str(e))
            finally:
                if update_order_id is not None:
                    try:
                        self._ctx.cancel_order(order_id=update_order_id)
                    except Exception:
                        pass

        # client_order_id_roundtrip - verify OKX echoes back the client_id we send
        if self._should_run("client_order_id_roundtrip"):
            await self._test_client_order_id_roundtrip(instr, amount, safe_price)

    async def _test_client_order_id_roundtrip(self, instr, amount, safe_price):
        """
        Verify that the broker's client_order_id is preserved through the OKX WS roundtrip.

        The broker creates a synthetic PENDING order keyed by client_id (e.g. "qubx_SYMBOL_...").
        When OKX confirms the order via WS push, _match_pending_request looks up that key in
        _pending_order_requests and removes the PENDING entry, re-keying it under the exchange ordId.

        If the broker passes the client_id via the wrong HTTP parameter name (e.g. Binance's
        "newClientOrderId" instead of OKX's "clientOrderId"), OKX ignores it and generates its
        own clOrdId. The WS push then arrives with OKX's clOrdId, which doesn't match the key
        in _pending_order_requests → the PENDING synthetic order is never removed, and both a
        PENDING and an OPEN entry coexist in _active_orders for the same logical order.

        PASS: after WS confirmation, only 1 non-PENDING order remains (synthetic cleaned up).
        FAIL: PENDING synthetic order persists alongside the confirmed OPEN order (clOrdId mismatch).
        """
        assert self._ctx is not None
        t0 = time.time()
        exchange_order_id = None
        try:
            # Snapshot existing orders before placing
            orders_before = set(self._ctx.account.get_orders(instr).keys())

            # Use trade_async with an explicit client_id (like the LOE does) — this creates a
            # synthetic PENDING order in _active_orders keyed by sent_client_id
            sent_client_id = f"cov_cid_test_{int(time.time() * 1000)}"
            returned_client_id = self._ctx.trade_async(instr, amount, price=safe_price, client_id=sent_client_id)  # type: ignore
            # trade_async may pass the id through make_client_id; use the returned value as the authoritative key
            sent_client_id = returned_client_id or sent_client_id
            logger.info(f"[client_order_id_roundtrip] broker registered client_id='{sent_client_id}'")

            # Wait for OKX WS confirmation: an OPEN order should appear (non-PENDING, non-UNKNOWN)
            def _has_open_order() -> bool:
                orders = self._ctx.account.get_orders(instr)  # type: ignore
                return any(
                    oid not in orders_before and o.status == "OPEN"
                    for oid, o in orders.items()
                )

            confirmed = await _wait(_has_open_order, self.timeout)
            if not confirmed:
                self._record(
                    "client_order_id_roundtrip", "Trading", TestStatus.FAIL, time.time() - t0,
                    f"No OPEN order arrived via WS within {self.timeout}s"
                )
                return

            # Inspect final state: PENDING should be gone, only OPEN should remain
            final_orders = {
                oid: o for oid, o in self._ctx.account.get_orders(instr).items()
                if oid not in orders_before
            }
            open_orders = {oid: o for oid, o in final_orders.items() if o.status == "OPEN"}
            pending_remaining = {oid: o for oid, o in final_orders.items() if o.status == "PENDING"}

            # Grab exchange order id for cleanup
            if open_orders:
                exchange_order_id = next(iter(open_orders.keys()))
                echoed_client_id = next(iter(open_orders.values())).client_id
                logger.info(
                    f"[client_order_id_roundtrip] OKX WS returned client_id='{echoed_client_id}' "
                    f"for exchange order '{exchange_order_id}'"
                )
            else:
                echoed_client_id = None

            if pending_remaining and open_orders:
                # Synthetic PENDING was NOT cleaned up — clOrdId mismatch confirmed
                pending_cids = [o.client_id for o in pending_remaining.values()]
                self._record(
                    "client_order_id_roundtrip", "Trading", TestStatus.FAIL, time.time() - t0,
                    f"clOrdId mismatch: sent='{sent_client_id}' OKX_returned='{echoed_client_id}'. "
                    f"Synthetic PENDING order still in _active_orders: {pending_cids}. "
                    f"Broker likely uses wrong param name ('newClientOrderId' instead of 'clientOrderId')."
                )
            elif not pending_remaining and open_orders:
                # Synthetic was properly cleaned up — clOrdId matched
                self._record(
                    "client_order_id_roundtrip", "Trading", TestStatus.PASS, time.time() - t0,
                    f"clOrdId preserved: sent='{sent_client_id}' matched OKX_returned='{echoed_client_id}'"
                )
            else:
                self._record(
                    "client_order_id_roundtrip", "Trading", TestStatus.ERROR, time.time() - t0,
                    f"Unexpected state: open={list(open_orders.keys())}, pending={list(pending_remaining.keys())}"
                )

        except Exception as e:
            self._record("client_order_id_roundtrip", "Trading", TestStatus.ERROR, time.time() - t0, str(e))
        finally:
            # Cancel by exchange order id if we got one; fall back to cancel_orders
            try:
                if exchange_order_id is not None:
                    self._ctx.cancel_order(order_id=exchange_order_id)  # type: ignore
                else:
                    remaining = {
                        oid: o for oid, o in self._ctx.account.get_orders(instr).items()  # type: ignore
                        if o.status in ("OPEN", "PENDING")
                    }
                    for oid in remaining:
                        try:
                            self._ctx.cancel_order(order_id=oid)  # type: ignore
                        except Exception:
                            pass
            except Exception:
                pass

    async def _run_subscription_management_tests(self):
        category = "Subscription Management"

        assert self._ctx is not None and self._strategy is not None

        ob_dtype = DataType.ORDERBOOK[0, 1]
        ob_base = DataType.ORDERBOOK  # key used by on_market_data (data.type is the base type)

        # Snapshot instruments at entry — tests may mutate ctx.instruments
        initial_instruments = list(self._ctx.instruments)
        has_two = len(initial_instruments) >= 2

        if self._should_run("unsubscribe_orderbook"):
            t0 = time.time()
            try:
                self._ctx.subscribe(ob_dtype)
                self._ctx.commit()

                instr = initial_instruments[0]
                c0 = self._strategy.get_count(ob_base, instr.symbol)
                ok = await _wait(lambda: self._strategy.get_count(ob_base, instr.symbol) > c0, self.timeout)
                if not ok:
                    self._record("unsubscribe_orderbook", category, TestStatus.FAIL, time.time() - t0,
                                 f"No {ob_dtype} data received before unsubscription")
                else:
                    self._ctx.unsubscribe(ob_dtype)
                    self._ctx.commit()
                    await asyncio.sleep(2.0)
                    c1 = self._strategy.get_count(ob_base, instr.symbol)
                    await asyncio.sleep(2.0)
                    c2 = self._strategy.get_count(ob_base, instr.symbol)
                    if c2 == c1:
                        self._record("unsubscribe_orderbook", category, TestStatus.PASS, time.time() - t0,
                                     f"Data stopped after unsubscription (count frozen at {c1})")
                    else:
                        self._record("unsubscribe_orderbook", category, TestStatus.FAIL, time.time() - t0,
                                     f"Data still arriving after unsubscription: {c2 - c1} events in 2s")
            except Exception as e:
                self._record("unsubscribe_orderbook", category, TestStatus.ERROR, time.time() - t0, str(e))

        for name in ("remove_instrument_subscription", "add_instrument_subscription"):
            if self._should_run(name) and not has_two:
                self._record(name, category, TestStatus.SKIP, 0.0,
                             "Requires 2 symbols — pass --symbols SYM1 SYM2")

        if self._should_run("remove_instrument_subscription") and has_two:
            t0 = time.time()
            instr1, instr2 = initial_instruments[0], initial_instruments[1]
            try:
                # Ensure both instruments are in the universe before the test
                self._ctx.add_instruments([instr1, instr2])
                self._ctx.subscribe(ob_dtype)
                self._ctx.commit()

                c0 = self._strategy.get_count(ob_base, instr2.symbol)
                ok = await _wait(lambda: self._strategy.get_count(ob_base, instr2.symbol) > c0, self.timeout)
                if not ok:
                    self._record("remove_instrument_subscription", category, TestStatus.FAIL, time.time() - t0,
                                 f"No {ob_dtype} data for {instr2.symbol} before removal")
                else:
                    self._ctx.remove_instruments([instr2])
                    await _wait(lambda: instr2 not in self._ctx.instruments, self.timeout)  # type: ignore
                    await asyncio.sleep(2.0)
                    c1 = self._strategy.get_count(ob_base, instr2.symbol)
                    await asyncio.sleep(2.0)
                    c2 = self._strategy.get_count(ob_base, instr2.symbol)
                    if c2 == c1:
                        self._record("remove_instrument_subscription", category, TestStatus.PASS, time.time() - t0,
                                     f"Data stopped for {instr2.symbol} after remove_instruments (count frozen at {c1})")
                    else:
                        self._record("remove_instrument_subscription", category, TestStatus.FAIL, time.time() - t0,
                                     f"Data still arriving for {instr2.symbol} after remove_instruments: {c2 - c1} events in 2s")
            except Exception as e:
                self._record("remove_instrument_subscription", category, TestStatus.ERROR, time.time() - t0, str(e))

        if self._should_run("add_instrument_subscription") and has_two:
            t0 = time.time()
            instr1, instr2 = initial_instruments[0], initial_instruments[1]
            try:
                # Ensure we start with only instr1 so add_instruments has something to do
                if instr2 in self._ctx.instruments:
                    self._ctx.remove_instruments([instr2])
                    await _wait(lambda: instr2 not in self._ctx.instruments, self.timeout)  # type: ignore

                self._ctx.subscribe(ob_dtype, instr1)
                self._ctx.commit()

                c0_1 = self._strategy.get_count(ob_base, instr1.symbol)
                ok = await _wait(lambda: self._strategy.get_count(ob_base, instr1.symbol) > c0_1, self.timeout)
                if not ok:
                    self._record("add_instrument_subscription", category, TestStatus.FAIL, time.time() - t0,
                                 f"No {ob_dtype} data for {instr1.symbol} before add_instruments")
                else:
                    # Add instr2 — triggers stream restart for {instr1, instr2}
                    c0_1 = self._strategy.get_count(ob_base, instr1.symbol)
                    c0_2 = self._strategy.get_count(ob_base, instr2.symbol)
                    self._ctx.add_instruments([instr2])

                    ok2 = await _wait(lambda: self._strategy.get_count(ob_base, instr2.symbol) > c0_2, self.timeout)
                    ok1 = self._strategy.get_count(ob_base, instr1.symbol) > c0_1

                    if ok1 and ok2:
                        c1_1 = self._strategy.get_count(ob_base, instr1.symbol) - c0_1
                        c1_2 = self._strategy.get_count(ob_base, instr2.symbol) - c0_2
                        self._record("add_instrument_subscription", category, TestStatus.PASS, time.time() - t0,
                                     f"Both instruments receiving data after add_instruments "
                                     f"({instr1.symbol}: +{c1_1} events, {instr2.symbol}: +{c1_2} events)")
                    elif not ok2:
                        self._record("add_instrument_subscription", category, TestStatus.FAIL, time.time() - t0,
                                     f"No data for newly added {instr2.symbol} after add_instruments")
                    else:
                        self._record("add_instrument_subscription", category, TestStatus.FAIL, time.time() - t0,
                                     f"No data for {instr1.symbol} after add_instruments (stream restart dropped existing subscription)")
            except Exception as e:
                self._record("add_instrument_subscription", category, TestStatus.ERROR, time.time() - t0, str(e))


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
