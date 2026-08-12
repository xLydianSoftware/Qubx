"""
Stress tests for CcxtStorage rate-limit behavior — xLydianSoftware/Qubx#264.

The production incident that motivated this file:
    At warmup start the exchange (OKX) rate-limited 5 of ~20 symbols with
    ``50011 Too Many Requests``. The CcxtStorage layer did not apply the
    configured rate limiter (it was never injected into the warmup data
    storage), did not retry, and silently returned ``[]``. Downstream ATR
    couldn't compute, FixedRiskSizer produced inf-sized targets, and live
    state was corrupted.

The tests in this file exercise the warmup OHLCV fetch path against a
``FakeCcxtExchange`` that enforces its own req/sec budget. They verify:

1. **Positive path** — with the rate limiter attached, concurrent multi-symbol
   fetches never trigger the exchange's 429 budget.

2. **Negative control** — without the rate limiter, the same load bursts past
   the budget and triggers 429s. This proves the mock is exercising the
   code path and that the positive test isn't vacuously passing.

3. **Wiring regression** — every page fetch is charged through the throttle
   hook ``_ensure_exchange`` installs, at the real ccxt cost, and every
   response drives a header sync through ``on_rest_response``.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any
from unittest.mock import Mock

import ccxt
import pytest

from qubx.connectors.ccxt import factory
from qubx.connectors.ccxt.exchange_manager import ExchangeManager
from qubx.connectors.ccxt.rate_limits import install_rate_limiter_hooks
from qubx.core.basics import LiveTimeProvider
from qubx.data.storages.ccxt import CcxtFetchExhausted, CcxtStorage, _retryable_fetch
from qubx.health.dummy import DummyHealthMonitor
from qubx.rate_limiting import (
    EndpointCosts,
    ExchangeRateLimitConfig,
    ExchangeRateLimiter,
    PoolConfig,
)

TF_MS = 3_600_000  # 1h in milliseconds


class FakeCcxtExchange:
    """
    Minimal CCXT-like async exchange with a strict sliding-window req/s budget.

    Raises ``ccxt.RateLimitExceeded`` when arrival rate exceeds ``max_rps``
    over the last ``window_sec``. Every ``fetch_ohlcv`` call is counted
    regardless of whether it's rejected, so ``call_count`` reflects traffic
    pressure while ``rate_limit_hits`` reflects failures.
    """

    def __init__(
        self,
        *,
        max_rps: float,
        window_sec: float = 1.0,
        bars_per_page: int = 10,
        latency_ms: float = 5.0,
        id: str = "fake",
        server_budget: int | None = None,
        throttle_cost: float = 1.0,
        stale_headers: dict[str, str] | None = None,
    ) -> None:
        self.id = id
        self.name = id
        self.max_rps = max_rps
        self.window_sec = window_sec
        self.bars_per_page = bars_per_page
        self._latency_s = latency_ms / 1000.0
        self._arrivals: list[float] = []
        self.call_count = 0
        self.rate_limit_hits = 0
        # Cost CCXT computes for this endpoint and hands to ``throttle`` (klines with limit=1000
        # is weight 5 on binance.um, 1.2 on kraken, ...).
        self.throttle_cost = throttle_cost
        # Server-reported budget, published in the per-response headers handed to
        # ``on_rest_response``. When set, each successful call decrements it so the header
        # value reflects real usage.
        self._server_budget = server_budget
        self._used_weight = 0
        # When set, the exchange-wide attribute deliberately diverges from the per-request
        # headers — a sync that reads it instead of the hook argument is then visible.
        self.stale_headers = stale_headers
        self.last_response_headers: dict[str, str] = {}
        # CCXT-side hooks — real CCXT invokes ``self.throttle(cost)`` inside
        # ``fetch2()`` and ``self.on_rest_response(...)`` after the HTTP response.
        # Defaults are no-ops; ``ExchangeManager.attach_rate_limiter`` replaces
        # them with versions that acquire from / sync to our rate limiter.
        self.enableRateLimit = True
        self.throttle = self._default_throttle
        self.on_rest_response = self._default_on_rest_response
        # ExchangeManager reads ``asyncio_loop.set_exception_handler`` during init —
        # provide a Mock so that path is satisfied without a real loop.
        self.asyncio_loop: Any = Mock()

    async def _default_throttle(self, cost: float | None = None) -> None:
        return None

    def _default_on_rest_response(
        self, code, reason, url, method, headers, body, req_headers, req_body
    ) -> None:
        return None

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int | None = None,
        **_: Any,
    ) -> list[list[float]]:
        self.call_count += 1
        # 1) Pre-request throttle (CCXT calls this inside fetch2 with the endpoint's cost).
        #    Once the hooks are installed this acquires ``throttle_cost`` from the rate limiter.
        await self.throttle(self.throttle_cost)

        now = time.monotonic()
        # drop arrivals outside the sliding window
        cutoff = now - self.window_sec
        self._arrivals = [t for t in self._arrivals if t > cutoff]
        if len(self._arrivals) >= self.max_rps * self.window_sec:
            self.rate_limit_hits += 1
            raise ccxt.RateLimitExceeded(
                f"fake: Too Many Requests "
                f"({len(self._arrivals)}/{int(self.max_rps * self.window_sec)} "
                f"in {self.window_sec}s)"
            )
        self._arrivals.append(now)
        await asyncio.sleep(self._latency_s)

        # Publish rate-limit headers as a real exchange would. Casing is deliberately neither
        # the canonical constant nor lowercase — the parsers' lookup must be case-insensitive.
        response_headers: dict[str, str] = {}
        if self._server_budget is not None:
            self._server_budget = max(0, self._server_budget - 1)
            self._used_weight += 1
            response_headers = {
                # no parser reads this one — it rides along as a negative control
                "x-ratelimit-remaining": str(self._server_budget),
                "X-Mbx-Used-Weight-1m": str(self._used_weight),  # Binance style
            }
            self.last_response_headers = self.stale_headers or response_headers

        # 2) Post-response hook (CCXT calls this inside fetch2 with THIS request's headers).
        #    Once the hooks are installed this invokes the header parser, which calls
        #    ``rate_limiter.sync_from_exchange(...)``.
        self.on_rest_response(
            200, "OK", f"https://fake/{symbol}", "GET",
            response_headers, "{}", {}, None,
        )

        # Simulate OKX-style behavior: cap each page at ``bars_per_page`` regardless of
        # caller's ``limit`` — forces pagination to iterate.
        n = min(limit or self.bars_per_page, self.bars_per_page)
        start = since or 0
        return [
            [start + i * TF_MS, 50_000.0, 50_100.0, 49_900.0, 50_050.0, 100.0]
            for i in range(n)
        ]


def _build_rate_limiter(*, capacity: int, refill_rate: float, cooldown: float = 0.2) -> ExchangeRateLimiter:
    """Single-pool IP-scoped rate limiter on the ``ccxt_rest`` endpoint."""
    config = ExchangeRateLimitConfig(
        pools={
            "ccxt_rest": PoolConfig(
                "ccxt_rest", "ip", capacity, refill_rate, cooldown=cooldown
            ),
        },
        endpoint_map={"rest": EndpointCosts([("ccxt_rest", 1)])},
        default_costs=EndpointCosts([("ccxt_rest", 1)]),
    )
    return ExchangeRateLimiter("FAKE.X", config)


def _build_storage(
    *,
    exchange: FakeCcxtExchange,
    limiter: ExchangeRateLimiter | None,
    retry_max_attempts: int = 5,
    retry_base_delay: float = 0.0,
    retry_jitter: float = 0.0,
    strict_fetch: bool = False,
) -> CcxtStorage:
    """
    Wire a FakeCcxtExchange into a CcxtStorage without touching the network.

    Retry defaults are zeroed so tests don't spend real wall-clock time in
    backoff. Production defaults live on the ``CcxtStorage`` constructor.

    ``strict_fetch`` defaults to False here so that tests which deliberately
    trigger 429s (the negative control in ``TestCcxtStorageRateLimitBudget``)
    don't explode with ``CcxtFetchExhausted``; each test opts-in when it
    wants the strict behavior.

    Injecting ``_exchanges`` bypasses ``_ensure_exchange``, so the hook install it does must be
    mirrored here — without it nothing throttles the storage path and the whole file goes vacuous.
    """
    storage = CcxtStorage(
        retry_max_attempts=retry_max_attempts,
        retry_base_delay=retry_base_delay,
        retry_jitter=retry_jitter,
        strict_fetch=strict_fetch,
    )
    storage._exchanges = {"FAKE.X": exchange}  # type: ignore[assignment]
    storage._rate_limiters = {"FAKE.X": limiter} if limiter is not None else {}
    install_rate_limiter_hooks(exchange, limiter, label="FAKE.X")
    return storage


def _bar_span(n_bars: int) -> tuple[int, int]:
    """Pick a [since, until] window that forces exactly n_bars across pagination."""
    since = 1_700_000_000_000  # arbitrary fixed epoch-ms
    until = since + (n_bars - 1) * TF_MS
    return since, until


def _span_for_pages(n_pages: int, bars_per_page: int = 10) -> tuple[int, int]:
    """Window that makes the pagination loop issue exactly *n_pages* fetches.

    Page k ends at ``since + (k*bpp - 1)*TF + (k-1)`` (the cursor bumps by 1ms per page) and the
    loop stops on the first page whose last bar reaches ``until``; half a page of slack lands
    ``until`` strictly between page n-1 and page n.
    """
    since = 1_700_000_000_000
    return since, since + (n_pages * bars_per_page - bars_per_page // 2) * TF_MS


def _instruments(*qubx_syms: str) -> list[tuple[str, str]]:
    """Build a list of (ccxt_symbol, qubx_symbol) tuples for the multi-fetch entry point."""
    return [(f"{s[:-4]}/{s[-4:]}", s) for s in qubx_syms]


# ─── tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestCcxtStorageRateLimitBudget:
    """
    Target scenario: 3 symbols × 5 pages each = 15 fetch calls. The mock
    enforces a 10 req/s budget over a 1-s sliding window. The rate limiter is
    configured to 5 tokens burst + 5/s refill — strictly below the mock's
    threshold so a correctly-applied limiter never trips the mock.

    Under the default ``asyncio.Semaphore(5)`` (concurrency, not rate) the
    unprotected path bursts ~5 calls per few milliseconds and blows past 10
    in 1 s, so the negative control does trip the mock. If it ever stopped
    doing so the positive test would be vacuously passing — hence the paired
    control.
    """

    BARS_PER_PAGE = 10
    PAGES_PER_SYMBOL = 5
    SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

    MOCK_BUDGET_RPS = 10.0
    LIMITER_CAPACITY = 5
    LIMITER_REFILL_RPS = 5.0

    def _spans(self) -> tuple[int, int]:
        return _bar_span(self.BARS_PER_PAGE * self.PAGES_PER_SYMBOL)

    def _fake_exchange(self) -> FakeCcxtExchange:
        return FakeCcxtExchange(
            max_rps=self.MOCK_BUDGET_RPS,
            window_sec=1.0,
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=5.0,
        )

    async def test_rate_limiter_prevents_429_under_concurrent_load(self):
        fake = self._fake_exchange()
        limiter = _build_rate_limiter(
            capacity=self.LIMITER_CAPACITY, refill_rate=self.LIMITER_REFILL_RPS
        )
        storage = _build_storage(exchange=fake, limiter=limiter)

        since, until = self._spans()
        result = await storage._async_fetch_ohlcv_multi(
            fake, _instruments(*self.SYMBOLS), "1h", since, until
        )

        assert set(result.keys()) == set(self.SYMBOLS)
        for sym in self.SYMBOLS:
            assert len(result[sym]) >= self.BARS_PER_PAGE * self.PAGES_PER_SYMBOL, (
                f"{sym}: got {len(result[sym])} bars, "
                f"expected ≥ {self.BARS_PER_PAGE * self.PAGES_PER_SYMBOL} — "
                "rate-limiter may have forced a premature exit"
            )
        assert fake.rate_limit_hits == 0, (
            f"rate limiter failed to prevent 429s: "
            f"{fake.rate_limit_hits} hits out of {fake.call_count} calls"
        )

    async def test_without_rate_limiter_triggers_429s(self):
        """Negative control: same load without the limiter overwhelms the budget."""
        fake = self._fake_exchange()
        storage = _build_storage(exchange=fake, limiter=None)

        since, until = self._spans()
        # ``gather(return_exceptions=True)`` inside the storage swallows 429s into [],
        # which is exactly the silent-degradation bug from #264. We only care here
        # that the mock *observed* the 429 pressure.
        await storage._async_fetch_ohlcv_multi(
            fake, _instruments(*self.SYMBOLS), "1h", since, until
        )

        assert fake.rate_limit_hits > 0, (
            "test harness is broken: without rate limiter the mock should observe "
            f"429s, but got 0 (call_count={fake.call_count})"
        )

    async def test_limiter_acquire_called_for_every_page_fetch(self, monkeypatch):
        fake = self._fake_exchange()
        limiter = _build_rate_limiter(
            capacity=self.LIMITER_CAPACITY, refill_rate=self.LIMITER_REFILL_RPS
        )
        storage = _build_storage(exchange=fake, limiter=limiter)

        acquire_calls: list[str] = []
        real_acquire = limiter.acquire

        async def _spy(endpoint: str, **kw):
            acquire_calls.append(endpoint)
            await real_acquire(endpoint, **kw)

        monkeypatch.setattr(limiter, "acquire", _spy)

        since, until = self._spans()
        await storage._async_fetch_ohlcv_multi(
            fake, _instruments(*self.SYMBOLS), "1h", since, until
        )

        expected_min = len(self.SYMBOLS) * self.PAGES_PER_SYMBOL
        assert len(acquire_calls) >= expected_min, (
            f"expected ≥{expected_min} acquire calls, got {len(acquire_calls)}"
        )
        assert all(e == "rest" for e in acquire_calls), f"unexpected endpoints: {set(acquire_calls) - {'rest'}}"


@pytest.mark.asyncio
class TestStorageChargesRealCcxtCost:
    """
    The storage no longer charges a flat 1 per page: CCXT hands ``throttle`` the endpoint's real
    cost (binance.um klines at limit=1000 is weight 5) and the override forwards it verbatim as
    ``weight_override``. Charging 1 there under-counts the IP budget five-fold.
    """

    BARS_PER_PAGE = 10
    PAGES = 3

    @pytest.mark.parametrize("cost", [5.0, 1.0])
    async def test_consumed_is_pages_times_the_ccxt_cost(self, cost: float):
        fake = FakeCcxtExchange(
            max_rps=1000, bars_per_page=self.BARS_PER_PAGE, latency_ms=0.0, throttle_cost=cost
        )
        limiter = _build_rate_limiter(capacity=100, refill_rate=1000.0)
        storage = _build_storage(exchange=fake, limiter=limiter)

        since, until = _span_for_pages(self.PAGES, self.BARS_PER_PAGE)
        await storage._async_fetch_ohlcv_multi(fake, [("BTC/USDT", "BTCUSDT")], "1h", since, until)

        assert fake.call_count == self.PAGES
        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None
        assert state["consumed"] == pytest.approx(self.PAGES * cost)

    async def test_nothing_is_consumed_without_a_limiter(self):
        """Negative control: the charge originates in the hook, so no limiter means no hook."""
        fake = FakeCcxtExchange(
            max_rps=1000, bars_per_page=self.BARS_PER_PAGE, latency_ms=0.0, throttle_cost=5.0
        )
        limiter = _build_rate_limiter(capacity=100, refill_rate=1000.0)
        storage = _build_storage(exchange=fake, limiter=None)

        since, until = _span_for_pages(self.PAGES, self.BARS_PER_PAGE)
        await storage._async_fetch_ohlcv_multi(fake, [("BTC/USDT", "BTCUSDT")], "1h", since, until)

        assert fake.call_count == self.PAGES
        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None
        assert state["consumed"] == 0.0


class TestEnsureExchangeInstallsHooks:
    """``load_markets()`` is itself a REST call (a heavy one on binance), so the hooks have to be
    on the exchange before it is submitted."""

    def _fake_with_load_markets(self, monkeypatch) -> tuple[FakeCcxtExchange, list[bool]]:
        fake = FakeCcxtExchange(max_rps=1000, latency_ms=0.0)
        default_throttle = fake.throttle
        throttled_at_load: list[bool] = []

        async def _load_markets(*_a: Any, **_kw: Any) -> dict:
            throttled_at_load.append(fake.throttle is not default_throttle)
            return {}

        fake.load_markets = _load_markets  # type: ignore[attr-defined]
        # ``_ensure_exchange`` imports the factory inside the function, so patch the module attr
        monkeypatch.setattr(factory, "get_ccxt_exchange", lambda *_a, **_kw: fake)
        return fake, throttled_at_load

    def test_hooks_are_installed_before_load_markets(self, monkeypatch):
        fake, throttled_at_load = self._fake_with_load_markets(monkeypatch)
        limiter = _build_rate_limiter(capacity=100, refill_rate=100.0)
        storage = CcxtStorage(rate_limiters={"FAKE.X": limiter})
        try:
            assert storage._ensure_exchange("FAKE.X") is fake
        finally:
            storage.close()

        assert throttled_at_load == [True]

    def test_no_limiter_leaves_the_exchange_untouched(self, monkeypatch):
        """Negative control: the flag above tracks the install, not the mere act of connecting."""
        fake, throttled_at_load = self._fake_with_load_markets(monkeypatch)
        storage = CcxtStorage()
        try:
            storage._ensure_exchange("FAKE.X")
        finally:
            storage.close()

        assert throttled_at_load == [False]


# ─── retry helper unit tests ──────────────────────────────────────────────────


class _ThrottlingFailingExchange(FakeCcxtExchange):
    """Fails the first ``fail_times`` calls *after* paying the throttle, like a real 429 does."""

    def __init__(self, *, fail_times: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fail_times = fail_times

    async def fetch_ohlcv(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        if self.call_count < self._fail_times:
            self.call_count += 1
            await self.throttle(self.throttle_cost)
            raise ccxt.NetworkError(f"boom on attempt {self.call_count}")
        return await super().fetch_ohlcv(*args, **kwargs)


@pytest.mark.asyncio
class TestRetryableFetch:
    """
    Exercises the ``_retryable_fetch`` helper directly, using an injected
    ``sleep`` to keep the tests deterministic (no wall-clock waits).
    """

    async def _no_sleep(self, _delay: float) -> None:
        return None

    async def test_returns_result_on_first_attempt(self):
        calls = 0

        async def call():
            nonlocal calls
            calls += 1
            return "ok"

        result = await _retryable_fetch(call, sleep=self._no_sleep)
        assert result == "ok"
        assert calls == 1

    async def test_retries_then_succeeds_on_rate_limit(self):
        attempts = 0

        async def call():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ccxt.RateLimitExceeded(f"attempt {attempts}: too many")
            return "eventually"

        result = await _retryable_fetch(
            call, max_attempts=5, base_delay=0.0, jitter=0.0, sleep=self._no_sleep
        )
        assert result == "eventually"
        assert attempts == 3

    async def test_retries_on_generic_network_error(self):
        attempts = 0

        async def call():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ccxt.NetworkError("transient DNS blip")
            return 42

        result = await _retryable_fetch(
            call, max_attempts=3, base_delay=0.0, jitter=0.0, sleep=self._no_sleep
        )
        assert result == 42
        assert attempts == 2

    async def test_retries_on_timeout_error(self):
        attempts = 0

        async def call():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise asyncio.TimeoutError()
            return "done"

        result = await _retryable_fetch(
            call, max_attempts=3, base_delay=0.0, jitter=0.0, sleep=self._no_sleep
        )
        assert result == "done"
        assert attempts == 2

    async def test_exhausts_attempts_and_raises_last_error(self):
        attempts = 0

        async def call():
            nonlocal attempts
            attempts += 1
            raise ccxt.RateLimitExceeded(f"attempt {attempts}")

        with pytest.raises(ccxt.RateLimitExceeded, match="attempt 4"):
            await _retryable_fetch(
                call, max_attempts=4, base_delay=0.0, jitter=0.0, sleep=self._no_sleep
            )
        assert attempts == 4

    async def test_permanent_error_is_not_retried(self):
        attempts = 0

        async def call():
            nonlocal attempts
            attempts += 1
            raise ccxt.BadSymbol("unknown symbol XYZ")

        with pytest.raises(ccxt.BadSymbol):
            await _retryable_fetch(
                call, max_attempts=5, base_delay=0.0, jitter=0.0, sleep=self._no_sleep
            )
        assert attempts == 1  # no retries on permanent error

    async def test_non_ccxt_exception_is_not_retried(self):
        attempts = 0

        async def call():
            nonlocal attempts
            attempts += 1
            raise ValueError("programmer error")

        with pytest.raises(ValueError):
            await _retryable_fetch(
                call, max_attempts=5, base_delay=0.0, jitter=0.0, sleep=self._no_sleep
            )
        assert attempts == 1

    async def test_does_not_acquire_the_limiter_directly(self):
        """Acquisition lives in the throttle override inside ``fetch2``; a second acquire here
        would double-charge every attempt."""
        attempts = 0

        class _ExplodingLimiter:
            async def acquire(self, *a: Any, **kw: Any) -> None:
                raise AssertionError("_retryable_fetch must not acquire — the throttle hook does")

            def report_limit_hit(self, **kw: Any) -> None:
                pass

        async def call():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ccxt.NetworkError("boom")
            return "ok"

        result = await _retryable_fetch(
            call,
            rate_limiter=_ExplodingLimiter(),
            max_attempts=5,
            base_delay=0.0,
            jitter=0.0,
            sleep=self._no_sleep,
        )
        assert result == "ok"
        assert attempts == 3

    async def test_every_attempt_is_charged_through_the_throttle(self):
        """Retries are not free: each attempt re-enters ``fetch2`` and pays the endpoint's cost."""
        fake = _ThrottlingFailingExchange(fail_times=2, max_rps=1000, latency_ms=0.0, throttle_cost=2.0)
        limiter = _build_rate_limiter(capacity=100, refill_rate=1000.0)
        install_rate_limiter_hooks(fake, limiter, label="FAKE.X")

        result = await _retryable_fetch(
            lambda: fake.fetch_ohlcv("BTC/USDT", "1h"),
            rate_limiter=limiter,
            max_attempts=5,
            base_delay=0.0,
            jitter=0.0,
            sleep=self._no_sleep,
        )
        assert len(result) > 0
        assert fake.call_count == 3
        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None
        assert state["consumed"] == pytest.approx(3 * 2.0)

    async def test_nothing_is_charged_without_the_throttle_hook(self):
        """Negative control for the test above: the charge comes from the installed hook, not
        from ``_retryable_fetch``."""
        fake = _ThrottlingFailingExchange(fail_times=2, max_rps=1000, latency_ms=0.0, throttle_cost=2.0)
        limiter = _build_rate_limiter(capacity=100, refill_rate=1000.0)

        await _retryable_fetch(
            lambda: fake.fetch_ohlcv("BTC/USDT", "1h"),
            rate_limiter=limiter,
            max_attempts=5,
            base_delay=0.0,
            jitter=0.0,
            sleep=self._no_sleep,
        )
        assert fake.call_count == 3
        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None
        assert state["consumed"] == 0.0

    async def test_reports_rate_limit_hit_to_limiter(self):
        hits: list[dict[str, Any]] = []

        class _Limiter:
            async def acquire(self, endpoint: str) -> None:
                pass

            def report_limit_hit(self, **kw: Any) -> None:
                hits.append(kw)

        attempts = 0

        async def call():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ccxt.RateLimitExceeded("50011: Too Many Requests")
            return "ok"

        await _retryable_fetch(
            call,
            rate_limiter=_Limiter(),
            rate_limit_pool="ccxt_rest",
            context="OKX BTCUSDT",
            max_attempts=3,
            base_delay=0.0,
            jitter=0.0,
            sleep=self._no_sleep,
        )
        assert len(hits) == 1
        assert hits[0]["pool_name"] == "ccxt_rest"
        # ``endpoint=`` would close every pool in that endpoint's cost list — including ``orders``,
        # stalling the write path on a read-path 429.
        assert "endpoint" not in hits[0]
        assert "OKX BTCUSDT" in hits[0]["reason"]

    async def test_non_rate_limit_errors_do_not_report_to_limiter(self):
        """NetworkError but not RateLimitExceeded should retry without closing the gate."""
        hits: list[dict] = []

        class _Limiter:
            async def acquire(self, endpoint: str) -> None:
                pass

            def report_limit_hit(self, **kw: Any) -> None:
                hits.append(kw)

        attempts = 0

        async def call():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ccxt.NetworkError("DNS glitch")
            return "ok"

        await _retryable_fetch(
            call, rate_limiter=_Limiter(), max_attempts=3, base_delay=0.0, jitter=0.0,
            sleep=self._no_sleep,
        )
        assert hits == []

    async def test_backoff_delays_are_exponential_with_cap(self):
        """Delay sequence should be base * 2^(attempt-1), capped at max_delay."""
        delays: list[float] = []

        async def capture_sleep(d: float) -> None:
            delays.append(d)

        attempts = 0

        async def call():
            nonlocal attempts
            attempts += 1
            raise ccxt.NetworkError(f"attempt {attempts}")

        with pytest.raises(ccxt.NetworkError):
            await _retryable_fetch(
                call,
                max_attempts=6,
                base_delay=1.0,
                max_delay=8.0,
                jitter=0.0,
                sleep=capture_sleep,
            )
        # attempts 1..5 each sleep before next (attempt 6 fails-final, no sleep).
        # expected base*2^0, base*2^1, ..., capped at max_delay
        assert delays == [1.0, 2.0, 4.0, 8.0, 8.0]


# ─── end-to-end flaky exchange test (task #3) ────────────────────────────────


class FlakyFakeExchange(FakeCcxtExchange):
    """
    Fake exchange that randomly fails a configurable fraction of ``fetch_ohlcv``
    calls with ``ccxt.RateLimitExceeded`` (independent of the req/s budget).
    Used to verify that retry logic drives every symbol to completion.
    """

    def __init__(
        self,
        *,
        failure_rate: float,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._rng = random.Random(seed)
        self._failure_rate = failure_rate
        self.injected_failures = 0

    async def fetch_ohlcv(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        # Budget-check and normal fetch first — if the real budget trips, let it propagate.
        if self._rng.random() < self._failure_rate:
            self.call_count += 1       # this is a real request attempt from the exchange's POV
            self.injected_failures += 1
            raise ccxt.RateLimitExceeded(
                f"flaky: injected 50011 at call #{self.call_count}"
            )
        return await super().fetch_ohlcv(*args, **kwargs)


@pytest.mark.asyncio
class TestCcxtStorageFlakyExchangeRetries:
    """
    End-to-end retry behavior: the exchange rejects ~40% of fetch calls with
    a RateLimitExceeded. Every symbol must still return full OHLCV after the
    retry loop takes over.
    """

    BARS_PER_PAGE = 10
    PAGES_PER_SYMBOL = 3
    SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

    def _spans(self) -> tuple[int, int]:
        return _bar_span(self.BARS_PER_PAGE * self.PAGES_PER_SYMBOL)

    async def test_all_symbols_succeed_despite_flaky_exchange(self):
        flaky = FlakyFakeExchange(
            failure_rate=0.4,
            seed=1234,
            max_rps=1000,  # effectively unlimited; we want to isolate injected failures
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=1.0,
        )
        # Rate limiter with a generous bucket — not the subject under test here.
        limiter = _build_rate_limiter(capacity=1000, refill_rate=1000.0)
        storage = _build_storage(exchange=flaky, limiter=limiter)

        since, until = self._spans()
        result = await storage._async_fetch_ohlcv_multi(
            flaky, _instruments(*self.SYMBOLS), "1h", since, until
        )

        assert flaky.injected_failures > 0, (
            "test harness: failure injection is expected to fire at least once"
        )
        for sym in self.SYMBOLS:
            assert len(result[sym]) >= self.BARS_PER_PAGE * self.PAGES_PER_SYMBOL, (
                f"{sym}: got {len(result[sym])} bars despite retry — injected={flaky.injected_failures} "
                f"total_calls={flaky.call_count}"
            )

    async def test_persistent_failures_raise_explicitly_in_strict_mode(self):
        """In strict_fetch mode (default for warmup), exhausted retries raise
        :class:`CcxtFetchExhausted` — callers can't silently proceed on zero bars.
        This is the primary fix for the #264 silent-degradation bug.
        """
        flaky = FlakyFakeExchange(
            failure_rate=1.0,  # 100% — every call rejected
            seed=7,
            max_rps=1000,
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=0.0,
        )
        limiter = _build_rate_limiter(capacity=1000, refill_rate=1000.0)
        storage = _build_storage(exchange=flaky, limiter=limiter, strict_fetch=True)

        since, until = self._spans()
        with pytest.raises(CcxtFetchExhausted) as excinfo:
            await storage._async_fetch_ohlcv_multi(
                flaky, _instruments(*self.SYMBOLS), "1h", since, until
            )

        assert set(excinfo.value.failures) == set(self.SYMBOLS)
        assert excinfo.value.total_requested == len(self.SYMBOLS)
        # each symbol should report a RateLimitExceeded as its cause
        for sym, exc in excinfo.value.failures.items():
            assert isinstance(exc, ccxt.RateLimitExceeded), (
                f"{sym}: expected RateLimitExceeded, got {type(exc).__name__}"
            )

    async def test_lenient_mode_returns_empty_on_exhaustion(self):
        """With ``strict_fetch=False`` the old behavior is preserved: failed
        symbols get ``[]`` and the caller can inspect logs to diagnose.
        """
        flaky = FlakyFakeExchange(
            failure_rate=1.0,
            seed=7,
            max_rps=1000,
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=0.0,
        )
        limiter = _build_rate_limiter(capacity=1000, refill_rate=1000.0)
        storage = _build_storage(exchange=flaky, limiter=limiter, strict_fetch=False)

        since, until = self._spans()
        result = await storage._async_fetch_ohlcv_multi(
            flaky, [("BTC/USDT", "BTCUSDT")], "1h", since, until
        )

        assert result == {"BTCUSDT": []}
        from qubx.data.storages.ccxt import _RETRY_MAX_ATTEMPTS
        assert flaky.call_count == _RETRY_MAX_ATTEMPTS, (
            f"expected exactly {_RETRY_MAX_ATTEMPTS} attempts, got {flaky.call_count}"
        )

    async def test_partial_failure_raises_in_strict_mode(self):
        """Even if *some* symbols succeed, strict mode raises for the failing ones.
        The ``failures`` map on the exception names exactly the problem symbols.
        """
        # Seeded rng + 50% rate + the scheduled symbol order → deterministic mix
        flaky = FlakyFakeExchange(
            failure_rate=1.0,
            seed=11,
            max_rps=1000,
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=0.0,
        )
        # Patch: fail only for one specific symbol to get a partial outcome.

        async def selective_fetch(symbol, timeframe, **kw):
            if symbol == "LINK/USDT":
                flaky.call_count += 1
                flaky.injected_failures += 1
                raise ccxt.RateLimitExceeded("flaky: LINK/USDT always fails")
            return await FakeCcxtExchange.fetch_ohlcv(flaky, symbol, timeframe, **kw)

        flaky.fetch_ohlcv = selective_fetch  # type: ignore[assignment]

        limiter = _build_rate_limiter(capacity=1000, refill_rate=1000.0)
        storage = _build_storage(exchange=flaky, limiter=limiter, strict_fetch=True)

        since, until = self._spans()
        with pytest.raises(CcxtFetchExhausted) as excinfo:
            await storage._async_fetch_ohlcv_multi(
                flaky,
                [("BTC/USDT", "BTCUSDT"), ("LINK/USDT", "LINKUSDT"), ("ETH/USDT", "ETHUSDT")],
                "1h", since, until,
            )

        # only LINKUSDT failed; others completed normally (proof failures dict is precise)
        assert set(excinfo.value.failures) == {"LINKUSDT"}
        assert excinfo.value.total_requested == 3


# ─── response-header sync (Option C from the #264 investigation) ─────────────


@pytest.mark.asyncio
class TestResponseHeaderSync:
    """
    Verify that the ``on_rest_response`` hook installed by ``_ensure_exchange`` invokes the
    exchange's header parser for every successful page fetch, and that the parsed budget
    actually moves the rate limiter's modeled state.

    The hook runs inside CCXT's fetch pipeline with THIS request's headers — unlike the deleted
    best-effort read of the exchange-wide ``last_response_headers``, which under concurrency
    could belong to a neighbouring request.
    """

    BARS_PER_PAGE = 10
    PAGES_PER_SYMBOL = 3

    def _spans(self) -> tuple[int, int]:
        return _bar_span(self.BARS_PER_PAGE * self.PAGES_PER_SYMBOL)

    @staticmethod
    def _spy_sync(limiter: ExchangeRateLimiter) -> list[dict[str, Any]]:
        """Record every ``sync_from_exchange`` call while still applying it."""
        calls: list[dict[str, Any]] = []
        real_sync = limiter.sync_from_exchange

        def _spy(pool_name: str, **kw: Any) -> None:
            calls.append({"pool_name": pool_name, **kw})
            real_sync(pool_name, **kw)

        limiter.sync_from_exchange = _spy  # type: ignore[method-assign]
        return calls

    @staticmethod
    def _drive_hook(fake: FakeCcxtExchange, headers: dict[str, str]) -> None:
        """Invoke the installed post-response hook exactly as CCXT's fetch pipeline does."""
        fake.on_rest_response(200, "OK", "https://fake/BTC-USDT", "GET", headers, "{}", {}, None)

    async def test_headers_drive_sync_once_per_page(self):
        fake = FakeCcxtExchange(
            id="binanceusdm",
            max_rps=1000,
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=0.0,
            server_budget=20,
        )
        limiter = _build_rate_limiter(capacity=1000, refill_rate=1000.0)
        storage = _build_storage(exchange=fake, limiter=limiter)
        # The sync must ride the installed hook — a bound method would still be the fake's no-op.
        assert getattr(fake.on_rest_response, "__self__", None) is None

        sync_calls = self._spy_sync(limiter)

        since, until = self._spans()
        await storage._async_fetch_ohlcv_multi(
            fake, [("BTC/USDT", "BTCUSDT")], "1h", since, until
        )

        # Sync happens once per successful page. Pagination issues at least
        # ``PAGES_PER_SYMBOL`` pages for the requested window (and sometimes one
        # extra tail page due to the ``last_ts + 1`` cursor bump — we only care
        # that it's > 1 and matches the exchange's actual call count).
        assert len(sync_calls) == fake.call_count
        assert len(sync_calls) >= self.PAGES_PER_SYMBOL
        assert all(c["pool_name"] == "ccxt_rest" for c in sync_calls)
        # used weight climbs by exactly one per successful call
        assert [c["used"] for c in sync_calls] == list(range(1, fake.call_count + 1))

    async def test_binance_headers_drive_sync_from_exchange(self):
        """Binance reports *used* weight, in whatever casing aiohttp handed ccxt."""
        fake = FakeCcxtExchange(
            id="binanceusdm",
            max_rps=1000,
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=0.0,
            server_budget=500,
            throttle_cost=5.0,  # so the pinned level differs from what the acquires alone leave
        )
        limiter = _build_rate_limiter(capacity=100, refill_rate=1.0)
        storage = _build_storage(exchange=fake, limiter=limiter)
        sync_calls = self._spy_sync(limiter)

        since, until = _span_for_pages(3, self.BARS_PER_PAGE)
        await storage._async_fetch_ohlcv_multi(
            fake, [("BTC/USDT", "BTCUSDT")], "1h", since, until
        )

        # neither the canonical constant nor lowercase — the parser's lookup is case-insensitive
        assert "X-Mbx-Used-Weight-1m" in fake.last_response_headers
        assert "X-MBX-USED-WEIGHT-1M" not in fake.last_response_headers
        assert fake.call_count == 3
        assert [c["used"] for c in sync_calls] == [1, 2, 3]
        assert all(c["pool_name"] == "ccxt_rest" for c in sync_calls)

        await asyncio.sleep(0.01)  # RatePool.sync applies through a task
        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None
        assert state["consumed"] == pytest.approx(3 * 5.0)
        # pinned to capacity - used by the header, not to the 100 - 15 the acquires alone leave
        assert state["remaining"] == pytest.approx(97.0, abs=2.0)

    async def test_sync_reads_the_hook_headers_not_the_stale_attribute(self):
        """``last_response_headers`` is exchange-wide and lags; the hook argument is this
        request's truth. Point them at different budgets and the limiter must see the hook's."""
        fake = FakeCcxtExchange(
            id="binanceusdm",
            max_rps=1000,
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=0.0,
            server_budget=50,
            stale_headers={"X-Mbx-Used-Weight-1m": "999"},
        )
        limiter = _build_rate_limiter(capacity=1000, refill_rate=1000.0)
        storage = _build_storage(exchange=fake, limiter=limiter)
        sync_calls = self._spy_sync(limiter)

        since, until = _span_for_pages(3, self.BARS_PER_PAGE)
        await storage._async_fetch_ohlcv_multi(
            fake, [("BTC/USDT", "BTCUSDT")], "1h", since, until
        )

        assert fake.last_response_headers["X-Mbx-Used-Weight-1m"] == "999"  # the decoy is in place
        assert [c["used"] for c in sync_calls] == [1, 2, 3]

    async def test_header_sync_moves_pool_state(self):
        fake = FakeCcxtExchange(id="binanceusdm", max_rps=1000, latency_ms=0.0)
        limiter = _build_rate_limiter(capacity=100, refill_rate=1.0)
        install_rate_limiter_hooks(fake, limiter, label="FAKE.X")

        await limiter.acquire("rest")  # the bucket only exists once something acquires
        before = await limiter.get_pool_state("ccxt_rest")
        assert before is not None
        assert before["remaining"] == pytest.approx(99.0, abs=1.0)

        self._drive_hook(fake, {"X-Mbx-Used-Weight-1m": "93"})  # 100 capacity - 93 used
        await asyncio.sleep(0.01)

        after = await limiter.get_pool_state("ccxt_rest")
        assert after is not None
        assert after["remaining"] == pytest.approx(7.0, abs=1.0)

    async def test_headerless_response_leaves_pool_state_alone(self):
        """Negative control for the two tests above: no headers, no movement."""
        fake = FakeCcxtExchange(id="binanceusdm", max_rps=1000, latency_ms=0.0)
        limiter = _build_rate_limiter(capacity=100, refill_rate=1.0)
        install_rate_limiter_hooks(fake, limiter, label="FAKE.X")

        await limiter.acquire("rest")
        self._drive_hook(fake, {})
        await asyncio.sleep(0.01)

        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None
        assert state["remaining"] == pytest.approx(99.0, abs=1.0)

    async def test_header_sync_creates_a_never_acquired_bucket(self):
        """``set_remaining`` now seeds the bucket from the pool config, so a header that lands
        before the first acquire is applied instead of discarded."""
        fake = FakeCcxtExchange(id="binanceusdm", max_rps=1000, latency_ms=0.0)
        limiter = _build_rate_limiter(capacity=100, refill_rate=1.0)
        install_rate_limiter_hooks(fake, limiter, label="FAKE.X")

        untouched = await limiter.get_pool_state("ccxt_rest")
        assert untouched is not None
        assert untouched["remaining"] == 100.0  # nothing acquired yet -> capacity fallback

        self._drive_hook(fake, {"X-Mbx-Used-Weight-1m": "88"})  # 100 capacity - 88 used
        await asyncio.sleep(0.01)

        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None
        assert state["remaining"] == pytest.approx(12.0, abs=1.0)

    async def test_unknown_exchange_skips_sync_silently(self):
        """If the exchange id has no registered parser, sync is a no-op (no error)."""
        fake = FakeCcxtExchange(
            id="exchange-without-a-parser",
            max_rps=1000,
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=0.0,
            server_budget=20,
        )
        limiter = _build_rate_limiter(capacity=1000, refill_rate=1000.0)
        storage = _build_storage(exchange=fake, limiter=limiter)

        sync_calls: list[Any] = []
        limiter.sync_from_exchange = lambda *a, **kw: sync_calls.append((a, kw))  # type: ignore[method-assign]

        since, until = self._spans()
        result = await storage._async_fetch_ohlcv_multi(
            fake, [("BTC/USDT", "BTCUSDT")], "1h", since, until
        )
        # fetch still succeeded; sync just didn't happen
        assert len(result["BTCUSDT"]) >= self.BARS_PER_PAGE * self.PAGES_PER_SYMBOL
        assert sync_calls == []

    async def test_empty_headers_skip_sync_silently(self):
        """When the exchange hasn't populated headers yet, sync is a no-op."""
        fake = FakeCcxtExchange(
            id="binanceusdm",
            max_rps=1000,
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=0.0,
            server_budget=None,  # disables header publishing
        )
        limiter = _build_rate_limiter(capacity=1000, refill_rate=1000.0)
        storage = _build_storage(exchange=fake, limiter=limiter)

        sync_calls: list[Any] = []
        limiter.sync_from_exchange = lambda *a, **kw: sync_calls.append((a, kw))  # type: ignore[method-assign]

        since, until = self._spans()
        await storage._async_fetch_ohlcv_multi(
            fake, [("BTC/USDT", "BTCUSDT")], "1h", since, until
        )
        assert sync_calls == []

    async def test_parser_exception_is_swallowed(self, monkeypatch):
        """A buggy parser must not break the fetch — failures are DEBUG-level."""
        from qubx.connectors.ccxt import rate_limits

        def _broken_parser(headers: dict, rl: Any) -> None:
            raise RuntimeError("simulated parser bug")

        monkeypatch.setitem(rate_limits.HEADER_PARSERS, "binanceusdm", _broken_parser)

        fake = FakeCcxtExchange(
            id="binanceusdm",
            max_rps=1000,
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=0.0,
            server_budget=20,
        )
        limiter = _build_rate_limiter(capacity=1000, refill_rate=1000.0)
        storage = _build_storage(exchange=fake, limiter=limiter)

        since, until = self._spans()
        # Should complete normally despite the broken parser.
        result = await storage._async_fetch_ohlcv_multi(
            fake, [("BTC/USDT", "BTCUSDT")], "1h", since, until
        )
        assert len(result["BTCUSDT"]) >= self.BARS_PER_PAGE * self.PAGES_PER_SYMBOL

    async def test_no_rate_limiter_skips_sync(self):
        """When no limiter is attached, don't even attempt to parse headers."""
        fake = FakeCcxtExchange(
            id="binanceusdm",
            max_rps=1000,
            bars_per_page=self.BARS_PER_PAGE,
            latency_ms=0.0,
            server_budget=20,
        )
        storage = _build_storage(exchange=fake, limiter=None)

        since, until = self._spans()
        # Exercise the code path — should simply not error.
        result = await storage._async_fetch_ohlcv_multi(
            fake, [("BTC/USDT", "BTCUSDT")], "1h", since, until
        )
        assert len(result["BTCUSDT"]) >= self.BARS_PER_PAGE * self.PAGES_PER_SYMBOL


# ─── cross-subsystem E2E stress test (xLydianSoftware/Qubx#264) ──────────────


@pytest.mark.asyncio
class TestRateLimiterE2ECrossSubsystem:
    """
    Full-stack stress test of the rate limiter with BOTH consumers:

    * ``CcxtStorage`` (warmup / historical REST fetch path, test-wired directly).
    * ``ExchangeManager`` (live REST path, attaches its throttle + on_rest_response
      hooks to the shared fake exchange).

    Both hang on the same ``ExchangeRateLimiter`` instance and the same budget-
    enforcing ``FakeCcxtExchange``. Fires concurrent REST traffic from both paths
    and verifies end-to-end:

    * Combined arrival rate never trips the exchange's 429 budget.
    * ``limiter.acquire`` fires for every request across both paths — both now go
      through the throttle override, the storage's via ``install_rate_limiter_hooks``.
    * ``limiter.sync_from_exchange`` is called from both paths and decrements
      a shared remaining-budget view.

    The test is self-contained — no network, no real CCXT exchange instance.
    """

    def _wire_manager(self, fake: FakeCcxtExchange, limiter: ExchangeRateLimiter) -> ExchangeManager:
        """Attach *limiter* through the real ``attach_rate_limiter`` code path."""
        manager = ExchangeManager(
            exchange_name="binance.um",
            factory_params={"exchange": "binance.um"},
            initial_exchange=fake,
            health_monitor=DummyHealthMonitor(),
            time_provider=LiveTimeProvider(),
        )
        manager.attach_rate_limiter(limiter)
        return manager

    async def test_shared_limiter_across_storage_and_manager(self):
        # --- Budget design -----------------------------------------------------
        # Mock exchange: 20 req/s budget. Limiter: 5 burst + 5/s refill, so peak
        # throughput over any 1s window is ≤ 10 (half of the mock's 20/s threshold).
        # Gap must stay generous: scheduling jitter can make the limiter's sliding
        # throughput briefly higher than its steady refill rate.
        # No server budget: the header sync re-pins the bucket to ``min(capacity, reported)`` on
        # every response, which would fight the pacing this test measures. Sync is covered below.
        fake = FakeCcxtExchange(
            id="binanceusdm",
            max_rps=20,
            window_sec=1.0,
            bars_per_page=10,
            latency_ms=2.0,
            server_budget=None,
        )
        limiter = _build_rate_limiter(capacity=5, refill_rate=5.0, cooldown=0.2)

        storage = _build_storage(exchange=fake, limiter=limiter)
        # Spy on limiter.acquire to count how many times each path hit the bucket.
        acquire_count = 0
        real_acquire = limiter.acquire

        async def _spy_acquire(endpoint, **kw):
            nonlocal acquire_count
            acquire_count += 1
            await real_acquire(endpoint, **kw)

        limiter.acquire = _spy_acquire  # type: ignore[method-assign]
        manager = self._wire_manager(fake, limiter)

        # --- Concurrent traffic -----------------------------------------------
        storage_symbols = _instruments("BTCUSDT", "ETHUSDT", "SOLUSDT")
        since = 1_700_000_000_000
        until = since + 29 * TF_MS  # ~3 pages per symbol

        async def storage_worker() -> dict[str, list]:
            return await storage._async_fetch_ohlcv_multi(
                fake, storage_symbols, "1h", since, until
            )

        async def manager_worker() -> int:
            """Burst 10 direct REST calls through the ExchangeManager-wrapped exchange."""
            hits = 0
            for i in range(10):
                bars = await manager.exchange.fetch_ohlcv(
                    "BTC/USDT", "1h", since=since + i * TF_MS, limit=1,
                )
                hits += len(bars)
            return hits

        storage_result, manager_bars = await asyncio.gather(
            storage_worker(), manager_worker()
        )

        # --- Assertions -------------------------------------------------------
        # Exchange was never rate-limited — the token bucket stayed below budget.
        assert fake.rate_limit_hits == 0, (
            f"rate-limiter failed to protect the exchange: "
            f"{fake.rate_limit_hits} 429s observed out of {fake.call_count} calls"
        )
        # Both paths produced data.
        assert set(storage_result) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        for sym, bars in storage_result.items():
            assert len(bars) > 0, f"storage path: no bars for {sym}"
        assert manager_bars > 0, "manager path: no bars returned"

        # Every exchange call must have gone through the limiter's acquire() — the
        # throttle override is now the single charging point for both paths.
        assert acquire_count >= fake.call_count, (
            f"acquire fired {acquire_count} times for {fake.call_count} exchange "
            "calls — some requests bypassed the limiter"
        )

    async def test_both_paths_drive_header_sync(self):
        """Every response on either path feeds the shared limiter's modeled budget.

        The limiter is deliberately generous here — this pins the sync wiring, not pacing.
        """
        fake = FakeCcxtExchange(
            id="binanceusdm",
            max_rps=1000,
            window_sec=1.0,
            bars_per_page=10,
            latency_ms=0.0,
            server_budget=1_000,
        )
        limiter = _build_rate_limiter(capacity=1000, refill_rate=1000.0)
        storage = _build_storage(exchange=fake, limiter=limiter)

        sync_count = 0
        real_sync = limiter.sync_from_exchange

        def _spy_sync(pool_name, **kw):
            nonlocal sync_count
            sync_count += 1
            real_sync(pool_name, **kw)

        limiter.sync_from_exchange = _spy_sync  # type: ignore[method-assign]
        manager = self._wire_manager(fake, limiter)

        since, until = _span_for_pages(3)
        await storage._async_fetch_ohlcv_multi(fake, _instruments("BTCUSDT"), "1h", since, until)
        await manager.exchange.fetch_ohlcv("BTC/USDT", "1h", since=since, limit=1)

        assert fake.call_count == 4  # 3 storage pages + 1 manager call
        # Each response drives at least one sync; the shared fake is wrapped by both wirings,
        # so the parser may run more than once per response (separate exchanges in production).
        assert sync_count >= fake.call_count

    async def test_parserless_exchange_drives_no_sync(self):
        """Negative control for the test above: no parser for the id, no sync at all."""
        fake = FakeCcxtExchange(
            id="exchange-without-a-parser", max_rps=1000, latency_ms=0.0, server_budget=1_000
        )
        limiter = _build_rate_limiter(capacity=1000, refill_rate=1000.0)
        storage = _build_storage(exchange=fake, limiter=limiter)

        sync_calls: list[Any] = []
        limiter.sync_from_exchange = lambda *a, **kw: sync_calls.append((a, kw))  # type: ignore[method-assign]
        self._wire_manager(fake, limiter)

        since, until = _span_for_pages(3)
        await storage._async_fetch_ohlcv_multi(fake, _instruments("BTCUSDT"), "1h", since, until)

        assert fake.call_count == 3
        assert sync_calls == []

    async def test_exchange_manager_path_alone_stays_under_budget(self):
        """Isolate the live path: rapid-fire REST calls through ExchangeManager.throttle
        must never exceed the exchange's budget.

        Budget gap matters: rate limiter's peak throughput (burst + refill over the
        mock's window) must stay strictly below the mock's budget, otherwise natural
        scheduling jitter will tip us over. Mock 20/s with limiter at 3 burst + 3/s
        keeps peak throughput ≤ 6/s over any 1-second window.
        """
        fake = FakeCcxtExchange(
            id="okx",
            max_rps=20,
            window_sec=1.0,
            bars_per_page=1,
            latency_ms=1.0,
            server_budget=200,
        )
        limiter = _build_rate_limiter(capacity=3, refill_rate=3.0, cooldown=0.2)

        manager = ExchangeManager(
            exchange_name="okx",
            factory_params={"exchange": "okx"},
            initial_exchange=fake,
            health_monitor=DummyHealthMonitor(),
            time_provider=LiveTimeProvider(),
        )
        manager.attach_rate_limiter(limiter)

        # Fire 20 concurrent requests — well above the mock's 10 req/s.
        N = 20
        results = await asyncio.gather(
            *[
                manager.exchange.fetch_ohlcv("BTC/USDT", "1h", since=1, limit=1)
                for _ in range(N)
            ],
            return_exceptions=True,
        )

        # All completed without raising.
        assert all(not isinstance(r, BaseException) for r in results), (
            f"some fetches raised: {[r for r in results if isinstance(r, BaseException)]}"
        )
        assert fake.rate_limit_hits == 0, (
            f"ExchangeManager path leaked {fake.rate_limit_hits} 429s — throttle "
            "override did not prevent budget breach"
        )
        assert fake.call_count == N
