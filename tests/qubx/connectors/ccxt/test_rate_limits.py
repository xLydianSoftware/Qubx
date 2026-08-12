"""Tests for ``qubx.connectors.ccxt.rate_limits``: venue configs, header parsers, hook installation."""

import inspect
from unittest.mock import Mock, patch

import ccxt.pro as cxp
import pytest

from qubx import logger
from qubx.connectors.ccxt.exchanges import EXCHANGE_ALIASES
from qubx.connectors.ccxt.rate_limits import (
    HEADER_PARSERS,
    _order_count_header,
    create_ccxt_rate_limit_config,
    get_header_parser,
    install_rate_limiter_hooks,
    parse_binance_headers,
)
from qubx.rate_limiting import ExchangeRateLimiter, PoolConfig

_ORDER_ENDPOINTS = ("create_order", "cancel_order", "edit_order")

_VENUES = [
    "binance",
    "binance.um",
    "binance.cm",
    "binance.pm",
    "binance.um.mm",
    "okx",
    "bybit",
    "kraken",
    "kraken.f",
    "not-a-real-venue",
]


class _RecordingLimiter:
    """Records ``sync_from_exchange`` calls. ``config`` is a real one so a parser that probes
    ``config.pools`` sees production shape."""

    def __init__(self, venue: str = "binance.um"):
        self.config = create_ccxt_rate_limit_config(venue)
        self.syncs: list[tuple[str, dict]] = []

    def sync_from_exchange(self, pool_name: str, **kwargs) -> None:
        self.syncs.append((pool_name, kwargs))


class _RecordingThrottle:
    """Records what the installed throttle forwards to ``acquire``."""

    def __init__(self):
        self.calls: list[tuple[str, float | None]] = []

    async def acquire(self, endpoint: str, weight_override: float | None = None) -> None:
        self.calls.append((endpoint, weight_override))


def _mock_exchange(id: str = "no-parser-for-this-id") -> Mock:
    """Mirrors ``_make_bare_mock_exchange`` in test_exchange_manager. ``throttle`` /
    ``on_rest_response`` must be explicit so a replacement is detectable, and ``enableRateLimit``
    starts ``False`` so the installer turning it on is observable."""
    mock = Mock()
    mock.id = id
    mock.enableRateLimit = False
    mock.throttle = Mock(name="throttle_original")
    mock.on_rest_response = Mock(name="on_rest_response_original")
    return mock


@pytest.fixture
def warnings_logged():
    records: list = []
    sink_id = logger.add(lambda m: records.append(m.record), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink_id)


class TestEndpointMap:
    @pytest.mark.parametrize("venue", _VENUES)
    @pytest.mark.parametrize("endpoint", _ORDER_ENDPOINTS)
    def test_order_endpoint_is_mapped(self, venue: str, endpoint: str):
        # unmapped means falling back to default_costs, which would charge the IP pool twice
        assert endpoint in create_ccxt_rate_limit_config(venue).endpoint_map

    # kraken.f has no orders pool — it surcharges the shared budget instead (see TestKrakenConfig)
    @pytest.mark.parametrize("venue", [v for v in _VENUES if v != "kraken.f"])
    @pytest.mark.parametrize("endpoint", _ORDER_ENDPOINTS)
    def test_order_endpoint_charges_only_the_orders_pool(self, venue: str, endpoint: str):
        assert create_ccxt_rate_limit_config(venue).endpoint_map[endpoint].costs == [("orders", 1)]

    @pytest.mark.parametrize("venue", _VENUES)
    def test_rest_endpoint_charges_only_the_ip_pool(self, venue: str):
        # endpoint "rest" -> pool "ccxt_rest"; the 1 is a placeholder ccxt's cost overrides
        assert create_ccxt_rate_limit_config(venue).endpoint_map["rest"].costs == [("ccxt_rest", 1)]

    @pytest.mark.parametrize("venue", _VENUES)
    def test_orders_is_never_reachable_from_default_costs(self, venue: str):
        # negative control: order endpoints must be the only route into the orders pool
        assert [pool for pool, _ in create_ccxt_rate_limit_config(venue).default_costs.costs] == ["ccxt_rest"]

    async def test_order_endpoint_is_a_noop_when_the_venue_has_no_orders_pool(self):
        config = create_ccxt_rate_limit_config("not-a-real-venue")
        assert "orders" not in config.pools
        limiter = ExchangeRateLimiter("not-a-real-venue", config)

        await limiter.acquire("create_order")

        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None and state["consumed"] == 0


class TestBinanceConfigs:
    def test_pm_gets_papi_numbers(self):
        pools = create_ccxt_rate_limit_config("binance.pm").pools
        assert (pools["ccxt_rest"].capacity, pools["ccxt_rest"].refill_rate) == (6000, 100.0)
        assert (pools["orders"].capacity, pools["orders"].refill_rate) == (1200, 20.0)

    def test_pm_order_budget_is_not_the_spot_one(self):
        pm = create_ccxt_rate_limit_config("binance.pm").pools["orders"]
        spot = create_ccxt_rate_limit_config("binance").pools["orders"]
        assert pm.capacity != spot.capacity

    def test_spot_order_budget(self):
        pools = create_ccxt_rate_limit_config("binance").pools
        assert (pools["orders"].capacity, pools["orders"].refill_rate) == (100, 10.0)
        assert (pools["ccxt_rest"].capacity, pools["ccxt_rest"].refill_rate) == (6000, 100.0)

    def test_coin_m_ip_budget_matches_dapi(self):
        # dapi publishes ORDERS MINUTE/1 only — no 10s window, unlike fapi
        pools = create_ccxt_rate_limit_config("binance.cm").pools
        assert (pools["ccxt_rest"].capacity, pools["ccxt_rest"].refill_rate) == (2400, 40.0)
        assert (pools["orders"].capacity, pools["orders"].refill_rate) == (1200, 20.0)

    def test_usd_m_ip_budget_matches_fapi(self):
        pools = create_ccxt_rate_limit_config("binance.um").pools
        assert (pools["ccxt_rest"].capacity, pools["ccxt_rest"].refill_rate) == (2400, 40.0)
        assert (pools["orders"].capacity, pools["orders"].refill_rate) == (300, 30.0)

    def test_market_maker_venue_routes_to_usd_m(self):
        assert create_ccxt_rate_limit_config("binance.um.mm").pools == create_ccxt_rate_limit_config("binance.um").pools

    @pytest.mark.parametrize(
        "ccxt_id, venue",
        [
            ("binancepm", "binance.pm"),
            ("binanceqv_usdm", "binance.um"),
            ("binance_um_mm", "binance.um"),
            ("binanceusdm", "binance.um"),
            ("binancecoinm", "binance.cm"),
        ],
    )
    def test_ccxt_id_shaped_name_gets_the_same_config_as_the_venue_name(self, ccxt_id: str, venue: str):
        # binanceusdm/binancecoinm were already dispatched before the alias tuples; the others were not
        assert create_ccxt_rate_limit_config(ccxt_id).pools == create_ccxt_rate_limit_config(venue).pools

    @pytest.mark.parametrize("ccxt_id", ["binancepm", "binanceqv_usdm", "binance_um_mm"])
    def test_ccxt_id_shaped_name_does_not_fall_through_to_the_default(self, ccxt_id: str):
        # _default_config ships a single pool; a binance config always carries orders too
        assert set(create_ccxt_rate_limit_config(ccxt_id).pools) == {"ccxt_rest", "orders"}

    @pytest.mark.parametrize("venue, capacity", [("okx", 20), ("bybit", 120), ("kraken", 20)])
    def test_binance_dispatch_does_not_swallow_other_venues(self, venue: str, capacity: float):
        # negative control for the startswith("binance") dispatch
        assert create_ccxt_rate_limit_config(venue).pools["ccxt_rest"].capacity == capacity


class TestOrderCountHeaderDerivation:
    @pytest.mark.parametrize(
        "venue, header",
        # windows verified against live exchangeInfo rateLimits (2026-08-11): spot ORDERS SECOND/10,
        # fapi ORDERS SECOND/10 + MINUTE/1, dapi ORDERS MINUTE/1 only, papi 1200/min per docs
        [
            ("binance", "X-MBX-ORDER-COUNT-10S"),
            ("binance.um", "X-MBX-ORDER-COUNT-10S"),
            ("binance.cm", "X-MBX-ORDER-COUNT-1M"),
            ("binance.pm", "X-MBX-ORDER-COUNT-1M"),
        ],
    )
    def test_pool_window_derives_the_header_binance_actually_emits(self, venue: str, header: str):
        assert _order_count_header(create_ccxt_rate_limit_config(venue).pools["orders"]) == header

    @pytest.mark.parametrize(
        "capacity, refill, header",
        [(300, 30.0, "X-MBX-ORDER-COUNT-10S"), (1200, 20.0, "X-MBX-ORDER-COUNT-1M"), (60, 60.0, "X-MBX-ORDER-COUNT-1S")],
    )
    def test_window_to_interval_letter(self, capacity: float, refill: float, header: str):
        assert _order_count_header(PoolConfig("orders", "account", capacity, refill)) == header

    @pytest.mark.parametrize("venue", ["binance", "binance.um", "binance.cm", "binance.pm"])
    def test_pool_window_is_one_binance_publishes(self, venue: str):
        # capacity/refill_rate is the header window; retuning either alone yields a name like
        # "-15S" that Binance never emits, silently disabling the sync
        pool = create_ccxt_rate_limit_config(venue).pools["orders"]
        assert pool.capacity / pool.refill_rate in (10.0, 60.0)


class TestKrakenConfig:
    def test_kraken_futures_gets_the_derivatives_budget(self):
        config = create_ccxt_rate_limit_config("kraken.f")
        pool = config.pools["ccxt_rest"]
        assert set(config.pools) == {"ccxt_rest"}  # no separate order limit is documented
        assert (pool.capacity, pool.refill_rate, pool.scope) == (125, 12.5, "account")

    def test_kraken_futures_worst_window_matches_the_documented_budget(self):
        # half-tokens: the bucket starts full, so capacity + 10*refill is the worst 10s window and
        # must equal the venue's 500 cost units per 10s
        pool = create_ccxt_rate_limit_config("kraken.f").pools["ccxt_rest"]
        assert (pool.capacity + 10 * pool.refill_rate) * 2 == 500

    @pytest.mark.parametrize("endpoint", _ORDER_ENDPOINTS)
    async def test_kraken_futures_order_costs_five_units(self, endpoint: str):
        # ccxt charges a flat 1 for every krakenfutures endpoint, so the map tops order ops up from
        # a read's 1 unit to the 5 half-tokens the venue bills (10 cost units)
        config = create_ccxt_rate_limit_config("kraken.f")
        limiter = ExchangeRateLimiter("kraken.f", config)

        await limiter.acquire("rest", weight_override=1.0)  # the throttle leg of the same call
        await limiter.acquire(endpoint)

        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None and state["consumed"] == 5

    async def test_kraken_futures_read_costs_one_unit(self):
        # negative control: a plain read is not surcharged
        limiter = ExchangeRateLimiter("kraken.f", create_ccxt_rate_limit_config("kraken.f"))

        await limiter.acquire("rest", weight_override=1.0)

        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None and state["consumed"] == 1

    def test_kraken_futures_venue_name_and_ccxt_id_agree(self):
        assert create_ccxt_rate_limit_config("kraken.f").pools == create_ccxt_rate_limit_config("krakenfutures").pools

    def test_kraken_spot_keeps_the_counter_pools(self):
        # negative control: the futures branch must not capture spot
        pools = create_ccxt_rate_limit_config("kraken").pools
        assert (pools["ccxt_rest"].capacity, pools["ccxt_rest"].refill_rate) == (20, 0.33)
        assert (pools["orders"].capacity, pools["orders"].refill_rate) == (60, 1.0)


class TestBinanceHeaderParser:
    def test_used_weight_syncs_the_ip_pool(self):
        limiter = _RecordingLimiter()
        parse_binance_headers({"X-MBX-USED-WEIGHT-1M": "1234"}, limiter)
        assert limiter.syncs == [("ccxt_rest", {"used": 1234})]

    @pytest.mark.parametrize("header", ["x-mbx-used-weight-1m", "X-MBX-USED-WEIGHT-1M", "X-Mbx-Used-Weight-1M"])
    def test_lookup_is_case_insensitive(self, header: str):
        limiter = _RecordingLimiter()
        parse_binance_headers({header: "7"}, limiter)
        assert limiter.syncs == [("ccxt_rest", {"used": 7})]

    @pytest.mark.parametrize(
        "venue, expected_used",
        # each surface publishes a different order window, and um/pm share the ccxt id binanceusdm,
        # so the header has to be chosen from the venue's own pool rather than hardcoded
        [("binance", 7), ("binance.um", 7), ("binance.cm", 9), ("binance.pm", 9)],
    )
    def test_orders_pool_syncs_from_the_window_matched_header(self, venue: str, expected_used: int):
        limiter = _RecordingLimiter(venue)
        parse_binance_headers(
            {
                "X-MBX-USED-WEIGHT-1M": "10",
                "X-MBX-ORDER-COUNT-1S": "3",
                "X-MBX-ORDER-COUNT-10S": "7",
                "X-MBX-ORDER-COUNT-1M": "9",
                "X-MBX-ORDER-COUNT-1D": "900",
            },
            limiter,
        )

        assert limiter.syncs == [("ccxt_rest", {"used": 10}), ("orders", {"used": expected_used})]

    def test_real_order_response_does_not_reset_the_ip_pool(self):
        # verbatim headers from a testnet POST /fapi/v1/order: Binance sends -1 for "not
        # applicable", and treating it as a count gives remaining = capacity + 1
        limiter = _RecordingLimiter("binance.um")
        parse_binance_headers(
            {"x-mbx-used-weight-1m": "-1", "x-mbx-order-count-10s": "1", "x-mbx-order-count-1m": "1"}, limiter
        )
        assert limiter.syncs == [("orders", {"used": 1})]

    @pytest.mark.parametrize("value", ["-1", "-100"])
    def test_negative_weight_is_not_a_count(self, value: str):
        limiter = _RecordingLimiter("binance.um")
        parse_binance_headers({"X-MBX-USED-WEIGHT-1M": value}, limiter)
        assert limiter.syncs == []

    def test_zero_is_a_real_count_not_a_sentinel(self):
        # negative control: 0 used is legitimate at the start of a window and must still sync
        limiter = _RecordingLimiter("binance.um")
        parse_binance_headers({"X-MBX-USED-WEIGHT-1M": "0"}, limiter)
        assert limiter.syncs == [("ccxt_rest", {"used": 0})]

    def test_order_count_for_another_window_is_ignored(self):
        # negative control: reading the wrong window would re-pin a 10s pool from a 1s count and
        # silently defeat it — only the matching header may sync
        limiter = _RecordingLimiter("binance.um")
        parse_binance_headers({"X-MBX-ORDER-COUNT-1S": "3", "X-MBX-ORDER-COUNT-1M": "9"}, limiter)
        assert limiter.syncs == []

    def test_order_count_header_is_case_insensitive(self):
        limiter = _RecordingLimiter("binance.pm")
        parse_binance_headers({"x-mbx-order-count-1m": "9"}, limiter)
        assert limiter.syncs == [("orders", {"used": 9})]

    def test_venue_without_an_orders_pool_syncs_nothing(self):
        # negative control: krakenfutures-shaped configs have no orders pool
        limiter = _RecordingLimiter("kraken.f")
        assert "orders" not in limiter.config.pools
        parse_binance_headers({"X-MBX-ORDER-COUNT-1M": "9"}, limiter)
        assert limiter.syncs == []

    def test_weight_header_alone_syncs_only_the_ip_pool(self):
        limiter = _RecordingLimiter("binance.um")
        parse_binance_headers({"X-MBX-USED-WEIGHT-1M": "9"}, limiter)
        assert limiter.syncs == [("ccxt_rest", {"used": 9})]

    @pytest.mark.parametrize("value", ["abc", "", None, "12.5"])
    def test_malformed_value_is_silent(self, value):
        limiter = _RecordingLimiter()
        parse_binance_headers({"X-MBX-USED-WEIGHT-1M": value}, limiter)
        assert limiter.syncs == []

    def test_silent_without_a_limiter(self):
        parse_binance_headers({"X-MBX-USED-WEIGHT-1M": "10"}, None)


def _constructed_ccxt_ids() -> dict[str, str]:
    """``exchange.id`` of every ccxt class the factory can build, keyed by the name it resolves.

    Classes that cannot be built bare are skipped — bitfinex_f's ``__init__`` starts a websocket on
    ``self.asyncio_loop``, which is None outside a running loop. The callers still assert that every
    registered parser id was resolved, so a skip cannot hide a dead registration.
    """
    names = set(EXCHANGE_ALIASES.values()) | {"binance", "binanceusdm", "binancecoinm", "okx", "bybit", "kraken"}
    ids = {}
    for name in sorted(names):
        cls = getattr(cxp, name, None)
        if cls is None:
            continue  # optional-dependency exchanges are absent unless their extra is installed
        try:
            ids[name] = cls().id
        except Exception:
            continue
    return ids


class TestHeaderParserRegistry:
    def test_binanceusdm_resolves_to_the_binance_parser(self):
        assert get_header_parser("binanceusdm") is parse_binance_headers

    def test_binance_portfolio_margin_inherits_the_binanceusdm_id(self):
        # binance.pm's header sync exists only because the subclass chain keeps ccxt's id
        assert cxp.binancepm().id == "binanceusdm"
        assert get_header_parser(cxp.binancepm().id) is parse_binance_headers

    def test_binance_portfolio_margin_chain_overrides_describe_without_setting_id(self):
        from qubx.connectors.ccxt.exchanges.binance.exchange import BinancePortfolioMargin, BinanceQVUSDM

        for cls in (BinancePortfolioMargin, BinanceQVUSDM):
            assert "describe" in cls.__dict__, f"{cls.__name__} no longer overrides describe()"
            assert cls().id == "binanceusdm", f"{cls.__name__} now sets its own id — re-check HEADER_PARSERS"

    def test_lookup_is_case_insensitive(self):
        assert get_header_parser("BinanceUsdm") is parse_binance_headers

    def test_unknown_id_has_no_parser(self):
        assert get_header_parser("not-an-exchange") is None

    def test_lookup_is_by_ccxt_id_not_venue_name(self):
        # negative control: no ccxt id contains a dot, so the venue name is not a key
        assert get_header_parser("binance.um") is None
        assert get_header_parser("binance.pm") is None

    def test_every_registered_key_is_an_id_qubx_constructs(self):
        ids = set(_constructed_ccxt_ids().values())
        assert set(HEADER_PARSERS) <= ids, f"dead parser registrations: {sorted(set(HEADER_PARSERS) - ids)}"

    def test_factory_alias_names_are_not_ccxt_ids(self):
        # negative control for the audit above: registering under a factory alias would be dead code
        ids = _constructed_ccxt_ids()
        assert ids["custom_krakenfutures"] == "krakenfutures"
        assert ids["gateio_futures"] == "gate"
        assert ids["binancepm"] == "binanceusdm"

    def test_only_binance_ids_are_registered(self):
        assert set(HEADER_PARSERS) == {"binance", "binanceusdm", "binancecoinm"}

    @pytest.mark.parametrize("ccxt_id", ["okx", "bybit", "kraken", "krakenfutures"])
    def test_venues_metering_per_endpoint_have_no_parser(self, ccxt_id: str):
        # okx/bybit meter per endpoint (okx also publishes no rate-limit header at all); syncing such
        # a signal into the venue-wide ccxt_rest pool would re-pin it to full on every response
        assert get_header_parser(ccxt_id) is None


class TestInstallRateLimiterHooks:
    async def test_ccxt_cost_is_forwarded_as_weight_override(self):
        exchange, limiter = _mock_exchange(), _RecordingThrottle()
        install_rate_limiter_hooks(exchange, limiter)

        await exchange.throttle(7)

        assert limiter.calls == [("rest", 7.0)]

    async def test_omitted_cost_becomes_one(self):
        exchange, limiter = _mock_exchange(), _RecordingThrottle()
        install_rate_limiter_hooks(exchange, limiter)

        await exchange.throttle()

        assert limiter.calls == [("rest", 1.0)]

    async def test_none_cost_becomes_one(self):
        exchange, limiter = _mock_exchange(), _RecordingThrottle()
        install_rate_limiter_hooks(exchange, limiter)

        await exchange.throttle(None)

        assert limiter.calls == [("rest", 1.0)]

    async def test_zero_cost_charges_zero(self):
        limiter = ExchangeRateLimiter("binance.um", create_ccxt_rate_limit_config("binance.um"))
        exchange = _mock_exchange()
        install_rate_limiter_hooks(exchange, limiter)

        await exchange.throttle(0)
        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None and state["consumed"] == 0.0

        # positive control: the same path does charge a non-zero cost
        await exchange.throttle(4)
        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None and state["consumed"] == 4.0

    async def test_consumed_accumulates_exactly(self):
        # consumed, not remaining — remaining refills by wall clock and is not deterministic
        limiter = ExchangeRateLimiter("binance.um", create_ccxt_rate_limit_config("binance.um"))
        exchange = _mock_exchange()
        install_rate_limiter_hooks(exchange, limiter)

        for cost in (2, 3, 0.5):
            await exchange.throttle(cost)

        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None and state["consumed"] == pytest.approx(5.5)

    def test_binance_klines_at_limit_1000_really_costs_five(self):
        ccxt_exchange = cxp.binanceusdm()
        config = ccxt_exchange.api["fapiPublic"]["get"]["klines"]
        assert ccxt_exchange.calculate_rate_limiter_cost("fapiPublic", "GET", "klines", {"limit": 1000}, config) == 5
        # negative control: the 5 is limit-dependent, not a constant
        assert ccxt_exchange.calculate_rate_limiter_cost("fapiPublic", "GET", "klines", {}, config) == 1

    async def test_real_klines_cost_is_charged_end_to_end(self):
        ccxt_exchange = cxp.binanceusdm()
        config = ccxt_exchange.api["fapiPublic"]["get"]["klines"]
        cost = ccxt_exchange.calculate_rate_limiter_cost("fapiPublic", "GET", "klines", {"limit": 1000}, config)

        limiter = ExchangeRateLimiter("binance.um", create_ccxt_rate_limit_config("binance.um"))
        exchange = _mock_exchange()
        install_rate_limiter_hooks(exchange, limiter)
        await exchange.throttle(cost)

        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None and state["consumed"] == 5.0

    async def test_ccxt_own_throttler_is_never_invoked(self):
        ccxt_exchange = cxp.binanceusdm()
        throttler_calls: list = []

        async def _recording_throttler(cost=None):
            throttler_calls.append(cost)

        ccxt_exchange.throttler = _recording_throttler

        # negative control: unhooked, ccxt's REST path does reach its own throttler
        await ccxt_exchange.throttle(3)
        assert throttler_calls == [3]

        limiter = _RecordingThrottle()
        install_rate_limiter_hooks(ccxt_exchange, limiter)
        await ccxt_exchange.throttle(5)

        assert throttler_calls == [3]
        assert limiter.calls == [("rest", 5.0)]

    def test_enable_rate_limit_stays_on(self):
        # fetch2 only calls exchange.throttle when the flag is set — turning it off disables all
        # REST throttling instead of just ccxt's
        exchange = _mock_exchange()
        install_rate_limiter_hooks(exchange, _RecordingThrottle())
        assert exchange.enableRateLimit is True

        ccxt_exchange = cxp.binanceusdm()
        install_rate_limiter_hooks(ccxt_exchange, _RecordingThrottle())
        assert ccxt_exchange.enableRateLimit is True

    def test_no_limiter_is_a_noop(self):
        # an id that does have a parser, so the header hook would land if anything were installed
        exchange = _mock_exchange("binanceusdm")
        original_throttle, original_hook = exchange.throttle, exchange.on_rest_response

        install_rate_limiter_hooks(exchange, None)

        assert exchange.throttle is original_throttle
        assert exchange.on_rest_response is original_hook
        assert exchange.enableRateLimit is False

    def test_unknown_id_leaves_on_rest_response_untouched(self):
        # negative control for the header hook: the throttle is installed either way
        exchange = _mock_exchange("no-parser-for-this-id")
        original_hook, original_throttle = exchange.on_rest_response, exchange.throttle

        install_rate_limiter_hooks(exchange, _RecordingThrottle())

        assert exchange.on_rest_response is original_hook
        assert exchange.throttle is not original_throttle
        assert exchange.enableRateLimit is True

    def test_header_hook_syncs_and_preserves_the_original_return_value(self):
        exchange = _mock_exchange("binanceusdm")
        original_hook = exchange.on_rest_response
        original_hook.return_value = "orig-return-value"
        limiter = _RecordingLimiter("binance.um")
        install_rate_limiter_hooks(exchange, limiter)

        args = (200, "OK", "https://binance", "GET", {"X-MBX-USED-WEIGHT-1M": "9"}, "{}", {}, None)
        result = exchange.on_rest_response(*args)

        assert result == "orig-return-value"
        original_hook.assert_called_once_with(*args)
        assert limiter.syncs == [("ccxt_rest", {"used": 9})]

    def test_header_hook_tolerates_a_raising_parser(self):
        def _boom(headers, rate_limiter):
            raise RuntimeError("simulated parser bug")

        exchange = _mock_exchange("binanceusdm")
        original_hook = exchange.on_rest_response
        original_hook.return_value = "orig-return-value"

        # patch the registry entry, not the module attribute — HEADER_PARSERS holds a direct
        # reference, so patching the module-level name would not be seen by get_header_parser
        with patch.dict(HEADER_PARSERS, {"binanceusdm": _boom}):
            install_rate_limiter_hooks(exchange, _RecordingLimiter("binance.um"))
            result = exchange.on_rest_response(200, "OK", "u", "GET", {"X-MBX-USED-WEIGHT-1M": "9"}, "{}", {}, None)

        assert result == "orig-return-value"
        original_hook.assert_called_once()

    @pytest.mark.parametrize("headers", [None, {}])
    def test_header_hook_skips_empty_headers(self, headers):
        exchange = _mock_exchange("binanceusdm")
        original_hook = exchange.on_rest_response
        limiter = _RecordingLimiter("binance.um")
        install_rate_limiter_hooks(exchange, limiter)

        exchange.on_rest_response(200, "OK", "u", "GET", headers, "{}", {}, None)

        assert limiter.syncs == []
        original_hook.assert_called_once()


class TestInstallDoesNotPinTheExchange:
    def test_hooked_exchange_is_still_collectable(self):
        # the install registry must not hold anything referencing the exchange, or every recreated
        # exchange (markets, currencies, closures) stays alive for the life of the process
        import gc
        import weakref

        from qubx.connectors.ccxt.rate_limits import _INSTALLED

        exchange = _mock_exchange("binanceusdm")
        exchange.on_rest_response = exchange.__class__.__call__  # a bound method, as ccxt's is
        ref = weakref.ref(exchange)
        install_rate_limiter_hooks(exchange, _RecordingThrottle(), label="x")
        before = len(_INSTALLED)

        del exchange
        gc.collect()

        assert ref() is None
        assert len(_INSTALLED) < before


class TestInstallIsIdempotent:
    """A second install must not stack a wrapper on top of the first — ExchangeManager reinstalls
    on every exchange recreation, and each extra layer would re-charge the pool per request."""

    def test_second_install_does_not_re_wrap_on_rest_response(self):
        exchange = _mock_exchange("binanceusdm")
        limiter = _RecordingLimiter("binance.um")

        install_rate_limiter_hooks(exchange, limiter)
        install_rate_limiter_hooks(exchange, limiter)

        exchange.on_rest_response(200, "OK", "u", "GET", {"X-MBX-USED-WEIGHT-1M": "9"}, "{}", {}, None)

        # a doubled wrapper would run the parser twice for this one response
        assert limiter.syncs == [("ccxt_rest", {"used": 9})]

    def test_second_install_keeps_the_first_throttle(self):
        exchange = _mock_exchange("binanceusdm")
        limiter = _RecordingLimiter("binance.um")

        install_rate_limiter_hooks(exchange, limiter)
        first_throttle, first_hook = exchange.throttle, exchange.on_rest_response

        install_rate_limiter_hooks(exchange, limiter)

        assert exchange.throttle is first_throttle
        assert exchange.on_rest_response is first_hook

    async def test_second_install_charges_the_pool_once_per_request(self):
        limiter = ExchangeRateLimiter("binance.um", create_ccxt_rate_limit_config("binance.um"))
        exchange = _mock_exchange("binanceusdm")

        install_rate_limiter_hooks(exchange, limiter)
        install_rate_limiter_hooks(exchange, limiter)

        await exchange.throttle(3)

        state = await limiter.get_pool_state("ccxt_rest")
        assert state is not None and state["consumed"] == 3.0

    def test_mock_exchange_still_gets_hooks_on_the_first_call(self):
        # negative control for the guard: Mock answers any getattr with a truthy child, so a
        # truthiness test would read it as already-installed and skip installation entirely
        exchange = _mock_exchange("binanceusdm")
        limiter = _RecordingLimiter("binance.um")
        marker = getattr(exchange.throttle, "_qubx_rate_limiter", None)
        assert bool(marker) is True and marker is not limiter
        original_throttle, original_hook = exchange.throttle, exchange.on_rest_response

        install_rate_limiter_hooks(exchange, limiter)

        assert exchange.throttle is not original_throttle
        assert exchange.on_rest_response is not original_hook
        assert exchange.enableRateLimit is True

    async def test_a_different_limiter_re_points_the_hooks(self):
        # negative control for the skip: the guard keys on limiter identity, so a genuine swap must
        # rewire — and must replace our previous wrapper, not stack a second one on top of it
        exchange = _mock_exchange("binanceusdm")
        original_hook = exchange.on_rest_response
        first, second = _RecordingLimiter("binance.um"), _RecordingLimiter("binance.um")

        install_rate_limiter_hooks(exchange, first)
        install_rate_limiter_hooks(exchange, second)

        exchange.on_rest_response(200, "OK", "u", "GET", {"X-MBX-USED-WEIGHT-1M": "5"}, "{}", {}, None)

        assert first.syncs == []
        assert second.syncs == [("ccxt_rest", {"used": 5})]
        original_hook.assert_called_once()  # the venue's own handler still runs exactly once

    async def test_a_different_limiter_re_points_the_throttle(self):
        exchange = _mock_exchange("binanceusdm")
        first, second = _RecordingThrottle(), _RecordingThrottle()

        install_rate_limiter_hooks(exchange, first)
        install_rate_limiter_hooks(exchange, second)
        await exchange.throttle(2)

        assert first.calls == []
        assert second.calls == [("rest", 2.0)]

    async def test_second_install_on_a_real_ccxt_exchange_is_a_noop(self):
        ccxt_exchange = cxp.binanceusdm()
        limiter = _RecordingThrottle()

        install_rate_limiter_hooks(ccxt_exchange, limiter)
        first_throttle = ccxt_exchange.throttle
        install_rate_limiter_hooks(ccxt_exchange, limiter)

        assert ccxt_exchange.throttle is first_throttle
        await ccxt_exchange.throttle(2)
        assert limiter.calls == [("rest", 2.0)]


class TestDefaultConfig:
    def test_unknown_venue_warns_naming_the_venue_and_the_guess(self, warnings_logged):
        create_ccxt_rate_limit_config("some-brand-new-venue")

        messages = [record["message"] for record in warnings_logged]
        assert any("some-brand-new-venue" in m and "20 req/s" in m for m in messages), messages

    def test_known_venue_does_not_warn(self, warnings_logged):
        # negative control
        for venue in ("binance.um", "binance.pm", "okx", "bybit", "kraken", "kraken.f"):
            create_ccxt_rate_limit_config(venue)
        assert warnings_logged == []

    def test_default_config_shape(self):
        config = create_ccxt_rate_limit_config("some-brand-new-venue")
        assert set(config.pools) == {"ccxt_rest"}
        assert (config.pools["ccxt_rest"].capacity, config.pools["ccxt_rest"].refill_rate) == (1200.0, 20.0)

    def test_signature_takes_only_the_venue_name(self):
        # the dead ccxt_exchange auto-derivation parameter is gone
        assert list(inspect.signature(create_ccxt_rate_limit_config).parameters) == ["exchange_name"]
