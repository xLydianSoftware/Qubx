"""
CCXT exchange rate limit configurations.

Provides default rate limit configs per exchange, differentiated by market type
(spot vs futures). Also includes response header parsers for syncing modeled
state with exchange-reported usage.

References:
- Binance: https://binance-docs.github.io/apidocs/spot/en/#limits
- Binance Futures: https://binance-docs.github.io/apidocs/futures/en/#limits
- Binance Portfolio Margin: https://developers.binance.com/docs/derivatives/portfolio-margin/general-info
- OKX: https://www.okx.com/docs-v5/en/#overview-rate-limit
- Kraken Spot REST: https://docs.kraken.com/api/docs/guides/spot-rest-ratelimits
- Kraken Spot trading: https://docs.kraken.com/api/docs/guides/spot-ratelimits
- Kraken Futures: https://docs.kraken.com/api/docs/guides/futures-rate-limits
- Bybit: https://bybit-exchange.github.io/docs/v5/rate-limit
"""

from weakref import WeakKeyDictionary

from qubx import logger
from qubx.rate_limiting import EndpointCosts, ExchangeRateLimitConfig, PoolConfig

_BINANCE_UM = ("binance.um", "binance.um.mm", "binanceusdm", "binanceqv_usdm", "binance_um_mm")
_BINANCE_CM = ("binance.cm", "binancecoinm")
_BINANCE_PM = ("binance.pm", "binancepm")


# Placeholder weight for the "rest" endpoint: the throttle passes ccxt's real per-endpoint cost as
# weight_override, which replaces it. The weight on order endpoints is literal — order-count limits
# count orders, not weight.
_CCXT_SUPPLIES_COST = 1


def _default_endpoint_costs() -> dict[str, EndpointCosts]:
    """Which pools each operation debits — routing, not limits. The limits are the ``pools``.

    The same on every ccxt venue (Kraken Futures overrides it), since only the numbers differ, not
    the topology. Order endpoints are listed even for venues without an ``orders`` pool: an unmapped
    endpoint falls back to ``default_costs`` and would double-charge the IP pool, whereas a cost
    naming an absent pool is simply skipped by the engine.
    """
    order = EndpointCosts([("orders", 1)])
    return {
        "rest": EndpointCosts([("ccxt_rest", _CCXT_SUPPLIES_COST)]),
        "create_order": order,
        "cancel_order": order,
        "edit_order": order,
    }


def create_ccxt_rate_limit_config(exchange_name: str) -> ExchangeRateLimitConfig:
    """Create rate limit config for a CCXT exchange.

    Accepts both framework venue names (``BINANCE.UM``) and ccxt exchange ids (``binanceusdm``).
    """
    name = exchange_name.lower()
    base = name.split(".")[0]

    if base.startswith("binance"):
        return _binance_config(name)
    elif base == "okx":
        return _okx_config()
    elif base == "bybit":
        return _bybit_config()
    elif base in ("kraken", "krakenfutures"):
        return _kraken_config(name)
    else:
        return _default_config(exchange_name)


# === Binance ===


def _binance_config(exchange_name: str) -> ExchangeRateLimitConfig:
    """Binance rate limits per surface (verified against live ``exchangeInfo``, 2026-08-10).

    | surface       | IP weight | orders   | order-count header |
    |---------------|-----------|----------|--------------------|
    | spot (api)    | 6000/1m   | 100/10s  | -10S               |
    | USD-M (fapi)  | 2400/1m   | 300/10s  | -10S               |
    | COIN-M (dapi) | 2400/1m   | 1200/1m  | -1M                |
    | PM (papi)     | 6000/1m   | 1200/1m  | -1M                |

    fapi publishes both 300/10s and 1200/1m; one bucket models the 10s one, since a burst breaches it
    long before the 1m. The residual gap is sustained flow: 30/s passes the 10s model but exceeds
    1200/1m, and only the venue's own -1015 (which closes the gate) catches it. dapi publishes only
    1200/1m, spot only 100/10s (plus 200000/1d, unmodelled).

    ``capacity`` must equal the modelled window's allowance and ``capacity / refill_rate`` its length:
    ``sync_from_exchange`` derives ``remaining = capacity - used``, and ``_order_count_header`` picks
    the header from that ratio. Changing either without the other silently disables the sync.
    """
    if exchange_name in _BINANCE_UM:
        return ExchangeRateLimitConfig(
            pools={
                "ccxt_rest": PoolConfig("ccxt_rest", "ip", 2400, 40.0, cooldown=30.0),
                "orders": PoolConfig("orders", "account", 300, 30.0, cooldown=10.0),
            },
            endpoint_map=_default_endpoint_costs(),
            default_costs=EndpointCosts([("ccxt_rest", 1)]),
        )
    elif exchange_name in _BINANCE_CM:
        return ExchangeRateLimitConfig(
            pools={
                "ccxt_rest": PoolConfig("ccxt_rest", "ip", 2400, 40.0, cooldown=30.0),
                "orders": PoolConfig("orders", "account", 1200, 20.0, cooldown=10.0),
            },
            endpoint_map=_default_endpoint_costs(),
            default_costs=EndpointCosts([("ccxt_rest", 1)]),
        )
    elif exchange_name in _BINANCE_PM:
        return ExchangeRateLimitConfig(
            pools={
                "ccxt_rest": PoolConfig("ccxt_rest", "ip", 6000, 100.0, cooldown=30.0),
                "orders": PoolConfig("orders", "account", 1200, 20.0, cooldown=10.0),
            },
            endpoint_map=_default_endpoint_costs(),
            default_costs=EndpointCosts([("ccxt_rest", 1)]),
        )
    else:
        return ExchangeRateLimitConfig(
            pools={
                "ccxt_rest": PoolConfig("ccxt_rest", "ip", 6000, 100.0, cooldown=30.0),
                "orders": PoolConfig("orders", "account", 100, 10.0, cooldown=10.0),
            },
            endpoint_map=_default_endpoint_costs(),
            default_costs=EndpointCosts([("ccxt_rest", 1)]),
        )


# === OKX ===


def _okx_config() -> ExchangeRateLimitConfig:
    """OKX v5 rate limits (unified across spot/futures).

    - REST: varies heavily by endpoint (2-60 req/2sec per endpoint)
    - Market data: 20/2sec per IP
    - place / amend / cancel order: 60 req/2sec each, scoped per user id + instrument id. The three
      counters are independent and per instrument; pooling them per account is conservative in both
      directions, and keeps us under the 1000/2sec sub-account cap by construction.

    No ``HEADER_PARSERS`` entry: OKX publishes no rate-limit header at all and meters per endpoint /
    instrument id — see the note above ``HEADER_PARSERS``.
    """
    return ExchangeRateLimitConfig(
        pools={
            "ccxt_rest": PoolConfig("ccxt_rest", "ip", 20, 10.0, cooldown=15.0),
            "orders": PoolConfig("orders", "account", 60, 30.0, cooldown=10.0),
        },
        endpoint_map=_default_endpoint_costs(),
        default_costs=EndpointCosts([("ccxt_rest", 1)]),
    )


# === Bybit ===


def _bybit_config() -> ExchangeRateLimitConfig:
    """Bybit v5 rate limits.

    - IP rule modelled by ``ccxt_rest``: 600 req/5s per IP; breaching it returns HTTP 403 and bans
      the IP for 10 minutes.
    - Orders: 10 req/s per UID for create/amend/cancel on inverse/linear/option (spot allows 20/s
      for create and cancel). Bybit meters per endpoint; pooling the three is conservative. Not
      modelled: option ``cancel-all`` is 1/s, stricter than this pool.

    No ``HEADER_PARSERS`` entry: v5's ``X-Bapi-Limit-*`` headers are per-endpoint, not per-IP — see
    the note above ``HEADER_PARSERS``.
    """
    return ExchangeRateLimitConfig(
        pools={
            "ccxt_rest": PoolConfig("ccxt_rest", "ip", 120, 24.0, cooldown=15.0),
            "orders": PoolConfig("orders", "account", 10, 10.0, cooldown=10.0),
        },
        endpoint_map=_default_endpoint_costs(),
        default_costs=EndpointCosts([("ccxt_rest", 1)]),
    )


# === Kraken ===


def _kraken_config(exchange_name: str) -> ExchangeRateLimitConfig:
    """Kraken rate limits.

    Kraken Spot runs two independent counters, both decaying at a tier-dependent rate.

    REST counter (``ccxt_rest``), threshold / decay per second: starter 15 / 0.33, intermediate
    20 / 0.5, pro 20 / 1.0. AddOrder and CancelOrder are explicitly excluded from it. ccxt supplies
    the real per-call cost (public 1.0-1.2, balance 3, ledgers 6, orders 0).

    Trading counter (``orders``), threshold / decay per second: starter 60 / 1.0, intermediate
    125 / 2.34, pro 180 / 3.75. Starter is modelled — the tier is not discoverable from the API.
    Two known distortions, both documented rather than modelled: Kraken meters this counter per
    *pair* while the pool is per account (over-throttles a multi-pair strategy), and its cost rises
    with how young the order is (cancel costs 8 under 5s, decaying to 1 over 300s) while
    ``_endpoint_map`` charges a flat 1 (under-throttles cancel-heavy flow).

    Kraken Futures runs one account-wide ``/derivatives`` budget of 500 cost units per 10s. Reads
    (openorders, openpositions, accounts, fills) cost 2, order ops (sendorder, cancelorder,
    editorder) cost 10, public endpoints (charts, instruments, tickers, orderbook) cost 0. There is
    no separate order-count limit, so no ``orders`` pool. ``/history`` has its own 100-per-10min
    pool; nothing here calls it.

    Neither the Spot REST, Spot trading nor the Futures guide documents any rate-limit response
    header, and ccxt parses none — hence no Kraken entry in ``HEADER_PARSERS``; breaches surface
    reactively as ``EOrder:Rate limit exceeded`` (spot) / ``apiLimitExceeded`` (futures).
    """
    if exchange_name in ("kraken.f", "krakenfutures"):
        # ccxt declares no per-endpoint costs for krakenfutures and charges a flat 1 per request, so
        # the pool counts half-tokens: a read is exactly 1 unit, and order endpoints add 4 on top of
        # that 1 to reach the venue's 10. 250 units/10s; the bucket starts full, so capacity +
        # 10*refill == 250 keeps the worst 10s window at exactly the documented 500 tokens.
        # Under-counted (unused by our call paths): batch orders, cancelallorders, leveragepreferences.
        order_surcharge = EndpointCosts([("ccxt_rest", 4)])
        return ExchangeRateLimitConfig(
            pools={
                "ccxt_rest": PoolConfig("ccxt_rest", "account", 125, 12.5, cooldown=10.0),
            },
            endpoint_map={
                "rest": EndpointCosts([("ccxt_rest", _CCXT_SUPPLIES_COST)]),
                "create_order": order_surcharge,
                "cancel_order": order_surcharge,
                "edit_order": order_surcharge,
            },
            default_costs=EndpointCosts([("ccxt_rest", 1)]),
        )

    # Counter threshold → capacity, decay → refill; the decay is the lowest (starter) tier's.
    return ExchangeRateLimitConfig(
        pools={
            "ccxt_rest": PoolConfig("ccxt_rest", "ip", 20, 0.33, cooldown=30.0),
            "orders": PoolConfig("orders", "account", 60, 1.0, cooldown=15.0),
        },
        # ccxt's per-endpoint cost (Balance=3, Ledgers=6, orders=0) always overrides the static 1.
        endpoint_map=_default_endpoint_costs(),
        default_costs=EndpointCosts([("ccxt_rest", 1)]),
    )


# === Default ===


def _default_config(exchange_name: str) -> ExchangeRateLimitConfig:
    """Fallback for venues with no hand-tuned config — the numbers below are a guess, not a limit."""
    rps = 20.0
    capacity = rps * 60
    logger.warning(
        f"No rate limit config for '{exchange_name}' — guessing {rps:.0f} req/s "
        f"(capacity {capacity:.0f}); add a config in qubx.connectors.ccxt.rate_limits"
    )

    return ExchangeRateLimitConfig(
        pools={
            "ccxt_rest": PoolConfig("ccxt_rest", "ip", capacity, rps, cooldown=15.0),
        },
        endpoint_map=_default_endpoint_costs(),
        default_costs=EndpointCosts([("ccxt_rest", 1)]),
    )


# === Response Header Parsers ===


def _header(headers: dict, name: str) -> str | None:
    """Case-insensitive header lookup — ccxt copies aiohttp's raw casing, it does not normalize."""
    value = headers.get(name)
    if value is not None:
        return value
    lowered = name.lower()
    for key, val in headers.items():
        if key.lower() == lowered:
            return val
    return None


def _order_count_header(pool: PoolConfig) -> str:
    """The ``X-MBX-ORDER-COUNT-*`` header matching *pool*'s window.

    Binance names the window ``(intervalNum)(intervalLetter)``, and a pool's window is
    ``capacity / refill_rate`` under this module's denomination rule. Deriving it is what lets one
    parser serve venues with different order windows — ``binance.um`` (10s) and ``binance.pm`` (1m)
    share the ccxt id ``binanceusdm``, so a hardcoded name would be wrong for one of them.
    """
    seconds = pool.capacity / pool.refill_rate
    if seconds < 60:
        return f"X-MBX-ORDER-COUNT-{int(seconds)}S"
    if seconds < 3600:
        return f"X-MBX-ORDER-COUNT-{int(seconds // 60)}M"
    return f"X-MBX-ORDER-COUNT-{int(seconds // 3600)}H"


def _used_count(headers: dict, name: str) -> int | None:
    """Header value as a usage count, or None when absent, unparseable, or negative.

    Binance sends ``-1`` for "not applicable" — notably ``X-MBX-USED-WEIGHT-1M: -1`` on order
    responses (observed on testnet). Treating it as a count yields ``remaining = capacity + 1`` and
    resets the pool to full on every order placement.
    """
    raw = _header(headers, name)
    if raw is None:
        return None
    try:
        used = int(raw)
    except (TypeError, ValueError):
        return None
    return used if used >= 0 else None


def parse_binance_headers(headers: dict, rate_limiter) -> None:
    """Sync the IP weight pool from ``X-MBX-USED-WEIGHT-1M`` and the order pool from its
    window-matched ``X-MBX-ORDER-COUNT-*``.

    Order-count headers ride only on order *placement* responses — cancels carry none — so every
    other response skips that branch. Syncing them is what catches order flow we did not place
    ourselves (manual trading, a second bot on the same key); our own count is otherwise exact,
    though charging cancels locally makes it conservative until the next placement re-pins it.
    """
    if rate_limiter is None:
        return

    used_weight = _used_count(headers, "X-MBX-USED-WEIGHT-1M")
    if used_weight is not None:
        rate_limiter.sync_from_exchange("ccxt_rest", used=used_weight)

    orders = rate_limiter.config.pools.get("orders")
    if orders is None or orders.refill_rate <= 0:
        return
    order_count = _used_count(headers, _order_count_header(orders))
    if order_count is not None:
        rate_limiter.sync_from_exchange("orders", used=order_count)


# A venue gets a parser only when it publishes a header whose scope matches the pool being pinned:
# a per-endpoint or per-instrument signal must never re-pin a venue-wide IP pool. OKX, Bybit and
# Kraken all fail that test — see each ``_*_config`` docstring for which way.
HEADER_PARSERS = {
    "binance": parse_binance_headers,
    "binanceusdm": parse_binance_headers,
    "binancecoinm": parse_binance_headers,
}


def get_header_parser(exchange_id: str):
    """Get the response header parser for a ccxt exchange id (``exchange.id``)."""
    return HEADER_PARSERS.get(exchange_id.lower())


# exchange -> its limiter. The hooks resolve it here at call time, so re-attaching a different
# limiter re-points them rather than stacking a second wrapper. The value must not reference the
# exchange, or the weak key could never be collected.
_INSTALLED: WeakKeyDictionary = WeakKeyDictionary()


def install_rate_limiter_hooks(exchange, rate_limiter, label: str = "") -> None:
    """Route a ccxt exchange's REST throttle and response headers through *rate_limiter*.

    ``fetch2`` only calls ``exchange.throttle`` when ``enableRateLimit`` is set, so the flag stays
    on; shadowing the bound method is what takes ccxt's own throttler out of the REST path.
    ccxt.pro's per-Client throttler is separate and still paces subscribe frames.
    """
    if rate_limiter is None:
        return

    already_installed = exchange in _INSTALLED
    _INSTALLED[exchange] = rate_limiter
    if already_installed:
        return

    exchange.enableRateLimit = True
    original_on_rest = exchange.on_rest_response

    async def _rate_limit_throttle(cost=None):
        limiter = _INSTALLED.get(exchange)
        if limiter is not None:
            await limiter.acquire("rest", weight_override=float(1.0 if cost is None else cost))

    exchange.throttle = _rate_limit_throttle

    parser = get_header_parser(exchange.id or "")
    if parser is None:
        return

    def _header_sync_hook(code, reason, url, method, headers, body, req_headers, req_body):
        limiter = _INSTALLED.get(exchange)
        try:
            if headers and limiter is not None:
                parser(headers, limiter)
        except Exception as e:
            logger.debug(f"[{label}] header sync failed (non-fatal): {e}")
        return original_on_rest(code, reason, url, method, headers, body, req_headers, req_body)

    exchange.on_rest_response = _header_sync_hook
