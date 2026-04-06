"""
CCXT exchange rate limit configurations.

Provides default rate limit configs per exchange, differentiated by market type
(spot vs futures). Also includes response header parsers for syncing modeled
state with exchange-reported usage.

References:
- Binance: https://binance-docs.github.io/apidocs/spot/en/#limits
- Binance Futures: https://binance-docs.github.io/apidocs/futures/en/#limits
- OKX: https://www.okx.com/docs-v5/en/#overview-rate-limit
- Kraken: https://docs.kraken.com/api/docs/guides/rate-limits
- Bybit: https://bybit-exchange.github.io/docs/v5/rate-limit
"""

from qubx.connectors.registry import rate_limit_config
from qubx.rate_limiting import EndpointCosts, ExchangeRateLimitConfig, PoolConfig


@rate_limit_config("ccxt")
def create_ccxt_rate_limit_config(exchange_name: str, ccxt_exchange=None) -> ExchangeRateLimitConfig:
    """Create rate limit config for a CCXT exchange.

    Handles exchange.market_type differentiation (spot vs futures).

    Args:
        exchange_name: Exchange name (e.g., "binance", "binance.um", "kraken")
        ccxt_exchange: Optional CCXT exchange instance for auto-detection
    """
    name = exchange_name.lower()
    base = name.split(".")[0]

    if base in ("binance", "binanceusdm", "binancecoinm"):
        return _binance_config(name)
    elif base == "okx":
        return _okx_config()
    elif base == "bybit":
        return _bybit_config()
    elif base in ("kraken", "krakenfutures"):
        return _kraken_config(name)
    else:
        return _default_config(exchange_name, ccxt_exchange)


# === Binance ===


def _binance_config(exchange_name: str) -> ExchangeRateLimitConfig:
    """Binance rate limits differentiated by market type.

    Spot:
    - IP weight: 6000/min (recently increased from 1200)
    - UID weight: 180,000/min
    - Orders: 10/sec, 200,000/day per account

    USDM Futures (binance.um):
    - IP weight: 2400/min
    - Orders: 300/10sec per account

    COIN-M Futures (binance.cm):
    - IP weight: 6000/min
    - Orders: 300/10sec per account

    Headers: X-MBX-USED-WEIGHT-1M, X-MBX-ORDER-COUNT-1S, X-MBX-ORDER-COUNT-1D
    """
    if exchange_name in ("binance.um", "binanceusdm"):
        # USDM Futures
        return ExchangeRateLimitConfig(
            pools={
                "ccxt_rest": PoolConfig("ccxt_rest", "ip", 2400, 40.0, cooldown=30.0),
                "orders": PoolConfig("orders", "account", 300, 30.0, cooldown=10.0),
            },
            endpoint_map={"ccxt_rest": EndpointCosts([("ccxt_rest", 1)])},
            default_costs=EndpointCosts([("ccxt_rest", 1)]),
        )
    elif exchange_name in ("binance.cm", "binancecoinm"):
        # COIN-M Futures
        return ExchangeRateLimitConfig(
            pools={
                "ccxt_rest": PoolConfig("ccxt_rest", "ip", 6000, 100.0, cooldown=30.0),
                "orders": PoolConfig("orders", "account", 300, 30.0, cooldown=10.0),
            },
            endpoint_map={"ccxt_rest": EndpointCosts([("ccxt_rest", 1)])},
            default_costs=EndpointCosts([("ccxt_rest", 1)]),
        )
    else:
        # Spot
        return ExchangeRateLimitConfig(
            pools={
                "ccxt_rest": PoolConfig("ccxt_rest", "ip", 6000, 100.0, cooldown=30.0),
                "orders": PoolConfig("orders", "account", 10, 10.0, cooldown=10.0),
            },
            endpoint_map={"ccxt_rest": EndpointCosts([("ccxt_rest", 1)])},
            default_costs=EndpointCosts([("ccxt_rest", 1)]),
        )


# === OKX ===


def _okx_config() -> ExchangeRateLimitConfig:
    """OKX rate limits (unified across spot/futures).

    - REST: varies heavily by endpoint (2-60 req/2sec per endpoint)
    - Trade endpoints: 60/2sec per account
    - Market data: 20/2sec per IP

    Headers: x-ratelimit-remaining, x-ratelimit-limit, x-ratelimit-reset
    """
    return ExchangeRateLimitConfig(
        pools={
            "ccxt_rest": PoolConfig("ccxt_rest", "ip", 20, 10.0, cooldown=15.0),
            "orders": PoolConfig("orders", "account", 60, 30.0, cooldown=10.0),
        },
        endpoint_map={"ccxt_rest": EndpointCosts([("ccxt_rest", 1)])},
        default_costs=EndpointCosts([("ccxt_rest", 1)]),
    )


# === Bybit ===


def _bybit_config() -> ExchangeRateLimitConfig:
    """Bybit rate limits (unified v5 API).

    - REST: 10 req/sec per endpoint per IP (varies by endpoint)
    - Orders: 10/sec per account
    - Rate limit info in response body: rate_limit_status, rate_limit_reset_ms
    """
    return ExchangeRateLimitConfig(
        pools={
            "ccxt_rest": PoolConfig("ccxt_rest", "ip", 120, 24.0, cooldown=15.0),
            "orders": PoolConfig("orders", "account", 10, 10.0, cooldown=10.0),
        },
        endpoint_map={"ccxt_rest": EndpointCosts([("ccxt_rest", 1)])},
        default_costs=EndpointCosts([("ccxt_rest", 1)]),
    )


# === Kraken ===


def _kraken_config(exchange_name: str) -> ExchangeRateLimitConfig:
    """Kraken rate limits.

    Kraken Spot uses a unique "call counter" system:
    - Each API call adds to a counter (1-6 depending on endpoint)
    - Counter decays at a fixed rate based on verification level
    - Intermediate verified: max counter 20, decay 0.33/sec
    - Pro verified: max counter 20, decay 1.0/sec

    Per-endpoint costs from CCXT:
    - Public: 1.0-1.2 per call
    - Private: 0-6 per call (orders=0, balance=3, ledgers=6)
    - Orders: cost 0 (separate matching engine limit)

    Kraken Futures:
    - Simple: ~1.67 req/sec (600ms rateLimit)
    - No per-endpoint weights documented

    Matching engine (order) limits (separate from REST):
    - Spot: based on tier, typically 60/min for intermediate
    - Futures: separate per-instrument limits
    """
    if exchange_name in ("krakenfutures",):
        return ExchangeRateLimitConfig(
            pools={
                "ccxt_rest": PoolConfig("ccxt_rest", "ip", 10, 1.67, cooldown=15.0),
            },
            endpoint_map={"ccxt_rest": EndpointCosts([("ccxt_rest", 1)])},
            default_costs=EndpointCosts([("ccxt_rest", 1)]),
        )

    # Kraken Spot — counter-decay system maps to token bucket:
    # capacity=20 (max counter), refill=0.33/sec (intermediate) or 1.0/sec (pro)
    # Using intermediate as default — can be overridden via config
    return ExchangeRateLimitConfig(
        pools={
            # Main REST counter (per IP, decays at 0.33/sec for intermediate tier)
            "ccxt_rest": PoolConfig("ccxt_rest", "ip", 20, 0.33, cooldown=30.0),
            # Order matching engine (separate limit per account)
            "orders": PoolConfig("orders", "account", 60, 1.0, cooldown=15.0),
        },
        endpoint_map={
            # CCXT defines per-endpoint costs for Kraken — these are the weights
            # that get passed through CCXT's calculateRateLimiterCost → our throttle override.
            # Key costs: Balance=3, ClosedOrders=3, TradesHistory=6, Ledgers=6
            # Orders have cost=0 in the REST counter (separate matching engine limit)
            "ccxt_rest": EndpointCosts([("ccxt_rest", 1)]),
        },
        default_costs=EndpointCosts([("ccxt_rest", 1)]),
    )


# === Default ===


def _default_config(exchange_name: str, ccxt_exchange=None) -> ExchangeRateLimitConfig:
    """Default conservative config for unknown exchanges.

    Derives from CCXT's rateLimit property if exchange instance provided.
    """
    rate_limit_ms = 50  # default: 20 req/sec
    if ccxt_exchange and hasattr(ccxt_exchange, "rateLimit"):
        rate_limit_ms = ccxt_exchange.rateLimit or 50

    rps = 1000.0 / rate_limit_ms
    capacity = rps * 60  # 1 minute capacity

    return ExchangeRateLimitConfig(
        pools={
            "ccxt_rest": PoolConfig("ccxt_rest", "ip", capacity, rps, cooldown=15.0),
        },
        endpoint_map={"ccxt_rest": EndpointCosts([("ccxt_rest", 1)])},
        default_costs=EndpointCosts([("ccxt_rest", 1)]),
    )


# === Response Header Parsers ===


def parse_binance_headers(headers: dict, rate_limiter) -> None:
    """Parse Binance rate limit headers and sync with rate limiter.

    Binance returns:
    - X-MBX-USED-WEIGHT-1M: weight used in current 1-minute window
    - X-MBX-ORDER-COUNT-1S: orders placed in current 1-second window
    - X-MBX-ORDER-COUNT-1D: orders placed in current day
    """
    if rate_limiter is None:
        return

    # Try both casing conventions (CCXT normalizes to lowercase)
    used_weight = headers.get("x-mbx-used-weight-1m") or headers.get("X-MBX-USED-WEIGHT-1M")
    if used_weight:
        try:
            rate_limiter.sync_from_exchange("ccxt_rest", used=int(used_weight))
        except Exception:
            pass

    order_count = headers.get("x-mbx-order-count-1s") or headers.get("X-MBX-ORDER-COUNT-1S")
    if order_count and "orders" in rate_limiter.config.pools:
        try:
            rate_limiter.sync_from_exchange("orders", used=int(order_count))
        except Exception:
            pass


def parse_okx_headers(headers: dict, rate_limiter) -> None:
    """Parse OKX rate limit headers and sync with rate limiter.

    OKX returns:
    - x-ratelimit-remaining: remaining requests for this endpoint
    - x-ratelimit-limit: total limit for this endpoint
    - x-ratelimit-reset: timestamp when limit resets (ms)
    """
    if rate_limiter is None:
        return

    remaining = headers.get("x-ratelimit-remaining") or headers.get("X-RateLimit-Remaining")
    if remaining:
        try:
            rate_limiter.sync_from_exchange("ccxt_rest", remaining=float(remaining))
        except Exception:
            pass


HEADER_PARSERS = {
    "binance": parse_binance_headers,
    "binanceusdm": parse_binance_headers,
    "binancecoinm": parse_binance_headers,
    "okx": parse_okx_headers,
}


def get_header_parser(exchange_name: str):
    """Get the response header parser for an exchange."""
    name = exchange_name.lower().split(".")[0]
    return HEADER_PARSERS.get(name)
