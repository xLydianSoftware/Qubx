"""
Lighter exchange rate limit configurations.

Based on official documentation: https://apidocs.lighter.xyz/docs/rate-limits

Rate Limits:
- Premium accounts: 24000 weighted REST API requests per minute
- Standard accounts: 60 requests per minute
- WebSocket: 100 sessions, 1000 subscriptions, 10 unique accounts per IP
"""

from qubx.utils.rate_limiter import RateLimiterRegistry, TokenBucketRateLimiter

# Account type rate limits (requests per minute)
PREMIUM_REST_LIMIT = 24000
STANDARD_REST_LIMIT = 60

# Default WebSocket subscription rate (subscriptions per second)
DEFAULT_WS_SUB_LIMIT = 5

# REST API Endpoint Weights (per Lighter documentation)
# https://apidocs.lighter.xyz/docs/rate-limits
WEIGHT_SEND_TX = 6  # /api/v1/sendTx, /api/v1/sendTxBatch, /api/v1/nextNonce
WEIGHT_ROOT_INFO = 100  # /, /info
WEIGHT_PUBLIC_POOLS = 50  # /api/v1/publicPools, /api/v1/txFromL1TxHash, /api/v1/candlesticks
WEIGHT_CANDLESTICKS = 50  # Same as PUBLIC_POOLS
WEIGHT_FUNDING = 300  # Funding endpoints (same weight as candlesticks)
WEIGHT_ACCOUNT_ORDERS = 100  # /api/v1/accountInactiveOrders, /api/v1/deposit/latest, /api/v1/pnl
WEIGHT_API_KEYS = 150  # /api/v1/apikeys
WEIGHT_DEFAULT = 300  # All other endpoints

# Transaction-specific limits for standard accounts (for future implementation)
# These are separate per-endpoint limits, not weight-based
TX_LIMIT_L2_WITHDRAW = 2  # requests per minute
TX_LIMIT_L2_UPDATE_LEVERAGE = 1  # requests per minute
TX_LIMIT_L2_CREATE_SUB_ACCOUNT = 2  # requests per minute
TX_LIMIT_L2_CREATE_PUBLIC_POOL = 2  # requests per minute
TX_LIMIT_L2_CHANGE_PUB_KEY = 6  # requests per 10 seconds (converted: 0.6/min)
TX_LIMIT_L2_TRANSFER = 1  # requests per minute


def create_lighter_rate_limiters(
    account_type: str = "premium",
    rest_rate_limit: int | None = None,
    ws_subscription_rate_limit: int | None = None,
) -> RateLimiterRegistry:
    """
    Create rate limiter registry for Lighter exchange.

    Creates two rate limiters:
    - "rest": For REST API requests (weight-based)
    - "ws_sub": For WebSocket subscriptions

    Args:
        account_type: "premium" or "standard" (default: "premium")
        rest_rate_limit: Override REST rate limit (requests per minute)
        ws_subscription_rate_limit: Override WS subscription rate (subscriptions per second)

    Returns:
        RateLimiterRegistry with configured limiters

    Example:
        >>> registry = create_lighter_rate_limiters(account_type="premium")
        >>> rest_limiter = registry.get_limiter("rest")
        >>> await rest_limiter.acquire(weight=50)
    """
    registry = RateLimiterRegistry()

    # Determine REST API rate limit
    if rest_rate_limit is not None:
        rest_limit = rest_rate_limit
    elif account_type == "premium":
        rest_limit = PREMIUM_REST_LIMIT
    elif account_type == "standard":
        rest_limit = STANDARD_REST_LIMIT
    else:
        raise ValueError(f"Invalid account_type: {account_type}. Must be 'premium' or 'standard'")

    # Create REST API rate limiter (requests per minute -> per second)
    rest_limiter = TokenBucketRateLimiter(
        capacity=rest_limit,
        refill_rate=rest_limit / 60.0,
        name=f"lighter_rest_{account_type}",
    )
    registry.register_limiter("rest", rest_limiter)

    # Create WebSocket subscription rate limiter
    ws_limit = ws_subscription_rate_limit if ws_subscription_rate_limit is not None else DEFAULT_WS_SUB_LIMIT
    ws_limiter = TokenBucketRateLimiter(
        capacity=float(ws_limit),
        refill_rate=float(ws_limit),
        name="lighter_ws_sub",
    )
    registry.register_limiter("ws_sub", ws_limiter)

    return registry
