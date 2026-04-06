"""
Rate limiting configuration data classes.

Defines the declarative configuration that each exchange connector provides
to describe its rate limit pools, endpoint weights, and scoping rules.
"""

from dataclasses import dataclass, field


@dataclass
class PoolConfig:
    """Configuration for a single rate limit pool.

    A pool represents one independent rate limit enforced by the exchange.
    Exchanges typically have multiple pools (e.g., request weight + order count).

    Args:
        name: Pool identifier (e.g., "request_weight", "orders", "sendtx")
        scope: What the limit is scoped to: "ip", "account", "address", or "local"
        capacity: Maximum tokens in the bucket
        refill_rate: Tokens replenished per second (0 for quota pools)
        pool_type: "rate" for time-based refill, "quota" for externally-managed
        cooldown: Seconds to close gate when rate limit is hit
    """

    name: str
    scope: str
    capacity: float
    refill_rate: float
    pool_type: str = "rate"
    cooldown: float = 15.0


@dataclass
class EndpointCosts:
    """Cost of an endpoint across multiple pools.

    Each tuple is (pool_name, weight) — the endpoint consumes `weight` tokens
    from the named pool. A single request can consume from multiple pools.

    Example:
        >>> # Binance create_order: 1 weight from request pool + 1 from order pool
        >>> EndpointCosts([("request_weight", 1), ("orders", 1)])
    """

    costs: list[tuple[str, float]]


@dataclass
class ExchangeRateLimitConfig:
    """Complete rate limit configuration for an exchange.

    Connectors provide this declaratively. The ExchangeRateLimiter uses it
    to create pools and map endpoint calls to the correct pools/weights.

    Args:
        pools: Named pool configurations
        endpoint_map: Maps endpoint names to their costs across pools
        default_costs: Fallback costs for unmapped endpoints
        gate_max_wait: Max seconds to wait for a closed gate before raising timeout
        metrics_interval: Seconds between metric emissions (0 to disable)
    """

    pools: dict[str, PoolConfig] = field(default_factory=dict)
    endpoint_map: dict[str, EndpointCosts] = field(default_factory=dict)
    default_costs: EndpointCosts = field(default_factory=lambda: EndpointCosts([]))
    gate_max_wait: float = 15.0
    metrics_interval: float = 60.0
