# Exchange Rate Limiting

Qubx provides a multi-pool rate limiting engine that connectors use to comply with exchange API limits. The engine handles token bucket acquisition, gate-based cooldowns, exchange state synchronization, and metric collection.

## Concepts

### Pools

A **pool** represents one independent rate limit enforced by an exchange. Most exchanges have multiple pools (e.g., REST request weight + order count + WebSocket message rate).

There are two pool types:

- **Rate pool** (`pool_type="rate"`) — Time-based token bucket. Tokens refill at a constant rate. When depleted, a gate closes and reopens on a timer. Used for request weight limits, message rate limits, etc.

- **Quota pool** (`pool_type="quota"`) — Externally managed budget with no time-based refill. The exchange controls replenishment (e.g., Lighter's volume quota which is replenished by trading volume). When depleted, the gate closes and stays closed until the exchange reports positive remaining via `sync_from_exchange()`.

### Endpoints

An **endpoint** maps an API call to one or more pool costs. A single request can consume tokens from multiple pools simultaneously.

```python
# Creating an order consumes from three pools:
"send_tx:create_order": EndpointCosts([
    ("ws_messages", 1),      # 1 token from WS message rate pool
    ("sendtx_rate", 1),      # 1 token from sendTx rate pool
    ("volume_quota", 1),     # 1 token from volume quota pool
])

# Cancelling only consumes from two (no volume quota):
"send_tx:cancel_order": EndpointCosts([
    ("ws_messages", 1),
    ("sendtx_rate", 1),
])
```

### Gates

Each pool has a **gate** (an `asyncio.Event`). When open, `acquire()` proceeds immediately. When closed, all requests for that pool block until the gate reopens.

- **Rate pool gates** reopen automatically after a cooldown timer.
- **Quota pool gates** never reopen on a timer — only when the exchange reports positive remaining via `sync_from_exchange()`, or on reconnection via `reset_gates()`.

### Scopes

Pools are scoped to what the exchange rate-limits by:

- `"ip"` — Per IP address (REST API weight limits)
- `"account"` — Per trading account
- `"address"` — Per L1/wallet address (Lighter sendTx limits)

The scope determines the backend storage key, enabling future cross-bot coordination via a shared Redis backend.

## Defining Configuration

Each connector declares its rate limit configuration using `PoolConfig`, `EndpointCosts`, and `ExchangeRateLimitConfig`. Use the `@rate_limit_config` registry decorator so the engine can be instantiated automatically.

```python
from qubx.connectors.registry import rate_limit_config
from qubx.rate_limiting import EndpointCosts, ExchangeRateLimitConfig, PoolConfig


@rate_limit_config("myexchange")
def create_my_rate_limit_config(
    exchange_name: str = "MYEXCHANGE",
    rate_limit_cooldown: float = 15.0,
    gate_max_wait: float = 15.0,
) -> ExchangeRateLimitConfig:
    pools = {
        # REST API weight pool — 1200 weight/min, scoped per IP
        "rest_weight": PoolConfig(
            name="rest_weight",
            scope="ip",
            capacity=1200,
            refill_rate=1200 / 60.0,  # 20 tokens/sec
            pool_type="rate",
            cooldown=rate_limit_cooldown,
        ),
        # Order rate pool — 10 orders/sec, scoped per account
        "orders": PoolConfig(
            name="orders",
            scope="account",
            capacity=10,
            refill_rate=10.0,
            pool_type="rate",
            cooldown=rate_limit_cooldown,
        ),
    }

    endpoint_map = {
        "fetch_candles": EndpointCosts([("rest_weight", 50)]),
        "fetch_orderbook": EndpointCosts([("rest_weight", 100)]),
        "create_order": EndpointCosts([("rest_weight", 1), ("orders", 1)]),
        "cancel_order": EndpointCosts([("rest_weight", 1)]),
    }

    return ExchangeRateLimitConfig(
        pools=pools,
        endpoint_map=endpoint_map,
        default_costs=EndpointCosts([("rest_weight", 10)]),
        gate_max_wait=gate_max_wait,
    )
```

### PoolConfig fields

| Field | Description |
|-------|-------------|
| `name` | Pool identifier (must match key in `pools` dict) |
| `scope` | What the limit is scoped to: `"ip"`, `"account"`, `"address"` |
| `capacity` | Maximum tokens in the bucket |
| `refill_rate` | Tokens replenished per second (0 for quota pools) |
| `pool_type` | `"rate"` (time-based) or `"quota"` (externally managed) |
| `cooldown` | Default seconds to close gate on rate limit hit |

### ExchangeRateLimitConfig fields

| Field | Description |
|-------|-------------|
| `pools` | Dict of pool name to `PoolConfig` |
| `endpoint_map` | Dict of endpoint name to `EndpointCosts` |
| `default_costs` | Fallback costs for unmapped endpoints |
| `gate_max_wait` | Max seconds to wait for a closed gate before raising `RateLimitGateTimeout` |

## Connector Integration

### Acquiring budget before API calls

Call `acquire(endpoint)` before every exchange request. It blocks until all pools for that endpoint have sufficient budget.

```python
# In your WebSocket manager or REST client:
async def send_order(self, order_params):
    if self._rate_limiter:
        await self._rate_limiter.acquire("create_order")

    await self._ws.send(order_params)
```

The engine iterates the endpoint's pool costs in order, acquiring from each pool. If any pool's gate is closed, `acquire()` blocks (for rate pools) or raises immediately (for quota pools).

### Syncing state from exchange responses

When the exchange reports actual usage in response headers or WebSocket messages, call `sync_from_exchange()` to correct drift between the model and reality.

```python
# From a REST response with rate limit headers:
remaining = int(response.headers["X-RateLimit-Remaining"])
limiter.sync_from_exchange("rest_weight", remaining=remaining)

# From a WebSocket message with quota info:
async def _handle_tx_response(self, message: dict):
    if "volume_quota_remaining" in message and self._rate_limiter:
        remaining = int(message["volume_quota_remaining"])
        self._rate_limiter.sync_from_exchange("volume_quota", remaining=remaining)
```

For quota pools, `sync_from_exchange()` also:

- **Reopens the gate** when remaining goes positive (the key fix for quota storms)
- **Grows `capacity`** to `max(capacity, remaining)` when no explicit capacity is provided (tracks real account quota when the exchange only reports remaining)

### Reporting rate limit hits

When the exchange explicitly tells you you've been rate limited (HTTP 429, WebSocket error code), call `report_limit_hit()` to close gates:

```python
async def _handle_error(self, error: dict):
    match error["code"]:
        case 23000:
            # General rate limit hit
            self._rate_limiter.report_limit_hit(
                pool_name="rest_weight",
                reason=f"error 23000: {error['message']}",
            )
        case 30009:
            # WebSocket message rate limit
            self._rate_limiter.report_limit_hit(
                pool_name="ws_messages",
                reason="error 30009",
            )
```

You can target a specific pool, an endpoint (closes all pools in its costs), or pass nothing to close all rate pools.

### Resetting on reconnection

Call `reset_gates()` when the WebSocket reconnects to clear stale gate state:

```python
async def connect(self):
    await super().connect()
    if self._rate_limiter:
        self._rate_limiter.reset_gates()
```

## Handling RateLimitGateTimeout

When a gate doesn't reopen in time (rate pools) or a quota pool is depleted, `acquire()` raises `RateLimitGateTimeout`. The exception has a `pool_name` attribute so connectors can implement pool-specific fallback logic.

### Fallback pattern for quota pools

Some exchanges allow limited operations even when quota is exhausted. For example, Lighter allows 1 free transaction per 15 seconds when volume quota is depleted. Model this as a separate rate pool with fallback logic:

```python
# In rate limit config — add a free-tx pool:
"sendtx_free": PoolConfig(
    name="sendtx_free",
    scope="address",
    capacity=1,
    refill_rate=1.0 / 15.0,  # 1 free tx per 15s
    pool_type="rate",
    cooldown=15.0,
),

# Endpoint for fallback:
"send_tx_free": EndpointCosts([("sendtx_free", 1)]),
```

```python
# In the connector — catch and fall back:
async def send_tx(self, tx_type, tx_info):
    if self._rate_limiter:
        endpoint = endpoint_for_tx_type(tx_type)
        try:
            await self._rate_limiter.acquire(endpoint)
        except RateLimitGateTimeout as e:
            if e.pool_name == "volume_quota" and is_quota_consuming(tx_type):
                # ws_messages + sendtx_rate already consumed before
                # volume_quota was checked, so only acquire free-tx budget
                logger.info("Volume quota exhausted, using free sendTx allowance")
                await self._rate_limiter.acquire("send_tx_free")
            else:
                raise

    await self._ws.send(tx_info)
```

This pattern ensures that when quota is exhausted, orders still trickle through at the free rate instead of being completely blocked.

## Metrics

The engine collects per-pool metrics via `collect_metrics()`. Call it periodically and emit via the strategy context:

```python
for m in await rate_limiter.collect_metrics():
    ctx.emitter.emit(m["name"], m["value"], m["tags"])
```

Available metrics (all tagged with `exchange`, `pool`, `scope`):

| Metric | Description |
|--------|-------------|
| `rate_limit.remaining` | Current tokens available |
| `rate_limit.capacity` | Pool capacity (grows for quota pools) |
| `rate_limit.utilization` | `1 - remaining/capacity` |
| `rate_limit.gate_closed` | 1.0 if gate is closed, 0.0 if open |
| `rate_limit.hits` | Cumulative rate limit hits reported |
| `rate_limit.wait_seconds` | Cumulative time spent waiting for budget |
| `rate_limit.consumed` | Cumulative tokens consumed |

## Architecture

```
ExchangeRateLimiter (engine.py)
├── acquire(endpoint)          → delegates to pool.acquire()
├── report_limit_hit()         → delegates to pool.close_gate()
├── sync_from_exchange()       → delegates to pool.sync()
├── reset_gates()              → delegates to pool.reset_gate()
├── collect_metrics()          → delegates to pool.get_state()
│
├── RatePool (pools.py)        — time-based token bucket
│   ├── acquire()              waits for gate, then backend.acquire()
│   ├── close_gate()           clears gate, schedules timed reopen
│   ├── sync()                 updates backend token count
│   └── get_state()            reads remaining from backend
│
└── QuotaPool (pools.py)       — externally managed
    ├── acquire()              fails fast if depleted (RateLimitGateTimeout)
    ├── close_gate()           clears gate, NO timed reopen
    ├── sync()                 updates remaining, reopens gate if positive
    └── get_state()            returns in-memory remaining
```

The engine contains no pool-type branching — all behavior differences are encapsulated in the pool subclasses.
