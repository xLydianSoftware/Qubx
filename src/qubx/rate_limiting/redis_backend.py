"""
Redis-based rate limit backend for cross-bot coordination.

Uses Lua scripts for atomic token bucket operations in Redis.
Multiple bots sharing an exchange account or egress IP coordinate
through shared Redis keys.
"""

import time

from qubx import logger

from .backend import IRateLimitBackend

# Lua script: atomic token bucket acquire
# Returns: wait_time_seconds (0 if immediate, >0 if had to wait conceptually)
# The script refills tokens, checks availability, and decrements atomically.
_LUA_ACQUIRE = """
local key = KEYS[1]
local weight = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local refill_rate = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- Get current state or initialize
local tokens = tonumber(redis.call('HGET', key, 'tokens') or capacity)
local last_refill = tonumber(redis.call('HGET', key, 'last_refill') or now)

-- Refill tokens based on elapsed time
local elapsed = now - last_refill
local tokens_to_add = elapsed * refill_rate
tokens = math.min(capacity, tokens + tokens_to_add)

-- Check if we have enough tokens
if tokens >= weight then
    tokens = tokens - weight
    redis.call('HSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 300)  -- 5 min TTL to auto-cleanup stale keys
    return 0  -- no wait needed
else
    -- Calculate wait time
    local tokens_needed = weight - tokens
    local wait_time = tokens_needed / refill_rate

    -- Update last_refill to now (tokens were refilled but not enough)
    redis.call('HSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 300)
    return wait_time * 1000  -- return milliseconds
end
"""

_LUA_GET_REMAINING = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local tokens = tonumber(redis.call('HGET', key, 'tokens'))
if tokens == nil then return capacity * 1000 end

local last_refill = tonumber(redis.call('HGET', key, 'last_refill') or now)
local elapsed = now - last_refill
local tokens_to_add = elapsed * refill_rate
tokens = math.min(capacity, tokens + tokens_to_add)

return math.floor(tokens * 1000)  -- return as millitokens for precision
"""

_LUA_SET_REMAINING = """
local key = KEYS[1]
local remaining = tonumber(ARGV[1])
local now = tonumber(ARGV[2])

redis.call('HSET', key, 'tokens', remaining, 'last_refill', now)
redis.call('EXPIRE', key, 300)
return 1
"""


class RedisBackend(IRateLimitBackend):
    """Redis-based token bucket backend for cross-bot coordination.

    Uses atomic Lua scripts so multiple bots can safely share rate limit
    pools without races. Keys auto-expire after 5 minutes of inactivity.

    Args:
        redis_url: Redis connection URL (e.g., "redis://redis.platform.svc:6379/0")
    """

    def __init__(self, redis_url: str):
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._acquire_script = self._redis.register_script(_LUA_ACQUIRE)
        self._get_remaining_script = self._redis.register_script(_LUA_GET_REMAINING)
        self._set_remaining_script = self._redis.register_script(_LUA_SET_REMAINING)
        logger.info(f"Redis rate limit backend connected: {redis_url}")

    async def acquire(self, key: str, weight: float, capacity: float, refill_rate: float) -> float:
        now = time.time()
        while True:
            wait_ms = await self._acquire_script(
                keys=[key],
                args=[weight, capacity, refill_rate, now],
            )
            wait_ms = float(wait_ms)
            if wait_ms <= 0:
                return 0.0
            # Need to wait — sleep and retry
            wait_s = wait_ms / 1000.0
            await __import__("asyncio").sleep(wait_s)
            now = time.time()

    async def get_remaining(self, key: str) -> float | None:
        # We need capacity and refill_rate to compute remaining after refill,
        # but we don't have them here. Return raw stored tokens.
        raw = await self._redis.hget(key, "tokens")
        if raw is None:
            return None
        return float(raw)

    async def set_remaining(self, key: str, remaining: float) -> None:
        now = time.time()
        await self._set_remaining_script(
            keys=[key],
            args=[remaining, now],
        )

    async def close(self) -> None:
        await self._redis.close()
