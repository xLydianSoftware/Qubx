"""
Redis-based rate limit backend for cross-bot coordination.

Uses Lua scripts for atomic token bucket operations in Redis.
Multiple bots sharing an exchange account or egress IP coordinate
through shared Redis keys.
"""

import asyncio
import time
from weakref import WeakKeyDictionary

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
    redis.call('EXPIRE', key, 600)  -- 10 min TTL to auto-cleanup stale keys
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
if tokens == nil then return -1 end

-- Compute refilled value and persist (same logic as acquire)
if refill_rate > 0 then
    local last_refill = tonumber(redis.call('HGET', key, 'last_refill') or now)
    local elapsed = now - last_refill
    local tokens_to_add = elapsed * refill_rate
    tokens = math.min(capacity, tokens + tokens_to_add)
    -- Persist refilled state so raw Redis values reflect reality
    redis.call('HSET', key, 'tokens', tokens, 'last_refill', now)
end

-- Refresh TTL on read (keeps keys alive while bot is running)
redis.call('EXPIRE', key, 600)

return math.floor(tokens * 1000)  -- return as millitokens for precision
"""

_LUA_SET_REMAINING = """
local key = KEYS[1]
local remaining = tonumber(ARGV[1])
local now = tonumber(ARGV[2])

redis.call('HSET', key, 'tokens', remaining, 'last_refill', now)
redis.call('EXPIRE', key, 600)
return 1
"""


class RedisBackend(IRateLimitBackend):
    """Redis-based token bucket backend for cross-bot coordination.

    Uses atomic Lua scripts so multiple bots can safely share rate limit
    pools without races. Keys auto-expire after 10 minutes of inactivity.

    Args:
        redis_url: Redis connection URL (e.g., "redis://redis.platform.svc:6379/0")
    """

    def __init__(self, redis_url: str):
        # - fail fast on misconfiguration, but don't create the client here:
        #   the async Redis client (with single_connection_client=True) holds an
        #   internal asyncio.Lock that binds to whichever event loop first awaits
        #   it. The same backend is shared between the simulation warmup loop and
        #   the live ccxt.pro AsyncThreadLoop, so we cache one client per loop.
        import redis.asyncio as _aioredis  # noqa: F401

        self._redis_url = redis_url
        self._loop_clients: WeakKeyDictionary[asyncio.AbstractEventLoop, tuple] = WeakKeyDictionary()
        logger.info(f"Redis rate limit backend configured: {redis_url}")

    def _scripts_for_current_loop(self):
        """
        Return (acquire, get_remaining, set_remaining) Lua scripts bound to the
        Redis client for the currently-running event loop, creating the client
        lazily on first use per loop.
        """
        import redis.asyncio as aioredis

        loop = asyncio.get_running_loop()
        entry = self._loop_clients.get(loop)
        if entry is None:
            client = aioredis.from_url(
                self._redis_url, decode_responses=True, single_connection_client=True
            )
            scripts = (
                client.register_script(_LUA_ACQUIRE),
                client.register_script(_LUA_GET_REMAINING),
                client.register_script(_LUA_SET_REMAINING),
            )
            self._loop_clients[loop] = (client, scripts)
            return scripts
        return entry[1]

    async def acquire(self, key: str, weight: float, capacity: float, refill_rate: float) -> float:
        acquire_script, _, _ = self._scripts_for_current_loop()
        now = time.time()
        while True:
            wait_ms = await acquire_script(
                keys=[key],
                args=[weight, capacity, refill_rate, now],
            )
            wait_ms = float(wait_ms)
            if wait_ms <= 0:
                return 0.0
            # Need to wait — sleep and retry
            wait_s = wait_ms / 1000.0
            await asyncio.sleep(wait_s)
            now = time.time()

    async def get_remaining(self, key: str, capacity: float = 0, refill_rate: float = 0) -> float | None:
        _, get_remaining_script, _ = self._scripts_for_current_loop()
        now = time.time()
        result = await get_remaining_script(
            keys=[key],
            args=[capacity, refill_rate, now],
        )
        result = float(result)
        if result < 0:
            return None
        return result / 1000.0  # convert millitokens back

    async def set_remaining(self, key: str, remaining: float) -> None:
        _, _, set_remaining_script = self._scripts_for_current_loop()
        now = time.time()
        await set_remaining_script(
            keys=[key],
            args=[remaining, now],
        )

    async def close(self) -> None:
        """
        Close the Redis client for the currently-running loop, if one was
        created for it. Clients for other loops are left to be closed when
        those loops are finalized (their entries drop from the WeakKeyDictionary).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        entry = self._loop_clients.pop(loop, None)
        if entry is not None:
            try:
                await entry[0].close()
            except Exception:
                pass
