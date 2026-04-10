"""
Rate limit manager — owns the full rate limiting lifecycle.

Created once by the runner. Handles backend creation, egress IP discovery,
and per-exchange rate limiter creation via the connector registry.
"""

import asyncio
import concurrent.futures

from qubx import logger

from .backend import IRateLimitBackend, InMemoryBackend
from .engine import ExchangeRateLimiter


class RateLimitManager:
    """Central manager for exchange rate limiting.

    Encapsulates all rate limiting setup so the runner stays clean.
    Connectors receive a ready-to-use ExchangeRateLimiter via kwargs.

    Usage (runner):
        >>> manager = RateLimitManager(config.live.rate_limiting, loop)
        >>> rl = manager.get_or_create("LIGHTER", "xlighter")
        >>> exchange_config.params["rate_limiter"] = rl
    """

    def __init__(self, config, loop: asyncio.AbstractEventLoop):
        """
        Args:
            config: RateLimitingConfig from YAML (or None to disable)
            loop: Shared event loop for async operations
        """
        self._loop = loop
        self._config = config
        self._rate_limiters: dict[str, ExchangeRateLimiter] = {}
        self._ip_resolver = None

        if config is None:
            self._backend = None
            self._egress_ip = None
            return

        # Create backend
        self._backend = self._create_backend(config)

        # Resolve egress IP
        self._egress_ip = self._resolve_egress_ip(config, loop)

    def _create_backend(self, config) -> IRateLimitBackend:
        if config.backend == "redis" and config.redis_url:
            try:
                from .redis_backend import RedisBackend

                return RedisBackend(config.redis_url)
            except Exception as e:
                logger.error(f"Failed to create Redis rate limit backend: {e}, falling back to local")
        return InMemoryBackend()

    def _resolve_egress_ip(self, config, loop: asyncio.AbstractEventLoop) -> str | None:
        if config.egress_ip != "auto":
            return config.egress_ip if config.egress_ip else None

        from .ip_resolver import EgressIPResolver

        self._ip_resolver = EgressIPResolver(check_interval=config.ip_check_interval)

        # Initial discovery (synchronous on shared loop)
        try:
            future = asyncio.run_coroutine_threadsafe(self._ip_resolver.discover(), loop)
            ip = future.result(timeout=10)
            if ip:
                self._ip_resolver._current_ip = ip
                logger.info(f"Egress IP discovered: {ip}")
        except (concurrent.futures.TimeoutError, Exception) as e:
            logger.warning(f"Failed to discover egress IP: {e}")

        # Start periodic monitoring
        asyncio.run_coroutine_threadsafe(self._ip_resolver.start(), loop)

        # Register callback to update all rate limiters when IP changes
        self._ip_resolver.on_ip_changed(self._on_ip_changed)

        return self._ip_resolver.current_ip

    def _on_ip_changed(self, old_ip: str | None, new_ip: str) -> None:
        for rl in self._rate_limiters.values():
            rl.update_scope_id("ip", f"ip_{new_ip}")

    def get_or_create(self, exchange_name: str, connector_name: str) -> ExchangeRateLimiter | None:
        """Get or create a rate limiter for an exchange.

        Uses the connector registry to find the rate limit config factory.

        Args:
            exchange_name: Exchange name (e.g., "LIGHTER", "BINANCE.UM")
            connector_name: Connector type (e.g., "xlighter", "ccxt")

        Returns:
            ExchangeRateLimiter or None if rate limiting is disabled or no config registered
        """
        if self._backend is None:
            return None

        if exchange_name in self._rate_limiters:
            return self._rate_limiters[exchange_name]

        from qubx.connectors.registry import ConnectorRegistry

        rl_config = ConnectorRegistry.get_rate_limit_config(connector_name, exchange_name)
        if rl_config is None:
            return None

        scope_ids = {}
        if self._egress_ip:
            scope_ids["ip"] = f"ip_{self._egress_ip}"

        rate_limiter = ExchangeRateLimiter(
            exchange_name.lower(),
            rl_config,
            backend=self._backend,
            scope_ids=scope_ids,
            event_loop=self._loop,
        )
        self._rate_limiters[exchange_name] = rate_limiter
        logger.info(f"Rate limiter created for {exchange_name} ({connector_name}): {list(rl_config.pools.keys())}")
        return rate_limiter

    @property
    def rate_limiters(self) -> dict[str, ExchangeRateLimiter]:
        """All created rate limiters, keyed by exchange name."""
        return self._rate_limiters

    def stop(self) -> None:
        """Stop background tasks (IP resolver)."""
        if self._ip_resolver:
            self._ip_resolver.stop()
