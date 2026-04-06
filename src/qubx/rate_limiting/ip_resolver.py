"""
Egress IP resolver for rate limit scoping.

Discovers the bot's public egress IP address periodically, updating
rate limiter scope IDs when the IP changes (e.g., after Cilium egress
policy update on the platform).
"""

import asyncio
import socket

import aiohttp

from qubx import logger

# Reliable IP echo services (try in order)
_IP_ECHO_URLS = [
    "https://ifconfig.me/ip",
    "https://api.ipify.org",
    "https://checkip.amazonaws.com",
]



class EgressIPResolver:
    """Discovers and tracks this bot's egress IP.

    Call start() to begin periodic discovery. When the IP changes,
    registered callbacks are invoked with (old_ip, new_ip).

    Usage:
        >>> resolver = EgressIPResolver(check_interval=60)
        >>> resolver.on_ip_changed(lambda old, new: print(f"{old} → {new}"))
        >>> await resolver.start()
        >>> resolver.current_ip  # "1.2.3.4"
    """

    def __init__(self, check_interval: float = 60.0, initial_ip: str | None = None):
        self._check_interval = check_interval
        self._current_ip = initial_ip
        self._callbacks: list = []
        self._task: asyncio.Task | None = None

    @property
    def current_ip(self) -> str | None:
        return self._current_ip

    def on_ip_changed(self, callback) -> None:
        """Register callback: callback(old_ip: str | None, new_ip: str)"""
        self._callbacks.append(callback)

    async def discover(self) -> str | None:
        """Discover current egress IP (tries multiple services)."""
        for url in _IP_ECHO_URLS:
            try:
                connector = aiohttp.TCPConnector(family=socket.AF_INET)
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            ip = (await resp.text()).strip()
                            if ip:
                                return ip
            except Exception:
                continue
        logger.warning("Failed to discover egress IP from all services")
        return None

    async def start(self) -> None:
        """Start periodic IP discovery."""
        # Do initial discovery
        ip = await self.discover()
        if ip:
            self._current_ip = ip
            logger.info(f"Egress IP discovered: {ip}")

        # Start monitoring loop
        self._task = asyncio.ensure_future(self._monitor_loop())

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _monitor_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._check_interval)
                new_ip = await self.discover()
                if new_ip and new_ip != self._current_ip:
                    old_ip = self._current_ip
                    self._current_ip = new_ip
                    logger.warning(f"Egress IP changed: {old_ip} → {new_ip}")
                    for cb in self._callbacks:
                        try:
                            cb(old_ip, new_ip)
                        except Exception as e:
                            logger.opt(colors=False).error(f"IP change callback error: {e}")
        except asyncio.CancelledError:
            pass
