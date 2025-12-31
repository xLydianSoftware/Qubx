import asyncio
import time

from .client import LighterClient


class LighterNonceProvider:
    def __init__(self, client: LighterClient, resync_cooldown: float = 1.0):
        self.client = client
        self._nonce = None
        self._lock = asyncio.Lock()
        self._last_resync_time: float = 0.0
        self._resync_cooldown = resync_cooldown

    async def get_nonce(self) -> int:
        async with self._lock:
            if self._nonce is None:
                self._nonce = await self.client.next_nonce()
            else:
                self._nonce += 1
            return self._nonce

    async def resync(self) -> None:
        current_time = time.monotonic()
        if current_time - self._last_resync_time < self._resync_cooldown:
            # Skip resync if we recently resynced
            return
        self._last_resync_time = current_time
        self._nonce = None
