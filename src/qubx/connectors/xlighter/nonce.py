import asyncio

from .client import LighterClient


class LighterNonceProvider:
    def __init__(self, client: LighterClient):
        self.client = client
        self._nonce = None
        self._lock = asyncio.Lock()

    async def get_nonce(self) -> int:
        async with self._lock:
            if self._nonce is None:
                self._nonce = await self.client.next_nonce()
            else:
                self._nonce += 1
            return self._nonce

    async def resync(self) -> None:
        self._nonce = None
