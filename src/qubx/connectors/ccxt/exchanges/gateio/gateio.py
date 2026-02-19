from typing import Any

import ccxt.pro as cxp

from qubx import logger

from ...adapters.polling_adapter import PollingConfig, PollingToWebSocketAdapter
from ..base import CcxtFuturePatchMixin

FUNDING_RATE_DEFAULT_POLL_MINUTES = 5


class GateioFutures(CcxtFuturePatchMixin, cxp.gate):
    """
    Custom Gate.io futures exchange class with polling-based funding rate support.

    CCXT's Gate.io does not implement watchFundingRates, so we use the
    PollingToWebSocketAdapter to convert fetchFundingRates into a watch-compatible
    interface, following the same pattern as HyperliquidEnhanced.
    """

    def __init__(self, config=None):
        super().__init__(config or {})
        self._funding_rate_adapter: PollingToWebSocketAdapter | None = None

    async def watch_funding_rates(
        self, symbols: list[str] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Watch funding rates using polling adapter (CCXT awaitable pattern).

        Polls fetchFundingRates at regular intervals and transforms the response
        to match ccxt_convert_funding_rate expectations.
        """
        if params is None:
            params = {}

        await self.load_markets()

        poll_interval_minutes = params.get("poll_interval_minutes", FUNDING_RATE_DEFAULT_POLL_MINUTES)
        poll_interval_seconds = poll_interval_minutes * 60

        if self._funding_rate_adapter is None:
            logger.debug(f"Starting Gate.io funding rate adapter, poll interval: {poll_interval_minutes}min")
            self._funding_rate_adapter = PollingToWebSocketAdapter(
                fetch_method=self.fetch_funding_rates,
                symbols=symbols or [],
                params=params or {},
                config=PollingConfig(poll_interval_seconds=poll_interval_seconds),
            )
        else:
            if symbols is not None:
                await self._funding_rate_adapter.update_symbols(symbols)

        funding_data = await self._funding_rate_adapter.get_next_data()

        # Transform to match ccxt_convert_funding_rate expectations:
        # - timestamp: must be set (Gate.io returns None)
        # - fundingRate: pass through
        # - interval: pass through (already parsed to "8h" etc by CCXT)
        # - nextFundingTime: mapped from fundingTimestamp
        # - markPrice, indexPrice: pass through
        transformed = {}
        current_time_ms = self.milliseconds()

        if isinstance(funding_data, dict):
            for idx, (symbol, rate_info) in enumerate(funding_data.items()):
                if not isinstance(rate_info, dict):
                    continue

                transformed_info = rate_info.copy()
                # Unique timestamp per symbol (Gate.io returns None for timestamp)
                transformed_info["timestamp"] = current_time_ms + idx
                # Map fundingTimestamp -> nextFundingTime
                transformed_info["nextFundingTime"] = rate_info.get("fundingTimestamp")
                transformed[symbol] = transformed_info

        return transformed

    async def un_watch_funding_rates(self, symbols: list[str] | None = None) -> None:
        """Stop watching funding rates."""
        if self._funding_rate_adapter:
            if symbols:
                await self._funding_rate_adapter.remove_symbols(symbols)
                if not self._funding_rate_adapter.is_watching():
                    await self._funding_rate_adapter.stop()
                    self._funding_rate_adapter = None
            else:
                await self._funding_rate_adapter.stop()
                self._funding_rate_adapter = None
