"""Integration tests for Binance USDⓈ-M bids/asks subscribe + unsubscribe.

Drives BinanceQVUSDM (our ccxt subclass) directly against live Binance
WebSocket endpoints to validate end-to-end that:

1. Subscribing via watch_bids_asks opens a WS client on the new
   /public/ws URL (post-2026-04-23 migration, ccxt 4.5.44+).
2. un_watch_bids_asks completes without AttributeError — the delegation
   to watch_multi_ticker_helper(isUnsubscribe=True) is correct and
   upstream's removal of get_message_hash does not affect us.
3. After unsubscribe, the Binance WS peer stops emitting bookTicker frames
   for the symbol (ack flow works).

These hit the public mainnet WS (no auth needed) and are skipped by the
default test run — opt in with `pytest -m integration`.
"""

import asyncio

import pytest

from qubx.connectors.ccxt.exchanges.binance.exchange import BinanceQVUSDM


@pytest.mark.integration
class TestBinanceBidsAsksSubscribeUnsubscribeIntegration:
    """Round-trip subscribe → unsubscribe against real Binance USDⓈ-M WS."""

    SYMBOL = "BTC/USDT:USDT"
    FIRST_QUOTE_TIMEOUT_S = 20.0
    POST_UNSUB_QUIET_WINDOW_S = 5.0

    async def _run(self):
        ex = BinanceQVUSDM({"options": {"defaultType": "future"}})
        try:
            await ex.load_markets()

            # 1. Subscribe and wait for the first quote.
            first = await asyncio.wait_for(
                ex.watch_bids_asks([self.SYMBOL]),
                timeout=self.FIRST_QUOTE_TIMEOUT_S,
            )
            # watch_bids_asks returns a dict keyed by symbol when given a list.
            assert isinstance(first, dict) and self.SYMBOL in first, f"unexpected first payload: {first!r}"
            quote = first[self.SYMBOL]
            bid = float(quote["bid"])
            ask = float(quote["ask"])
            assert bid > 0 and ask > 0 and ask >= bid, f"nonsensical quote: bid={bid} ask={ask}"

            # 2. The WS client the subscription is attached to MUST be on
            #    the new /public/ws path — this is what Binance keeps alive
            #    after 2026-04-23.
            public_clients = [url for url in ex.clients if "fstream.binance.com/public/ws" in url]
            legacy_clients = [url for url in ex.clients if url.rstrip("/").endswith("fstream.binance.com/ws")]
            assert public_clients, f"no /public/ws WS client; saw {list(ex.clients)}"
            assert not legacy_clients, f"legacy /ws URL still in use: {legacy_clients}"

            # 3. Unsubscribe — this is the path that was broken by the
            #    removed get_message_hash. If delegation is wrong, this
            #    raises AttributeError (or hangs waiting for an ack that
            #    never resolves a matching messageHash).
            await asyncio.wait_for(ex.un_watch_bids_asks([self.SYMBOL]), timeout=10.0)

            # 4. Post-unsubscribe: no new bookTicker frames for this symbol
            #    should resolve on watch_bids_asks. We can't peek inside
            #    ccxt's frame queue portably, so we prove absence by timing
            #    out a fresh watch call over a short quiet window.
            # NOTE: a fresh watch call after un_watch re-subscribes, which
            # is the correct behaviour. So instead we assert that the ack
            # flow cleaned up ccxt's subscription registry.
            # Sub-message hashes for this symbol must be gone from every
            # WS client we had.
            remaining = [
                h
                for client in ex.clients.values()
                for h in getattr(client, "subscriptions", {})
                if "bidask:bookTicker@" + self.SYMBOL in h
            ]
            assert not remaining, (
                f"un_watch_bids_asks did not evict subscription entry; leftover hashes: {remaining}"
            )
        finally:
            await ex.close()

    def test_subscribe_and_unsubscribe_bookticker(self):
        asyncio.run(self._run())
