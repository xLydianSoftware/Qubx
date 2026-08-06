"""Unit tests for Qubx's Hyperliquid exchange overrides.

Pins the cid-only edit guard: HL modify addresses the order by venue oid — there is
no cloid-amend endpoint, so the base ccxt fallback (edit_order('')) would crash
inside parse_to_int(''). HyperliquidEnhanced overrides edit_order_with_client_order_id
to raise ccxt NotSupported before any network call, so the connector's cid-only edit
path (used before the venue ack is seen) gets a clean reject instead of a crash.

Fully offline — no markets/network needed, since the override raises immediately.
"""

import asyncio

import pytest
from ccxt.base.errors import NotSupported

from qubx.connectors.ccxt.exchanges.hyperliquid.hyperliquid import HyperliquidEnhanced


def run(coro):
    # NOT asyncio.run: that clears the thread's current event loop on exit, breaking
    # later tests in the same worker that rely on asyncio.get_event_loop()
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestHyperliquidAmendDialect:
    def test_amend_quantity_dialect_is_replacement(self):
        # HL modify is a venue-side cancel+replace: the amend amount is the replacement
        # order's size (remaining), not the new total — the connector must translate.
        assert HyperliquidEnhanced.AMEND_QUANTITY_DIALECT == "replacement"


class TestHyperliquidCidOnlyEditRejected:
    def test_edit_order_with_client_order_id_raises_not_supported(self):
        exchange = HyperliquidEnhanced()

        async def _attempt_edit():
            try:
                await exchange.edit_order_with_client_order_id(
                    "qubx-cid-1", "BTC/USDC:USDC", "limit", "buy", 1.0, 100.0
                )
            finally:
                await exchange.close()

        with pytest.raises(NotSupported):
            run(_attempt_edit())
