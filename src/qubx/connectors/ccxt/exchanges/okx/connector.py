"""OKX CcxtConnector subclass.

Adds the OKX-specific behavior on top of the generic ``CcxtConnector``:

- **Split orders/fills streams** (via ``_TwoStreamCcxtConnector``): OKX's
  ``watch_orders`` carries only status, not fills; the fills come on a separate
  ``watch_my_trades`` stream. The base runs both concurrently — status events ride
  with ``fill=None`` and each trade arrives as a ``DealEvent``; the AccountManager
  correlates them by trade id (see the two-stream base docstring).
- **Third stream for the algo book**: trigger/conditional orders are pushed on OKX's
  "orders-algo" channel, not "orders" — see ``_account_streams``.
- **Balance extraction**: ccxt's OKX balance mapping is wrong for the framework — see
  ``_convert_balances``.
- **Venue account figures**: OKX's trading-balance payload carries account-level
  figures (``totalEq`` / ``mgnRatio`` / ``adjEq`` / ``imr`` / ``mmr`` in ``info.data[0]``) — see
  ``_extract_venue_figures``; AM prefers them per metric over its derived values.
- **make_client_id / cid_framework_prefix**: OKX clOrdId is case-sensitive
  alphanumeric only, 1-32 chars — the underscore in ``qubx_`` is stripped, so origin
  classification keys on the sanitized prefix (``qubx``), derived from the same
  regex the producer uses.

There is no real-time ``watch_balance`` stream — AM's snapshot cadence covers
balance refresh.
"""

import asyncio
import re
from functools import partial
from typing import Any, Coroutine

from qubx import logger
from qubx.core.basics import FRAMEWORK_CID_PREFIX, Balance, Instrument

from ...connector import _LeverageInfo
from ...utils import info_float, instrument_to_ccxt_symbol
from .._two_stream import _TwoStreamCcxtConnector

_OKX_CLIENT_ID_RE = re.compile(r"[^a-zA-Z0-9]")
_OKX_CLIENT_ID_MAX_LEN = 32


def _account_data(raw_balance: dict[str, Any]) -> dict[str, Any]:
    """First element of ``info.data`` from an OKX balance payload, ``{}`` when absent/malformed.

    ccxt's own ``parse_trading_balance`` treats ``data: []`` / missing ``data`` as a valid
    response shape (``safe_dict(data, 0, {})``), so it reaches us; these extractors run
    outside ``_snapshot_async``'s per-leg isolation, so raising here would sink the whole
    snapshot — degrade to empty instead.
    """
    info = raw_balance.get("info")
    if not isinstance(info, dict):
        return {}
    data = info.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return {}
    return data[0]


class OkxCcxtConnector(_TwoStreamCcxtConnector):
    """OKX connector: split orders/fills streams + OKX balance/clOrdId rules."""

    # OKX strips "_" from cids (see make_client_id), so framework orders echo back as
    # "qubx..." — classify with the prefix produced by the SAME sanitizing regex, so
    # producer and classifier can never drift. Residual caveat: an external cid that
    # happens to start with "qubx" reads as RECOVERED (unavoidable given the charset).
    cid_framework_prefix = _OKX_CLIENT_ID_RE.sub("", FRAMEWORK_CID_PREFIX)

    # - ccxt symbols whose leverage fill is in flight, so a burst of reads for the same
    #   symbol between the miss and the store schedules one venue call, not one per read
    _leverage_fills: set[str]
    # - ccxt symbols the read path has already asked the venue about. NOT keyed on the cache:
    #   the write path's adopt-on-send seats an entry with `maximum` still None, which would
    #   otherwise suppress the tier read for the whole universe until the hourly sweep.
    _leverage_probed: set[str]

    # Seconds between one fill's venue reads and the next. ccxt's REST throttle is one queue
    # shared with order placement, so the first post-warmup tick — which asks for the whole
    # universe at once — must trickle rather than queue ahead of the strategy's orders.
    _leverage_fill_spacing_s: float = 1.0

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._leverage_fills = set()
        self._leverage_probed = set()
        self._leverage_fill_lock = asyncio.Lock()

    def _account_streams(self) -> list[Coroutine[Any, Any, None]]:
        """
        Watch the algo book as well: OKX streams trigger/conditional orders on their own channel.

        ``watch_orders`` covers channel "orders" only, so a stop's terminal state never arrives
        over the socket and the order sits in PENDING_CANCEL until the next order-bearing
        snapshot resolves it — measured at 43s on a live close. ccxt subscribes to
        "orders-algo" when the watch is asked for trigger orders.
        """
        streams = super()._account_streams()
        streams.append(
            self._run_ws_loop(
                watch=partial(self._em.exchange.watch_orders, params={"trigger": True}),
                handle=self._handle_ws_order,
                stream="orders_algo",
                mark_ready=False,
            )
        )
        return streams

    async def _read_configured_leverage(self, symbol: str) -> int | None:
        """
        One symbol's configured leverage from OKX's own endpoint.

        ccxt asks for cross margin mode unless told otherwise, which is the mode
        ``set_leverage`` writes by default, so read and write agree.
        """
        try:
            row = await self._em.exchange.fetch_leverage(symbol)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{self.exchange_name}] fetch_leverage {symbol}: {e}")
            return None
        value = (row.get("longLeverage") or row.get("shortLeverage")) if row else None
        return int(value) if value is not None else None

    async def _read_max_leverage(self, symbol: str) -> int | None:
        """One symbol's venue cap: OKX publishes tiers one market at a time."""
        try:
            tiers = await self._em.exchange.fetch_market_leverage_tiers(symbol)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{self.exchange_name}] leverage tiers {symbol}: {e}")
            return None
        levels = [t["maxLeverage"] for t in (tiers or []) if t.get("maxLeverage") is not None]
        return int(max(levels)) if levels else None

    def _store(self, symbol: str, configured: int | None = None, maximum: int | None = None) -> None:
        held = self._leverage_cache.get(symbol)
        self._leverage_cache[symbol] = _LeverageInfo(
            configured=configured if configured is not None else (held.configured if held else None),
            maximum=maximum if maximum is not None else (held.maximum if held else None),
        )

    async def _refresh_leverage_cache(self) -> None:
        """
        Refresh the symbols the cache already holds, one at a time.

        The base sweep reads ``fetch_leverages`` and ``fetch_leverage_tiers``; okx has neither,
        so it fills nothing and every getter falls through to a venue round trip — measured at
        0.93s per ``get_instrument_leverage`` against 16us on Binance, where the sweep works.
        Entries land here on first ask, and this keeps them as fresh as the hourly sweep does
        elsewhere.
        """
        for symbol in list(self._leverage_cache):
            self._store(
                symbol,
                configured=await self._read_configured_leverage(symbol),
                maximum=await self._read_max_leverage(symbol),
            )

    def _schedule_leverage_fill(self, symbol: str) -> None:
        """Ask OKX for one symbol's settings off-thread, at most once per symbol.

        A read on the snapshot path must never raise, so a loop that is gone (shutdown,
        teardown) is logged and left alone rather than aborting the whole tick.
        """
        if symbol in self._leverage_probed or symbol in self._leverage_fills:
            return
        coro = self._fill_leverage_entry(symbol)
        try:
            self._spawn(coro)
        except Exception as exc:  # noqa: BLE001 — the caller is a snapshot read
            # close it, as _run_sync does, so a rejected schedule does not also surface as
            # "coroutine was never awaited"
            coro.close()
            logger.warning(f"[{self.exchange_name}] leverage fill for {symbol} not scheduled: {exc}")
            return
        # After the spawn: a fill that already finished has marked the symbol probed, which is
        # what makes it final — the in-flight set only covers the window before that.
        self._leverage_fills.add(symbol)

    async def _fill_leverage_entry(self, symbol: str) -> None:
        """One symbol's two venue reads, serialized against every other fill.

        Marked probed whatever happens: ``_store`` leaves a cache entry behind even when both
        reads answer None, so the hourly refresh — which iterates the cache — owns retries from
        here, and a symbol the venue publishes nothing for is asked once, not every 5s tick.
        """
        try:
            async with self._leverage_fill_lock:
                configured = await self._read_configured_leverage(symbol)
                await asyncio.sleep(self._leverage_fill_spacing_s)
                maximum = await self._read_max_leverage(symbol)
                self._store(symbol, configured=configured, maximum=maximum)
                # still holding the lock: this is what spaces the NEXT fill's first read
                await asyncio.sleep(self._leverage_fill_spacing_s)
        finally:
            self._leverage_probed.add(symbol)
            self._leverage_fills.discard(symbol)

    def get_instrument_leverage(self, instrument: Instrument) -> float | None:
        """Cache only: entries land asynchronously on first ask, None until then."""
        leverage = super().get_instrument_leverage(instrument)
        if leverage is None:
            self._schedule_leverage_fill(instrument_to_ccxt_symbol(instrument))
        return leverage

    def get_max_instrument_leverage(self, instrument: Instrument) -> float | None:
        """Cache only: entries land asynchronously on first ask, None until then."""
        maximum = super().get_max_instrument_leverage(instrument)
        if maximum is None:
            self._schedule_leverage_fill(instrument_to_ccxt_symbol(instrument))
        return maximum

    def _convert_balances(self, raw_balance: dict[str, Any]) -> list[Balance]:
        """Use OKX ``cashBal``/``frozenBal`` per currency from the raw response.

        ccxt maps OKX's ``eq`` (equity = cashBal + unrealizedPnL) to balance ``total``;
        we want the cash leg, so we read ``cashBal`` (total) and ``frozenBal`` (locked)
        straight from ``info.data[0].details``. Currencies with a zero cash balance are
        skipped.
        """
        details = _account_data(raw_balance).get("details") or []
        balances: list[Balance] = []
        for detail in details:
            cash_bal = float(detail.get("cashBal", 0) or 0)
            if not cash_bal:
                continue
            frozen_bal = float(detail.get("frozenBal", 0) or 0)
            balances.append(
                Balance(
                    exchange=self.exchange_name,
                    currency=detail["ccy"],
                    free=cash_bal - frozen_bal,
                    locked=frozen_bal,
                    total=cash_bal,
                )
            )
        return balances

    def _extract_venue_figures(
        self, raw_balance: dict[str, Any]
    ) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
        """OKX account-level figures from ``info.data[0]`` of the trading-balance payload.

        - equity: ``totalEq`` — total account equity. USD-denominated; reported as-is
          against the USDT base (USD≈USDT, a bp-level basis difference).
        - available_margin: ``adjEq − imr`` (adjusted equity minus initial margin
          requirement) — both populated only in multi-currency/portfolio margin modes.
        - margin_ratio: ``mgnRatio`` — same coverage-multiple convention as the derived
          ``AccountState.margin_ratio``, but the venue value is not capped at 100.
        - withdrawable: deliberately None — OKX reports max-withdrawal only on a
          separate ``account/max-withdrawal`` endpoint, outside the snapshot seam,
          so AM derives it (= available).
        - total_maint_margin / total_initial_margin: ``mmr`` / ``imr`` — account-level
          maintenance / initial margin requirements (cross positions + pending orders),
          populated only in multi-currency/portfolio margin modes like ``adjEq``.

        Not-applicable fields arrive as ``""`` → None → AM derives that metric.
        """
        acct = _account_data(raw_balance)
        equity = info_float(acct, "totalEq")
        margin_ratio = info_float(acct, "mgnRatio")
        adj_eq = info_float(acct, "adjEq")
        imr = info_float(acct, "imr")
        available_margin = adj_eq - imr if adj_eq is not None and imr is not None else None
        return equity, available_margin, margin_ratio, None, info_float(acct, "mmr"), imr

    def make_client_id(self, suggested: str) -> str:
        """OKX clOrdId: case-sensitive alphanumeric only, 1-32 chars.

        Enforce the base ``qubx_`` prefix first, then strip the underscore (and any
        other non-alphanumeric character) and truncate to 32. The ``qubx`` lead
        survives the strip (alphanumeric), and origin classification keys on that
        sanitized form via ``cid_framework_prefix``.
        """
        prefixed = super().make_client_id(suggested)
        sanitized = _OKX_CLIENT_ID_RE.sub("", prefixed)
        sanitized = sanitized[:_OKX_CLIENT_ID_MAX_LEN]
        return sanitized if sanitized else prefixed[:_OKX_CLIENT_ID_MAX_LEN]
