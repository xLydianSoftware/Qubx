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
_OKX_LEVERAGE_INFO_MAX_IDS = 20


def _configured_levers(response: dict[str, Any]) -> dict[str, int]:
    """instId -> configured leverage from a leverage-info payload.

    One row per (instId, posSide): "net" on a one-way account, "long"/"short" on a hedged one.
    The long side wins, the rule the single-symbol read applied to longLeverage.
    """
    levers: dict[str, int] = {}
    for row in (response or {}).get("data") or []:
        inst_id = row.get("instId")
        lever = info_float(row, "lever")
        if not inst_id or lever is None or (inst_id in levers and row.get("posSide") != "long"):
            continue
        levers[inst_id] = int(lever)
    return levers


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

    # - ccxt symbols queued for the next batched read, and those a batch has already covered.
    #   Probed is NOT keyed on the cache: the write path's adopt-on-send seats an entry of its
    #   own, which would otherwise suppress the read for the whole universe until the sweep.
    _leverage_pending: set[str]
    _leverage_probed: set[str]
    _leverage_flush_scheduled: bool

    # A snapshot tick asks for every universe instrument in a tight loop; the flush waits this
    # long so they leave as one batch instead of one call per instrument.
    _leverage_flush_debounce_s: float = 0.2

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._leverage_pending = set()
        self._leverage_probed = set()
        self._leverage_flush_scheduled = False

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

    def _store(self, symbol: str, configured: int | None) -> None:
        held = self._leverage_cache.get(symbol)
        self._leverage_cache[symbol] = _LeverageInfo(
            configured=configured if configured is not None else (held.configured if held else None),
            maximum=held.maximum if held else None,
        )

    async def _refresh_leverage_cache(self) -> None:
        """Re-read the configured leverage of every symbol the cache holds, in the same batches.

        The base sweep reads ``fetch_leverages`` and ``fetch_leverage_tiers``; okx has neither.
        The venue cap needs no sweep at all — it comes from the loaded market metadata.
        """
        await self._read_leverage_batched(sorted(self._leverage_cache))

    def _schedule_leverage_fill(self, symbol: str) -> None:
        """Queue one symbol for the next batched read.

        A read on the snapshot path must never raise, so a loop that is gone (shutdown,
        teardown) is logged and left alone rather than aborting the whole tick.
        """
        if symbol in self._leverage_probed or symbol in self._leverage_pending:
            return
        # Queued and flagged BEFORE the spawn: the loop thread can run the flush to completion
        # inside this call, and a clear that lands before the add would strand either one.
        self._leverage_pending.add(symbol)
        if self._leverage_flush_scheduled:
            return
        self._leverage_flush_scheduled = True
        coro = self._flush_leverage_pending()
        try:
            self._spawn(coro)
        except Exception as exc:  # noqa: BLE001 — the caller is a snapshot read
            self._leverage_flush_scheduled = False
            self._leverage_pending.clear()
            # close it, as _run_sync does, so a rejected schedule does not also surface as
            # "coroutine was never awaited"
            coro.close()
            logger.warning(f"[{self.exchange_name}] leverage flush not scheduled: {exc}")

    async def _flush_leverage_pending(self) -> None:
        """Drain the queued symbols, re-draining whatever arrived while the venue answered."""
        try:
            while True:
                await asyncio.sleep(self._leverage_flush_debounce_s)
                symbols = sorted(self._leverage_pending)
                if not symbols:
                    return
                self._leverage_pending.difference_update(symbols)
                await self._read_leverage_batched(symbols)
        finally:
            # a cancelled flush drops its queue rather than holding symbols the gate then skips;
            # they are unprobed, so the next snapshot tick queues them again
            self._leverage_pending.clear()
            self._leverage_flush_scheduled = False

    async def _read_leverage_batched(self, symbols: list[str]) -> None:
        """Configured leverage for many symbols at once.

        ``GET /api/v5/account/leverage-info`` takes up to 20 comma-separated instIds (21 is
        error 50025) and a batch of 20 costs what one symbol does — 270ms against 279ms,
        measured on mainnet with ccxt 4.5.50. ``mgnMode=cross`` is what ``set_leverage`` writes,
        so read and write agree.

        A symbol is marked probed only once a call came back for it, and always with a cache
        entry — the hourly refresh iterates the cache, so probing without one would strand it.
        """
        markets = self._em.exchange.markets or {}
        for start in range(0, len(symbols), _OKX_LEVERAGE_INFO_MAX_IDS):
            by_id = {
                market["id"]: symbol
                for symbol in symbols[start : start + _OKX_LEVERAGE_INFO_MAX_IDS]
                if (market := markets.get(symbol)) is not None
            }
            if not by_id:
                continue  # markets not loaded yet; the next tick asks again
            try:
                response = await self._em.exchange.privateGetAccountLeverageInfo(
                    {"instId": ",".join(by_id), "mgnMode": "cross"}
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[{self.exchange_name}] leverage-info for {len(by_id)} instruments: {e}")
                continue
            levers = _configured_levers(response)
            for inst_id, symbol in by_id.items():
                self._store(symbol, configured=levers.get(inst_id))
                self._leverage_probed.add(symbol)

    def get_instrument_leverage(self, instrument: Instrument) -> float | None:
        """Cache only: entries land asynchronously on first ask, None until then."""
        leverage = super().get_instrument_leverage(instrument)
        if leverage is None:
            self._schedule_leverage_fill(instrument_to_ccxt_symbol(instrument))
        return leverage

    def get_max_instrument_leverage(self, instrument: Instrument) -> float | None:
        """The venue cap off the loaded market metadata — no venue call.

        OKX publishes it per instrument as ``lever``, which ccxt normalises into
        ``limits.leverage.max``; it matched the top of the ``fetch_market_leverage_tiers``
        ladder on every sampled symbol, and that call cost ~275ms each.
        """
        market = (self._em.exchange.markets or {}).get(instrument_to_ccxt_symbol(instrument))
        if market is None:
            return None
        maximum = (market.get("limits") or {}).get("leverage", {}).get("max")
        return float(maximum) if maximum is not None else None

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
