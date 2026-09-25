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
import time
from functools import partial
from typing import Any, Coroutine

from qubx import logger
from qubx.core.basics import (
    FRAMEWORK_CID_PREFIX,
    Balance,
    Instrument,
    Position,
    VenueSettingsUpdate,
    create_venue_settings_event,
)

from ...connector import VenueFigures, _LeverageInfo
from ...utils import info_float, instrument_to_ccxt_symbol
from .._two_stream import _TwoStreamCcxtConnector

_OKX_CLIENT_ID_RE = re.compile(r"[^a-zA-Z0-9]")
_OKX_CLIENT_ID_MAX_LEN = 32
_OKX_LEVERAGE_INFO_MAX_IDS = 20
_OKX_LEVERAGE_BACKOFF_S = 60.0
_OKX_TIER_READS_PER_FLUSH = 5


def _configured_levers(response: dict[str, Any]) -> dict[str, float]:
    """instId -> configured leverage from a leverage-info payload.

    One row per (instId, posSide): "net" on a one-way account, "long"/"short" on a hedged one.
    The long side wins, the rule the single-symbol read applied to longLeverage.

    Kept as the venue reports it, fractions and all: rounding 50.5x down to 50x would pick a
    deeper notional tier than the account actually has.
    """
    levers: dict[str, float] = {}
    for row in (response or {}).get("data") or []:
        inst_id = row.get("instId")
        lever = info_float(row, "lever")
        if not inst_id or lever is None or (inst_id in levers and row.get("posSide") != "long"):
            continue
        levers[inst_id] = lever
    return levers


def _parse_tiers(rows: list[dict[str, Any]] | None) -> list[tuple[float, float]]:
    """(maxLever, maxSz) per position tier, in the venue's own tier order.

    ``maxSz`` is read RAW: ccxt copies it into its unified ``maxNotional`` field, but it is a
    position size in CONTRACTS, not a notional.
    """
    tiers: list[tuple[float, float, float]] = []
    for row in rows or []:
        raw = row.get("info") or {}
        max_lever = info_float(raw, "maxLever")
        max_size = info_float(raw, "maxSz")
        if max_lever is None or max_size is None:
            continue
        tiers.append((info_float(raw, "tier") or float(len(tiers) + 1), max_lever, max_size))
    # keyed on the tier number alone, and a stable sort — rows sharing one keep the venue's
    # order, where a tiebreak on maxLever would inverted the descending-leverage table
    # `_max_size_at` walks.
    return [(lever, size) for _, lever, size in sorted(tiers, key=lambda tier: tier[0])]


def _max_size_at(tiers: list[tuple[float, float]], leverage: float) -> float | None:
    """Contracts allowed at ``leverage``: the LAST tier still permitting it.

    Tier 1 is the smallest size at the highest leverage, and both fall away down the table, so
    the cap is the deepest tier whose own maximum still covers the configured leverage. A
    leverage above tier 1's own maximum cannot be configured at all, and reads as tier 1.
    """
    allowed = [size for max_lever, size in tiers if max_lever >= leverage]
    if allowed:
        return allowed[-1]
    return tiers[0][1] if tiers else None


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
    # - monotonic stamp; no flush starts before it. Set when a batch fails.
    _leverage_flush_backoff_until: float
    # - position tiers per ccxt symbol, (maxLever, maxSz) in tier order. Static per instrument,
    #   so a symbol is fetched once per process; the queue rides the same flush task.
    _tiers: dict[str, list[tuple[float, float]]]
    _tiers_pending: set[str]

    # A snapshot tick asks for every universe instrument in a tight loop; the flush waits this
    # long so they leave as one batch instead of one call per instrument.
    _leverage_flush_debounce_s: float = 0.2

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._leverage_pending = set()
        self._leverage_probed = set()
        self._leverage_flush_scheduled = False
        self._leverage_flush_backoff_until = 0.0
        self._tiers = {}
        self._tiers_pending = set()

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

    def _store(self, symbol: str, configured: float | None) -> None:
        held = self._leverage_cache.get(symbol)
        self._leverage_cache[symbol] = _LeverageInfo(
            configured=configured if configured is not None else (held.configured if held else None),
            maximum=held.maximum if held else None,
            max_notional=held.max_notional if held else None,
        )
        # The base emits this from its own sweep, which okx replaces wholesale — and both the
        # batched read and the hourly refresh land here, so a value that moved on the venue UI
        # reaches the Position from either. A first fill (nothing held) is not a change.
        if configured is None or held is None or held.configured is None or held.configured == configured:
            return
        try:
            instrument = self._instrument_for_symbol(symbol)
        except Exception:  # noqa: BLE001 — not ours; nothing to update
            return
        self.channel.send(create_venue_settings_event(VenueSettingsUpdate(instrument, leverage=float(configured))))

    async def _refresh_leverage_cache(self) -> None:
        """Re-read the configured leverage of every symbol the cache holds, in the same batches.

        The base sweep reads ``fetch_leverages`` and ``fetch_leverage_tiers``; okx has neither.
        The venue cap needs no sweep at all — it comes from the loaded market metadata.
        """
        await self._read_leverage_batched(sorted(self._leverage_cache))

    def _schedule_leverage_fill(self, symbol: str) -> None:
        """Queue one symbol for the next batched read."""
        if symbol in self._leverage_probed:
            return
        # Queued BEFORE the spawn: the loop thread can run the flush to completion inside this
        # call, and an add landing after its drain would sit in the queue unnoticed. Re-adding a
        # symbol already queued is a no-op, so the queue is never consulted to decide anything.
        self._leverage_pending.add(symbol)
        self._ensure_flush_scheduled()

    def _ensure_flush_scheduled(self) -> None:
        """Start the flush unless one is already running or the venue is in backoff.

        A read on the snapshot path must never raise, so a loop that is gone (shutdown,
        teardown) is logged and left alone rather than aborting the whole tick.
        """
        if self._leverage_flush_scheduled or time.monotonic() < self._leverage_flush_backoff_until:
            return
        self._leverage_flush_scheduled = True
        coro = self._flush_leverage_pending()
        try:
            self._spawn(coro)
        except Exception as exc:  # noqa: BLE001 — the caller is a snapshot read
            self._leverage_flush_scheduled = False
            self._leverage_pending.clear()
            self._tiers_pending.clear()
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
                # tiers are read one symbol at a time, so a whole universe's backlog is minutes
                # long. Taking a slice per iteration keeps a leverage miss arriving mid-run to
                # one slice of waiting instead of the whole backlog.
                tier_symbols = sorted(self._tiers_pending)[:_OKX_TIER_READS_PER_FLUSH]
                if not symbols and not tier_symbols:
                    return
                self._leverage_pending.difference_update(symbols)
                self._tiers_pending.difference_update(tier_symbols)
                if symbols:
                    await self._read_leverage_batched(symbols)
                if tier_symbols:
                    await self._read_tiers(tier_symbols)
        except BaseException:
            # abnormal exit (cancellation): drop the queues rather than leave them to a task that
            # no longer exists. Nothing was recorded, so the next snapshot tick queues them again.
            self._leverage_pending.clear()
            self._tiers_pending.clear()
            raise
        finally:
            self._leverage_flush_scheduled = False
            # A miss that arrived while the flag was still set saw no reason to spawn, so with the
            # flag down this task is the only thing that can notice it.
            if self._leverage_pending or self._tiers_pending:
                self._ensure_flush_scheduled()

    async def _read_leverage_batched(self, symbols: list[str]) -> None:
        """Configured leverage for many symbols at once.

        ``GET /api/v5/account/leverage-info`` takes up to 20 comma-separated instIds (21 is
        error 50025) and a batch of 20 costs what one symbol does — 270ms against 279ms,
        measured on mainnet with ccxt 4.5.50. ``mgnMode=cross`` is what ``set_leverage`` writes,
        so read and write agree.

        A symbol is marked probed only once a call came back for it, and always with a cache
        entry — the hourly refresh iterates the cache, so probing without one would strand it.
        A failure probes nothing and backs the flush off, so a permanently broken endpoint (a key
        without account read, an account type that rejects cross) costs one call a minute rather
        than one per symbol per 5s tick.

        The endpoint's own budget is 20 requests / 2s, which a universe past ~400 instruments
        would brush: 20 chunks leave over ~2.1s at ccxt's 110ms global throttle.
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
                self._leverage_flush_backoff_until = time.monotonic() + _OKX_LEVERAGE_BACKOFF_S
                logger.warning(
                    f"[{self.exchange_name}] leverage-info failed for {len(by_id)} instruments, "
                    f"retrying in {_OKX_LEVERAGE_BACKOFF_S:g}s: {e}"
                )
                return
            levers = _configured_levers(response)
            for inst_id, symbol in by_id.items():
                self._store(symbol, configured=levers.get(inst_id))
                self._leverage_probed.add(symbol)

    async def _read_tiers(self, symbols: list[str]) -> None:
        """One ``fetch_market_leverage_tiers`` per symbol (~275ms), through ccxt's throttle.

        Static per instrument, so a symbol that answers is never asked again; one that fails
        backs the flush off with the rest and is re-queued by the next snapshot. The caller
        hands over a slice, not the whole backlog — see the flush loop.
        """
        for symbol in symbols:
            try:
                rows = await self._em.exchange.fetch_market_leverage_tiers(symbol)
            except Exception as e:  # noqa: BLE001
                self._leverage_flush_backoff_until = time.monotonic() + _OKX_LEVERAGE_BACKOFF_S
                logger.warning(
                    f"[{self.exchange_name}] leverage tiers failed for {symbol}, "
                    f"retrying in {_OKX_LEVERAGE_BACKOFF_S:g}s: {e}"
                )
                return
            tiers = _parse_tiers(rows)
            if rows is None or (rows and not tiers):
                # NOT cached: `[]` is checked as "asked and answered", so an unreadable response
                # would leave the symbol capless for the life of the process. An empty LIST is
                # different — the venue really has no tiers — and is cached; None is no answer.
                self._leverage_flush_backoff_until = time.monotonic() + _OKX_LEVERAGE_BACKOFF_S
                logger.warning(
                    f"[{self.exchange_name}] leverage tiers for {symbol} came back unreadable "
                    f"({'None' if rows is None else 'no maxLever/maxSz'}); "
                    f"retrying in {_OKX_LEVERAGE_BACKOFF_S:g}s"
                )
                return
            self._tiers[symbol] = tiers

    def _schedule_tiers_fetch(self, symbol: str) -> None:
        if symbol in self._tiers:
            return
        self._tiers_pending.add(symbol)
        self._ensure_flush_scheduled()

    def _configured_leverage(self, symbol: str) -> float | None:
        cached = self._leverage_cache.get(symbol)
        return float(cached.configured) if cached is not None and cached.configured is not None else None

    def _tier_notional(self, symbol: str, leverage: float | None, multiplier: float, price: float) -> float | None:
        """``maxSz × contracts→tokens × price`` at ``leverage``, or None while anything is unknown.

        Denominated in the quote currency, with no conversion rate applied — where
        ``Position.notional_value`` divides by ``last_update_conversion_rate``. The two are
        directly comparable only while that rate is 1, which is every USDT-settled instrument;
        Binance's own ``maxNotionalValue`` carries the same caveat.
        """
        tiers = self._tiers.get(symbol)
        if tiers is None:
            self._schedule_tiers_fetch(symbol)
            return None
        if leverage is None or not price > 0:  # NaN price (never marked) fails this too
            return None
        max_size = _max_size_at(tiers, leverage)
        return max_size * multiplier * price if max_size is not None else None

    async def _fill_leverage_settings(self, positions: list[Position]) -> None:
        """Set ``max_notional`` on each snapshot position from the venue's position tiers.

        OKX caps a position by SIZE, tier by tier: tier 1 is the smallest size at the highest
        leverage, and the cap at the configured leverage is the maxSz of the last tier that
        still permits it. That maxSz is in CONTRACTS — ccxt copies it into a field it calls
        ``maxNotional``, which it is not — so it becomes a notional only after the contract
        multiplier and the mark price.

        Held positions only, refreshed every snapshot, the same coverage Binance has. A symbol
        whose tiers have not been read yet queues the fetch and keeps its previous value; a
        value the payload itself supplied is never overwritten.
        """
        for pos in positions:
            if pos.max_notional is not None:
                continue
            symbol = instrument_to_ccxt_symbol(pos.instrument)
            notional = self._tier_notional(
                symbol,
                pos.leverage if pos.leverage is not None else self._configured_leverage(symbol),
                pos.instrument.quantity_multiplier,
                pos.last_update_price,
            )
            if notional is not None:
                pos.max_notional = notional

    def get_max_instrument_notional(self, instrument: Instrument) -> float:
        """The tier cap at the configured leverage, ``inf`` while anything it needs is unknown.

        Cache and last-quote lookups only — the base reads it off a blocking single-symbol
        position pull, which the 5s snapshot cannot afford.
        """
        symbol = instrument_to_ccxt_symbol(instrument)
        quote = self._data_provider.get_quote(instrument)
        notional = self._tier_notional(
            symbol,
            self._configured_leverage(symbol),
            instrument.quantity_multiplier,
            quote.mid_price() if quote is not None else float("nan"),
        )
        return notional if notional is not None else float("inf")

    def get_instrument_leverage(self, instrument: Instrument) -> float | None:
        """Cache only: entries land asynchronously on first ask, None until then."""
        leverage = super().get_instrument_leverage(instrument)
        if leverage is None:
            self._schedule_leverage_fill(instrument_to_ccxt_symbol(instrument))
        return leverage

    def get_max_instrument_leverage(self, instrument: Instrument) -> float | None:
        """The venue cap off the loaded market metadata — no venue call.

        OKX publishes it per instrument as ``lever``, and it matched the top of the
        ``fetch_market_leverage_tiers`` ladder on every sampled symbol — a ~275ms call per
        symbol. Read RAW, never ccxt's ``limits.leverage.max``: that defaults an absent or
        empty ``lever`` to 1.0, which the write path would clamp to and silently de-leverage
        the instrument, where None means "unknown, send it unclamped".
        """
        market = (self._em.exchange.markets or {}).get(instrument_to_ccxt_symbol(instrument))
        return info_float((market or {}).get("info") or {}, "lever")

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

    def _extract_venue_figures(self, raw_balance: dict[str, Any]) -> VenueFigures:
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
        - collateral_equity: ``adjEq`` — discount-adjusted (haircut) equity; None outside
          multi-currency/portfolio margin modes.

        Not-applicable fields arrive as ``""`` → None → AM derives that metric.
        """
        acct = _account_data(raw_balance)
        equity = info_float(acct, "totalEq")
        margin_ratio = info_float(acct, "mgnRatio")
        adj_eq = info_float(acct, "adjEq")
        imr = info_float(acct, "imr")
        available_margin = adj_eq - imr if adj_eq is not None and imr is not None else None
        return VenueFigures(
            equity=equity,
            available_margin=available_margin,
            margin_ratio=margin_ratio,
            withdrawable=None,
            total_maint_margin=info_float(acct, "mmr"),
            total_initial_margin=imr,
            collateral_equity=adj_eq,
        )

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
