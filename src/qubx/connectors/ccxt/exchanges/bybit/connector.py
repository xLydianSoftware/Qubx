"""Bybit CcxtConnector subclass.

Bybit, like OKX and Bitfinex, splits the account feed: ``bybit.parse_order`` sets
``trades: None``, so fills arrive only on the ``execution`` topic ``watch_my_trades`` reads
(same /v5/private socket). On top of that split, it fills ADL rank, margin mode and the venue
account figures from Bybit surfaces ccxt's unified shapes do not reach.
"""

from typing import Any, Literal

from qubx import logger
from qubx.core.basics import Instrument, Position, RejectCause

from ...utils import info_float, instrument_to_ccxt_symbol, normalize_margin_mode
from .._two_stream import _TwoStreamCcxtConnector


def _account_block(raw_balance: dict[str, Any]) -> dict[str, Any]:
    """``info.result.list[0]`` of a Bybit wallet-balance payload, ``{}`` when absent/malformed.

    Runs outside ``_snapshot_async``'s per-leg isolation, so raising would sink the snapshot.
    """
    info = raw_balance.get("info")
    if not isinstance(info, dict):
        return {}
    result = info.get("result")
    if not isinstance(result, dict):
        return {}
    rows = result.get("list")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return {}
    return rows[0]


class BybitCcxtConnector(_TwoStreamCcxtConnector):
    """Bybit connector: split orders/fills streams, Bybit account surface, base behavior otherwise."""

    # /v5/order/realtime defaults to 20 rows and ccxt hardcodes limit=200 on /v5/position/list,
    # so without the cursor walk the differ takes a truncated list as venue truth
    #
    # LIMITATION: a symbol-less /v5 read carries ccxt's settleCoin=USDT default and the endpoint
    # takes one settle coin per call, so the snapshot covers USDT-settled instruments only —
    # a USDC-settled perp reads as flat/no-orders to reconcile.
    _snapshot_fetch_params = {"paginate": True}

    # account-wide on a UTA, so one value serves every instrument
    _margin_mode: Literal["cross", "isolated"] | None = None

    _adl_levels: dict[str, int]

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._adl_levels = {}

    async def _fill_leverage_settings(self, positions: list[Position]) -> None:
        await super()._fill_leverage_settings(positions)
        await self._fill_margin_mode(positions)
        self._fill_adl_levels(positions)

    async def _fill_margin_mode(self, positions: list[Position]) -> None:
        """Stamp ``Position.margin_mode`` from ONE account-wide ``/v5/account/info`` read.

        ccxt's ``bybit.parse_position`` hardcodes ``marginMode: None``; the symbol only feeds
        ``safe_symbol``, so any held instrument answers for all of them.

        Served from the cache once it is warm — the read shares ccxt's throttle with order
        placement, and ``set_margin_mode`` drops the cache when the value can have changed.
        """
        if not positions:
            return
        mode = self._margin_mode or await self._read_margin_mode(instrument_to_ccxt_symbol(positions[0].instrument))
        if mode is None:
            return
        self._margin_mode = mode
        for pos in positions:
            pos.margin_mode = mode

    async def _read_margin_mode(self, symbol: str) -> Literal["cross", "isolated"] | None:
        """PORTFOLIO_MARGIN has no framework equivalent and normalizes to None."""
        try:
            row = await self._em.exchange.fetch_margin_mode(symbol)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{self.exchange_name}] fetch_margin_mode for {symbol}: {e}")
            return None
        return normalize_margin_mode(row.get("marginMode"))

    def _fill_adl_levels(self, positions: list[Position]) -> None:
        """Stamp ``Position.adl_level`` from the ranks ``BybitF`` kept off the snapshot's own
        positions read."""
        self._adl_levels = self._em.exchange.adl_ranks if positions else {}
        for pos in positions:
            level = self._adl_levels.get(instrument_to_ccxt_symbol(pos.instrument))
            if level is not None:
                pos.adl_level = level

    def get_adl_level(self, instrument: Instrument) -> int | None:
        # local cache, refreshed each snapshot — never a blocking venue call
        return self._adl_levels.get(instrument_to_ccxt_symbol(instrument))

    def set_margin_mode(self, instrument: Instrument, mode: str) -> bool:
        """Drop the cached mode on a successful write so the next read goes back to the venue."""
        ok = super().set_margin_mode(instrument, mode)
        if ok:
            self._margin_mode = None
        return ok

    def get_margin_mode(self, instrument: Instrument) -> str | None:
        """Serve what the snapshot cached; a cold cache costs one REST hop on the strategy
        thread, every later call a field read. Never raises at the caller: a venue read that
        fails or times out reports None, as the base connector's does."""
        if self._margin_mode is not None:
            return self._margin_mode
        try:
            self._margin_mode = self._run_sync(self._read_margin_mode(instrument_to_ccxt_symbol(instrument)))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{self.exchange_name}] margin mode read for {instrument.symbol}: {e}")
            return None
        return self._margin_mode

    def _extract_venue_figures(
        self, raw_balance: dict[str, Any]
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """(equity, available_margin, margin_ratio, withdrawable) from ``info.result.list[0]``.

        ``accountMMRate`` is maintenance margin OVER equity — the reciprocal of the framework's
        ratio — and ``""``/``"0"`` map to None, as does withdrawable (per coin only), so that AM
        derives those metrics.
        """
        acct = _account_block(raw_balance)
        mm_rate = info_float(acct, "accountMMRate")
        return (
            info_float(acct, "totalMarginBalance"),
            info_float(acct, "totalAvailableBalance"),
            1.0 / mm_rate if mm_rate is not None and mm_rate > 0 else None,
            None,
        )

    def _reject_details(self, raw: dict[str, Any]) -> tuple[str | None, RejectCause]:
        """Bybit accepts then cancels a refused order, carrying the verdict in ``rejectReason``."""
        reason = (raw.get("info") or {}).get("rejectReason") or None
        if reason is None:
            return None, RejectCause.UNKNOWN
        # EC_PostOnlyWillTakeLiquidity / EC_NoImmediateQtyToFill: the price could not rest or fill
        cause = (
            RejectCause.NOT_FILLABLE if reason.startswith(("EC_PostOnly", "EC_NoImmediateQty")) else RejectCause.UNKNOWN
        )
        return reason, cause
