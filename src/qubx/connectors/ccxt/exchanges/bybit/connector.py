"""Bybit CcxtConnector subclass.

Bybit, like OKX and Bitfinex, splits the account feed: ``bybit.parse_order`` sets
``trades: None``, so fills arrive only on the ``execution`` topic ``watch_my_trades`` reads
(same /v5/private socket). On top of that split, it fills margin mode and the venue account
figures from Bybit surfaces ccxt's unified shapes do not reach.
"""

from typing import Any, Literal

from qubx import logger
from qubx.core.basics import Balance, Instrument, Position, RejectCause

from ...connector import VenueFigures
from ...utils import info_float, instrument_to_ccxt_symbol, merge_funding_wallets, normalize_margin_mode
from .._two_stream import _TwoStreamCcxtConnector
from .bybit import _POST_ONLY_REFUSAL


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

    # account-wide on a UTA, so one value serves every instrument
    _margin_mode: Literal["cross", "isolated"] | None = None

    def _trade_client_id(self, raw: dict[str, Any]) -> str | None:
        """Bybit's execution frame carries ``orderLinkId``; ccxt keeps it on ``info``.

        Empty for an order placed outside the framework, which then materializes as EXTERNAL.
        """
        return (raw.get("info") or {}).get("orderLinkId") or None

    async def _fill_leverage_settings(self, positions: list[Position]) -> None:
        await super()._fill_leverage_settings(positions)
        await self._fill_margin_mode(positions)

    async def _fill_margin_mode(self, positions: list[Position]) -> None:
        """Stamp ``Position.margin_mode`` from ONE account-wide ``/v5/account/info`` read.

        ccxt's ``bybit.parse_position`` hardcodes ``marginMode: None``; the symbol only feeds
        ``safe_symbol``, so any held instrument answers for all of them.

        Served from the cache once warm; the hourly sweep re-reads it.
        """
        if not positions:
            return
        if self._margin_mode is None:
            try:
                self._margin_mode = await self._read_margin_mode(instrument_to_ccxt_symbol(positions[0].instrument))
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[{self.exchange_name}] fetch_margin_mode: {e}")
                return
        if self._margin_mode is None:
            return
        for pos in positions:
            pos.margin_mode = self._margin_mode

    async def _read_margin_mode(self, symbol: str) -> Literal["cross", "isolated"] | None:
        """PORTFOLIO_MARGIN has no framework equivalent and normalizes to None. Raises on a
        venue error, which a caller must not confuse with that None."""
        row = await self._em.exchange.fetch_margin_mode(symbol)
        return normalize_margin_mode(row.get("marginMode"))

    async def _refresh_leverage_cache(self) -> None:
        await super()._refresh_leverage_cache()
        # changeable from the venue UI; nothing else invalidates it
        symbol = next(iter(self._leverage_cache), None) or next(iter(self._symbol_to_instrument), None)
        if symbol is None:
            return
        try:
            self._margin_mode = await self._read_margin_mode(symbol)
        except Exception as e:  # noqa: BLE001 — keep the cached mode rather than blanking it
            logger.debug(f"[{self.exchange_name}] margin mode re-read: {e}")

    def set_margin_mode(self, instrument: Instrument, mode: str) -> bool:
        """Adopt the written value; the getter is cache-only."""
        ok = super().set_margin_mode(instrument, mode)
        if ok:
            self._margin_mode = normalize_margin_mode(mode)
        return ok

    def get_margin_mode(self, instrument: Instrument) -> str | None:
        return self._margin_mode

    def _convert_balances(self, raw_balance: dict[str, Any]) -> list[Balance]:
        """Base rows plus ``borrowAmount`` as debt and the FUND wallet rows grafted by
        ``BybitF.fetch_balance`` as the ``funding`` wallet, outside ``total``."""
        balances = super()._convert_balances(raw_balance)
        coins = _account_block(raw_balance).get("coin")
        rows = {c["coin"]: c for c in coins if isinstance(c, dict) and "coin" in c} if isinstance(coins, list) else {}
        for bal in balances:
            if (row := rows.get(bal.currency)) is not None:
                bal.debt = info_float(row, "borrowAmount") or 0.0
        info = raw_balance.get("info")
        funding = info.get("funding") if isinstance(info, dict) else None
        return merge_funding_wallets(
            balances, funding, self.exchange_name, main_wallet="unified", ccy_field="coin", amount_field="walletBalance"
        )

    def _extract_venue_figures(self, raw_balance: dict[str, Any]) -> VenueFigures:
        """Venue figures from ``info.result.list[0]``.

        ``accountMMRate`` is maintenance margin OVER equity — the reciprocal of the framework's
        ratio — and ``""``/``"0"`` map to None, as does withdrawable (per coin only), so that AM
        derives those metrics.

        Equity is ``totalEquity`` (NAV). ``totalMarginBalance`` is ``totalWalletBalance +
        totalPerpUPL`` with the collateral discount on non-USDT holdings folded in — the haircut
        figure, reported as collateral_equity. available_margin and margin_ratio come from the
        venue separately, so they keep the discount the venue applies.
        """
        acct = _account_block(raw_balance)
        mm_rate = info_float(acct, "accountMMRate")
        return VenueFigures(
            equity=info_float(acct, "totalEquity"),
            available_margin=info_float(acct, "totalAvailableBalance"),
            margin_ratio=1.0 / mm_rate if mm_rate is not None and mm_rate > 0 else None,
            withdrawable=None,
            total_maint_margin=info_float(acct, "totalMaintenanceMargin"),
            total_initial_margin=info_float(acct, "totalInitialMargin"),
            collateral_equity=info_float(acct, "totalMarginBalance"),
        )

    def _reject_details(self, raw: dict[str, Any]) -> tuple[str | None, RejectCause]:
        """The venue's verdict on a rejection seen on the read path.

        Bybit either reports ``orderStatus=Rejected`` outright or accepts and then cancels,
        which ``BybitF.parse_order`` normalises to rejected; both carry ``rejectReason``.
        """
        reason = (raw.get("info") or {}).get("rejectReason") or None
        if reason is None:
            return None, RejectCause.UNKNOWN
        # the price could not rest (post-only crossing) or fill (IOC/FOK with nothing to take)
        not_fillable = reason.startswith((_POST_ONLY_REFUSAL, "EC_NoImmediateQty"))
        return reason, RejectCause.NOT_FILLABLE if not_fillable else RejectCause.UNKNOWN
