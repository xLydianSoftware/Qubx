"""Binance portfolio-margin CcxtConnector subclass.

Everything PM-specific on the Qubx side lives here, keeping the base ``CcxtConnector``
(BINANCE.UM) free of portfolio-margin branches. Resolved from ``CUSTOM_CONNECTORS`` by
the configured VENUE name (``binance.pm``) — the canonical name both venues share
(``binance.um``) still resolves to the base class.

- **Venue account figures**: PM's ``papiGetBalance`` is a per-asset list with no
  account-level figures, so ``BinancePortfolioMargin.fetch_balance`` grafts
  ``GET /papi/v1/account`` into ``info`` — see ``_extract_venue_figures``. The papi
  figures are USD-denominated (vs USDT on fapi); the difference is treated as
  negligible, same as the base class treats fapi's USDT figures.
- **ADL level**: papi positionRisk carries no adl field; PM exposes it on a dedicated
  bulk ``GET /papi/v1/um/adlQuantile`` endpoint. One account-wide call per snapshot
  stamps ``Position.adl_level`` (so ``ctx.get_adl_level`` — an AccountManager dict
  read — works on PM) and refreshes a local cache that ``get_adl_level`` serves
  without any network call.
"""

from typing import Any

from qubx.core.basics import Instrument, Position

from ...connector import CcxtConnector
from ...utils import info_float, instrument_to_ccxt_symbol


def _account_figures(raw_balance: dict[str, Any]) -> dict[str, Any]:
    """The grafted ``GET /papi/v1/account`` dict from a PM balance payload, ``{}`` when
    absent (graft failed or a non-PM payload reached us) — degrade to derived figures
    rather than sink the snapshot."""
    info = raw_balance.get("info")
    if not isinstance(info, dict):
        return {}
    account = info.get("account")
    return account if isinstance(account, dict) else {}


def _parse_adl_quantiles(rows: Any) -> dict[str, int]:
    """Market id -> worst (max) ADL quantile across sides, from the papi adlQuantile
    response: ``[{"symbol": ..., "adlQuantile": {"LONG": n, "SHORT": n} | {"BOTH": n}}]``."""
    levels: dict[str, int] = {}
    for row in rows if isinstance(rows, list) else []:
        symbol = row.get("symbol")
        quantiles = row.get("adlQuantile")
        if symbol and isinstance(quantiles, dict) and quantiles:
            try:
                levels[symbol] = int(max(float(v) for v in quantiles.values()))
            except (TypeError, ValueError):
                continue
    return levels


class BinancePmCcxtConnector(CcxtConnector):
    """BINANCE.PM connector: papi account figures + papi ADL on top of the base."""

    _adl_levels: dict[str, int]

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._adl_levels = {}

    def _extract_venue_figures(
        self, raw_balance: dict[str, Any]
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """(equity, available_margin, margin_ratio, withdrawable) from ``papiGetAccount``:
        ``accountEquity`` (collateral + uPnL across um/cm/margin, USD),
        ``totalAvailableBalance`` (margin available for new positions),
        ``uniMMR`` (accountEquity / accountMaintMargin — same shape as AM's derived
        ratio; reported as a 99999999 sentinel when maint margin is 0, mapped to None
        so AM applies its own no-positions handling) and ``virtualMaxWithdrawAmount``."""
        account = _account_figures(raw_balance)
        maint = info_float(account, "accountMaintMargin")
        margin_ratio = info_float(account, "uniMMR") if maint is not None and maint > 0 else None
        return (
            info_float(account, "accountEquity"),
            info_float(account, "totalAvailableBalance"),
            margin_ratio,
            info_float(account, "virtualMaxWithdrawAmount"),
        )

    async def _fill_leverage_settings(self, positions: list[Position]) -> None:
        # snapshot post-processing hook: leverage/max_notional from the base, then ADL
        await super()._fill_leverage_settings(positions)
        await self._fill_adl_levels(positions)

    async def _fill_adl_levels(self, positions: list[Position]) -> None:
        """Stamp ``Position.adl_level`` from ONE account-wide adlQuantile call per
        snapshot and refresh the cache ``get_adl_level`` reads. Best-effort: a failure
        leaves the previous cache and the positions' levels unchanged."""
        if not positions:
            return
        try:
            rows = await self._em.exchange.papiGetUmAdlQuantile()
        except Exception as e:  # noqa: BLE001
            self._dbg.debug("fetch adl quantiles failed: {}", e)
            return
        self._adl_levels = _parse_adl_quantiles(rows)
        for pos in positions:
            level = self._adl_levels.get(self._market_id(pos.instrument))
            if level is not None:
                pos.adl_level = level

    def _market_id(self, instrument: Instrument) -> str:
        try:
            return self._em.exchange.market(instrument_to_ccxt_symbol(instrument))["id"]
        except Exception:  # noqa: BLE001 — markets not loaded yet; Binance ids equal the symbol
            return instrument.symbol

    def get_adl_level(self, instrument: Instrument) -> int | None:
        # local cache read (refreshed each snapshot) — never a blocking venue call;
        # strategies normally read ctx.get_adl_level -> Position.adl_level anyway
        return self._adl_levels.get(self._market_id(instrument))
