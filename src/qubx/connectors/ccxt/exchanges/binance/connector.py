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
  ``GET /papi/v1/um/adlQuantile`` endpoint — see ``get_adl_level``.
"""

from typing import Any

from qubx import logger
from qubx.core.basics import Instrument

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


class BinancePmCcxtConnector(CcxtConnector):
    """BINANCE.PM connector: papi account figures + papi ADL on top of the base."""

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

    async def _fetch_trigger_open_orders(self) -> list[dict]:
        """PM has no working venue-wide conditional open-orders endpoint: ccxt 4.5.50 maps
        the ``trigger: True`` sweep to ``GET /papi/v1/um/conditional/openOrders``, which
        404s with an HTML error page (live-verified 2026-07-22). Treat as "no trigger
        orders" so the full snapshot sweep survives; regular open orders still reconcile.
        Caveat: conditional/stop orders on PM are unverified end-to-end — if the papi
        conditional API is dead, placing them likely fails too."""
        return []

    def get_adl_level(self, instrument: Instrument) -> int | None:
        # papi positionRisk has no adl field — query the dedicated endpoint. Response:
        # [{"symbol": ..., "adlQuantile": {"LONG": n, "SHORT": n} | {"BOTH": n}}]; the
        # worst (max) quantile across sides is the conservative per-instrument answer.
        try:
            market_id = self._em.exchange.market(instrument_to_ccxt_symbol(instrument))["id"]
            rows = self._run_sync(self._em.exchange.papiGetUmAdlQuantile({"symbol": market_id}))
            for row in rows if isinstance(rows, list) else []:
                if row.get("symbol") != market_id:
                    continue
                quantiles = row.get("adlQuantile")
                if isinstance(quantiles, dict) and quantiles:
                    return int(max(float(v) for v in quantiles.values()))
            return None
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.exchange_name}] fetch adl quantile for {instrument.symbol}: {e}")
            return None
