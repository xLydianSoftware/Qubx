"""Binance portfolio-margin CcxtConnector subclass.

Everything PM-specific on the Qubx side lives here, keeping the base ``CcxtConnector``
(BINANCE.UM) free of portfolio-margin branches. Resolved from ``CUSTOM_CONNECTORS`` by
the configured VENUE name (``binance.pm``) — the canonical name both venues share
(``binance.um``) still resolves to the base class.

- **Venue account figures**: PM's ``papiGetBalance`` is a per-asset list with no
  account-level figures, so ``BinancePortfolioMargin.fetch_balance`` grafts
  ``GET /papi/v1/account`` into ``info`` — see ``_extract_venue_figures``. Equity is
  ``actualEquity`` (NAV, no collateral haircut); the haircut ``accountEquity`` is
  reported as ``collateral_equity``. The papi figures are USD-denominated (vs USDT on
  fapi); the difference is treated as negligible, same as the base class treats fapi's
  USDT figures.
- **Wallet breakdown**: PM money sits in the cross-margin wallet plus the UM/CM futures
  wallets; ``total`` is their sum (``totalWalletBalance``), ``Balance.wallets`` carries
  the split.
- **WS balance push**: disabled (``_wants_ws_balance_push``). The papi user-data
  ACCOUNT_UPDATE carries the UM sub-wallet in ``wb``, not the account wallet — see
  the class attribute. Balance refresh rides the papi snapshot instead.
- **ADL level**: papi positionRisk carries no adl field; PM exposes it on a dedicated
  bulk ``GET /papi/v1/um/adlQuantile`` endpoint. One account-wide call per snapshot
  stamps ``Position.adl_level`` (so ``ctx.get_adl_level`` — an AccountManager dict
  read — works on PM) and refreshes a local cache that ``get_adl_level`` serves
  without any network call.
- **Wallet moves**: futures→margin collection (``asset-collection`` / ``auto-collection``)
  and margin→futures negative-balance repay; both amount-less on the venue side.
- **Debt repayment**: ``repayLoan`` pays cross-margin ``borrowed`` + ``interest`` from the
  margin wallet.
"""

import uuid
from decimal import Decimal
from typing import Any

from qubx import logger
from qubx.core.basics import Balance, DebtRepaid, FundsMoved, Instrument, Position, WalletMove
from qubx.core.events import DebtRepaidEvent, FundsMovedEvent

from ...connector import CcxtConnector, VenueFigures
from ...utils import info_float, instrument_to_ccxt_symbol, set_liabilities

_PM_WALLET_FIELDS = (
    ("margin", "crossMarginAsset"),
    ("futures_um", "umWalletBalance"),
    ("futures_cm", "cmWalletBalance"),
)


def _pm_rows(raw_balance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    info = raw_balance.get("info")
    rows = info.get("balance") if isinstance(info, dict) else None
    return {r["asset"]: r for r in rows if isinstance(r, dict) and "asset" in r} if isinstance(rows, list) else {}


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

    # PM has no usable WS balance push. The papi /pm/ws ACCOUNT_UPDATE reports the UM
    # SUB-wallet in `a.B[].wb`, not the account: PM collateral sits in the cross-margin
    # wallet, so wb tracks cumulative UM realized PnL/fees (~0 on a fresh account) and
    # an absolute push overwrites the real balance until the next snapshot. Same trap
    # the REST path already sidesteps in BinanceQV.parse_balance_custom, which reads
    # whole-account totalWalletBalance. Snapshot-only here.
    _wants_ws_balance_push = False

    _adl_levels: dict[str, int]

    _COLLECT = WalletMove("futures_um", "margin", False)
    _REPAY = WalletMove("margin", "futures_um", False)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._adl_levels = {}

    def _convert_balances(self, raw_balance: dict[str, Any]) -> list[Balance]:
        balances = super()._convert_balances(raw_balance)
        rows = _pm_rows(raw_balance)
        for bal in balances:
            row = rows.get(bal.currency)
            if row is None:
                continue
            legs = {key: v for key, field in _PM_WALLET_FIELDS if (v := info_float(row, field))}
            bal.wallets = legs if set(legs) - {"margin"} else None
            # negativeBalance sign is undocumented
            set_liabilities(
                bal,
                borrowed=info_float(row, "crossMarginBorrowed"),
                interest=info_float(row, "crossMarginInterest"),
                negative=abs(info_float(row, "negativeBalance") or 0.0),
            )
        return balances

    def _extract_venue_figures(self, raw_balance: dict[str, Any]) -> VenueFigures:
        """Venue figures from ``papiGetAccount``.

        - equity: ``actualEquity`` — account equity without collateral rate (NAV), the
          same basis as Bybit ``totalEquity`` / OKX ``totalEq``. Absent → None; never
          falls back to the haircut figure.
        - collateral_equity: ``accountEquity`` — collateral-rate (haircut) equity plus
          uPnL across um/cm/margin, USD; what ``uniMMR`` is computed from.
        - available_margin: ``totalAvailableBalance`` (margin available for new positions).
        - margin_ratio: ``uniMMR`` (accountEquity / accountMaintMargin); reported as a
          99999999 sentinel when maint margin is 0, mapped to None so AM applies its own
          no-positions handling.
        - withdrawable: ``virtualMaxWithdrawAmount``.
        - total_maint_margin / total_initial_margin: ``accountMaintMargin`` /
          ``accountInitialMargin`` — account-wide across um/cm/margin.
        """
        account = _account_figures(raw_balance)
        maint = info_float(account, "accountMaintMargin")
        margin_ratio = info_float(account, "uniMMR") if maint is not None and maint > 0 else None
        return VenueFigures(
            equity=info_float(account, "actualEquity"),
            available_margin=info_float(account, "totalAvailableBalance"),
            margin_ratio=margin_ratio,
            withdrawable=info_float(account, "virtualMaxWithdrawAmount"),
            total_maint_margin=maint,
            total_initial_margin=info_float(account, "accountInitialMargin"),
            collateral_equity=info_float(account, "accountEquity"),
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

    def wallet_moves(self) -> list[WalletMove]:
        return [self._COLLECT, self._REPAY]

    def move_funds(self, currency: str | None, src: str, dst: str, amount: float | None = None) -> str:
        if (src, dst) not in {(m.src, m.dst) for m in self.wallet_moves()}:
            raise ValueError(f"[{self.exchange_name}] unsupported wallet move {src} -> {dst}")
        if amount is not None:
            raise ValueError(f"[{self.exchange_name}] Binance PM wallet moves take no amount")
        move_id = f"mv-{uuid.uuid4().hex[:12]}"
        self._spawn(self._move_funds(move_id, currency, src, dst))
        return move_id

    async def _move_funds(self, move_id: str, currency: str | None, src: str, dst: str) -> None:
        ex = self._em.exchange
        try:
            if (src, dst) == (self._REPAY.src, self._REPAY.dst):
                # repays every negative futures balance; currency is recorded, not sent
                resp = await ex.papiPostRepayFuturesNegativeBalance()
            elif currency is None:
                resp = await ex.papiPostAutoCollection()
            else:
                resp = await ex.papiPostAssetCollection({"asset": currency.upper()})
            msg = (resp or {}).get("msg")
            status, reason = ("DONE", None) if msg == "success" else ("FAILED", str(resp if msg is None else msg))
        except Exception as e:  # noqa: BLE001 — every failure is reported, not raised
            status, reason = "FAILED", f"{type(e).__name__}: {e}"
        record = FundsMoved(
            move_id=move_id,
            exchange=self.exchange_name,
            currency=currency,
            src=src,
            dst=dst,
            requested=None,
            status=status,
            failure_reason=reason,
        )
        (logger.info if status == "DONE" else logger.error)(f"[{self.exchange_name}] funds move {record.to_dict()}")
        self.send(FundsMovedEvent(instrument=None, moved=record))
        self.request_snapshot(include_orders=False)

    def debt_repayments(self) -> list[str]:
        return ["borrowed", "interest"]

    def repay_debt(self, currency: str, amount: float | None = None) -> str:
        if not currency:
            raise ValueError(f"[{self.exchange_name}] repay_debt needs a currency")
        if amount is not None and not amount > 0:
            raise ValueError(f"[{self.exchange_name}] repay amount must be positive, got {amount}")
        repay_id = f"rp-{uuid.uuid4().hex[:12]}"
        self._spawn(self._repay_debt(repay_id, currency, amount))
        return repay_id

    async def _repay_debt(self, repay_id: str, currency: str, amount: float | None) -> None:
        ex = self._em.exchange
        asset = currency.upper()
        venue_ref = None
        try:
            if amount is None:
                owed = await self._owed(asset)
            else:
                owed = format(Decimal(str(amount)), "f")
            if owed is None:
                status, reason = "FAILED", "nothing to repay"
            else:
                resp = await ex.papiPostRepayLoan({"asset": asset, "amount": owed})
                tran_id = resp.get("tranId") if isinstance(resp, dict) else None
                if tran_id is not None:
                    status, reason, venue_ref = "DONE", None, str(tran_id)
                else:
                    status, reason = "FAILED", str(resp)
        except Exception as e:  # noqa: BLE001 — every failure is reported, not raised
            status, reason = "FAILED", f"{type(e).__name__}: {e}"
        record = DebtRepaid(
            repay_id=repay_id,
            exchange=self.exchange_name,
            currency=currency,
            requested=amount,
            status=status,
            venue_ref=venue_ref,
            failure_reason=reason,
        )
        (logger.info if status == "DONE" else logger.error)(f"[{self.exchange_name}] debt repay {record.to_dict()}")
        self.send(DebtRepaidEvent(instrument=None, repaid=record))
        self.request_snapshot(include_orders=False)

    async def _owed(self, asset: str) -> str | None:
        """Borrowed + interest owed in ``asset`` as the venue's decimal string; None when nothing is owed."""
        rows = await self._em.exchange.papiGetBalance()
        row = next((r for r in rows or [] if isinstance(r, dict) and r.get("asset") == asset), {})
        # Decimal keeps the venue's digits exact — a float round trip could over- or under-pay
        owed = Decimal(row.get("crossMarginBorrowed") or "0") + Decimal(row.get("crossMarginInterest") or "0")
        return format(owed.normalize(), "f") if owed > 0 else None
