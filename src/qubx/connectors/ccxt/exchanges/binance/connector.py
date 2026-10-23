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
- **Spot leg** (``markets: [BINANCE.UM, BINANCE]``): BINANCE spot orders go to the papi
  cross-margin surface (ccxt routes a spot market under portfolioMargin there; cancels, order
  queries and trades follow, margin fills arrive on the same ``/pm/ws`` stream). A holding is a
  balance, not a venue position: the snapshot derives one position per non-stable asset from its
  exact net cross-margin balance (sub-lot dust included), priced at the current mark (the venue
  reports no entry price).
"""

import json
import math
import uuid
from decimal import Decimal
from typing import Any

import ccxt
import pandas as pd

from qubx import logger
from qubx.core.basics import (
    STABLE_CURRENCIES,
    Balance,
    DebtRepaid,
    FundsMoved,
    Instrument,
    OrderRequest,
    Position,
    WalletMove,
)
from qubx.core.events import DebtRepaidEvent, FundsMovedEvent
from qubx.core.exceptions import InvalidOrderParameters
from qubx.utils.marketdata.ccxt import ccxt_symbol_to_instrument

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


def _failure(e: Exception, sent: bool) -> tuple[str, str]:
    """(status, reason) for an exception from a wallet move / repayment. UNKNOWN when the
    request may have reached the venue: timeouts, dropped connections, 5xx and the venue's own
    "execution status unknown" replies (papi -1007 -> BadResponse, -1000 -> OperationFailed).
    Rate-limit and timestamp rejections are refused before execution, so they stay FAILED."""
    reason = f"{type(e).__name__}: {e}"
    maybe_applied = isinstance(e, ccxt.OperationFailed) and not isinstance(
        e, (ccxt.ExchangeError, ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.InvalidNonce)
    )
    return ("UNKNOWN" if sent and maybe_applied else "FAILED"), reason


def _net_margin_holding(row: dict[str, Any]) -> float:
    """Net cross-margin holding of an asset: asset (free + locked) less what is borrowed and owed."""
    return (
        (info_float(row, "crossMarginAsset") or 0.0)
        - (info_float(row, "crossMarginBorrowed") or 0.0)
        - (info_float(row, "crossMarginInterest") or 0.0)
    )


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
    # - papi margin: a client id is unique among OPEN orders only (live-checked 2026-10-12: a duplicate
    #   is refused with -2010 while the first is open, accepted once it is cancelled)
    _replaces_with_same_client_id = True

    _adl_levels: dict[str, int]
    # - markets a PM account can trade besides BINANCE.UM: BINANCE spot, in the cross-margin wallet
    EXTRA_MARKETS = ("BINANCE",)
    # - the account trades BINANCE spot (configured in its markets)
    _spot_margin: bool
    # - quote of the spot instrument a held asset maps to
    _spot_quote: str = "USDT"

    _COLLECT = WalletMove("futures_um", "margin", False)
    _REPAY = WalletMove("margin", "futures_um", False)

    def __init__(self, *, markets: tuple[str, ...] = (), ignore_assets: tuple[str, ...] = (), **kwargs: Any):
        super().__init__(**kwargs)
        self._adl_levels = {}
        self._spot_margin = "BINANCE" in markets
        # - coins left as plain balances: never a spot position, never traded (e.g. BNB for the fee discount)
        self._ignored_assets = frozenset(a.upper() for a in ignore_assets)
        self._unpriced_warned: set[str] = set()

    def submit_order(self, request: OrderRequest) -> None:
        instrument = request.instrument
        if instrument is not None and instrument.is_spot() and instrument.base.upper() in self._ignored_assets:
            raise InvalidOrderParameters(f"{instrument.base} is in the account's ignore_assets: not traded")
        super().submit_order(request)

    def _convert_balances(self, raw_balance: dict[str, Any]) -> list[Balance]:
        balances = super()._convert_balances(raw_balance)
        rows = _pm_rows(raw_balance)
        for bal in balances:
            row = rows.get(bal.currency)
            if row is None:
                continue
            bal.wallets = {key: info_float(row, field) or 0.0 for key, field in _PM_WALLET_FIELDS}
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

    async def _balance_positions(self, raw_balance: dict[str, Any]) -> list[Position]:
        """Spot positions from the cross-margin balances already in the snapshot payload: one per
        non-stable asset with a spot market against ``_spot_quote``, sized by the exact net holding
        (a base-asset fee leaves it off the lot grid), marked and averaged at the current price.

        A holding the venue will not trade — below the market's min amount or min notional — stays a
        balance, not a position: it can never be closed, and as a position it would keep a strategy
        retrying the close. A position held locally that the venue reduces to such dust reconciles to
        flat. A missing price raises for a holding of at least a lot: the snapshot is authoritative,
        and dropping a held asset would read as a sell. Sub-lot dust is priced from the live quote
        only (a coin the bot does not trade has none) and left out without one. An asset whose spot market is inactive, or that
        the venue refuses to price, is left out with a warning: one halted coin in the wallet must not
        fail every snapshot (a traded instrument is priced from its live quote and never gets there).
        """
        if not self._spot_margin:
            return []
        ex = self._em.exchange
        held: list[tuple[Instrument, float, dict[str, Any]]] = []
        for asset, row in _pm_rows(raw_balance).items():
            if asset.upper() in STABLE_CURRENCIES or asset.upper() in self._ignored_assets:
                continue
            market = ex.markets.get(f"{asset}/{self._spot_quote}")
            if market is None or not market.get("spot"):
                continue
            if market.get("active") is False:
                self._warn_unpriced(asset, "its spot market is inactive")
                continue
            if (net := round(_net_margin_holding(row), 12)) == 0.0:
                continue
            instrument = ccxt_symbol_to_instrument(ex.name, market)
            if abs(net) < instrument.min_size:
                continue
            held.append((instrument, net, row))
        if not held:
            return []
        prices = await self._spot_prices([i for i, q, _ in held if abs(q) >= i.lot_size])
        positions = []
        for instrument, quantity, row in held:
            if (price := prices.get(instrument)) is None:
                quote = self._data_provider.get_quote(instrument)
                if quote is None or quote.mid_price() <= 0:
                    continue
                price = quote.mid_price()
            if abs(quantity) * price < instrument.min_notional:
                continue
            pos = Position(instrument=instrument, quantity=quantity, pos_average_price=price)
            updated = row.get("updateTime")
            pos.update_market_price(
                pd.Timestamp(int(updated), unit="ms").asm8 if updated else self._time.time(), price, 1
            )
            positions.append(pos)
        return positions

    async def _spot_prices(self, instruments: list[Instrument]) -> dict[Instrument, float]:
        prices: dict[Instrument, float] = {}
        for instrument in instruments:
            quote = self._data_provider.get_quote(instrument)
            if quote is not None and quote.mid_price() > 0:
                prices[instrument] = quote.mid_price()
        missing = {self._market_id(i): i for i in instruments if i not in prices}
        if missing:
            refused: set[str] = set()
            try:
                rows = await self._spot_tickers(list(missing))
            except ccxt.NetworkError:
                raise
            except ccxt.ExchangeError:
                # - the venue refuses the batch for one bad symbol (-1121): price them one by one
                rows = []
                for market_id in missing:
                    try:
                        rows.extend(await self._spot_tickers([market_id]))
                    except ccxt.NetworkError:
                        raise
                    except ccxt.ExchangeError as e:
                        refused.add(market_id)
                        self._warn_unpriced(missing[market_id].symbol, f"the venue refuses to price it ({e})")
            for row in rows:
                if (instrument := missing.get(row.get("symbol"))) is not None and (price := info_float(row, "price")):
                    prices[instrument] = price
            if unpriced := [i.symbol for m, i in missing.items() if i not in prices and m not in refused]:
                raise ValueError(f"no price for spot holding(s) {unpriced}")
        return prices

    async def _spot_tickers(self, market_ids: list[str]) -> list[dict[str, Any]]:
        # - spot ticker endpoint directly: ccxt's fetch_tickers on this class takes the linear path
        rows = await self._em.exchange.publicGetTickerPrice({"symbols": json.dumps(market_ids, separators=(",", ":"))})
        return rows if isinstance(rows, list) else []

    def _warn_unpriced(self, name: str, why: str) -> None:
        if name not in self._unpriced_warned:
            self._unpriced_warned.add(name)
            logger.warning(f"[{self.exchange_name}] spot holding {name} left out of the snapshot: {why}")

    async def _fetch_extra_open_orders(self) -> list[dict]:
        orders = await super()._fetch_extra_open_orders()
        if not self._spot_margin:
            return orders
        ex = self._em.exchange
        return [*orders, *ex.parse_orders(await ex.papiGetMarginOpenOrders())]

    def set_instrument_leverage(self, instrument: Instrument, leverage: float) -> None:
        if instrument.is_spot():
            logger.debug(f"[{self.exchange_name}] {instrument.symbol}: spot has no leverage setting; skipped")
            return
        super().set_instrument_leverage(instrument, leverage)

    def _amends_in_place(self, instrument: Instrument) -> bool:
        # - papi has no margin amend; ccxt's spot edit is the SPOT account's cancelReplace, which
        #   cannot see a margin order. Spot amends run as cancel + recreate (connector._replace_async).
        return not instrument.is_spot() and super()._amends_in_place(instrument)

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
            if not pos.instrument.is_futures():
                continue
            level = self._adl_levels.get(self._market_id(pos.instrument))
            if level is not None:
                pos.adl_level = level

    def _market_id(self, instrument: Instrument) -> str:
        try:
            return self._em.exchange.market(instrument_to_ccxt_symbol(instrument))["id"]
        except Exception:  # noqa: BLE001 — markets not loaded yet; Binance ids equal the symbol
            return instrument.symbol

    def get_adl_level(self, instrument: Instrument) -> int | None:
        if not instrument.is_futures():
            return None
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
        try:
            ex = self._em.exchange
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
            status, reason = _failure(e, sent=True)
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
        try:
            (logger.info if status == "DONE" else logger.error)(
                "[{}] funds move {}", self.exchange_name, record.to_dict()
            )
        finally:
            self.send(FundsMovedEvent(instrument=None, moved=record))
            self.request_snapshot(include_orders=False)

    def debt_repayments(self) -> list[str]:
        return ["borrowed", "interest"]

    def repay_debt(self, currency: str, amount: float | None = None) -> str:
        if not currency:
            raise ValueError(f"[{self.exchange_name}] repay_debt needs a currency")
        if amount is not None and not (math.isfinite(amount) and amount > 0):
            raise ValueError(f"[{self.exchange_name}] repay amount must be positive and finite, got {amount}")
        repay_id = f"rp-{uuid.uuid4().hex[:12]}"
        self._spawn(self._repay_debt(repay_id, currency, amount))
        return repay_id

    async def _repay_debt(self, repay_id: str, currency: str, amount: float | None) -> None:
        asset = currency.upper()
        venue_ref = None
        requested = amount
        sent = False
        try:
            ex = self._em.exchange
            if amount is None:
                owed = await self._owed(asset)
            else:
                owed = format(Decimal(str(amount)), "f")
            if owed is None:
                status, reason = "FAILED", "nothing to repay"
            else:
                requested = float(owed)
                sent = True
                resp = await ex.papiPostRepayLoan({"asset": asset, "amount": owed})
                tran_id = resp.get("tranId") if isinstance(resp, dict) else None
                if tran_id is not None:
                    status, reason, venue_ref = "DONE", None, str(tran_id)
                else:
                    status, reason = "FAILED", str(resp)
        except Exception as e:  # noqa: BLE001 — every failure is reported, not raised
            status, reason = _failure(e, sent)
        record = DebtRepaid(
            repay_id=repay_id,
            exchange=self.exchange_name,
            currency=currency,
            requested=requested,
            status=status,
            venue_ref=venue_ref,
            failure_reason=reason,
        )
        try:
            (logger.info if status == "DONE" else logger.error)(
                "[{}] debt repay {}", self.exchange_name, record.to_dict()
            )
        finally:
            self.send(DebtRepaidEvent(instrument=None, repaid=record))
            self.request_snapshot(include_orders=False)

    async def _owed(self, asset: str) -> str | None:
        """Borrowed + interest owed in ``asset`` as the venue's decimal string; None when nothing is owed."""
        rows = await self._em.exchange.papiGetBalance()
        row = next((r for r in rows or [] if isinstance(r, dict) and r.get("asset") == asset), {})
        # Decimal keeps the venue's digits exact — a float round trip could over- or under-pay
        owed = Decimal(str(row.get("crossMarginBorrowed") or "0")) + Decimal(str(row.get("crossMarginInterest") or "0"))
        return format(owed.normalize(), "f") if owed > 0 else None
