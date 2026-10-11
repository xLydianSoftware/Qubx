# STATUS — Qubx PM spot (agent `qubx-pm-spot`, branch `feat/pm-spot-margin`)

## Checkpoints
- [x] C1 design notes (this file)
- [x] C2 Q1–Q3 implemented + unit tests
- [ ] C3 live smoke
- [ ] C4 Q4 simulator

## Live account inventory (before any change, 2026-10-12)
`tools/inventory.py` in `~/projects/test_strategies/pm-spot-smoke` (read-only, keys never printed):
- papi account: actualEquity 454.397, accountEquity 454.352, IM 0, MM 0, status NORMAL
- balances (cross-margin wallet): **USDT 41.9268**, **USDC 412.57**; nothing else
- UM positions: none. UM / margin / algo open orders: none.
- Note: USDC was already there; "leave flat (only USDT)" is read as "no non-stable assets, no positions".
  Only ~$42 USDT is free to buy USDT-quoted spot → spot legs stay ≤ $20.

## C1 — design (after reading the code)

### What exists
- `BINANCE.PM` venue → `BinancePortfolioMargin` (ccxt binanceusdm subclass, `portfolioMargin: True`,
  `defaultType: swap`, markets spot+linear+inverse) + `BinancePmCcxtConnector` (papi account figures, ADL,
  wallet moves, repay). Runner canonicalizes the venue to `BINANCE.UM`: one connector, one AccountState.
- `ccxt_build_qubx_exchange_name("BINANCE.PM", "spot")` already returns `BINANCE` — a spot fill from the
  PM stream already parses into a `BINANCE:SPOT` instrument, but the AM has no `BINANCE` state → dropped.
- Universe notation `BINANCE:DOGEUSDT` inside the `BINANCE.PM` section already resolves the spot instrument.

### ccxt 4.5.50 routing (verified in the installed source, `async_support/binance.py`)
- `create_order` (l.6283): spot market + portfolioMargin → `papiPostMarginOrder` (`/papi/v1/margin/order`).
  `create_order_request` keeps `reduceOnly` in params for a spot market without marginMode → would be sent
  to papi → **Q3 strip in connector**.
- `cancel_order` (l.7421) → `papiDeleteMarginOrder`; `fetch_order` (l.6688) → `papiGetMarginOrder`;
  `fetch_my_trades` (l.7768) spot+PM → `papiGetMarginMyTrades`. All need the symbol (they do).
- `fetch_open_orders()` without symbol → UM (defaultSubType linear wins). Venue-wide margin open orders:
  call `papiGetMarginOpenOrders()` + `parse_orders` directly.
- `parse_order` marks rows without `positionSide`/`cumQuote` as spot → margin orders resolve to the spot symbol.
- WS: `watch_orders` with defaultType swap → papi listenKey `/pm/ws`. `handle_order_update` serves
  `executionReport` (margin) and `ORDER_TRADE_UPDATE` (UM); `parse_ws_order` resolves spot when `ps` is
  absent. One stream carries both → no extra subscription (live-verify in C3).

### Decision: one PM AccountState, `BINANCE` as an alias of `BINANCE.UM`
A PM account is one account: one equity, one margin, one balance sheet. Two AccountStates would
double count capital/balances or need zeroed venue figures, and break leverage. So:
- AM gets `exchange_aliases={"BINANCE": "BINANCE.UM"}`; every state/connector lookup resolves through it.
  `get_positions/get_orders(exchange=…)` filter by `instrument.exchange`; capital/balances of `BINANCE`
  are the shared account's.
- TradingManager / processing resolve the connector via the AM alias.
- Data: `BINANCE` instruments fall back to the `BINANCE.PM` data provider (like `BINANCE.UM`).
- Opt-in: new `ExchangeConfig.spot_margin: bool = False` (validated: only `BINANCE.PM`). Prod PM bots
  (aggregator, frab-olereon…) are untouched unless they set it.
- Reconciler task key `instrument.symbol` → `instrument` (spot and perp DOGEUSDT would collide).

### Connector (`BinancePmCcxtConnector`, spot enabled)
- Snapshot positions += spot positions from the papi balance rows already fetched: base asset net =
  `crossMarginAsset − crossMarginBorrowed − crossMarginInterest`, floored to lot size, for assets with a
  `<ASSET>/<base ccy>` spot market. Avg price = mark (quote mid, else ticker) — venues report none.
- Open orders += `papiGetMarginOpenOrders`.
- Leverage/ADL post-processing and leverage set: futures only.
- Q3: `reduceOnly` stripped for spot instruments in `submit_order` payload.
- Spot fee charged in the base asset → converted to quote value on the Deal (commission accounting);
  the holding difference (gross deal vs net balance) is corrected by snapshot reconcile.

### AM rule for spot positions
Venue balances carry no entry price → for spot instruments the snapshot is authoritative for **size only**:
the Differ skips avg-price diffs, `reconcile_position_from_snapshot` keeps the local avg (seeds the
snapshot's mark-based avg only when materializing).

## Decisions
1. Single shared AccountState with exchange alias (above) rather than a second spot state.
2. Opt-in via `spot_margin: true` on the `BINANCE.PM` exchange config.
3. Spot leg is long-only in practice: no borrow (`sideEffectType` left default NO_SIDE_EFFECT); a sell
   larger than the holding is refused by the venue.
4. Smoke uses TRX/ADA (pm-smoke notes box-test shares this account on DOGE/XRP).

## C2 — done (Q1–Q3 + unit tests)
- `ExchangeConfig.spot_margin` (runner validates: only BINANCE.PM) → `exchange_aliases={"BINANCE": "BINANCE.UM"}`
  into the AM, spot tcc registered under `BINANCE`, `ConnectorBuildContext.spot_exchange` → PM connector.
- AM: every state/connector lookup resolves through `_key`; `resolve_exchange()` public; exchange-scoped
  `get_positions/get_orders` filter by `instrument.exchange`; fee calculator prefers the alias's own.
- TradingManager / paper feed resolve the connector through the AM; data-provider fallback
  `DATA_PROVIDER_FALLBACKS` (BINANCE → BINANCE.PM, BINANCE.UM → BINANCE.PM).
- CcxtConnector: spot orders drop `reduceOnly` (Q3); hooks `_fetch_extra_open_orders` / `_balance_positions`;
  `spot_fee_in_quote` restates base-asset fees in quote on WS fills and hist deals.
- BinancePmCcxtConnector (spot on): spot positions from cross-margin net balances (floored to lot, stables
  skipped, priced from the quote or a ticker fetch — a missing price fails the snapshot rather than flattening),
  `papiGetMarginOpenOrders` in full snapshots, no leverage/ADL/amend for spot (amend → OrderUpdateRejected:
  ccxt's spot edit is the SPOT account's cancelReplace).
- BinancePortfolioMargin.fetch_open_orders(spot symbol) → papi margin (ccxt picked the UM book via defaultSubType).
- AM spot rule: Differ skips avg-price diffs for spot; reconcile keeps the ledger avg on same-side size changes.
- Reconciler confirm-task key = `str(instrument)` (spot/perp same symbol no longer collide).
- Tests: `tests/qubx/connectors/ccxt/test_binance_pm_spot.py` (18), `tests/qubx/core/account_manager/spot_alias_test.py` (7),
  runner `test_pm_spot_margin_serves_spot_from_the_pm_account` + `test_spot_margin_rejected_off_portfolio_margin`.
- Full unit suite: 3998 passed; 3 failed + 10 errors are pre-existing env issues (no local BITFINEX.F instrument
  data → `test_subscription_watchdog`/`test_subscription_intent`; tardis tests) — unrelated to this branch.

## In progress
- C3 live smoke (harness `~/projects/test_strategies/pm-spot-smoke`, strategy `pm_smoke.spot_smoke`, port 8092).

## Blockers
- none

## Live smoke log
- (none yet)
