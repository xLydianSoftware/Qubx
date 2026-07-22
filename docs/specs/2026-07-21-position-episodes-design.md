# Position Episodes — Design

- **Date:** 2026-07-21
- **Status:** Draft (pending approval)
- **Repo:** `Qubx` (core). Downstream consumer: `frab` (separate PR — pair-level `*_current` P&L from leg episodes; retires its tick-sampled baseline map).
- **Branch:** `feat/position-episodes` (off `main`)

## Goal

`Position` accumulators (`r_pnl`, `commissions`, `cumulative_funding`) are **lifetime** values — they survive closes and restarts by design. Strategies and dashboards, however, routinely need P&L **scoped to the currently open position**: "what has this position earned since it was opened", including its own entry costs.

Today the only way to get that is strategy-side baseline sampling (frab stamps baselines on a 5 s scheduler tick, up to 5 s *after* the opening fills — so entry commissions, the crossed spread, and any funding settle inside the window leak into the baseline). This can never be exact.

Fix it where the truth lives: the `Position` itself tracks **episodes** — the span from one flat→open transition to the next return to flat — by stamping baselines synchronously in the deal-processing path.

## Concept

An **episode** starts when a flat position becomes non-flat and ends when it returns to flat. "Flat" is qubx's existing canonical predicate — `abs(quantity) < instrument.lot_size` — the same test used by `is_open()` and by `update_position`'s full-close snap (which zeroes quantity and avg price exactly). No new epsilon is introduced; sub-lot dust is already flat everywhere in qubx, and a full close lands on exact `0.0`. This satisfies the "episode ends at exact zero" decision without inventing a second flatness definition.

Episodes are a **view over the lifetime accumulators** — none of the existing fields change meaning.

## New `Position` state (4 fields)

```python
episode_start_time: dt_64 = NaT      # stamp of the opening deal
r_pnl_at_open: float = 0.0
commissions_at_open: float = 0.0
cumulative_funding_at_open: float = 0.0
```

Defaults make legacy state degrade gracefully: with zero baselines, episode accessors equal the lifetime values — exactly today's behavior.

## New accessors

```python
def episode_pnl(self) -> float:          # total P&L of the current episode:
    return self.pnl - self.r_pnl_at_open #   realized-since-open + unrealized + funding
                                         #   (pnl == r_pnl at the stamp instant, position flat)
def episode_funding(self) -> float:
    return self.cumulative_funding - self.cumulative_funding_at_open

def episode_commissions(self) -> float:
    return self.commissions - self.commissions_at_open

def episode_price_pnl(self) -> float:    # excludes funding (mirrors get_total_price_pnl)
    return self.episode_pnl() - self.episode_funding()

def episode_net_pnl(self) -> float:      # the "honest" number: costs included
    return self.episode_pnl() - self.episode_commissions()
```

`pnl`'s existing convention (includes funding via `r_pnl`, excludes commissions) carries over unchanged; `episode_net_pnl` is the all-in figure. A fresh entry honestly starts at ≈ −(entry fees + crossed spread) — intended.

## Stamping semantics (single mutation path: `Position.update_position`)

`update_position` already decomposes every deal into `qty_closing` / `qty_opening` and is the sole size-mutation path for both live and simulation — episodes work in backtests for free.

1. **Flat → open** (`not self.is_open()` before the deal, non-flat after): stamp all four fields from the **pre-deal** accumulator values (plus the deal timestamp) — the opening deal's own fee and any crossed spread land *inside* the episode.
2. **Partial close / add** (no flat crossing): no stamping. Realized P&L from trims accrues inside the episode; `position_avg_price` adjusts on adds as today.
3. **Full close** (existing snap to exact zero): no stamping; the episode's final values remain readable via the accessors until the next open (useful for "last episode" reporting).
4. **Sign flip through zero in one deal** (`qty_closing` and `qty_opening` both non-zero with direction change): the closing portion's `deal_pnl` and a **pro-rata share of the deal fee** (by `|qty_closing| : |qty_opening|`) belong to the *old* episode; the new baselines are stamped after applying those and before the opening portion.
5. **`realize_only=True`** (stale-deal recovery — books P&L/fee without size change): never stamps; deltas flow into the current episode's lifetime accumulators, which is the correct recovery semantic.
6. **`reconcile_size`** (authoritative venue snapshot): if it takes a flat position to non-flat (first-connect recovery of a pre-existing position), stamp an episode at the current accumulator values — "episode starts now", the honest lower bound when the true opening was never observed.
7. **`apply_funding_payment`**: unchanged. Settles booked while flat land *before* the next baseline stamp and are thus attributed to the closed episode (correct). A late settle arriving after a re-open is attributed to the new episode — known, accepted, documented (rare: requires close + settle-lag + reopen inside one funding interval).
8. **`flatten()`** (delisted-market reconcile): ends the episode by the flat predicate; accumulators preserved as today.
9. **`reset()`**: clears the four fields (as it clears everything). **`reset_by_position`**: copies them.

## Persistence & restore

Episode fields ride the same channel as `cumulative_funding`:

- **`core/loggers.py`** position record: add `episode_start_time`, `realized_pnl_at_open_quoted`, `commissions_at_open_quoted`, `funding_at_open_quoted` alongside the existing `realized_pnl_quoted` / `commissions_quoted` / `funding_pnl_quoted`.
- **`restorers/position.py`** — all three `IPositionRestorer` implementations (Csv, MongoDB, Postgres) read them back. **Migration:** rows written by an older qubx lack the fields → restorer stamps episode-at-restore (current accumulators, `episode_start_time = restore time`) — identical to today's behavior, self-heals when the position next turns over.

## What downstream gets (context, not in this PR)

frab: `funding_pnl_current = Σ leg.episode_funding()`, `net_pnl_current = Σ leg.episode_net_pnl()`; pair entry time = earliest leg `episode_start_time`; the persisted `frab_entries` baseline map dies (entry_time optionally retained for the pair-level notion). Dashboard flips its P&L columns to the `_current` values.

## Non-goals

- No changes to lifetime accumulator semantics, `AccountState`, or margin math.
- No multi-episode history on the object (one current episode; history is the position log's job).
- No pair/portfolio aggregation in core (strategy concern).
- No epsilon-flat configuration.

## Testing

- **Unit (`Position`)**: open-from-flat stamps pre-deal values (entry fee inside episode); partial trim/add keeps baselines; full close preserves final episode values; reopen re-stamps; sign-flip splits deal_pnl and pro-rates fee between episodes; funding settle while flat attributed to old episode; `realize_only` doesn't stamp; `reconcile_size` flat→open stamps at-now; `reset`/`reset_by_position`/`flatten` behaviors; legacy zero-baseline degradation.
- **Restorer round-trip**: log → restore reproduces episode fields; legacy rows (fields absent) stamp episode-at-restore.
- **Simulation integration**: a scripted backtest (open → funding → trim → close → reopen) asserts episode accessors at each step.

## Rollout

Additive, non-breaking. Qubx PR to `main` → CI releases a stable version to PyPI → frab repins in its own PR (already sequenced) → xrelease → user-controlled bot restart. No coordination hazard: until frab consumes the accessors, nothing reads them.
