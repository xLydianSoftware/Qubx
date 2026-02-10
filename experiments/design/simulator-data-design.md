# Simulator Data Layer: Adopting New IterableSimulationData

## Status: IN PROGRESS

## Context

Old architecture used `DataReader` + `DataFetcher` (one SQL per symbol per read).
New architecture uses `IReader/IStorage` + `DataPump` + `RawSymbolBuffer` + `MemReader` (one batched read for all symbols).

The new `IterableSimulationData` is now in `simulated_data.py` (replaced old class).
Need to update consumers: `SimulatedDataProvider` and `SimulationRunner`.

---

## Design Principles

1. **Single IterableSimulationData per simulation** — created in `SimulationRunner`, shared across all data providers. It owns the slicer and iteration loop.

2. **One SimulatedDataProvider per exchange** — created in the per-exchange loop in `SimulationRunner._create_backtest_context()`. Each wraps the shared IterableSimulationData and handles exchange-specific concerns (quotes, account notifications).

3. **Exchange list comes from SimulationSetup** — `setup.exchanges` defines what exchanges exist in the sim.
   - Open question: Can a strategy subscribe to an exchange not in the original config?

4. **SimulationDataConfig updated**:
   - `data_storage: IStorage` — main data storage
   - `customized_data_storages: dict[str, IStorage]` — overrides per subscription type

---

## Decided: IterableSimulationData owns IStorage, resolves readers lazily

**IterableSimulationData receives IStorage (+ custom storages) directly.**
It resolves IReader internally on demand from instrument data.

### Constructor
```python
def __init__(self, storage: IStorage, custom_types_storages: dict[str, IStorage] | None = None, ...):
    self._storage = storage
    self._custom_storages = dict(custom_types_storages or {})
    self._readers = {}   # cache: reader_key -> IReader (lazy)
```

### Reader resolution: `_get_or_create_reader(data_type, exchange, market_type)`
- Custom storage checked first (`_custom_storages.get(data_type)`), fallback to `_storage`
- Custom readers cached as `"{data_type}:{exchange}:{market_type}"`
- Main readers cached as `"{exchange}:{market_type}"` (shared across data types)

### Pump granularity: one per (subscription, exchange, market_type)
- Pump key: `"{access_key}.{exchange}:{market_type}"` e.g. `"ohlc.1h.BINANCE.UM:SWAP"`
- `add_instruments_for_subscription()` groups instruments by `(exchange, market_type)`
- Each group gets its own pump with its own IReader

### Slicer key: includes exchange scope
- Format: `"{requested_data_type}.{exchange}:{market_type}:{symbol}"`
- e.g. `"ohlc(1h).BINANCE.UM:SWAP:BTCUSDT"` — prevents cross-exchange collisions
- DataPump stores `_exchange` + `_market_type` and uses them in `_make_slicer_key()`

---

## Changes Summary

### `DataPump` (simulated_data.py)
- [x] Add `exchange: str` and `market_type: str` to constructor, stored as `_exchange`, `_market_type`
- [x] `_make_slicer_key()` now returns `f"{requested_data_type}.{exchange}:{market_type}:{symbol}"`
- [x] `__repr__` includes exchange scope

### `IterableSimulationData` (simulated_data.py)
- [x] Constructor takes `storage: IStorage` + `custom_types_storages: dict[str, IStorage] | None`
- [x] `_get_or_create_reader(data_type, exchange, market_type)` — lazy IReader resolution with caching
- [x] `_get_or_create_pump()` — takes exchange/market_type, pump key includes exchange scope
- [x] `add_instruments_for_subscription()` — groups by (exchange, market_type), per-group pump creation
- [x] `remove_instruments_from_subscription()` — groups by (exchange, market_type) to find pump; ALL iterates all pumps
- [x] `get_instruments_for_subscription()` — iterates all pumps matching access_key prefix
- [x] `peek_historical_data()` — constructs pump_key from instrument's exchange + market_type
- [x] `__iter__` — iterates all pumps (no change needed, already works)

### Tests (simulated_data_test.py)
- [x] All 7 tests updated: `storage=CsvStorage(...)` / `storage=HandyStorage(...)` instead of `readers={"ohlc": reader}`
- [x] All 7 tests pass

### `SimulatedDataProvider` (data.py)
- [ ] Drop `readers: dict[str, DataReader]` from `__init__`
- [ ] Rework `get_ohlc()` — use IReader or deprecate

### `SimulationRunner` (runner.py)
- [ ] Update `IterableSimulationData(...)` construction — pass `data_config.data_storage`
- [ ] Update `SimulatedDataProvider(...)` construction — drop old `readers` param
- [ ] Fix custom subscription handling (line ~166 references old `data_config.data_providers`)

### `SimulationDataConfig` (utils.py)
- [x] Fields: `data_storage: IStorage`, `customized_data_storages: dict[str, IStorage]`
- [ ] Fix `get_timeguarded_aux_reader()` — still references `self.aux_data_provider` (old field)

---

## Open Questions

1. **get_ohlc() — still needed?** Used by `ctx.ohlc()` for lookback. New strategies use streaming OHLCV series. May be able to simplify or stub.
2. **Can strategy subscribe to exchange not in config?** `_get_or_create_reader()` will call `storage.get_reader(exchange, market_type)` — will raise if storage doesn't have that exchange. Probably fine as-is.
