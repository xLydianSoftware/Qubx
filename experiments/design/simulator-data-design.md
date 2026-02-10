# Simulator Data Layer: Adopting New IterableSimulationData

## Status: DRAFT

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

### Constructor (already updated)
```python
def __init__(
    self,
    storage: IStorage,
    custom_types_storages: dict[str, IStorage] | None = None,
    ...
):
    self._storage = storage
    self._custom_storages = dict(custom_types_storages or {})
    self._readers = {}   # cache: reader_key -> IReader (lazy, populated on first need)
```

### Reader resolution flow

When `add_instruments_for_subscription(subscription, instruments)` is called:

```
1. Parse subscription -> (access_key, data_type, params)
   e.g. "ohlc(1h)" -> ("ohlc.1h", "ohlc", {"timeframe": "1h"})

2. For each instrument, derive reader key from instrument:
   reader_key = f"{instrument.exchange}:{instrument.market_type}"
   e.g. "BINANCE.UM:SWAP"

3. Get-or-create reader (cached in self._readers):
   a) Check custom storage first: self._custom_storages.get(data_type)
      - If found -> custom_storage.get_reader(exchange, market_type)
   b) Otherwise -> self._storage.get_reader(exchange, market_type)
   c) Cache: self._readers[cache_key] = reader

4. Reader is expensive to create (may open DB connections) -> cache once, reuse.
```

### `_get_or_create_reader` (NEW method)

```python
def _get_or_create_reader(self, data_type: str, exchange: str, market_type: str) -> IReader:
    # - check custom storage first
    custom_storage = self._custom_storages.get(data_type)
    if custom_storage is not None:
        cache_key = f"{data_type}:{exchange}:{market_type}"
        if cache_key not in self._readers:
            self._readers[cache_key] = custom_storage.get_reader(exchange, market_type)
        return self._readers[cache_key]

    # - fallback to main storage
    cache_key = f"{exchange}:{market_type}"
    if cache_key not in self._readers:
        self._readers[cache_key] = self._storage.get_reader(exchange, market_type)
    return self._readers[cache_key]
```

Custom storage readers get data_type-prefixed cache key (different storages may return different readers for same exchange). Main storage reader is shared across all data types for same (exchange, market_type) since IReader handles all types.

---

## Pump Granularity

**One pump per (subscription, exchange, market_type).**

A pump groups all symbols from the same exchange+market under the same subscription.
E.g. `ohlc(1h)` + `BINANCE.UM:SWAP` = one pump serving BTCUSDT, ETHUSDT, SOLUSDT, etc.

Each pump has exactly one IReader. Pump internals stay simple — no multi-reader logic.

### Pump key format
```
pump_key = f"{access_key}.{exchange}:{market_type}"
```
Examples:
- `"ohlc.1h.BINANCE.UM:SWAP"` — 1h OHLC for all Binance perpetuals
- `"ohlc.4h.BINANCE.UM:SWAP"` — 4h OHLC for same exchange
- `"ohlc.1h.BYBIT:SWAP"` — 1h OHLC for Bybit perpetuals
- `"quote.BINANCE.UM:SWAP"` — raw quotes for Binance

### Slicer key format

**Must include exchange + market_type** to avoid collisions (BTCUSDT exists on multiple exchanges).

Current: `f"{requested_data_type}.{symbol}"` -> `"ohlc(1h).BTCUSDT"`

New: `f"{requested_data_type}.{exchange}:{market_type}:{symbol}"`
-> `"ohlc(1h).BINANCE.UM:SWAP:BTCUSDT"`

DataPump needs exchange + market_type context to build this key.
Options:
- a) Pass exchange/market_type to DataPump constructor, store as fields
- b) Derive from attached instruments (all share same exchange in one pump)

**Option (a)** is cleaner — pump already knows its scope at creation time.

```python
class DataPump:
    def __init__(self, reader, subscription_type, exchange, market_type, ...):
        self._exchange = exchange
        self._market_type = market_type
        ...

    def _make_slicer_key(self, symbol: str) -> str:
        return f"{self._requested_data_type}.{self._exchange}:{self._market_type}:{symbol}"
```

### Updated `_get_or_create_pump`

```python
def _get_or_create_pump(self, access_key: str, subscription: str, data_type: str,
                         exchange: str, market_type: str) -> DataPump:
    pump_key = f"{access_key}.{exchange}:{market_type}"

    if pump_key in self._pumps:
        return self._pumps[pump_key]

    reader = self._get_or_create_reader(data_type, exchange, market_type)

    pump = DataPump(
        reader=reader,
        subscription_type=subscription,
        exchange=exchange,
        market_type=market_type,
        warmup_period=self._warmups.get(access_key),
        chunksize=self._chunksize,
        open_close_time_indent_secs=self._open_close_time_indent_secs,
        trading_session=self._trading_session,
    )
    self._pumps[pump_key] = pump
    return pump
```

### Updated `add_instruments_for_subscription`

```python
def add_instruments_for_subscription(self, subscription, instruments):
    instruments = instruments if isinstance(instruments, list) else [instruments]
    access_key, data_type, _params = self._parse_subscription_spec(subscription)
    instruments = self._filter_instruments_for_subscription(data_type, instruments)
    if not instruments:
        return

    # - group instruments by (exchange, market_type) -> each group gets its own pump
    groups: dict[tuple[str, str], list[Instrument]] = {}
    for i in instruments:
        groups.setdefault((i.exchange, i.market_type), []).append(i)

    for (exchange, market_type), group_instruments in groups.items():
        pump = self._get_or_create_pump(access_key, subscription, data_type, exchange, market_type)

        new_instruments = []
        for i in group_instruments:
            if not pump.has_instrument(i):
                pump.attach_instrument(i)
                slicer_key = pump._make_slicer_key(i.symbol)
                self._instruments[slicer_key] = (i, pump, subscription)
                new_instruments.append(i)

        if self.is_running and new_instruments:
            new_mem_readers = pump.restart_read(
                pd.Timestamp(self._current_time, unit="ns"), self._stop
            )
            if new_mem_readers and self._slicer is not None:
                self._slicer.put(new_mem_readers)
```

### Updated `remove_instruments_from_subscription`

Need to find the correct pump for each instrument. Current code does `self._pumps.get(access_key)` — but now pumps are keyed by `access_key.exchange:market_type`.

Options:
- a) Look up pump from `self._instruments[slicer_key]` — already stores `(instrument, pump, subscription)`
- b) Reconstruct pump_key from instrument

**Option (a)** is simplest — we already have the pump reference:

```python
def remove_instruments_from_subscription(self, subscription, instruments):
    instruments = instruments if isinstance(instruments, list) else [instruments]

    if subscription == DataType.ALL:
        # - remove from ALL pumps (search by instrument in self._instruments)
        for i in instruments:
            self._remove_instrument_from_all_pumps(i)
        return

    access_key, data_type, _ = self._parse_subscription_spec(subscription)

    # - group by (exchange, market_type) to find pump
    groups: dict[tuple[str, str], list[Instrument]] = {}
    for i in instruments:
        groups.setdefault((i.exchange, i.market_type), []).append(i)

    for (exchange, market_type), group_instruments in groups.items():
        pump_key = f"{access_key}.{exchange}:{market_type}"
        pump = self._pumps.get(pump_key)
        if not pump:
            continue

        keys_to_remove = []
        for i in group_instruments:
            slicer_key = pump.remove_instrument(i)
            if slicer_key:
                self._instruments.pop(slicer_key, None)
                keys_to_remove.append(slicer_key)

        if self.is_running and keys_to_remove and self._slicer is not None:
            self._slicer.remove(keys_to_remove)

        pump.cleanup_inactive()
```

### Updated query methods

`get_instruments_for_subscription` currently finds pump by `access_key`. Now there can be multiple pumps per access_key (one per exchange). Need to iterate all matching pumps:

```python
def get_instruments_for_subscription(self, subscription):
    if subscription == DataType.ALL:
        return list(i for i, _, _ in self._instruments.values())

    access_key, _, _ = self._parse_subscription_spec(subscription)
    result = []
    # - find all pumps matching this access_key prefix
    for pump_key, pump in self._pumps.items():
        if pump_key.startswith(access_key + "."):
            result.extend(pump.get_instruments())
    return result
```

`peek_historical_data` needs to find the right pump for a specific instrument:

```python
def peek_historical_data(self, instrument, subscription):
    ...
    access_key, _, _ = self._parse_subscription_spec(subscription)
    pump_key = f"{access_key}.{instrument.exchange}:{instrument.market_type}"
    pump = self._pumps.get(pump_key)
    if pump is None:
        return []

    slicer_key = pump._make_slicer_key(instrument.symbol)
    return self._slicer.fetch_before_time(slicer_key, self._current_time)
```

---

## Changes Summary

### `DataPump` (simulated_data.py)
- [ ] Add `exchange: str` and `market_type: str` to constructor + store as fields
- [ ] Update `_make_slicer_key()` to include exchange:market_type

### `IterableSimulationData` (simulated_data.py)
- [x] Constructor: already takes `storage: IStorage` + `custom_types_storages`
- [ ] Add `_get_or_create_reader(data_type, exchange, market_type) -> IReader`
- [ ] Update `_get_or_create_pump()` — add exchange/market_type params, pump key includes exchange
- [ ] Update `add_instruments_for_subscription()` — group by (exchange, market_type), per-group pump
- [ ] Update `remove_instruments_from_subscription()` — find pump per (access_key, exchange, market_type)
- [ ] Update `get_instruments_for_subscription()` — iterate all matching pumps
- [ ] Update `peek_historical_data()` — construct pump_key from instrument
- [ ] Update `__iter__` initial pump read loop (already iterates all pumps — OK as-is)

### `SimulatedDataProvider` (data.py)
- [ ] Drop `readers: dict[str, DataReader]` from `__init__`
- [ ] Rework `get_ohlc()` — use IReader or deprecate

### `SimulationRunner` (runner.py)
- [ ] Update `IterableSimulationData(...)` construction — pass storage + custom storages
- [ ] Update `SimulatedDataProvider(...)` construction — drop old readers param
- [ ] Fix custom subscription handling (line ~166)

### `SimulationDataConfig` (utils.py)
- [x] Fields updated: `data_storage`, `customized_data_storages`
- [ ] Fix `get_timeguarded_aux_reader()` — references old field

---

## Open Questions

1. **get_ohlc() — still needed?** Check callers. If only used by old strategies, can deprecate.
2. **Can strategy subscribe to exchange not in config?** If yes, storage.get_reader() will attempt it — may fail if storage doesn't have that exchange. Probably fine (let it raise).

---

## Notes

<!-- push thoughts here -->
