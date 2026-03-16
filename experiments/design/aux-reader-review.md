# Aux Data System — Full Architecture Review

## Overview

The aux data system provides **auxiliary data** (funding rates, open interest, fundamental data, etc.)
to strategies via `ctx.get_aux_data(data_id, **kwargs)`. It is **entirely built on the old `DataReader` class**
with reflection-based `get_*()` method discovery — needs migration to `IStorage`/`IReader`.

---

## System Architecture

```mermaid
graph TB
    subgraph "YAML Configuration"
        YC["StrategyConfig.aux<br/><code>configs.py:220</code>"]
        YL["LiveConfig.aux<br/><code>configs.py:180</code>"]
        YS["SimulationConfig.aux<br/><code>configs.py:200</code>"]
        PF["PrefetchConfig<br/><code>configs.py:91-96</code>"]
    end

    subgraph "Config Resolution"
        RAC["resolve_aux_config()<br/><code>configs.py:244-260</code>"]
    end

    subgraph "Factory — Reader Construction"
        CAR["construct_aux_reader()<br/><code>factory.py:402-438</code>"]
        CR["construct_reader()<br/><code>factory.py</code>"]
    end

    subgraph "Old DataReader Hierarchy"
        DR["DataReader<br/><code>readers.py:112-162</code><br/>get_aux_data() / get_aux_data_ids()<br/><i>reflection: get_{data_id}()</i>"]
        COMP["CompositeReader<br/><code>readers.py:2249-2563</code><br/>merges N readers"]
        IMCR["InMemoryCachedReader<br/><code>helpers.py:25-297</code><br/>caches candle data"]
        TGW["TimeGuardedWrapper<br/><code>helpers.py:300-340</code><br/>prevents look-ahead"]
        CPR["CachedPrefetchReader<br/><code>helpers.py:462-2066</code><br/>prefetch + caching"]
    end

    subgraph "Core Interfaces & Wiring"
        IMM["IMarketManager.get_aux_data()<br/><code>interfaces.py:893</code>"]
        MM["MarketManager<br/><code>market.py:28-134</code><br/>_aux_data_provider: DataReader | None"]
    end

    subgraph "Strategy Context"
        SC["StrategyContext<br/><code>context.py:143</code> — aux_data_provider param<br/><code>context.py:181</code> — self._aux<br/><code>context.py:332</code> — ctx.aux property<br/><code>context.py:631</code> — get_aux_data()"]
    end

    subgraph "Strategy (User Code)"
        USR["strategy.on_event():<br/>ctx.get_aux_data('candles', ...)<br/>ctx.get_aux_data('funding_rates', ...)"]
    end

    subgraph "Runners"
        LR["Live Runner<br/><code>runner.py:450-493</code>"]
        SR["SimulationRunner<br/><code>backtester/runner.py:126-612</code>"]
        WR["Warmup Runner<br/><code>runner.py:847</code>"]
        CLR["CLI simulate()<br/><code>runner.py:1040-1043</code>"]
    end

    subgraph "Backtester Config"
        SDC["SimulationDataConfig<br/><code>backtester/utils.py:102-140</code><br/>aux_providers / prefetch_config"]
        RSDC["recognize_simulation_data_config()<br/><code>backtester/utils.py:604-638</code>"]
        SIM["simulate()<br/><code>simulator.py:44</code> — aux_data param"]
    end

    subgraph "Initializer"
        INIT["BasicStrategyInitializer<br/><code>initializer.py:144-168</code><br/>set_data_cache_config()<br/>get_data_cache_config()"]
    end

    %% Config flow
    YC --> RAC
    YL --> RAC
    YS --> RAC
    RAC --> CAR

    %% Factory builds readers
    CAR --> CR
    CR -->|single| DR
    CR -->|multiple| COMP
    COMP --> DR

    %% Live runner path
    CAR -->|"_aux_reader"| LR
    PF -->|"if prefetch enabled"| LR
    LR -->|"wrap"| CPR
    CPR --> DR
    LR -->|"aux_data_provider="| SC

    %% Simulation runner path (CLI)
    CAR -->|"sim_params['aux_data']"| CLR
    CLR --> SIM
    SIM --> RSDC
    RSDC --> SDC

    %% Simulation runner path (direct simulate())
    SIM -->|"aux_data="| RSDC

    %% SimulationRunner internal
    SDC -->|"get_timeguarded_aux_reader()"| SR
    SR -->|"self._aux_data_reader"| TGW
    TGW --> CPR
    SR -->|"_prefetch_aux_data()"| CPR
    SR -->|"aux_data_provider="| SC

    %% Warmup runner
    WR -->|"ctx.aux →<br/>recognize_simulation_data_config()"| RSDC

    %% Context internal wiring
    SC -->|"aux_data_provider="| MM
    MM -->|"_aux_data_provider"| DR

    %% Strategy calls
    SC --> IMM
    IMM --> MM
    MM -->|"get_aux_data()"| DR
    USR --> SC

    %% Initializer
    INIT -.->|"cache config for CachedPrefetchReader"| CPR

    %% Styling
    classDef old fill:#ffcccc,stroke:#cc0000,color:#333
    classDef iface fill:#cce5ff,stroke:#0066cc,color:#333
    classDef config fill:#fff3cd,stroke:#cc9900,color:#333
    classDef runner fill:#d4edda,stroke:#28a745,color:#333
    classDef user fill:#e2d5f1,stroke:#6f42c1,color:#333

    class DR,COMP,IMCR,TGW,CPR old
    class IMM,MM iface
    class YC,YL,YS,PF,RAC,SDC,RSDC,SIM config
    class LR,SR,WR,CLR,CAR,CR runner
    class USR,SC user
```

---

## Detailed Call Chains

### 1. Live Runner Path

```mermaid
sequenceDiagram
    participant YAML as YAML Config
    participant RC as resolve_aux_config()<br/>configs.py:244
    participant CAR as construct_aux_reader()<br/>factory.py:402
    participant LR as Live Runner<br/>runner.py:448-493
    participant CPR as CachedPrefetchReader<br/>helpers.py:462
    participant SC as StrategyContext<br/>context.py:143
    participant MM as MarketManager<br/>market.py:42
    participant S as Strategy

    YAML->>RC: config.aux / config.live.aux
    RC->>CAR: list[ReaderConfig]
    CAR->>CAR: construct_reader() × N
    CAR-->>LR: DataReader | CompositeReader

    alt prefetch enabled
        LR->>CPR: wrap with CachedPrefetchReader
        CPR-->>LR: CachedPrefetchReader
    end

    LR->>SC: StrategyContext(aux_data_provider=_aux_reader)
    SC->>MM: MarketManager(aux_data_provider=...)
    Note over SC: self._aux = aux_data_provider

    S->>SC: ctx.get_aux_data("funding_rates", ...)
    SC->>MM: get_aux_data("funding_rates", ...)
    MM->>CPR: get_aux_data("funding_rates", ...)
    CPR->>CAR: _reader.get_aux_data(...)
    Note over CAR: reflection: getattr(self, "get_funding_rates")()
```

### 2. Simulation Runner Path (via `simulate()`)

```mermaid
sequenceDiagram
    participant U as User Code
    participant SIM as simulate()<br/>simulator.py:44
    participant RSDC as recognize_simulation_data_config()<br/>backtester/utils.py:604
    participant SDC as SimulationDataConfig<br/>backtester/utils.py:102
    participant SR as SimulationRunner<br/>backtester/runner.py
    participant TGW as TimeGuardedWrapper<br/>helpers.py:300
    participant CPR as CachedPrefetchReader
    participant SC as StrategyContext

    U->>SIM: simulate(strategy, data, aux_data={...})
    SIM->>RSDC: recognize_simulation_data_config(data, aux_data, prefetch_config)
    RSDC->>SDC: SimulationDataConfig(data_storage, aux_providers, prefetch_config)
    SDC-->>SR: data_config

    SR->>SDC: get_timeguarded_aux_reader(time_provider)
    Note over SDC: Currently returns None! (commented out)
    SDC-->>SR: None

    SR->>SR: _prefetch_aux_data()
    Note over SR: Skipped if _aux_data_reader is None

    SR->>SC: StrategyContext(aux_data_provider=self._aux_data_reader)
    Note over SC: aux_data_provider = None (broken path)
```

### 3. CLI Simulate Path (via YAML)

```mermaid
sequenceDiagram
    participant CLI as CLI simulate<br/>runner.py:1040
    participant RC as resolve_aux_config()<br/>configs.py:244
    participant CAR as construct_aux_reader()<br/>factory.py:402
    participant SIM as simulate()<br/>simulator.py:44
    participant RSDC as recognize_simulation_data_config()

    CLI->>RC: resolve_aux_config(cfg.aux, cfg.simulation.aux)
    RC-->>CLI: list[ReaderConfig]
    CLI->>CAR: construct_aux_reader(aux_configs)
    CAR-->>CLI: DataReader | CompositeReader | None
    CLI->>SIM: simulate(..., aux_data=_aux_reader)
    Note over SIM: But aux_data param expects dict[str, IStorage]!
    SIM->>RSDC: recognize_simulation_data_config(data, aux_data)
    Note over RSDC: Type mismatch: gets DataReader,<br/>validates isinstance(IStorage) → skip
```

### 4. Warmup Runner Path

```mermaid
sequenceDiagram
    participant WR as Warmup Runner<br/>runner.py:845-857
    participant CTX as StrategyContext
    participant RSDC as recognize_simulation_data_config()
    participant SR as SimulationRunner

    WR->>CTX: ctx.aux
    Note over CTX: Returns DataReader | None (from live runner)
    WR->>RSDC: recognize_simulation_data_config(decls, aux_data=ctx.aux)
    Note over RSDC: ctx.aux is DataReader, not dict[str, IStorage]<br/>→ silent skip (wrong type)
    RSDC-->>SR: SimulationDataConfig(aux_providers={})
```

---

## File Reference Index

| File | Lines | Role |
|------|-------|------|
| [`configs.py`](../../src/qubx/utils/runner/configs.py) | 91-96, 180, 200, 220, 244-260 | YAML config models, `PrefetchConfig`, `resolve_aux_config()` |
| [`factory.py`](../../src/qubx/utils/runner/factory.py) | 402-438 | `construct_aux_reader()` — builds `DataReader`/`CompositeReader` from config |
| [`runner.py`](../../src/qubx/utils/runner/runner.py) | 258, 448-493, 847, 1040-1043 | Live runner wiring, warmup, CLI simulate |
| [`readers.py`](../../src/qubx/data/readers.py) | 112-162, 2249-2563 | `DataReader` base (reflection dispatch), `CompositeReader` |
| [`helpers.py`](../../src/qubx/data/helpers.py) | 25-297, 300-340, 462-2066 | `InMemoryCachedReader`, `TimeGuardedWrapper`, `CachedPrefetchReader` |
| [`interfaces.py`](../../src/qubx/core/interfaces.py) | 893, 2430-2451 | `IMarketManager.get_aux_data()`, `IStrategyInitializer.set_data_cache_config()` |
| [`market.py`](../../src/qubx/core/mixins/market.py) | 28-134 | `MarketManager._aux_data_provider: DataReader \| None` |
| [`context.py`](../../src/qubx/core/context.py) | 143, 181, 332, 631 | `StrategyContext._aux`, `ctx.aux`, `ctx.get_aux_data()` |
| [`backtester/utils.py`](../../src/qubx/backtester/utils.py) | 102-140, 604-638 | `SimulationDataConfig`, `recognize_simulation_data_config()` |
| [`backtester/runner.py`](../../src/qubx/backtester/runner.py) | 126, 391, 466, 575-612 | `SimulationRunner._aux_data_reader`, `_prefetch_aux_data()` |
| [`simulator.py`](../../src/qubx/backtester/simulator.py) | 44, 107 | `simulate(aux_data=...)` entry point |
| [`initializer.py`](../../src/qubx/core/initializer.py) | 144-168 | `BasicStrategyInitializer.set_data_cache_config()` |

---

## Key Problems

### 1. Type Mismatch — simulation vs live
- **Live runner** (`runner.py:493`): passes `DataReader` as `aux_data_provider` → works
- **CLI simulate** (`runner.py:1043`): passes `DataReader` as `aux_data` param to `simulate()` which expects `dict[str, IStorage] | None` → **type mismatch**, silently ignored in `recognize_simulation_data_config()`
- **Warmup** (`runner.py:847`): passes `ctx.aux` (a `DataReader`) as `aux_data` → same type mismatch → silently ignored

### 2. `get_timeguarded_aux_reader()` is Dead Code
- `SimulationDataConfig.get_timeguarded_aux_reader()` (`backtester/utils.py:117`) — **entirely commented out**, always returns `None`
- This means `SimulationRunner._aux_data_reader` is always `None` in simulation
- `_prefetch_aux_data()` always early-returns

### 3. Reflection-Based Dispatch
- `DataReader.get_aux_data()` uses `hasattr(self, f"get_{data_id}")` → fragile, no type safety
- `get_aux_data_ids()` scans class methods via `__dict__` → breaks with inheritance, dynamic methods

### 4. Deep Wrapper Nesting
- Live path can stack: `TimeGuardedWrapper(CachedPrefetchReader(CompositeReader([DR, DR, ...])))`
- `_prefetch_aux_data()` manually unwraps `isinstance` chains (`runner.py:581-584`)

### 5. Dual System — IStorage vs DataReader
- New data path uses `IStorage` → `IReader.read()`
- Aux data still entirely on old `DataReader.get_aux_data()` with `get_*()` reflection
- No bridge between the two systems

---

## Old DataReader Hierarchy (to refactor)

```mermaid
classDiagram
    class DataReader {
        +get_aux_data(data_id, **kwargs) Any
        +get_aux_data_ids() set~str~
        +read(data_id, start, stop, ...) list
        +get_names() list~str~
        +get_symbols(exchange, dtype) list~str~
        #reflection: get_{data_id}(**kwargs)
    }

    class InMemoryDataFrameReader {
        +read()
    }

    class InMemoryCachedReader {
        -_reader: DataReader
        -_external: dict
        +get_aux_data(data_id, **kwargs)
        +get_aux_data_ids()
    }

    class TimeGuardedWrapper {
        -_reader: DataReader
        -_time_guard_provider: ITimeProvider
        +read(data_id, ...)
        +get_aux_data(data_id, **kwargs)
        -_time_guarded_data(data)
    }

    class CachedPrefetchReader {
        -_reader: DataReader
        -_aux_cache: dict
        -_read_cache: dict
        +read(data_id, ...)
        +get_aux_data(data_id, **kwargs)
        +get_aux_data_ids()
        +prefetch_aux_data(names, ...)
        -_fetch_and_cache_aux_data()
        -_filter_aux_data_to_requested_range()
        -_merge_aux_data()
        -_detect_aux_data_overlap()
    }

    class CompositeReader {
        -readers: list~DataReader~
        +get_aux_data(data_id, **kwargs)
        +get_aux_data_ids()
        +read(data_id, ...)
        -_merge_aux_data()
    }

    DataReader <|-- InMemoryDataFrameReader
    InMemoryDataFrameReader <|-- InMemoryCachedReader
    DataReader <|-- TimeGuardedWrapper
    DataReader <|-- CachedPrefetchReader
    DataReader <|-- CompositeReader

    TimeGuardedWrapper o-- DataReader : wraps
    CachedPrefetchReader o-- DataReader : wraps
    CompositeReader o-- DataReader : wraps N
    InMemoryCachedReader o-- DataReader : wraps
```

---

## Summary: What Touches Aux Data

| Component | Type | Status | Notes |
|-----------|------|--------|-------|
| `StrategyConfig.aux` | Config | ✅ Active | YAML-level config |
| `LiveConfig.aux` | Config | ✅ Active | Section override |
| `SimulationConfig.aux` | Config | ✅ Active | Section override |
| `resolve_aux_config()` | Config util | ✅ Active | Merges global/section |
| `construct_aux_reader()` | Factory | ✅ Active | Builds DataReader hierarchy |
| `DataReader.get_aux_data()` | Base | ✅ Active | Reflection dispatch |
| `CompositeReader` | Wrapper | ✅ Active | Multi-reader merge |
| `CachedPrefetchReader` | Wrapper | ✅ Active | Caching + prefetch |
| `TimeGuardedWrapper` | Wrapper | ✅ Active (live only) | Look-ahead prevention |
| `InMemoryCachedReader` | Wrapper | ✅ Active | In-memory candle cache |
| `MarketManager._aux_data_provider` | Core | ✅ Active | Stores DataReader |
| `StrategyContext._aux` / `ctx.aux` | Core | ✅ Active | Stores & exposes DataReader |
| `ctx.get_aux_data()` | Core | ✅ Active | Strategy-facing API |
| `IMarketManager.get_aux_data()` | Interface | ✅ Active | Abstract interface |
| `IStrategyInitializer.set_data_cache_config()` | Interface | ⚠️ Partial | Config only, not wired to runtime |
| `SimulationDataConfig.get_timeguarded_aux_reader()` | Backtester | ❌ Dead | Commented out, returns None |
| `SimulationRunner._prefetch_aux_data()` | Backtester | ❌ Dead | Always skipped (reader=None) |
| `simulate(aux_data=...)` | Entry point | ⚠️ Broken | Expects `dict[str, IStorage]`, gets `DataReader` from CLI |
| Warmup runner `ctx.aux` pass-through | Runner | ⚠️ Broken | Type mismatch: DataReader → IStorage check fails |
