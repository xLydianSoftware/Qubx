# Connector Registry Redesign — Design Spec

- **Date:** 2026-06-23
- **Branch:** `feat/connector-registry-redesign` (off `account-mgmt-redesign`)
- **Tracking issue:** [xLydianSoftware/Qubx#318](https://github.com/xLydianSoftware/Qubx/issues/318)
- **Status:** approved design, pre-implementation

## Summary

Replace the connector plugin registration system — three import-side-effect decorators
(`@connector` / `@data_provider` / `@rate_limit_config`) feeding an untyped `**kwargs` factory — with:

1. A typed, two-part **build context** (`BuildContext` → `ConnectorBuildContext`).
2. A cohesive **`ExchangePlugin`** ABC that groups a venue's connector + data provider + rate-limit
   declaration into one object.
3. **Entry-point discovery** (`importlib.metadata`, group `qubx.exchange_plugins`) replacing the
   decorators, with lazy per-name loading.

This is done now, before the Hyperliquid connector, so external connectors (`qubx-hyperliquid`,
`qubx-lighter`) are authored once against the final contract. It rides the `account-mgmt-redesign`
breaking window so external plugins absorb a single break.

## Background & motivation

The `account-mgmt-redesign` branch already collapsed the old `IBroker`/`IAccountProcessor` split into a
single `IConnector`. Registration, however, still has three weaknesses (see #318):

1. **Untyped construction.** `ConnectorRegistry.get_connector(name, **kwargs)` calls `factory(**kwargs)`
   (`registry.py:126-140`); the required kwargs are a convention reverse-engineered from `runner.py:568`.
   A mismatch fails at runtime with `TypeError: unexpected keyword argument`.
2. **A venue is three scattered decorators** that must agree on a name string by hand.
3. **Import-for-side-effect registration** — nothing is registered until a module is imported; global
   mutable state populated by import order.

## Scope

**In scope (Qubx only):**
- New core types, registry, loader; runner two-phase build; migrate in-tree `ccxt` + `tardis`; move
  `read_only` enforcement to the framework; make `loop` required and retire `AsyncThreadLoop`; entry-point
  declarations.

**Out of scope (follow-ups):**
- Rewriting `qubx-hyperliquid` / `qubx-lighter` (done in the `exchanges` repo against this contract).
- The general plugin loader for `@storage` / `@reader` (unchanged).
- Market-data event typing (still rides tuples).
- Changes to `IConnector` / `IDataProvider` interfaces themselves.

## Architecture

### Core types — `src/qubx/connectors/plugin.py` (new)

```python
@dataclass(frozen=True, kw_only=True)
class BuildContext:
    exchange_name: str                          # the VENUE instance, e.g. "BINANCE.UM"
    time_provider: ITimeProvider
    channel: CtrlChannel
    credentials: CredentialsProvider            # registry.py:35
    health_monitor: IHealthMonitor
    loop: asyncio.AbstractEventLoop             # REQUIRED — connectors consume a loop, never create one
    rate_limiter: ExchangeRateLimiter | None = None   # shared, built by RateLimitManager; None = disabled

@dataclass(frozen=True, kw_only=True)
class ConnectorBuildContext(BuildContext):
    data_provider: IDataProvider                # the already-built provider the connector depends on

class ExchangePlugin(ABC):
    name: str                                   # connector KIND / registry key (e.g. "ccxt")
    def create_data_provider(self, ctx: BuildContext) -> IDataProvider | None: return None
    def create_connector(self, ctx: ConnectorBuildContext) -> IConnector | None: return None
    def rate_limits(self, exchange_name: str) -> ExchangeRateLimitConfig | None: return None
```

- **Two context types** model the build ordering: the data provider is built from `BuildContext`; the
  connector from `ConnectorBuildContext`, which adds the already-built `data_provider`. The type system
  guarantees a connector author never sees a `None` data provider. `ConnectorBuildContext` *is-a*
  `BuildContext`, so `create_data_provider` accepts either.
- **`ExchangePlugin` is an ABC with `None`-returning defaults.** A plugin overrides only what it provides
  (tardis overrides only `create_data_provider`). Partial-capability venues are first-class.
- **`read_only` is NOT in the context** — it is framework trading policy, enforced at the trading mixin
  (see below).

### Registry — `src/qubx/connectors/registry.py` (slimmed)

A single `dict[str, ExchangePlugin]`. The three decorators and their three dicts are deleted.

```python
class ConnectorRegistry:
    _plugins: dict[str, ExchangePlugin] = {}

    @classmethod
    def register(cls, plugin: ExchangePlugin) -> None:           # tests / programmatic
        cls._plugins[plugin.name.lower()] = plugin

    @classmethod
    def get_plugin(cls, name: str) -> ExchangePlugin:
        name = name.lower()
        if name not in cls._plugins:
            plugin = PluginLoader.load(name)                     # lazy entry-point import
            if plugin is None:
                raise ValueError(f"No connector plugin '{name}'. Available: {sorted(PluginLoader.available())}")
            cls._plugins[name] = plugin
        return cls._plugins[name]

    @classmethod
    def get_data_provider(cls, name: str, ctx: BuildContext) -> IDataProvider:
        dp = cls.get_plugin(name).create_data_provider(ctx)
        if dp is None:
            raise ValueError(f"Venue plugin '{name}' provides no data provider")
        return dp

    @classmethod
    def get_connector(cls, name: str, ctx: ConnectorBuildContext) -> IConnector:
        conn = cls.get_plugin(name).create_connector(ctx)
        if conn is None:
            raise ValueError(f"Venue plugin '{name}' provides no execution connector")  # e.g. tardis
        return conn

    @classmethod
    def get_rate_limit_config(cls, name: str, exchange_name: str) -> ExchangeRateLimitConfig | None:
        return cls.get_plugin(name).rate_limits(exchange_name)   # None is VALID (rate limiting off)
```

The convenience methods centralize the capability `None`-guard so callers (runner, `RateLimitManager`)
keep calling by name and only swap `**kwargs` for a typed `ctx`.

### Discovery — `src/qubx/plugins/loader.py` (`PluginLoader` added)

Entry-point group **`qubx.exchange_plugins`**. The existing `load_plugins(paths, modules)` stays
untouched for `@storage` / `@reader`.

- `available() -> set[str]` — entry-point **names** from `importlib.metadata.entry_points(group=...)`,
  **no plugin imported** (pure metadata). Used to validate `connector: foo` before anything loads.
- `load(name) -> ExchangePlugin | None` — finds the entry point whose name == `name`, calls `ep.load()`
  (the only import), validates `isinstance(obj, ExchangePlugin)`, asserts `obj.name == name`, returns it.
  Unknown name → `None`.
- The `EntryPoints` scan is cached once; loaded plugins cached in `ConnectorRegistry._plugins`. Loading is
  **lazy per name** — a Binance-only bot never imports the Hyperliquid/Lighter modules; the
  `qubx[connectors]` extra is only needed when a ccxt venue is actually selected.

**Name handling:** the entry-point key **is** the connector name (the config's `connector:` value), so the
lazy lookup needs no import. `plugin.name` is kept on the object (logging / self-identification) and the
loader **asserts `plugin.name == ep.name`** to catch drift.

**Errors:** unknown name → `ValueError` listing `available()`; `ModuleNotFoundError` on `ep.load()` (e.g.
missing extra) → wrapped with "install `qubx[connectors]`"; non-`ExchangePlugin` → `TypeError`.

**Caveat:** entry points require installed `.dist-info`. Qubx and its plugins are always installed
(editable counts); a raw uninstalled source tree would not be discovered.

### Built-ins (ccxt, tardis) — same path

A package can declare entry points pointing at itself, so Qubx declares its built-ins in its own
`pyproject.toml`:

```toml
[project.entry-points."qubx.exchange_plugins"]
ccxt   = "qubx.connectors.ccxt.plugin:PLUGIN"
tardis = "qubx.connectors.tardis.plugin:PLUGIN"
```

They are discovered alongside third-party plugins with no special-casing (the pytest-builtin-plugins
pattern). Lazy `.load()` respects the `[connectors]` extra; future extraction to a `qubx-ccxt` package
would only relocate the entry-point line.

## Construction flow (runner two-phase)

Per exchange (replaces `runner.py:541-578`, `:760-771`):

```python
limiter = _rl_manager.get_or_create(venue, conn_name)           # builds/caches one limiter per exchange
base = BuildContext(exchange_name=venue, time_provider=_time, channel=_chan,
                    credentials=account_manager, health_monitor=_health_monitor,
                    loop=loop, rate_limiter=limiter)
dp   = ConnectorRegistry.get_data_provider(conn_name, base)
cctx = ConnectorBuildContext(**vars(base), data_provider=dp)
conn = ConnectorRegistry.get_connector(conn_name, cctx)
```

The old `exchange_config.params["rate_limiter"]` injection (`runner.py:543`) and the `read_only=` connector
arg (`:576`) are removed.

## Rate-limit wiring

Ownership splits three ways — the plugin **declares**, the manager **owns**, the context **shares**:

```
RateLimitManager.get_or_create(venue, connector)            # manager.py:91
  ├─ if self._backend is None: return None                  # rate limiting off → no plugin call
  ├─ ConnectorRegistry.get_rate_limit_config(connector, venue)   # UNCHANGED call site (manager.py:111)
  │     └─ get_plugin(connector).rate_limits(venue)  ─►  ExchangeRateLimitConfig (pools/weights)
  ├─ builds ExchangeRateLimiter(config, backend, scope, loop); caches per-exchange
  └─ returns the limiter
        │  (runner places it in BuildContext.rate_limiter)
        ▼
  plugin.create_data_provider(ctx)  → attaches ctx.rate_limiter to the DP's exchange manager
  plugin.create_connector(cctx)     → attaches cctx.rate_limiter to the connector's exchange manager
```

- `manager.py:111` is **unchanged** — only the registry method it calls now delegates to the plugin.
- This closes a current gap: today the limiter is attached only to the data provider
  (`ccxt/data.py:46,61`); the authenticated connector exchange manager (`ccxt/factory.py:182-213`) never
  gets one. After this change both share the **same** `ExchangeRateLimiter` instance.

## `read_only` — single framework gate

`read_only` is framework trading policy, not connector construction input, and the trading mixin is the
**sole** caller of connector write methods (`trading.py:115` submit, `:327` cancel, `:387` update). So:

- Enforce `read_only` **once** in the trading mixin: raise `ReadOnlyConnector` at those three dispatch
  points before calling the connector. The flag is threaded from `config.live.read_only`
  (`configs.py`, `LiveConfig`) onto the trading mixin at context construction (one `self._read_only`).
- Delete the five per-connector guards (`ccxt/connector.py:302,404,572,701,713`) and the `read_only`
  ctor param. Templates and `LiveConfig.read_only` are **kept** (default `false`), so no config migration.

## `loop` ownership (and retiring `AsyncThreadLoop`)

Connectors consume a loop, never create one. `loop` is **required** on `BuildContext` (no default). The
runner already creates one shared loop on a dedicated thread (`runner.py:274-276`, the `SharedEventLoop`
thread) and passes it everywhere.

Two helpers in `qubx/utils/misc.py` make loop ownership and cross-thread submission explicit, and replace
the under-powered `AsyncThreadLoop`:

```python
def run_sync(loop: asyncio.AbstractEventLoop, coro, *, timeout: float | None = None):
    """Submit `coro` to `loop` from another thread, block for the result, propagate its exception.
    Guards the classic deadlock of being called from `loop`'s own thread."""
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        raise RuntimeError("run_sync called from the target loop's own thread — would deadlock")
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)

class BackgroundEventLoop:
    """Owns an asyncio loop on a dedicated daemon thread; submit/run coroutines onto it."""
    def __init__(self, name: str = "QubxConnectorLoop"):
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever, daemon=True, name=name)
        self._thread.start()
    @property
    def loop(self) -> asyncio.AbstractEventLoop: return self._loop
    def submit(self, coro) -> concurrent.futures.Future: return asyncio.run_coroutine_threadsafe(coro, self._loop)
    def run_sync(self, coro, *, timeout: float | None = None): return run_sync(self._loop, coro, timeout=timeout)
    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop); self._thread.join()
```

- `BackgroundEventLoop` is the single thing that **owns** a loop+thread (start/stop) — for REPL / notebook /
  conformance construction, and (follow-up) for external clients (HL/lighter) that own their loop. It replaces
  the raw `(loop, thread)` tuple helper idea.
- `run_sync(loop, coro)` is the blocking submit-and-wait that every connector currently hand-rolls as
  `_run_sync` — now one implementation, with exception propagation and a **reentrancy guard** (the deadlock
  `AsyncThreadLoop` left exposed).

**Retire `AsyncThreadLoop`** (`utils/misc.py:451`): it is a stateless one-line wrapper over
`run_coroutine_threadsafe` that holds no state and owns no lifecycle, yet its call sites/comments imply it
does; it omits `run_sync`, ships a mis-typed `run_in_executor`, and exposes the reentrancy deadlock. Replace:
- Call sites that submit to an **externally-owned** loop (ccxt `_loop` properties — `connector.py:211`,
  `data.py:105`, `connection_manager.py:85`, `subscription_orchestrator.py:51`, `warmup_service.py:49`;
  `tardis/data.py`; `data/storages/ccxt.py`) → call `asyncio.run_coroutine_threadsafe(coro, loop)` /
  `run_sync(loop, coro)` directly. They don't own the loop, so they shouldn't hold a wrapper that implies they do.
- Loop **owners** → `BackgroundEventLoop`.

Delete the `loop is None` self-spawn branch in `ccxt/factory.py:55-62`; `loop` becomes required in
`get_ccxt_exchange` / `get_ccxt_exchange_manager`. The runner's shared loop and `_cleanup_event_loop`
(`runner.py:274-276`, `:109`) can adopt `BackgroundEventLoop` (optional, behavior-preserving).

## Config & release interaction

The strategy YAML `plugins.modules` and `qubx release` are **kept as-is** — transparent to strategy
authors.

- `plugins.modules` does double duty today: runtime registration (`plugins/loader.py:45` imports modules
  to fire decorators) and release bundling (`release.py:1664` `_get_plugin_deps` maps each module → a
  package in `[project.optional-dependencies]` and bundles that wheel).
- After this change, for **connectors** the runtime-registration role moves to entry points; the
  `plugins.modules` role narrows to **bundling declaration** (and a now-harmless early import). Release
  bundling is unchanged: the bundled connector wheel is installed via the generated `pyproject.toml` +
  `uv.lock` + `uv sync`, so its `.dist-info` is present and entry-point discovery works in the deployed
  bot. No `release.py` change.
- Caveat (conceptual only): `plugins.modules` becomes mildly overloaded — a real import-to-activate for
  `@storage`/`@reader`, a bundle declaration for connectors. No behavior breaks; registration is
  idempotent (registry keyed by name); no double-registration since the connector decorators are gone.

## Migration mechanics (file-by-file)

- **`connectors/plugin.py` (new):** `ExchangePlugin`, `BuildContext`, `ConnectorBuildContext`.
- **`registry.py`:** delete `register_connector`/`register_data_provider`/`register_rate_limit_config` +
  the `connector`/`data_provider`/`rate_limit_config` decorators + their dicts; add `register` /
  `get_plugin` / the three convenience methods. Extend the existing tombstone `__getattr__`
  (`registry.py:184-194`) so stale `@connector`/`@data_provider`/`@rate_limit_config` imports raise a
  clear "use entry points + ExchangePlugin" error.
- **`plugins/loader.py`:** add `PluginLoader` (entry-point discovery, lazy load). `load_plugins` unchanged.
- **`runner.py`:** the two-phase build above (`:541-578`, `:760-771`); `rate_limiter` into the context
  instead of `exchange_config.params`; drop the `read_only=` connector arg; still call
  `load_plugins(stg_config.plugins)` for storage/readers.
- **`connectors/ccxt/plugin.py` (new):** `CcxtPlugin(ExchangePlugin)`, `PLUGIN = CcxtPlugin()`.
  `create_data_provider`/`create_connector` build today's `CcxtDataProvider` / connector and attach
  `ctx.rate_limiter` to **both** exchange managers (closing the gap); `rate_limits` →
  `create_ccxt_rate_limit_config`. Delete `@connector`/`@data_provider`/`@rate_limit_config`, the
  `connectors/__init__.py` import-for-side-effect, and the five `read_only` guards + ctor param.
- **`connectors/tardis/plugin.py` (new):** `TardisPlugin(ExchangePlugin)` overriding only
  `create_data_provider`; `PLUGIN = TardisPlugin()`.
- **`core/mixins/trading.py`:** `read_only` gate at `:115`/`:327`/`:387`; `self._read_only` from
  `config.live.read_only`.
- **`utils/misc.py`:** add `run_sync(loop, coro, timeout)` + `BackgroundEventLoop`; **delete `AsyncThreadLoop`**.
- **`AsyncThreadLoop` call sites:** migrate the ccxt stack (`connector.py:211`, `data.py:105`,
  `connection_manager.py:85`, `subscription_orchestrator.py:51`, `warmup_service.py:49`), `tardis/data.py`,
  and `data/storages/ccxt.py` to `run_coroutine_threadsafe` / `run_sync`; any owned-loop usage →
  `BackgroundEventLoop`.
- **`ccxt/factory.py`:** `loop` required; delete `loop is None` self-spawn branch (`:55-62`). Optionally route
  the runner's shared loop (`runner.py:274-276`) + `_cleanup_event_loop` (`:109`) through `BackgroundEventLoop`.
- **`pyproject.toml`:** the `[project.entry-points."qubx.exchange_plugins"]` block.
- **`RateLimitManager` (`manager.py`):** unchanged.

## Testing strategy

**Unit (no entry points):** contexts (required `loop`/`rate_limiter`; `ConnectorBuildContext` is-a
`BuildContext`); `ExchangePlugin` ABC defaults via a `FakePlugin`; registry (`get_plugin`,
`get_connector`/`get_data_provider`/`get_rate_limit_config`, `None`→clear error e.g. `get_connector("tardis")`);
`PluginLoader` (`available()` without import, lazy `load`, unknown→None, non-plugin→TypeError, name assert,
missing-extra wrap); `run_sync` (result / timeout / exception propagation / reentrancy-guard raise) and
`BackgroundEventLoop` (submit / run_sync / stop); trading-mixin `read_only` gate.

**Integration:** runner two-phase build for ccxt (mocked creds); **shared-limiter identity** — the same
`ExchangeRateLimiter` reaches both the DP and connector exchange managers; real entry-point discovery of
Qubx's own `ccxt`/`tardis` (installed editable in CI); **lazy-load guard** —
`qubx.connectors.ccxt.plugin` not in `sys.modules` until `get_plugin("ccxt")`.

**Migration / regression:** rewrite decorator-era registry tests to the plugin model; replace the five
connector `read_only` tests (`test_ccxt_connector_writes.py`) with trading-mixin gate tests; keep the ccxt
connector/data-provider/storage suites green after the `AsyncThreadLoop` migration; existing runner/e2e
suites pass (`just test`); stale `@connector` import → tombstone error test.

## Sequencing — two Qubx PRs

The release flow is untouched (we keep `plugins.modules`), so there is no release PR.

- **PR 1 — prep (behavior-preserving, green standalone):** move `read_only` to the trading-mixin gate
  (delete connector guards + ctor param, relocate tests); `loop` required; add `run_sync` +
  `BackgroundEventLoop` and **retire `AsyncThreadLoop`** (migrate all in-tree call sites); delete the
  `loop is None` self-spawn branch. (The `AsyncThreadLoop` retirement is a sizable mechanical migration —
  can be split into its own commit/PR if review gets large.)
- **PR 2 — the migration (atomic):** `ExchangePlugin`/contexts/`PluginLoader`; new `ConnectorRegistry`
  API; runner two-phase + `rate_limiter`-in-context; `ccxt → CcxtPlugin` (attach limiter to connector) and
  `tardis → TardisPlugin`; `pyproject` entry points; delete the three decorators + import-side-effect;
  extend the tombstone; migrate tests.

An optional shim variant of PR 2 (old decorators + new API coexisting for one cycle) is possible but not
recommended: external plugins are already broken by `account-mgmt-redesign` and ccxt is the only in-tree
connector, so an atomic flip is cleaner.

## Follow-ups (separate, after this lands)

- Update `exchanges/docs/hyperliquid-iconnector-migration.md` and `new_exchange.md` to the
  `ExchangePlugin` + entry-point shape.
- Author the Hyperliquid connector (and migrate `qubx-lighter`) against this final contract in the
  `exchanges` repo.
- Migrate `qubx-hyperliquid` / `qubx-lighter` off `AsyncThreadLoop` (and their bespoke `_run_sync`) to
  `BackgroundEventLoop` / `run_sync` while authoring them against this contract.

## Non-goals

- No change to `IConnector` / `IDataProvider` interfaces.
- No change to the config schema (`connector: <name>` stays a string key; `read_only` stays).
- No change to the market-data tuple path or to `@storage`/`@reader` plugins.

## Decisions log (resolved during brainstorming)

- **Scope:** full end state (typed contexts + `ExchangePlugin` + entry points).
- **Plugin shape:** ABC with `None`-returning defaults.
- **Build context:** two types (`BuildContext` + `ConnectorBuildContext`).
- **Rate limits:** plugin declares config; `RateLimitManager` owns the limiter; context shares it.
- **`read_only`:** kept, enforced as a single trading-mixin gate (not removed; not per-connector).
- **`loop`:** required; helper for non-runner construction.
- **Discovery:** entry points replace decorators; built-ins via self-declared entry points; name handling
  via entry-point key + `plugin.name` assert.
- **Config/release:** keep `plugins.modules`; release flow unchanged.
- **`AsyncThreadLoop`:** retired in favor of `BackgroundEventLoop` (owns loop+thread) + `run_sync(loop, coro)`
  (cross-thread blocking submit with reentrancy guard); non-owning call sites use `run_coroutine_threadsafe` /
  `run_sync` directly.
