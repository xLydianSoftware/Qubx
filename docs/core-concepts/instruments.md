# Instruments and Lookups

An `Instrument` is one tradable listing on one venue. Strategies get them from the global `lookup`:

```python
from qubx.core.lookups import lookup

btc = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
pepe = lookup.find_instruments("BINANCE.UM", base="PEPE")  # also matches 1000PEPEUSDT
```

## The `Instrument`

Identity is `(exchange, market_type, symbol)`: equality, hashing and `str(i)` (`BINANCE.UM:SWAP:BTCUSDT`) use only those three fields, so metadata can change without changing which instrument it is.

| Field | Meaning |
|---|---|
| `symbol`, `exchange`, `market_type`, `exchange_symbol` | Identity and the venue's own symbol |
| `base`, `quote`, `settle`, `inverse` | |
| `tick_size`, `lot_size`, `min_size`, `min_notional` | Venue trading rules |
| `contract_size` | Quantity per contract: base units for linear contracts, quote units for inverse ones. It is the product of every venue size and multiplier field (OKX `ctVal × ctMult`). `quantity_multiplier` is an alias. |
| `listing_id` | The instrument service's listing id; stable through renames |
| `underlying` | `Underlying(kind: AssetKind, code)`, what the listing is a contract on: `CRYPTO:PEPE`, `EQUITY:NVDA.XNAS`, `COMMODITY:XAU` |
| `calendar` | When this listing trades: `24/7`, a MIC such as `XNAS`, or `None` |
| `expiry`, `strike`, `option_right` | Futures and options only |
| `listed_at`, `delisted_at` | Lifecycle; a future `delisted_at` is a scheduled delisting |
| `margin_tradable` | Spot pairs that can be traded on margin |
| `venue_attributes` | Read-only mapping of venue labels (Binance `contractType`, spot permission groups, …). Filter on them; never branch on them |

Fields after `min_notional` are keyword-only.

`i.asset` is the venue coin with any multiplier prefix or suffix stripped from `base`: `1000PEPE`, `10000SATS` and `SHIB1000` answer `PEPE`, `SATS` and `SHIB`, and the bStock `NVDAB` answers `NVDAB`. It depends only on the listing, so an instrument a connector builds itself and the same instrument from the lookup always agree. Economic exposure is `i.underlying` (it can be `None`); `i.exposure_code` is `i.underlying.code`, or `i.asset` when there is no underlying.

Instruments carry no margin rates. Live positions use the margin the venue reports. In simulation the maintenance margin is `AccountManagerConfig.maint_margin_rate` (default 5% of notional) and the initial margin is 0.

## Lookup types

The instrument lookup is configured by `instrument_lookup` in the Qubx settings (`~/.qubx/config.json` or `QUBX_INSTRUMENT_LOOKUP__*` env vars).

| `type` | Source |
|---|---|
| `file` (default) | JSON files in `~/.qubx/instruments`, seeded from the packaged files on first use |
| `service` | The platform instrument service |

### `service`

| Setting | Env | Meaning |
|---|---|---|
| `url` | `QUBX_INSTRUMENT_LOOKUP__URL` | Service base including its route prefix: `http://control-api.platform.svc/internal/instrument-service` in-cluster, `https://api-dev.xlydian.com/instrument-service` off-cluster |
| `token` | `QUBX_INSTRUMENT_LOOKUP__TOKEN` | `xl` API token, sent as `Authorization: Bearer …`; not needed in-cluster |
| `reload_interval` | `QUBX_INSTRUMENT_LOOKUP__RELOAD_INTERVAL` | How often to re-read the snapshot, e.g. `1h`; unset means never |
| `exchanges` | `QUBX_INSTRUMENT_LOOKUP__EXCHANGES` | Load only these exchanges, e.g. `BINANCE.UM,OKX.F` (or a JSON list); unset loads all |

```json
{"instrument_lookup": {"type": "service", "url": "https://api-dev.xlydian.com/instrument-service", "token": "…", "reload_interval": "1h"}}
```

- The lookup reads `GET {url}/snapshot` on startup and builds every instrument from its listing's current version. It tries three times; if all fail, startup fails. There is no fallback to another lookup.
- With a `reload_interval`, a background thread re-reads the snapshot with `If-None-Match` and swaps the result in; lookups never wait on the network. A `304` keeps the current copy. A failed refresh logs a warning, keeps the copy and is retried after a full interval.
- `find_symbol(exchange, old_symbol)` resolves a renamed listing's former symbols to its current `Instrument`. A listing whose current symbol is the same string wins over an alias.
- Delisted listings are included, with `delisted_at` set.

The `mongo` lookup was removed in Qubx 4.0.

## Migrating from 3.x

| 3.x | Now |
|---|---|
| `onboard_date` | `listed_at` |
| `delist_date` | `delisted_at` |
| `delivery_date` | `expiry` |
| `contract_multiplier` | folded into `contract_size`; `quantity_multiplier` is `contract_size` |
| `initial_margin`, `maint_margin`, `liquidation_fee` | removed; see the margin note above |
| positional args after `min_notional` | keyword-only |
| `instrument_lookup.type = "mongo"`, `mongo_url` | `type = "service"` with `url` (and `token` off-cluster) |

Instrument files cached in `~/.qubx/instruments` by 3.x still load: the old field names are mapped when read.
