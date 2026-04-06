# Bot Control Protocol

The control protocol allows external systems (CLI, web UI, LLM agents) to interact with a running Qubx strategy via HTTP. It replaces the old health-only server with a full control API while remaining backward compatible with Kubernetes health probes.

## Enabling the Control Server

Set the `QUBX_CONTROL_PORT` environment variable:

```bash
QUBX_CONTROL_PORT=8080 qubx run config.yml --paper
```

The legacy `QUBX_HEALTH_PORT` also works as a fallback.

The server starts immediately (before strategy warmup) so that `/health` is available for K8s liveness probes. Action endpoints become available once the strategy context is attached.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe (always 200) |
| `GET` | `/ready` | Readiness probe (200 after warmup, 503 during) |
| `GET` | `/actions` | List all available actions with parameter schemas |
| `POST` | `/actions/{name}` | Execute an action |

### Request Format

```bash
curl -X POST http://localhost:8080/actions/get_positions \
  -H 'Content-Type: application/json' \
  -d '{"params": {}}'
```

Actions with parameters:

```bash
curl -X POST http://localhost:8080/actions/get_quote \
  -H 'Content-Type: application/json' \
  -d '{"params": {"symbol": "BTCUSDT"}}'
```

### Response Format

```json
{
  "status": "ok",
  "data": { ... },
  "message": null
}
```

On error:

```json
{
  "detail": "Unknown symbol: XYZUSDT"
}
```

## Built-in Actions

Every bot gets these actions automatically, regardless of whether the strategy implements any custom actions.

### Discovery

| Action | Description | Params |
|--------|-------------|--------|
| `get_available_instruments` | All tradable instruments on an exchange | `exchange`, `quote?`, `market_type?` |
| `get_instrument_details` | Tick size, lot size, min notional for instruments | `symbols`, `exchange?` |
| `get_top_instruments` | Top N by turnover, market cap, or funding rate | `exchange`, `count?`, `sort_by?`, `period?`, `timeframe?`, `quote?`, `market_type?` |

**Market types**: `SPOT`, `SWAP` (perpetual futures), `FUTURE` (dated futures), `OPTION`, `MARGIN`

Example — top 10 by turnover:

```bash
curl -X POST http://localhost:8080/actions/get_top_instruments \
  -d '{"params": {"exchange": "BINANCE.UM", "count": 10, "sort_by": "turnover", "period": "3d", "timeframe": "1d"}}'
```

!!! note
    `get_top_instruments` requires auxiliary storage configured (e.g., `aux: storage: "qdb::quantlab"`). Turnover and funding use the exchange aux reader; market cap uses `COINGECKO:FUNDAMENTAL`.

### Universe

| Action | Description | Params | Dangerous |
|--------|-------------|--------|-----------|
| `get_universe` | Current trading universe | — | — |
| `add_instruments` | Add instruments | `symbols`, `exchange?` | — |
| `remove_instruments` | Remove instruments | `symbols`, `exchange?`, `if_has_position?` | Yes |
| `set_universe` | Replace entire universe | `symbols`, `exchange?`, `if_has_position?` | Yes |

Symbols can include the exchange prefix for multi-exchange setups: `"BINANCE.UM:BTCUSDT"`. Or pass the `exchange` parameter to apply to all symbols in the list.

### Diagnostics

| Action | Description | Params |
|--------|-------------|--------|
| `get_positions` | Positions with unrealized PnL | — |
| `get_balances` | Account balances per exchange | — |
| `get_orders` | Open orders | `symbol?` |
| `get_quote` | Latest bid/ask | `symbol` |
| `get_ohlc` | Recent OHLC bars | `symbol`, `timeframe?`, `length?` |
| `get_state` | Full state dump (multi-exchange) | — |
| `get_health` | Connectivity, queue size, latencies | — |
| `get_total_capital` | Total capital across exchanges | — |
| `get_leverages` | Per-instrument and portfolio leverage | — |
| `get_subscriptions` | Active data subscriptions | `symbol?` |

### Trading

| Action | Description | Params | Dangerous |
|--------|-------------|--------|-----------|
| `trade` | Place an order | `symbol`, `amount`, `price?`, `time_in_force?` | Yes |
| `set_target_position` | Set target position size | `symbol`, `target`, `price?` | Yes |
| `set_target_leverage` | Set target leverage | `symbol`, `leverage`, `price?` | Yes |
| `close_position` | Close one position | `symbol` | Yes |
| `close_positions` | Close all positions | — | Yes |
| `cancel_orders` | Cancel open orders | `symbol?` | — |
| `emit_signal` | Emit a trading signal | `symbol`, `signal_value`, `price?`, `group?` | Yes |

## Custom Actions with `@action`

Strategies can expose custom actions using the `@action` decorator:

```python
from qubx.control import IControllable, action
from qubx.control.types import ActionResult
from qubx.core.interfaces import IStrategy, IStrategyContext

class MyStrategy(IStrategy, IControllable):
    threshold: float = 0.7
    paused: bool = False

    @action(description="Get strategy parameters", category="diagnostics", read_only=True)
    def get_params(self, ctx: IStrategyContext):
        return {"threshold": self.threshold, "paused": self.paused}

    @action(description="Update confidence threshold", category="config")
    def set_threshold(self, ctx: IStrategyContext, value: float):
        if not 0.0 <= value <= 1.0:
            return ActionResult(status="error", error="Must be between 0 and 1")
        old = self.threshold
        self.threshold = value
        return ActionResult(status="ok", data={"old": old, "new": value})

    @action(description="Pause signal generation", category="config")
    def pause(self, ctx: IStrategyContext):
        self.paused = True

    @action(description="Resume signal generation", category="config")
    def resume(self, ctx: IStrategyContext):
        self.paused = False
```

### Decorator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `description` | `str` | required | Human-readable description (also used by LLMs) |
| `category` | `str` | `"custom"` | Grouping: `trading`, `universe`, `diagnostics`, `config`, `custom` |
| `read_only` | `bool` | `False` | If `True`, executes directly on the server thread (fast). If `False`, executes on the strategy thread via command queue (safe). |
| `dangerous` | `bool` | `False` | Hint for UIs to show confirmation prompts |
| `hidden` | `bool` | `False` | If `True`, not listed in `GET /actions` |

### Parameter Inference

Action parameters are automatically inferred from the method signature:

```python
@action(description="Multi-param example")
def my_action(self, ctx, name: str, count: int, items: list, flag: bool = True):
    ...
```

| Python type | Schema type |
|-------------|------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` | `"array"` |
| `dict` | `"object"` |

Parameters with defaults are marked as `required: false`.

### Return Values

Actions can return:

- **A dict** — automatically wrapped in `ActionResult(status="ok", data=...)`
- **An `ActionResult`** — returned as-is (useful for error handling)
- **`None`** — returns `ActionResult(status="ok", data=None)`

## Custom State with `@state`

The `@state` decorator marks methods whose return values are automatically included in the `get_state` response under the `"custom"` key:

```python
from qubx.control import state

class MyStrategy(IStrategy):

    @state(description="MACD indicator values")
    def macd_values(self, ctx: IStrategyContext) -> dict:
        return {i.symbol: self._indicators[i].value for i in ctx.instruments}

    @state(description="Current market regime")
    def regime(self, ctx: IStrategyContext) -> str:
        return "trending" if self.vol > 0.5 else "ranging"
```

The `get_state` response will include:

```json
{
  "total_capital": 100000.0,
  "exchanges": { ... },
  "custom": {
    "macd_values": {"BTCUSDT": -7.98, "ETHUSDT": -1.10},
    "regime": "trending"
  }
}
```

### Guidelines

- `@state` methods must be **fast and read-only** — they're called on every `get_state` request
- If a `@state` method throws an exception, the error is captured as `"error: <message>"` without failing the entire state response
- `@state` and `@action` are independent — use `@action` for callable operations, `@state` for automatic state inclusion

## The `get_state` Response

`get_state` returns a multi-exchange snapshot matching the format used by the platform's state persistence:

```json
{
  "timestamp": "2026-04-05T10:30:00",
  "total_capital": 100000.0,
  "exchanges": {
    "BINANCE.UM": {
      "base_currency": "USDT",
      "capital": { "total": 100000.0, "available": 96500.0 },
      "net_leverage": 0.333,
      "gross_leverage": 1.0,
      "open_positions": 3,
      "positions": {
        "BTCUSDT": {
          "quantity": 0.5,
          "avg_price": 67500.0,
          "market_price": 68000.0,
          "unrealized_pnl": 250.0,
          "market_value": 34000.0,
          "leverage": 0.34
        }
      },
      "orders": {
        "BTCUSDT": [
          { "id": "...", "type": "LIMIT", "side": "SELL", "quantity": 0.5, "price": 69000.0, "status": "OPEN" }
        ]
      },
      "balances": {
        "USDT": { "total": 100000.0, "free": 96500.0, "locked": 3500.0 }
      }
    }
  },
  "instruments": ["BINANCE.UM:SWAP:BTCUSDT", "BINANCE.UM:SWAP:ETHUSDT"],
  "is_warmup": false,
  "is_simulation": false,
  "custom": { ... }
}
```

## Thread Safety

Actions are categorized as **read-only** or **write**:

- **Read-only actions** (`read_only=True`) execute directly on the HTTP server thread. They can read strategy state without blocking the data processing loop. This is safe because Python's GIL prevents data corruption — you may see slightly stale values, but the process won't crash.

- **Write actions** (`read_only=False`) are enqueued on a command queue and executed on the strategy's data processing thread. This ensures they don't race with market data processing. The HTTP request blocks until the command completes (up to 30s timeout).

All built-in diagnostic actions (`get_*`) are read-only. Trading and universe actions are write actions.

## LLM Tool Compatibility

The `/actions` endpoint returns a JSON schema that maps directly to LLM function/tool definitions:

```python
# Convert actions to Anthropic tool format
response = requests.get("http://localhost:8080/actions")
actions = response.json()["actions"]

tools = []
for action in actions:
    properties = {}
    required = []
    for param in action["params"]:
        properties[param["name"]] = {
            "type": param["type"],
            "description": param["description"],
        }
        if param.get("required", True):
            required.append(param["name"])

    tools.append({
        "name": action["name"],
        "description": action["description"],
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    })
```
