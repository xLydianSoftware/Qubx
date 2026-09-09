# Scheduling

## `on_fit()` schedule

Now it can use custom format

“M @ 23:59:55” - first day of every month at 23:59:55

“Q @ 15:00” - every quarter start at 15:00

“5D @ 10:00” - every 5 days at 10:00

“MON @ 9:30” - every Monday at 9:30 (TUE, WED, …..)

## On-demand callbacks: `register_handler` / `post_event`

`ctx.register_handler(name, method)` registers a strategy-thread callback under `name` with no cron armed — the on-demand counterpart of `schedule()`. `ctx.post_event(name, payload=None)` then wakes it, optionally handing it an object. Register the handler in `on_start`, before starting any thread that will call `post_event`.

- **Names.** `register_handler` raises `ValueError` on an empty name, a duplicate (in either registry — a name registered as a handler or minted by `schedule()`/`delay()` is taken), or any name that collides with a built-in event — either a non-data-type handler (`fit`, `event`, `time`, `error`, …) or *anything* the framework parses as a data type (`trade`, `ohlc(1h)`, `funding_rate`, `open_interest`, …). Use a namespaced name such as `my.wakeup`.
- **Payload.** The handler runs as `method(ctx, payload)` — the object passed to `post_event`, or `None` when it was called with just a name. (Callbacks registered with `schedule()`/`delay()` keep their `method(ctx)` shape; the two registries are separate.) The payload is **handed over, not copied**: after posting it the producing thread must not mutate it, and the handler must treat it as read-only. A consumer that can re-read its data from the source — a Redis stream, a queue you drain yourself — should post `None` and re-read on the strategy thread instead of shipping a snapshot; that is what quantkit's aggregator does.
- **Live vs simulation.** Live, `post_event` enqueues onto the strategy's data channel and returns immediately — thread-safe, callable from any thread, never blocking — and the handler runs on the strategy thread, exactly like a scheduled callback. In simulation the channel dispatches **synchronously on the calling thread**, so there `post_event` must only be called from the strategy thread, and a posting thread must not be started at all — it would drive the pipeline concurrently with the simulation loop. Guard any such thread with `if ctx.is_live:`.
- **Errors and shutdown.** `post_event` raises `ValueError` if no handler is registered under `name`, and `RuntimeError` if there is no data provider. An exception raised *inside* the handler is logged, not propagated (same as a scheduled method). Live, once the context has stopped `post_event` is a silent no-op rather than an error.
- **Boot.** Like scheduled custom methods, a registered handler runs as soon as its event is dequeued — it can fire before boot completes and before `on_start` returns; signals it emits are buffered until the pipeline drains them.
- **Inside a threaded `on_fit`.** `register_handler` validates the name eagerly and defers the registration to the fit commit (like `schedule()`); `post_event` passes straight through.

```python
def on_start(self, ctx):
    ctx.register_handler("my.wakeup", self._on_wakeup)
    if ctx.is_live:  # in simulation the channel dispatches inline — never post from a thread
        threading.Thread(target=self._watch, args=(ctx,), daemon=True).start()

def _watch(self, ctx):
    while True:
        update = external_blocking_wait()          # e.g. {"symbol": "BTCUSDT", "score": 0.7}
        # handed over: don't touch `update` after this line — build a fresh one next round
        ctx.post_event("my.wakeup", update)        # thread-safe; runs on the strategy thread

def _on_wakeup(self, ctx, payload):
    if payload is None:                            # posted with no payload (or by someone else)
        return
    self._scores[payload["symbol"]] = payload["score"]   # read-only use of the payload
```

## `set_universe()`

Accepts additional parameter if_has_position_then  

It describe what to do with assets requested to remove when they have open position.

It can have 3 possible values:

- “close” (default) - close position immediatelly and remove (unsubscribe) instrument from strategy
- “wait_for_close” - keep instrument and it’s position until it’s closed from strategy (or risk management), then remove instrument from strategy
- “wait_for_change” - keep instrument and position until strategy would try to change it - then close position and remove instrument

## `simulate()`

Start / stop can be also expressed in form: 

 `start="2023-06-01", stop="+10d"` - 10 days from start day

 `start="2023-06-01", stop="-5d"`  - start 5 days before start day

data parameter can be configurted to accept different data sources:

```python
l1 = loader(....)
custom_reader = ....
r = simulate({'CrossOver MA': TestB(...)}, 
    {                                     
      'ohlc(1h)': l1,
      'trade': l1,
      'quote': l1,
      '<r>MY_DATA</r>': custom_reader
    },
   1000, ['BINANCE.UM:BTCUSDT'], "vip0_usdt", "2023-07-10", "2023-07-11", debug="DEBUG",
)
```