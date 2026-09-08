# Scheduling

## `on_fit()` schedule

Now it can use custom format

“M @ 23:59:55” - first day of every month at 23:59:55

“Q @ 15:00” - every quarter start at 15:00

“5D @ 10:00” - every 5 days at 10:00

“MON @ 9:30” - every Monday at 9:30 (TUE, WED, …..)

## On-demand callbacks: `register_handler` / `post_event`

`ctx.register_handler(name, method)` registers a strategy-thread callback under `name`, with no cron armed — the on-demand counterpart of `schedule()` (raises `ValueError` on an empty or duplicate name, or one that shadows a built-in data-type handler such as `trade` or `ohlc(1h)`). `ctx.post_event(name)` wakes it: it takes no payload, the handler always runs as `method(ctx)`, and it raises `ValueError` if no handler is registered under `name`. Live, `post_event` enqueues onto the strategy's data channel and returns immediately — thread-safe, callable from any thread, never blocking — and the handler then runs on the strategy thread, just like a scheduled callback. In simulation the channel dispatches synchronously on the calling thread instead, so there `post_event` must only be called from the strategy thread. `post_event` raises `RuntimeError` if there's no data provider, and becomes a silent no-op once the context has stopped. Register the handler in `on_start` (or `on_init`), before starting any thread that will call `post_event`. (Inside a threaded `on_fit`, `register_handler` is deferred to the commit like `schedule()`; `post_event` passes straight through.)

```python
def on_start(self, ctx):
    ctx.register_handler("my.wakeup", self._on_wakeup)
    threading.Thread(target=self._watch, args=(ctx,), daemon=True).start()

def _watch(self, ctx):
    while True:
        external_blocking_wait()
        ctx.post_event("my.wakeup")   # thread-safe; _on_wakeup runs on the strategy thread

def _on_wakeup(self, ctx):
    ...
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