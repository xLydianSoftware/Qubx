# Scheduling

## `on_fit()` schedule

Now it can use custom format

“M @ 23:59:55” - first day of every month at 23:59:55

“Q @ 15:00” - every quarter start at 15:00

“5D @ 10:00” - every 5 days at 10:00

“MON @ 9:30” - every Monday at 9:30 (TUE, WED, …..)

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