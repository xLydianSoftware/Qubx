# Paper Trading

<!-- 
This page should include:
- Introduction to paper trading
- Setting up paper trading
- Differences from backtesting
- Differences from live trading
- Analyzing paper trading results
- Transitioning to live trading
-->

## Introduction to Paper Trading

Paper trading (also known as simulated trading) allows you to test your trading strategies in real-time market conditions without risking real money. It serves as a bridge between backtesting and live trading, helping you validate your strategy's performance in current market conditions.

## Why Use Paper Trading?

## Starting Capital and Commissions

A `--paper` run takes each exchange's starting capital and commission tier from a `paper` block under that
exchange in the `live:` section. The field names are the ones `simulation:` uses.

```yaml
live:
  exchanges:
    BINANCE.UM:
      connector: ccxt
      universe: [BTCUSDT, ETHUSDT]
      base_currency: USDT        # optional, also wins over the accounts file
      paper:
        capital: 10000           # in the exchange's base currency
        commissions: vip0_usdt   # a tier from qubx/resources/crypto-fees.ini, or maker=0.02,taker=0.05 (percent)
```

Each field is taken from the first place that sets it:

1. `live.exchanges.<EXCHANGE>.paper` in the strategy config
2. the accounts file (`~/.qubx/accounts.toml`, `<strategy dir>/accounts.toml`, `QUBX_ACCOUNT_FILE`):
   the exchange's `[[accounts]]` entry, then its `[[defaults]]` entry (`initial_capital`, `commissions`)
3. the built-in default: capital 100,000, no commissions (zero fees)

**The config wins over the accounts file when both set a field.** `capital` and `commissions` fall back
separately, so a config can set one and leave the other to the accounts file.

A live run ignores the `paper` block: its balances come from the venue and its commission tier from the
accounts file. With `--restore`, a restored balance in the base currency replaces the starting capital.
