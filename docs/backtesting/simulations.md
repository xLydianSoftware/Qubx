# Simulations

### Running simulations (experiments) from yml config

It’s possible to make a yml file where we can describe simulation setup. Let’s put config into test1.yml

```yaml
strategy: test.model.Strategy1  # strategy class

description: 
  - Description of this experiment
  - May be multiline

parameters:            # what startegy parameters to use in this run
  timeframe: "4h"
  parameter1: 123

data:                  # what data should be used for simulation
  ohlc(4h): (use_as_aux_loader:=loader("BINANCE.UM", "4h", source="mqdb::nebula"))

simulation:
  instruments: ["BINANCE.UM:BTCUSDT"]
  capital: 100000.0
  commissions: "vip0_usdt"
  start: "2021-01-01"
  stop: "2025-01-21"
  aux_data: use_as_aux_loader  # here we could use loader defined in data section !
  debug: ERROR
```

Now we can run it using qubx cli:

```bash
> qubx simulate test1.yml -o /backtests/tests
```

Result will be stored into /backtests/tests folder.

### Variations (hyperparameter optimization)

It’s possible to run variations for some parameters (kind of optimization preparation). For that it’s enough to add  variate section:

```yaml
strategy: test.model.Strategy1  # strategy class

description: 
  - Description of this experiment
  - May be multiline

parameters:            # what startegy parameters to use in this run
  timeframe: "4h"
  parameter1: 123
  parameter1: 333

variate:
  parameter1: [10, 20, 30, 40, 50, 50, 60, 70, 80, 90, 100]
  parameter2: [10, 20, 30, 40, 50, 50, 60, 70, 80, 90, 100]
  with:  # here we can apply constraints on parameters
    parameter1, parameter2: parameter2 > parameter1
    parameter2: parameter2 <= 90

data:                  # what data should be used for simulation
  ohlc(4h): (use_as_aux_loader:=loader("BINANCE.UM", "4h", source="mqdb::nebula"))

simulation:
  instruments: ["BINANCE.UM:BTCUSDT"]
  capital: 100000.0
  commissions: "vip0_usdt"
  start: "2021-01-01"
  stop: "2025-01-21"
  debug: ERROR
```

Run this using same command

```bash
> qubx simulate test1.yml -o /backtests/tests
```