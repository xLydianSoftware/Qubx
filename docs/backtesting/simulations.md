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

#### Instruments from a folder

Instruments that the configured instrument lookup does not have (for example US stocks read from Yahoo) can be taken from a folder of instrument JSON files for this one run:

```bash
> qubx simulate test1.yml --lookup file:configs/instruments -o /backtests/tests
```

- The folder path may be relative; it is resolved against the working directory.
- The option takes precedence over `~/.qubx/config.json` and the `QUBX_INSTRUMENT_LOOKUP__*` environment variables, and variations of the run that execute in parallel use the same folder.
- `qubx validate test1.yml --lookup file:<folder>` accepts the same option; there it only checks that the folder exists.

#### Data read errors

If reading market data fails during a simulation (for example a network error from a remote store), the read is reopened from where it started, up to 4 attempts with waits of 1, 2 and 3 seconds. Rows already received are skipped, so each row arrives once. If the last attempt fails, the simulation stops with that error instead of continuing without data for some symbols.

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