# Qubx

[![CI](https://github.com/xLydianSoftware/Qubx/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/xLydianSoftware/Qubx/actions/workflows/ci.yml)

## Next generation of Qube quantitative backtesting framework (QUBX)
```          
⠀⠀⡰⡖⠒⠒⢒⢦⠀⠀
⠀⢠⠃⠈⢆⣀⣎⣀⣱⡀  QUBX | Quantitative Backtesting Environment
⠀⢳⠒⠒⡞⠚⡄⠀⡰⠁         (c) 2024, by Dmytro Mariienko
⠀⠀⠱⣜⣀⣀⣈⣦⠃⠀⠀⠀
```                                          

## Documentation

See [Qubx Documentation](https://xlydiansoftware.github.io/Qubx/en/latest/)

## Installation
> pip install qubx

## How to run live trading (Only Binance spot tested)
1. cd experiments/
2. Edit strategy config file (zero_test.yaml). Testing strategy is just doing flip / flop trading once per minute (trading_allowed should be set for trading)
3. Modify accounts config file under ./configs/.env and provide your API binance credentials (see example in example-accounts.cfg):
```
[binance-mde]
apiKey = ...
secret = ...
base_currency = USDT
```
4. Run in console (-j key if want to run under jupyter console)

```
> python ..\src\qubx\utils\runner.py configs\zero_test.yaml -a configs\.env -j 
```

## Running tests
We use `pytest` for running tests. For running unit tests execute
```
just test
```

We also have several integration tests (marked with `@pytest.mark.integration`), which mainly make sure that the exchange connectors function properly. We test them on the corresponding testnets, so you will need to generate api credentials for the exchange testnets that you want to verify.

Once you have the testnet credentials store them in an `.env.integration` file in the root of the Qubx directory
```
# BINANCE SPOT test credentials
BINANCE_SPOT_API_KEY=...
BINANCE_SPOT_SECRET=...

# BINANCE FUTURES test credentials
BINANCE_FUTURES_API_KEY=...
BINANCE_FUTURES_SECRET=...
```

To run the tests simply call
```
just test-integration
```