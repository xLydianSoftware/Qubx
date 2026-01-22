# Qubx - Quantitative Trading Framework

[![CI](https://github.com/xLydianSoftware/Qubx/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/xLydianSoftware/Qubx/actions/workflows/ci.yml)

```
⠀⠀⡰⡖⠒⠒⢒⢦⠀⠀
⠀⢠⠃⠈⢆⣀⣎⣀⣱⡀  QUBX | Quantitative Backtesting Environment
⠀⢳⠒⠒⡞⠚⡄⠀⡰⠁         (c) 2026, by xLydian
⠀⠀⠱⣜⣀⣀⣈⣦⠃⠀⠀⠀
```

Qubx is a next-generation quantitative trading framework designed for efficient backtesting and live trading. Built with Python, it offers a robust environment for developing, testing, and deploying trading strategies.

**Qubx is under active development.** We are continuously improving the framework and will update our documentation in the coming days/weeks. This will include comprehensive end-to-end examples for running simulations and live trading.

### Supported Data Types

Qubx supports a wide range of market data:
- OHLC (candlestick data)
- L2 Orderbook
- Liquidations
- Funding rates
- And more...

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Create a Strategy

```bash
# Create a simple strategy template (default)
uv run qubx init

# Or specify a name and symbols
uv run qubx init --name my_strategy --symbols BTCUSDT,ETHUSDT
```

### 3. Run Your Strategy

```bash
cd my_strategy

# Run in paper trading mode
uv run qubx run config.yml --paper

# Or run in Jupyter mode for interactive development
./jpaper.sh
```

### Available Templates

```bash
# List available strategy templates
uv run qubx init --list-templates

# Create strategy with full project structure and MACD example
uv run qubx init --template project --name my_project
```

### Strategy Development Workflow

1. **Initialize**: `uv run qubx init` - Create strategy from template
2. **Develop**: Edit `strategy.py` to implement your trading logic
3. **Test**: `uv run qubx run config.yml --paper` - Run in paper mode
4. **Debug**: `./jpaper.sh` - Use Jupyter for interactive development
5. **Deploy**: Configure for live trading when ready

## Features

- High-performance backtesting engine
- Live trading support
- Advanced data analysis tools
- Integration with multiple exchanges
- Comprehensive strategy development toolkit
- Detailed performance analytics

## Documentation

For detailed documentation, visit [Qubx Documentation](https://xlydiansoftware.github.io/Qubx/en/latest/)

## Prerequisites

To build and run Qubx, you need:

- Python 3.11 or higher
- C/C++ compiler for Cython compilation
- uv for dependency management

## Installation

### Using pip

```bash
pip install qubx
```

### Development Setup

1. Clone the repository
2. Install dependencies using uv:

```bash
uv sync --all-extras
```

Example trading strategies can be found in the `examples/` directory.

## CLI Usage

Qubx comes with a command-line interface that provides several useful commands:

```bash
qubx --help  # Show all available commands
```

Available commands:

- `qubx init` - Create a new strategy from template
- `qubx run` - Start a strategy with given configuration
- `qubx simulate` - Run strategy simulation
- `qubx ls` - List all strategies in a directory
- `qubx release` - Package a strategy into a zip file
- `qubx deploy` - Deploy a strategy from a zip file
- `qubx browse` - Browse backtest results using interactive TUI

## Development

### Running Tests

Run the test suite:

```bash
just test
```

### Additional Commands

- Check code style: `just style-check`
- Build package: `just build`
- Run verbose tests: `just test-verbose`

## In Production

Qubx powers the [AllegedAlpha](https://app.lighter.xyz/public-pools/281474976625478) public pool on Lighter. Public pools allow users to deposit funds from their blockchain wallet into a smart contract. The pool operator manages the trading strategy, and a performance fee is taken from profits (X: [@allegedalpha](https://x.com/allegedalpha)).

## About xLydian

Qubx is developed by [xLydian](https://xlydian.com/).

- Website: [xlydian.com](https://xlydian.com/)
- X: [@xLydian_xyz](https://x.com/xLydian_xyz)
- Contact: [info@xlydian.com](mailto:info@xlydian.com)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
