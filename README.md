# Qubx - Quantitative Trading Framework

[![CI](https://github.com/xLydianSoftware/Qubx/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/xLydianSoftware/Qubx/actions/workflows/ci.yml)

```
⠀⠀⡰⡖⠒⠒⢒⢦⠀⠀
⠀⢠⠃⠈⢆⣀⣎⣀⣱⡀  QUBX | Quantitative Backtesting Environment
⠀⢳⠒⠒⡞⠚⡄⠀⡰⠁         (c) 2024, by Dmytro Mariienko
⠀⠀⠱⣜⣀⣀⣈⣦⠃⠀⠀⠀
```

Qubx is a next-generation quantitative trading framework designed for efficient backtesting and live trading. Built with Python, it offers a robust environment for developing, testing, and deploying trading strategies.

## Quick Start

### 1. Install Dependencies
```bash
poetry install
```

### 2. Create a Strategy
```bash
# Create a simple strategy template (default)
poetry run qubx init

# Or specify a name and symbols
poetry run qubx init --name my_strategy --symbols BTCUSDT,ETHUSDT
```

### 3. Run Your Strategy
```bash
cd my_strategy

# Run in paper trading mode
poetry run qubx run config.yml --paper

# Or run in Jupyter mode for interactive development
./jpaper.sh
```

### Available Templates
```bash
# List available strategy templates
poetry run qubx init --list-templates

# Create strategy with full project structure and MACD example
poetry run qubx init --template project --name my_project
```

### Strategy Development Workflow
1. **Initialize**: `poetry run qubx init` - Create strategy from template
2. **Develop**: Edit `strategy.py` to implement your trading logic
3. **Test**: `poetry run qubx run config.yml --paper` - Run in paper mode
4. **Debug**: `./jpaper.sh` - Use Jupyter for interactive development
5. **Deploy**: Configure for live trading when ready

## Features

- 🚀 High-performance backtesting engine
- 🔄 Live trading support
- 📊 Advanced data analysis tools
- 📈 Integration with multiple exchanges
- 🛠 Comprehensive strategy development toolkit
- 🔍 Detailed performance analytics

## Documentation

For detailed documentation, visit [Qubx Documentation](https://xlydiansoftware.github.io/Qubx/en/latest/)

## Prerequisites

To build and run Qubx, you need:

- Python 3.10 or higher
- C/C++ compiler for Cython compilation
- Poetry for dependency management

## Installation

### Using pip

```bash
pip install qubx
```

### Development Setup

1. Clone the repository
2. Install dependencies using Poetry:

```bash
poetry install
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
