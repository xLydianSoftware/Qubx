# XLighter Exchange Connector Example

This example demonstrates how to use the XLighter connector for trading on Lighter exchange.

## Prerequisites

1. **Lighter Account**: You need a Lighter account with:
   - Ethereum address (public key)
   - Private key
   - Account index (obtained from Lighter platform)
   - API key index (optional, defaults to 0)

2. **Account Configuration**: Create an account configuration file (e.g., `accounts.toml`):

```toml
[[accounts]]
name = "xlighter-main"
exchange = "XLIGHTER"
api_key = "0xYourEthereumAddress"
secret = "0xYourPrivateKey"
account_index = 225671  # Your Lighter account index
api_key_index = 2       # Optional, defaults to 0
base_currency = "USDC"
initial_capital = 100000.0
commissions = "maker=0.0,taker=0.0"
```

## Configuration

The `config.yml` file shows how to configure a strategy for Lighter:

- **Connector**: Set `connector: xlighter` under the exchange configuration
- **Universe**: List the instruments you want to trade (e.g., `BTC-USDC`, `ETH-USDC`)
- **Read-only mode**: Set `read_only: true` to monitor without trading

## Running the Example

### Paper Trading (Simulated)

```bash
poetry run qubx run config.yml --paper
```

This mode:
- Uses simulated account with initial capital
- No real trades are placed
- WebSocket data is live from Lighter
- Perfect for testing your strategy

### Live Trading

```bash
poetry run qubx run config.yml --account-file accounts.toml
```

This mode:
- Connects to your real Lighter account
- Places actual orders on the exchange
- **Use with caution!** Start with small position sizes

### Jupyter Mode

For interactive development:

```bash
poetry run qubx run config.yml --paper --jupyter
```

## Available Instruments

The Lighter exchange supports perpetual swaps. Common instruments:

- `BTC-USDC` - Bitcoin perpetual
- `ETH-USDC` - Ethereum perpetual
- `SOL-USDC` - Solana perpetual
- `ARB-USDC` - Arbitrum perpetual

Check Lighter's platform for the full list of available markets.

## Features

The XLighter connector provides:

- **Real-time WebSocket data**: Orderbook, trades, quotes, market stats
- **Order management**: Create, cancel, modify orders
- **Account tracking**: Positions, balances, fills
- **Funding payments**: Automatic processing of funding rates
- **Reconnection handling**: Automatic reconnection with exponential backoff

## Example Strategy

See `xlighter_example/strategy.py` for a simple example strategy that demonstrates:

- Subscribing to market data
- Accessing positions and balances
- Placing and canceling orders
- Handling fills and account updates

## Troubleshooting

### Common Issues

1. **"Connector xlighter is not supported"**: Make sure you're using the latest version of Qubx with XLighter support

2. **Authentication errors**: Verify your API key, private key, and account_index are correct

3. **"Instrument not found"**: The instrument loader may need to refresh. Instruments are cached locally.

4. **WebSocket disconnections**: The connector automatically reconnects with exponential backoff. Check your network connection.

### Debug Mode

Enable debug logging:

```bash
poetry run qubx run config.yml --paper --log-level DEBUG
```

## Support

For issues specific to:
- **Qubx framework**: Check the main Qubx documentation
- **Lighter exchange**: Visit https://lighter.xyz or check their documentation
- **XLighter connector**: See the connector source code in `/src/qubx/connectors/xlighter/`
