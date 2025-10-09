# Lighter WebSocket Test Data

This directory contains captured WebSocket messages from Lighter exchange for testing purposes.

## Directory Structure

```
test_data/
├── README.md                      # This file
├── samples/                       # Captured live samples
│   ├── capture_summary.json      # Capture session summary
│   ├── orderbook_samples.json    # All orderbook messages
│   ├── trades_samples.json       # All trade messages
│   ├── market_stats_samples.json # All market stats messages
│   ├── system_samples.json       # System messages (connected, subscribed, etc.)
│   ├── orderbook/                # Individual orderbook samples
│   │   ├── sample_01.json
│   │   ├── sample_02.json
│   │   └── ...
│   ├── trades/                   # Individual trade samples
│   │   └── ...
│   └── market_stats/             # Individual market stats samples
│       └── ...
└── mock/                         # Mock/synthetic data (future)
```

## Captured Message Types

### Order Book Messages
- **Snapshot**: Initial full orderbook state when subscribing
- **Updates**: Incremental updates (adds, removes, changes)
- **Markets**: BTC-USDC (market_id=0), ETH-USDC (market_id=1)

Format:
```json
{
  "type": "subscribed/order_book" or "update/order_book",
  "channel": "order_book/0",
  "data": {
    "bids": [[price, size], ...],
    "asks": [[price, size], ...],
    "market_id": 0,
    "timestamp": 1696800000000
  }
}
```

### Trade Messages
- **Real-time trades**: Executed trades on the market

Format:
```json
{
  "type": "trade",
  "channel": "trade/0",
  "data": {
    "market_id": 0,
    "price": "50000.00",
    "size": "0.1",
    "side": "B",  // "B" = Buy, "S" = Sell
    "timestamp": 1696800000000
  }
}
```

### Market Stats Messages
- **Statistics**: 24h volume, high, low, etc.

Format:
```json
{
  "type": "market_stats",
  "channel": "market_stats/all",
  "data": {
    "market_id": 0,
    "high_24h": "51000.00",
    "low_24h": "49000.00",
    "volume_24h": "1000.5",
    ...
  }
}
```

### System Messages
- **Connected**: WebSocket connection established
- **Subscribed**: Subscription confirmation

## Capturing New Samples

To capture fresh samples:

```bash
# Run capture script (captures for 30 seconds)
poetry run python scripts/capture_lighter_samples.py

# Samples will be saved to this directory
```

## Using Samples in Tests

```python
import json
from pathlib import Path

# Load orderbook samples
samples_dir = Path(__file__).parent / "test_data/samples"
with open(samples_dir / "orderbook_samples.json") as f:
    orderbook_samples = json.load(f)

# Use in tests
def test_orderbook_conversion():
    sample = orderbook_samples[0]["data"]
    converted = convert_lighter_orderbook(sample)
    assert converted is not None
```

## Notes

- Samples contain **real data** from Lighter mainnet
- Captured data is **read-only** (market data, no private account info)
- Use samples for **unit tests** and **validation**
- Recapture periodically if Lighter message format changes
