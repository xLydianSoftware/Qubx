# Signal Export Tutorial

This tutorial explains how to configure and use exporters in Qubx to send trading signals, target positions, and position changes to external systems.

## Overview

Qubx provides a flexible exporting system that allows you to send trading data to various destinations such as:

- Redis Streams
- Slack
- Custom exporters

Exporters can be configured in your strategy YAML configuration file and can be customized to export different types of data with different formatting options.

## Configuring Exporters

Exporters are configured in the `exporters` section of your strategy YAML file. Here's an example configuration:

```yaml
exporters:
  - exporter: SlackExporter
    parameters:
      signals_webhook_url: env:SLACK_WEBHOOK_URL
      export_signals: true
      export_targets: true
      export_position_changes: false
      strategy_emoji: ":rocket:"
      include_account_info: true
      formatter:
        class: SlackMessageFormatter
        args:
          strategy_emoji: ":chart_with_upwards_trend:"
          include_account_info: true
  - exporter: RedisStreamsExporter
    parameters:
      redis_url: env:REDIS_URL
      signals_stream: strategy_signals
      export_signals: true
      export_targets: false
      export_position_changes: true
      max_stream_length: 2000
```

Each exporter has a `parameters` section that configures its behavior. The parameters vary depending on the exporter type.

## Environment Variable Substitution

As shown in the example above, you can use environment variables in your configuration by prefixing the value with `env:`. This is particularly useful for sensitive information like URLs, API keys, and passwords.

For example:

```yaml
redis_url: env:REDIS_URL
```

This will substitute the value of the `REDIS_URL` environment variable at runtime. This approach allows you to:

1. Keep sensitive information out of your configuration files
2. Use different values in different environments (development, testing, production)
3. Easily change configuration without modifying files

Environment variable substitution works for any string parameter in the exporter configuration, including stream names, webhook URLs, and formatter parameters.

## Redis Streams Exporter

The Redis Streams Exporter is a powerful way to send trading data to Redis Streams, which can then be consumed by other applications in real-time.

### Configuration

Here's a complete example of a Redis Streams Exporter configuration:

```yaml
exporter: RedisStreamsExporter
parameters:
  redis_url: env:REDIS_URL
  signals_stream: strategy:mystrategy:signals
  targets_stream: strategy:mystrategy:targets
  position_changes_stream: strategy:mystrategy:position_changes
  export_signals: true
  export_targets: true
  export_position_changes: true
  max_stream_length: 1000
  formatter:
    class: IncrementalFormatter
    args:
      alert_name: "MyStrategy"
      exchange_mapping:
        "BINANCE.UM": "BINANCE_FUTURES"
```

### Parameters

- `redis_url`: Redis connection URL (e.g., "redis://localhost:6379/0")
- `strategy_name`: Name of the strategy (used in stream keys if custom stream names are not provided)
- `export_signals`: Whether to export signals (default: true)
- `export_targets`: Whether to export target positions (default: true)
- `export_position_changes`: Whether to export position changes (default: true)
- `signals_stream`: Custom stream name for signals (default: "strategy:{strategy_name}:signals")
- `targets_stream`: Custom stream name for target positions (default: "strategy:{strategy_name}:targets")
- `position_changes_stream`: Custom stream name for position changes (default: "strategy:{strategy_name}:position_changes")
- `max_stream_length`: Maximum length of each stream (default: 1000)
- `formatter`: Formatter to use for formatting data (default: DefaultFormatter)

### Stream Data Format

The Redis Streams Exporter sends data to Redis Streams in a key-value format. The keys and values depend on the formatter used, but the default formatter includes the following fields:

#### Signals

- `timestamp`: ISO-formatted timestamp when the signal was generated
- `instrument`: Symbol of the instrument (e.g., "BTC-USDT")
- `exchange`: Exchange name (e.g., "BINANCE")
- `direction`: Signal direction (-1.0 to 1.0)
- `reference_price`: Price at which the signal was generated
- `group`: Signal group (if any)

#### Target Positions

- `timestamp`: ISO-formatted timestamp when the target position was generated
- `instrument`: Symbol of the instrument
- `exchange`: Exchange name
- `target_size`: Target position size
- `price`: Price at which the target position was generated

#### Position Changes

- `timestamp`: ISO-formatted timestamp when the position change occurred
- `instrument`: Symbol of the instrument
- `exchange`: Exchange name
- `price`: Price at which the position change occurred
- `leverage`: Current leverage
- `previous_leverage`: Previous leverage

### Consuming Redis Streams

You can consume the Redis Streams data using any Redis client that supports Redis Streams. Here's a simple example in Python:

```python
import redis
import json

# Connect to Redis
r = redis.from_url("redis://localhost:6379/0")

# Get the last ID we processed (or "0" for the beginning of the stream)
last_id = "0"

while True:
    # Read new messages from the stream
    response = r.xread(
        {
            "strategy:mystrategy:signals": last_id,
            "strategy:mystrategy:targets": last_id,
            "strategy:mystrategy:position_changes": last_id,
        },
        count=100,
        block=1000,
    )
    
    # Process the messages
    for stream_name, messages in response:
        for message_id, data in messages:
            # Update the last ID
            last_id = message_id
            
            # Convert bytes to strings
            data = {k.decode(): v.decode() for k, v in data.items()}
            
            # Process the message
            print(f"Stream: {stream_name.decode()}, Data: {data}")
```

## Incremental Formatter

The Incremental Formatter is a specialized formatter for position changes that tracks leverage changes and generates entry/exit messages. This is particularly useful for integration with trading platforms or bots that need to know when to enter or exit positions.

### Configuration

```yaml
formatter:
  class: IncrementalFormatter
  args:
    alert_name: "MyStrategy"
    exchange_mapping:
      "BINANCE.UM": "BINANCE_FUTURES"
```

### Parameters

- `alert_name`: The name of the alert to include in the messages
- `exchange_mapping`: Optional mapping of exchange names to use in messages. If an exchange is not in the mapping, the instrument's exchange is used.

### How It Works

The Incremental Formatter tracks leverage changes for each instrument and generates entry/exit messages based on the change in leverage:

1. When leverage increases (in absolute terms), it generates an ENTRY message with the leverage change.
2. When leverage decreases (in absolute terms), it generates an EXIT message with the exit fraction.
3. When leverage changes sign (from long to short or vice versa), it generates an ENTRY message with the full current leverage.

### Message Format

#### Entry Messages

```json
{
  "type": "ENTRY",
  "data": "{
    'action':'ENTRY',
    'exchange':'BINANCE_FUTURES',
    'alertName':'MyStrategy',
    'symbol':'BTC-USDT',
    'side':'BUY',
    'leverage':0.5,
    'entryPrice':50000
  }"
}
```

#### Exit Messages

```json
{
  "type": "EXIT",
  "data": "{
    'action':'EXIT',
    'exchange':'BINANCE_FUTURES',
    'alertName':'MyStrategy',
    'symbol':'BTC-USDT',
    'exitFraction':0.5,
    'exitPrice':52000
  }"
}
```

## Custom Formatters

You can create custom formatters by extending the `DefaultFormatter` class or implementing the `IExportFormatter` interface. This allows you to format the data in any way you need for your specific use case.

Here's a simple example of a custom formatter:

```python
from qubx.exporters.formatters.base import DefaultFormatter
from qubx.core.basics import Instrument, Signal, TargetPosition, dt_64
from qubx.core.interfaces import IAccountViewer
from typing import Any, Dict

class MyCustomFormatter(DefaultFormatter):
    def __init__(self, include_extra_info: bool = False):
        super().__init__()
        self.include_extra_info = include_extra_info
    
    def format_signal(self, time: dt_64, signal: Signal, account: IAccountViewer) -> dict[str, Any]:
        # Get the default formatting
        data = super().format_signal(time, signal, account)
        
        # Add custom fields
        if self.include_extra_info:
            data["total_capital"] = str(account.get_total_capital())
            data["available_capital"] = str(account.get_available_capital())
        
        return data
```

To use your custom formatter, you would configure it in your YAML file:

```yaml
formatter:
  class: path.to.your.module.MyCustomFormatter
  args:
    include_extra_info: true
```

## Multiple Exporters

You can configure multiple exporters to send data to different destinations. For example, you might want to send signals to both Redis Streams and Slack:

```yaml
exporters:
  - exporter: RedisStreamsExporter
    parameters:
      redis_url: env:REDIS_URL
      export_signals: true
  - exporter: SlackExporter
    parameters:
      signals_webhook_url: env:SLACK_WEBHOOK_URL
      export_signals: true
```

When multiple exporters are configured, Qubx creates a `CompositeExporter` that forwards all export calls to each of the configured exporters.

## Conclusion

Exporters provide a flexible way to send trading data to external systems. By configuring exporters in your strategy YAML file, you can easily integrate your Qubx strategies with other systems such as monitoring dashboards, trading bots, or notification services.

The Redis Streams Exporter with the Incremental Formatter is particularly useful for real-time integration with trading systems, as it provides a standardized way to communicate entry and exit signals.

Remember to use environment variable substitution for sensitive information like URLs and API keys to keep your configuration secure and flexible.
