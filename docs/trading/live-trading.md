# Live Trading

<!-- 
This page should include:
- Setting up for live trading
- Exchange connectors
- API configuration
- Monitoring and management
- Risk management in live trading
- Deployment options
-->

## Introduction to Live Trading

To wake a strategy callback from your own thread (an external feed, a blocking client), see [On-demand callbacks: `register_handler` / `post_event`](../backtesting/scheduling.md#on-demand-callbacks-register_handler-post_event) — `post_event` is thread-safe live, but must not be posted from a thread in simulation.
