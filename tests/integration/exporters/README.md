# Redis Streams Exporter Integration Tests

This directory contains integration tests for the Redis Streams Exporter. The tests verify that the exporter correctly exports trading data to Redis Streams.

## Requirements

- Docker
- Python 3.8+
- Poetry

## Running the Tests

To run the tests, use the following command from the project root:

```bash
poetry run pytest tests/qubx/exporters/test_redis_streams.py -v
```

## Test Structure

The tests use pytest-docker to start a Redis container for testing. The tests verify:

1. Exporting signals with the default formatter
2. Exporting signals with a compact formatter
3. Exporting target positions
4. Exporting leverage changes

## Docker Compose Configuration

The tests use a Docker Compose file to start a Redis container. The configuration is in `docker-compose.yml`.

## Custom Fixtures

The tests use several custom fixtures:

- `redis_service`: Starts a Redis container and returns the connection URL
- `account_viewer`: Creates a mock account viewer for testing
- `instruments`: Creates test instruments
- `signals`: Creates test signals
- `target_positions`: Creates test target positions
