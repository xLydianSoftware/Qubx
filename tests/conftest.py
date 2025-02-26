import os

import dotenv
import pytest
from loguru import logger


def pytest_addoption(parser):
    add_env_option(parser)


def add_env_option(parser):
    parser.addoption(
        "--env",
        default=".env.integration",
        help="Path to the environment file to use for integration tests",
    )


@pytest.fixture
def exchange_credentials(request):
    EXCHANGE_MAPPINGS = {"BINANCE_SPOT": "BINANCE", "BINANCE_FUTURES": "BINANCE.UM"}
    env_path = request.config.getoption("--env")
    options = dotenv.dotenv_values(env_path)
    api_keys = {k: v for k, v in options.items() if k.endswith("_API_KEY")}
    api_secrets = {k: v for k, v in options.items() if k.endswith("_SECRET")}
    exchange_credentials = {}
    for key, api_key in api_keys.items():
        exchange = key[: -len("_API_KEY")]
        exchange_credentials[EXCHANGE_MAPPINGS.get(exchange, exchange)] = {
            "api_key": api_key,
            "secret": api_secrets[f"{exchange}_SECRET"],
        }
    return exchange_credentials


@pytest.fixture(autouse=True)
def setup_and_teardown_logger():
    """
    Fixture to properly handle loguru's file handlers between test runs.
    This ensures that file handlers are properly closed after each test.
    """
    # Store the IDs of the current handlers
    original_handlers = []

    # Get the IDs of all current handlers
    # We'll add a dummy handler and then remove it to get the current handler IDs
    dummy_id = logger.add(lambda _: None)
    for i in range(1, dummy_id):
        try:
            # Try to use the handler to see if it exists
            logger.debug("", _force_use_handler=i)
            original_handlers.append(i)
        except:
            pass
    logger.remove(dummy_id)

    # Let the test run
    yield

    # After the test, remove any handlers that were added during the test
    # Add another dummy handler to get the current maximum ID
    new_dummy_id = logger.add(lambda _: None)
    for i in range(1, new_dummy_id):
        if i not in original_handlers:
            try:
                logger.remove(i)
            except:
                pass
    logger.remove(new_dummy_id)
