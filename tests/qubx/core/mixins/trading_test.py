import pandas as pd
import pytest

from qubx.core.interfaces import ITimeProvider
from qubx.core.mixins.trading import ClientIdStore


class MockTimeProvider(ITimeProvider):
    def __init__(self, timestamp=None):
        self._timestamp = timestamp or pd.Timestamp("2023-01-01").asm8

    def time(self):
        return self._timestamp


@pytest.fixture
def time_provider():
    return MockTimeProvider()


@pytest.fixture
def client_id_store():
    return ClientIdStore()


def test_initialize_from_timestamp():
    """Test that order_id is initialized correctly from timestamp."""
    # Given a time provider with a known timestamp
    timestamp = pd.Timestamp("2023-01-01").asm8
    time_provider = MockTimeProvider(timestamp)
    store = ClientIdStore()

    # When initializing order_id from timestamp
    result = store._initialize_id_from_timestamp(time_provider)

    # Then the order_id is derived correctly
    expected = timestamp.astype("int64") // 100_000_000
    assert result == expected


def test_create_id(client_id_store):
    """Test ID creation from symbol and order ID."""
    # When creating an ID
    result = client_id_store._create_id("BTCUSD", 12345)

    # Then the ID follows the expected format
    assert result == "qubx_BTCUSD_12345"


def test_generate_id(time_provider, client_id_store):
    """Test that generated IDs are unique and incremental."""
    # When generating multiple IDs
    ids = [client_id_store.generate_id(time_provider, "BTCUSD") for _ in range(5)]

    # Then all IDs are unique
    assert len(ids) == len(set(ids)), "Generated IDs should be unique"

    # And they follow an incremental pattern
    for i in range(1, len(ids)):
        id1_parts = ids[i - 1].split("_")
        id2_parts = ids[i].split("_")
        assert int(id2_parts[2]) == int(id1_parts[2]) + 1, "IDs should increment by 1"


def test_unique_ids_across_symbols(time_provider, client_id_store):
    """Test that IDs are unique across different symbols."""
    # When generating IDs for different symbols
    id1 = client_id_store.generate_id(time_provider, "BTCUSD")
    id2 = client_id_store.generate_id(time_provider, "ETHUSD")

    # Then the IDs are unique
    assert id1 != id2

    # And they follow the expected format with different symbols
    assert "BTCUSD" in id1
    assert "ETHUSD" in id2

    # And the numerical part increments
    id1_num = int(id1.split("_")[2])
    id2_num = int(id2.split("_")[2])
    assert id2_num == id1_num + 1
