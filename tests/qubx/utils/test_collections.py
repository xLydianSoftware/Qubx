import numpy as np
import pytest
from numba import types
from numba.experimental import jitclass

from qubx.utils.collections import (
    DequeFloat32,
    DequeFloat64,
    DequeIndicator,
    DequeInt32,
    DequeProtocol,
    create_deque_class,
)


def test_deque_float32():
    """Test basic operations with float32 deque."""
    # Create a deque with capacity 3 for float32 elements
    dq: DequeProtocol = DequeFloat32(3)

    # Test 1: Push elements and check size
    dq.push_back(1.0)
    dq.push_back(2.0)
    assert dq.get_size() == 2, "Size should be 2 after 2 pushes"
    dq.push_back(3.0)
    assert dq.get_size() == 3, "Size should be 3 after filling the deque"

    # Test 2: Push when full (overwriting behavior)
    dq.push_back(4.0)  # This should overwrite the oldest (1.0)
    assert dq.get_size() == 3, "Size should remain 3 after overwriting"
    arr = dq.to_array()
    expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    assert np.allclose(arr, expected), f"Expected {expected}, got {arr}"

    # Test 3: Pop elements and check order
    oldest = dq.pop_front()
    assert oldest == 2.0, f"Expected 2.0, got {oldest}"
    assert dq.get_size() == 2, "Size should be 2 after one pop"

    # Test 4: Push front
    dq.push_front(5.0)  # This should push 5.0 to the front
    arr = dq.to_array()
    expected = np.array([5.0, 3.0, 4.0], dtype=np.float32)
    assert np.allclose(arr, expected), f"Expected {expected}, got {arr}"

    # Test 5: Pop back
    newest = dq.pop_back()
    assert newest == 4.0, f"Expected 4.0, got {newest}"
    assert dq.get_size() == 2, "Size should be 2 after popping back"

    # Test 6: Is empty and is full
    assert not dq.is_empty(), "Deque should not be empty"
    dq.pop_front()
    dq.pop_front()
    assert dq.is_empty(), "Deque should be empty after popping all elements"
    assert not dq.is_full(), "Deque should not be full when empty"

    # Test 7: Push to full and verify overwrite again
    dq.push_back(6.0)
    dq.push_back(7.0)
    dq.push_back(8.0)
    dq.push_back(9.0)  # Overwrite oldest
    arr = dq.to_array()
    expected = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    assert np.allclose(arr, expected), f"Expected {expected}, got {arr}"


def test_deque_inside_jitclass():
    """Test using deque inside another jitclass."""

    @jitclass(  # type: ignore
        [
            ("dq", DequeInt32.class_type.instance_type),  # type: ignore
            ("counter", types.int32),
        ]
    )
    class TestClass:
        def __init__(self):
            self.dq = DequeInt32(3)
            self.counter = 0

        def on_event(self):
            self.dq.push_back(self.counter)
            self.counter += 1

    tc = TestClass()
    for _ in range(4):
        tc.on_event()

    arr = tc.dq.to_array()
    expected = np.array([1, 2, 3], dtype=np.int32)
    assert np.allclose(arr, expected), f"Expected {expected}, got {arr}"


def test_deque_indicator():
    """Test structured dtype deque with timestamp and value fields."""
    dq: DequeProtocol = DequeIndicator(3)

    # Test push_back_fields
    dq.push_back_fields(1000, 1.5)  # timestamp, value
    dq.push_back_fields(2000, 2.5)
    assert dq.get_size() == 2, "Size should be 2 after pushing 2 records"

    # Test accessing fields
    record = dq[0]  # Get newest record
    assert record["timestamp"] == 2000, f"Expected timestamp 2000, got {record['timestamp']}"
    assert record["value"] == 2.5, f"Expected value 2.5, got {record['value']}"

    # Test overwriting behavior
    dq.push_back_fields(3000, 3.5)
    dq.push_back_fields(4000, 4.5)  # Should overwrite first record

    arr = dq.to_array()
    assert len(arr) == 3, "Array length should be 3"
    assert arr[0]["timestamp"] == 2000, f"Expected timestamp 2000, got {arr[0]['timestamp']}"
    assert arr[-1]["timestamp"] == 4000, f"Expected timestamp 4000, got {arr[-1]['timestamp']}"


def test_deque_errors():
    """Test error conditions."""
    dq: DequeProtocol = DequeFloat64(2)

    # Test empty deque errors
    with pytest.raises(IndexError, match="Deque is empty"):
        dq.pop_front()

    with pytest.raises(IndexError, match="Deque is empty"):
        dq.pop_back()

    with pytest.raises(IndexError, match="Deque is empty"):
        dq.front()

    with pytest.raises(IndexError, match="Deque is empty"):
        dq.back()

    # Test index out of bounds
    dq.push_back(1.0)
    with pytest.raises(IndexError, match="Index out of bounds"):
        _ = dq[1]  # Only 1 element, index 1 is out of bounds

    with pytest.raises(IndexError, match="Index out of bounds"):
        _ = dq[-1]  # Negative indices not supported


def test_unsupported_dtype():
    """Test creating deque with unsupported dtype."""
    with pytest.raises(ValueError, match="Unsupported scalar dtype"):
        create_deque_class(np.dtype(np.complex128))
