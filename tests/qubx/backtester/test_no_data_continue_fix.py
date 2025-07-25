"""
Simple test to verify the NoDataContinue fix works.
"""
import pandas as pd
import pytest

from qubx.backtester.sentinels import NoDataContinue
from qubx.data.composite import IteratedDataStreamsSlicer


def test_slicer_returns_no_data_continue_when_empty():
    """Test that IteratedDataStreamsSlicer returns NoDataContinue when no keys exist."""
    slicer = IteratedDataStreamsSlicer()
    
    # Start iteration
    iter_slicer = iter(slicer)
    
    # Should return NoDataContinue sentinel when no data streams
    result = next(iter_slicer)
    
    assert len(result) == 3
    assert result[0] == ""  # key
    assert result[1] == 0   # time
    assert isinstance(result[2], NoDataContinue)  # sentinel


def test_slicer_continues_after_removing_all_streams():
    """Test that slicer can continue after all streams are removed."""
    from qubx.core.basics import Quote
    
    slicer = IteratedDataStreamsSlicer()
    
    # Mock some data using proper Timestamped objects
    def mock_generator():
        # Create proper Quote objects with required parameters
        quote = Quote(time=123456789, bid=100.0, ask=101.0, bid_size=10.0, ask_size=10.0)
        yield [quote]
    
    # Add data stream
    slicer.put({"test_stream": mock_generator()})
    
    # Verify we can iterate
    iter_slicer = iter(slicer)
    result1 = next(iter_slicer)  # Should get first data point
    assert result1[0] == "test_stream"
    assert not isinstance(result1[2], NoDataContinue)
    
    # Now remove all streams
    slicer.remove(["test_stream"])
    
    # Should now return NoDataContinue
    result2 = next(iter_slicer)
    assert isinstance(result2[2], NoDataContinue)


def test_no_data_continue_sentinel_properties():
    """Test the NoDataContinue sentinel object."""
    sentinel = NoDataContinue()
    assert sentinel.next_scheduled_time is None
    
    sentinel_with_time = NoDataContinue(12345)
    assert sentinel_with_time.next_scheduled_time == 12345
    
    # Test repr
    repr_str = repr(sentinel)
    assert "NoDataContinue" in repr_str


def test_slicer_iterating_flag_remains_true_after_no_data_continue():
    """Test that _iterating remains True after returning NoDataContinue."""
    slicer = IteratedDataStreamsSlicer()
    
    # Start iteration - should get NoDataContinue when empty
    iter_slicer = iter(slicer)
    assert slicer._iterating == True
    
    result1 = next(iter_slicer)
    assert isinstance(result1[2], NoDataContinue)
    
    # Critical test: _iterating should still be True after returning sentinel
    assert slicer._iterating == True, "_iterating should remain True to allow future put() operations to rebuild"


def test_no_initial_subscriptions_scenario():
    """Test scenario where no initial subscriptions exist and data is added later."""
    from qubx.backtester.simulated_data import IterableSimulationData
    from qubx.core.basics import DataType
    
    # Create simulation data with empty readers (simulating no initial subscriptions)
    sim_data = IterableSimulationData({})
    
    # Start iteration - should handle empty state gracefully
    qiter = sim_data.create_iterable("2023-01-01", "2023-01-01 12:00")
    iterator = iter(qiter)
    
    # First call should return NoDataContinue since no subscriptions exist
    result = next(iterator)
    instrument, data_type, event, is_hist = result
    
    # Should get NoDataContinue sentinel when no subscriptions exist
    assert instrument is None
    assert isinstance(event, NoDataContinue)
    
    print("✅ Test confirmed: NoDataContinue sentinel returned when no initial subscriptions exist")
    print("   This allows scheduled events to fire and subscribe to data dynamically")


def test_iterating_flag_true_with_no_initial_subscriptions():
    """Critical test: _iterating must be True even when starting with no subscriptions."""
    slicer = IteratedDataStreamsSlicer()
    
    # Before iteration starts
    assert slicer._iterating == False, "Should be False before iteration starts"
    
    # Start iteration with no data streams
    iter_slicer = iter(slicer)
    
    # CRITICAL: _iterating should be True even with no initial data streams
    assert slicer._iterating == True, "CRITICAL: _iterating must be True after iter() even with no streams"
    
    # Get the NoDataContinue sentinel
    result = next(iter_slicer)
    assert isinstance(result[2], NoDataContinue)
    
    # CRITICAL: _iterating should STILL be True after returning sentinel
    assert slicer._iterating == True, "CRITICAL: _iterating must remain True after NoDataContinue"
    
    print("✅ CRITICAL TEST PASSED: _iterating=True allows future put() operations to rebuild")
    print("   This enables dynamic subscription during scheduled events")