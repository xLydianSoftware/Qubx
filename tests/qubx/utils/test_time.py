"""
Tests for time utility functions, especially interval_to_cron with offset intervals
"""

import pytest

from qubx.utils.time import interval_to_cron


def test_offset_interval_hour_based():
    """Test offset intervals for hour-based schedules"""
    assert interval_to_cron("1h -1min") == "59 * * * *"
    assert interval_to_cron("1h -1s") == "59 * * * * 59"
    assert interval_to_cron("1h -30min") == "30 * * * *"
    # - 2h -30min: fires at 1:30, 3:30, 5:30, etc. (30min before end of each 2h period)
    assert interval_to_cron("2h -30min") == "30 1,3,5,7,9,11,13,15,17,19,21,23 * * *"
    # - 3h -15min: fires at 2:45, 5:45, 8:45, etc. (15min before end of each 3h period)
    assert interval_to_cron("3h -15min") == "45 2,5,8,11,14,17,20,23 * * *"
    assert interval_to_cron("1h -59min") == "1 * * * *"


def test_offset_interval_day_based():
    """Test offset intervals for day-based schedules"""
    assert interval_to_cron("1d -5Min") == "55 23 * * *"
    assert interval_to_cron("1d -1h") == "0 23 * * *"
    assert interval_to_cron("2d -1h") == "0 23 */2 * *"
    assert interval_to_cron("1d -30min") == "30 23 * * *"


def test_offset_interval_minute_based():
    """Test offset intervals for minute-based schedules"""
    assert interval_to_cron("1min -30s") == "* * * * * 30"
    # - 5min -1s: fires at 4:59, 9:59, 14:59, etc. (1s before end of each 5min period)
    assert interval_to_cron("5min -1s") == "4,9,14,19,24,29,34,39,44,49,54,59 * * * * 59"
    # - 5min -1min: fires at 4:00, 9:00, 14:00, etc. (1min before end of each 5min period)
    assert interval_to_cron("5min -1min") == "4,9,14,19,24,29,34,39,44,49,54,59 * * * * 0"


def test_offset_interval_week_based():
    """Test offset intervals for week-based schedules"""
    # - 1 week minus 1 day = 6 days at midnight
    result = interval_to_cron("1w -1d")
    assert result == "0 0 */7 * *"


def test_offset_interval_errors():
    """Test that invalid offset intervals raise appropriate errors"""
    # - offset larger than base should fail
    with pytest.raises(ValueError, match="too large for base interval"):
        interval_to_cron("1h -2h")

    # - offset equal to base should fail
    with pytest.raises(ValueError, match="too large for base interval"):
        interval_to_cron("1h -1h")


def test_existing_timedelta_functionality():
    """Ensure existing timedelta parsing still works"""
    assert interval_to_cron("4h") == "0 */4 * * *"
    # - note: these may have seconds appended by timedelta_to_crontab
    result = interval_to_cron("2d")
    assert result.startswith("59 23 */2 * *")


def test_existing_schedule_format():
    """Ensure existing @ schedule format still works"""
    # - daily at specific time
    result = interval_to_cron("@10:30")
    assert result.startswith("30 10 * * *")

    # - monthly
    result = interval_to_cron("1M@15:00")
    assert "15 1 */1 *" in result

    # - day of week
    result = interval_to_cron("TUE@23:59")
    assert "59 23 * * 2" in result


def test_offset_with_complex_base_intervals():
    """Test offsets with complex base intervals"""
    # - 2 hours minus 15 minutes: fires at 1:45, 3:45, 5:45, etc.
    assert interval_to_cron("2h -15min") == "45 1,3,5,7,9,11,13,15,17,19,21,23 * * *"

    # - 15 minutes minus 10 seconds: fires at 14:50, 29:50, 44:50, 59:50
    assert interval_to_cron("15Min -10s") == "14,29,44,59 * * * * 50"

    # - 3 days minus 2 hours
    assert interval_to_cron("3d -2h") == "0 22 */3 * *"


def test_offset_boundary_conditions():
    """Test boundary conditions for offset intervals"""
    # - 1 hour minus 1 second (should be at 59:59)
    assert interval_to_cron("1h -1s") == "59 * * * * 59"

    # - 1 day minus 1 second (should be at 23:59:59)
    assert interval_to_cron("1d -1s") == "59 23 * * * 59"

    # - 1 minute minus 1 second
    assert interval_to_cron("1min -1s") == "* * * * * 59"
