from datetime import datetime

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from qubx.core.basics import Instrument
from qubx.core.detectors.delisting import DelistingDetector


@pytest.fixture
def mock_time_provider(mocker: MockerFixture):
    provider = mocker.Mock()
    # Set current time to 2024-01-15 00:00:00
    provider.time.return_value = pd.Timestamp("2024-01-15").value
    return provider


@pytest.fixture
def detector(mock_time_provider):
    return DelistingDetector(mock_time_provider, delisting_check_days=1)


def test_detect_delistings_empty_list(detector):
    result = detector.detect_delistings([])
    assert result == []


def test_detect_delistings_no_delist_dates(detector, mocker: MockerFixture):
    instruments = [
        mocker.Mock(spec=Instrument, symbol="BTC", delist_date=None),
        mocker.Mock(spec=Instrument, symbol="ETH", delist_date=None),
    ]
    result = detector.detect_delistings(instruments)
    assert result == []


def test_detect_delistings_within_check_period(detector, mocker: MockerFixture):
    # Current time: 2024-01-15, check_days=1, so check_ahead=2024-01-16
    delisting_today = mocker.Mock(spec=Instrument, symbol="BTC", delist_date=datetime(2024, 1, 15))
    delisting_tomorrow = mocker.Mock(spec=Instrument, symbol="ETH", delist_date=datetime(2024, 1, 16))
    safe_instrument = mocker.Mock(spec=Instrument, symbol="SOL", delist_date=datetime(2024, 1, 17))

    instruments = [delisting_today, delisting_tomorrow, safe_instrument]
    result = detector.detect_delistings(instruments)

    assert len(result) == 2
    assert delisting_today in result
    assert delisting_tomorrow in result
    assert safe_instrument not in result


def test_detect_delistings_exactly_at_boundary(detector, mocker: MockerFixture):
    # Current time: 2024-01-15, check_days=1, check_ahead=2024-01-16
    # Instrument delisting exactly at boundary should be detected
    boundary_instrument = mocker.Mock(spec=Instrument, symbol="BTC", delist_date=datetime(2024, 1, 16))

    result = detector.detect_delistings([boundary_instrument])
    assert len(result) == 1
    assert boundary_instrument in result


def test_detect_delistings_with_different_check_days(mock_time_provider, mocker: MockerFixture):
    detector = DelistingDetector(mock_time_provider, delisting_check_days=3)

    # Current time: 2024-01-15, check_days=3, check_ahead=2024-01-18
    within_range = mocker.Mock(spec=Instrument, symbol="BTC", delist_date=datetime(2024, 1, 17))
    at_boundary = mocker.Mock(spec=Instrument, symbol="ETH", delist_date=datetime(2024, 1, 18))
    beyond_range = mocker.Mock(spec=Instrument, symbol="SOL", delist_date=datetime(2024, 1, 19))

    instruments = [within_range, at_boundary, beyond_range]
    result = detector.detect_delistings(instruments)

    assert len(result) == 2
    assert within_range in result
    assert at_boundary in result
    assert beyond_range not in result


def test_detect_delistings_with_zero_check_days(mock_time_provider, mocker: MockerFixture):
    detector = DelistingDetector(mock_time_provider, delisting_check_days=0)

    delisting_instrument = mocker.Mock(spec=Instrument, symbol="BTC", delist_date=datetime(2024, 1, 15))
    result = detector.detect_delistings([delisting_instrument])

    assert result == []


def test_detect_delistings_with_nat_timestamp(detector, mocker: MockerFixture):
    nat_instrument = mocker.Mock(spec=Instrument, symbol="BTC", delist_date=pd.NaT)
    result = detector.detect_delistings([nat_instrument])
    assert result == []


def test_detect_delistings_with_timezone_aware_date(detector, mocker: MockerFixture):
    # Test with timezone-aware timestamp
    tz_aware_date = pd.Timestamp("2024-01-15", tz="UTC")
    delisting_instrument = mocker.Mock(spec=Instrument, symbol="BTC", delist_date=tz_aware_date)

    result = detector.detect_delistings([delisting_instrument])
    assert len(result) == 1
    assert delisting_instrument in result


def test_filter_delistings_empty_list(detector):
    result = detector.filter_delistings([])
    assert result == []


def test_filter_delistings_no_delist_dates(detector, mocker: MockerFixture):
    instruments = [
        mocker.Mock(spec=Instrument, symbol="BTC", delist_date=None),
        mocker.Mock(spec=Instrument, symbol="ETH", delist_date=None),
    ]
    result = detector.filter_delistings(instruments)
    assert result == instruments


def test_filter_delistings_within_check_period(detector, mocker: MockerFixture):
    # Current time: 2024-01-15, check_days=1, check_ahead=2024-01-16
    delisting_today = mocker.Mock(spec=Instrument, symbol="BTC", delist_date=datetime(2024, 1, 15))
    delisting_tomorrow = mocker.Mock(spec=Instrument, symbol="ETH", delist_date=datetime(2024, 1, 16))
    safe_instrument = mocker.Mock(spec=Instrument, symbol="SOL", delist_date=datetime(2024, 1, 17))
    no_delist = mocker.Mock(spec=Instrument, symbol="ADA", delist_date=None)

    instruments = [delisting_today, delisting_tomorrow, safe_instrument, no_delist]
    result = detector.filter_delistings(instruments)

    assert len(result) == 2
    assert delisting_today not in result
    assert delisting_tomorrow not in result
    assert safe_instrument in result
    assert no_delist in result


def test_filter_delistings_is_inverse_of_detect(detector, mocker: MockerFixture):
    instruments = [
        mocker.Mock(spec=Instrument, symbol="BTC", delist_date=datetime(2024, 1, 15)),
        mocker.Mock(spec=Instrument, symbol="ETH", delist_date=datetime(2024, 1, 16)),
        mocker.Mock(spec=Instrument, symbol="SOL", delist_date=datetime(2024, 1, 17)),
        mocker.Mock(spec=Instrument, symbol="ADA", delist_date=None),
    ]

    detected = detector.detect_delistings(instruments)
    filtered = detector.filter_delistings(instruments)

    # Every instrument should be in exactly one of the two lists
    assert len(detected) + len(filtered) == len(instruments)
    assert set(detected) & set(filtered) == set()  # No overlap
    assert set(detected) | set(filtered) == set(instruments)  # Complete coverage


def test_filter_delistings_with_zero_check_days(mock_time_provider, mocker: MockerFixture):
    detector = DelistingDetector(mock_time_provider, delisting_check_days=0)

    instruments = [
        mocker.Mock(spec=Instrument, symbol="BTC", delist_date=datetime(2024, 1, 15)),
        mocker.Mock(spec=Instrument, symbol="ETH", delist_date=datetime(2024, 1, 16)),
    ]
    result = detector.filter_delistings(instruments)

    # When check_days=0, all instruments should pass through
    assert result == instruments


def test_filter_delistings_with_nat_timestamp(detector, mocker: MockerFixture):
    nat_instrument = mocker.Mock(spec=Instrument, symbol="BTC", delist_date=pd.NaT)
    normal_instrument = mocker.Mock(spec=Instrument, symbol="ETH", delist_date=None)

    result = detector.filter_delistings([nat_instrument, normal_instrument])

    # NaT should be treated as no delist date
    assert len(result) == 2
    assert nat_instrument in result
    assert normal_instrument in result


def test_filter_delistings_with_timezone_aware_date(detector, mocker: MockerFixture):
    # Test with timezone-aware timestamp that should be filtered
    tz_aware_date = pd.Timestamp("2024-01-15", tz="UTC")
    delisting_instrument = mocker.Mock(spec=Instrument, symbol="BTC", delist_date=tz_aware_date)
    safe_instrument = mocker.Mock(spec=Instrument, symbol="ETH", delist_date=None)

    result = detector.filter_delistings([delisting_instrument, safe_instrument])

    assert len(result) == 1
    assert safe_instrument in result
    assert delisting_instrument not in result


def test_delisting_check_days_property(mock_time_provider):
    detector = DelistingDetector(mock_time_provider, delisting_check_days=5)
    assert detector.delisting_check_days == 5

    detector.delisting_check_days = 10
    assert detector.delisting_check_days == 10


def test_error_handling_in_detect_delistings(detector, mocker: MockerFixture):
    # Create an instrument with an invalid delist_date that will cause an exception
    bad_instrument = mocker.Mock(spec=Instrument, symbol="BAD", delist_date="invalid_date")
    good_instrument = mocker.Mock(spec=Instrument, symbol="GOOD", delist_date=datetime(2024, 1, 15))

    result = detector.detect_delistings([bad_instrument, good_instrument])

    # Bad instrument should be excluded, good one should be included
    assert len(result) == 1
    assert good_instrument in result
    assert bad_instrument not in result


def test_error_handling_in_filter_delistings(detector, mocker: MockerFixture):
    # Create an instrument with an invalid delist_date that will cause an exception
    bad_instrument = mocker.Mock(spec=Instrument, symbol="BAD", delist_date="invalid_date")
    good_instrument = mocker.Mock(spec=Instrument, symbol="GOOD", delist_date=None)

    result = detector.filter_delistings([bad_instrument, good_instrument])

    # On error in filter_delistings, instrument should be included (safe default)
    assert len(result) == 2
    assert bad_instrument in result
    assert good_instrument in result
