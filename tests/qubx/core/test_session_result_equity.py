import pandas as pd

from qubx.core.metrics import _transfer_offsets

INDEX = pd.date_range("2026-01-01", periods=4, freq="1h")


def _transfers(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("timestamp")


def test_off_grid_transfer_lands_on_next_bar():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:02"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 100.0,
                "to_amount": 100.0,
                "status": "completed",
            }
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [0.0, 0.0, 100.0, 100.0]
    assert list(_transfer_offsets(tl, "A", INDEX)) == [0.0, 0.0, -100.0, -100.0]


def test_two_transfers_in_one_bar_sum():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:02"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 100.0,
                "to_amount": 100.0,
                "status": "completed",
            },
            {
                "timestamp": pd.Timestamp("2026-01-01 01:30"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 50.0,
                "to_amount": 50.0,
                "status": "completed",
            },
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [0.0, 0.0, 150.0, 150.0]


def test_transfer_before_first_bar_counts_from_the_start():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2025-12-31 23:00"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 10.0,
                "to_amount": 10.0,
                "status": "completed",
            }
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [10.0, 10.0, 10.0, 10.0]


def test_transfer_after_last_bar_is_ignored():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2026-01-02 00:00"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 10.0,
                "to_amount": 10.0,
                "status": "completed",
            }
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [0.0, 0.0, 0.0, 0.0]


def test_non_completed_transfers_excluded():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:02"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 100.0,
                "to_amount": 100.0,
                "status": "pending",
            },
            {
                "timestamp": pd.Timestamp("2026-01-01 01:02"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 7.0,
                "to_amount": 7.0,
                "status": "failed",
            },
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [0.0, 0.0, 0.0, 0.0]


def test_converted_transfer_credits_destination_amount():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:02"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 100.0,
                "to_amount": 99.0,
                "status": "completed",
            }
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [0.0, 0.0, 99.0, 99.0]
    assert list(_transfer_offsets(tl, "A", INDEX)) == [0.0, 0.0, -100.0, -100.0]
