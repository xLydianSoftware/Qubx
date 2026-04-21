"""
Regression tests for xLydianSoftware/Qubx#264.

Verifies that rate limiters are injected into warmup storage configs so that
warmup OHLCV fetches are subject to the same per-exchange rate-limit budget as
the live path. Without this, transient exchange rate limits at warmup start
can leave an instrument with zero bars, which cascades through indicators to
corrupted live state (see issue for the production incident).
"""

from unittest.mock import MagicMock

from qubx.utils.runner.configs import StorageConfig, TypedStorageConfig, WarmupConfig
from qubx.utils.runner.runner import _inject_warmup_rate_limiters


def _make_warmup(custom_data: list[TypedStorageConfig] | None = None) -> WarmupConfig:
    return WarmupConfig(
        data=StorageConfig(storage="ccxt", args={}),
        custom_data=custom_data or [],
    )


class TestInjectWarmupRateLimiters:
    def test_injects_into_primary_data_storage_args(self):
        warmup = _make_warmup()
        rate_limiters = {"OKX.F": MagicMock(name="OKX.F-rl")}

        _inject_warmup_rate_limiters(warmup, rate_limiters)

        assert warmup.data.args["rate_limiters"] is rate_limiters

    def test_injects_into_every_custom_data_storage(self):
        custom = [
            TypedStorageConfig(
                data_type=["ohlc(1h)"],
                storages=[
                    StorageConfig(storage="ccxt", args={}),
                    StorageConfig(storage="mqdb", args={"host": "nebula"}),
                ],
            ),
            TypedStorageConfig(
                data_type="trades",
                storages=[StorageConfig(storage="ccxt", args={})],
            ),
        ]
        warmup = _make_warmup(custom_data=custom)
        rate_limiters = {"BINANCE.UM": MagicMock(name="binance-rl")}

        _inject_warmup_rate_limiters(warmup, rate_limiters)

        for typed_cfg in warmup.custom_data:
            for sc in typed_cfg.storages:
                assert sc.args["rate_limiters"] is rate_limiters

    def test_none_warmup_is_noop(self):
        # Should not raise when warmup config is missing entirely
        _inject_warmup_rate_limiters(None, {"X": MagicMock()})

    def test_none_or_empty_rate_limiters_is_noop(self):
        warmup = _make_warmup()
        _inject_warmup_rate_limiters(warmup, None)
        _inject_warmup_rate_limiters(warmup, {})
        assert "rate_limiters" not in warmup.data.args

    def test_preserves_existing_args(self):
        warmup = WarmupConfig(
            data=StorageConfig(storage="ccxt", args={"max_history": "30d"}),
        )
        rate_limiters = {"OKX.F": MagicMock()}

        _inject_warmup_rate_limiters(warmup, rate_limiters)

        assert warmup.data.args["max_history"] == "30d"
        assert warmup.data.args["rate_limiters"] is rate_limiters
