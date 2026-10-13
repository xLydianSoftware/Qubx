import numpy as np
import pandas as pd
from pytest import approx

from qubx.backtester.simulator import simulate
from qubx.core import metrics
from qubx.core.basics import Instrument, MarketType
from qubx.core.metrics import traded_notional
from qubx.data.registry import StorageRegistry


def _fills(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"exchange": "OKX.F", "market_type": "SWAP", "symbol": s, "filled_qty": q, "price": p} for s, q, p in rows],
        index=pd.date_range("2025-01-01", periods=len(rows), freq="1min", name="timestamp"),
    )


def _okx_swap(symbol: str, contract_size: float) -> Instrument:
    return Instrument(
        symbol,
        MarketType.SWAP,
        "OKX.F",
        symbol[:-4],
        "USDT",
        "USDT",
        symbol,
        0.01,
        1.0,
        1.0,
        contract_size=contract_size,
    )


def test_fills_in_contracts_are_converted_with_the_session_instrument_multiplier():
    fills = _fills([("ETHUSDT", 10, 2000.0), ("ETHUSDT", -10, 2100.0), ("BTCUSDT", 5, 90000.0)])
    instruments = [_okx_swap("ETHUSDT", 0.1), _okx_swap("BTCUSDT", 0.01)]
    assert traded_notional(fills, instruments) == approx(10 * 2000 * 0.1 + 10 * 2100 * 0.1 + 5 * 90000 * 0.01)


def test_a_symbol_neither_in_the_session_nor_in_the_lookup_counts_with_multiplier_one(monkeypatch):
    def unreachable(*args, **kwargs):
        raise RuntimeError("instrument service is unavailable")

    monkeypatch.setattr(metrics.lookup, "find_symbol", unreachable)
    fills = _fills([("ETHUSDT", 10, 2000.0), ("SOLUSDT", 4, 150.0)])
    assert traded_notional(fills, [_okx_swap("ETHUSDT", 0.1)]) == approx(10 * 2000 * 0.1 + 4 * 150.0)


def test_bps_volume_is_gross_so_the_fee_level_does_not_change_it():
    storage = StorageRegistry.get("csv::tests/data/storages/csv_longer")
    close = storage["BINANCE.UM", "SWAP"].read("ETHUSDT", "ohlc(1h)", "2024-01-01", "2024-03-01").to_pd()["close"]
    side = np.sign(close - close.rolling(24).mean()).dropna()
    signals = (side[side != side.shift()] * 0.5).to_frame("ETHUSDT")
    signals.index = signals.index + pd.Timedelta("1h")

    perf = {}
    for fees in ["vip0_usdt", "vip5_usdt"]:
        r = simulate(
            {fees: signals},
            data=storage,
            capital=100_000,
            instruments=["BINANCE.UM:SWAP:ETHUSDT"],
            commissions=fees,
            signal_timeframe="1h",
            start="2024-01-01",
            stop="2024-03-01",
            debug="WARNING",
            silent=True,
        )[0]
        perf[fees] = r.performance()

    assert perf["vip0_usdt"]["gain"] < perf["vip5_usdt"]["gain"]
    assert perf["vip0_usdt"]["bps_volume"] == approx(perf["vip5_usdt"]["bps_volume"], rel=1e-6)
