"""Save -> load round trip for SimulationResultsSaver.

pa.Table.from_pandas(df, schema=METADATA_SCHEMA) is strict: a writer key that does not
match the schema breaks every simulation save, so the metadata columns need a live pin.
"""

import pandas as pd

from qubx.core.metrics import TradingSessionResult
from qubx.utils.results import SimulationResultsSaver

INDEX = pd.date_range("2026-01-01", periods=3, freq="1h")


def _result(capital) -> TradingSessionResult:
    portfolio = pd.DataFrame(
        {
            "BINANCE.UM:BTCUSDT_PnL": [0.0, 10.0, 10.0],
            "BINANCE.UM:BTCUSDT_Commissions": [0.0, 1.0, 0.0],
            "BINANCE.UM:BTCUSDT_Value": [1000.0, 1000.0, 1000.0],
            "HYPERLIQUID.F:BTCUSDC_PnL": [0.0, -5.0, -5.0],
            "HYPERLIQUID.F:BTCUSDC_Commissions": [0.0, 1.0, 0.0],
            "HYPERLIQUID.F:BTCUSDC_Value": [-1000.0, -1000.0, -1000.0],
        },
        index=INDEX,
    )
    return TradingSessionResult(
        id=0,
        name="roundtrip",
        start=INDEX[0],
        stop=INDEX[-1],
        exchanges=["BINANCE.UM", "HYPERLIQUID.F"],
        instruments=[],
        capital=capital,
        base_currency="USDT",
        commissions=None,
        portfolio_log=portfolio,
        executions_log=pd.DataFrame(),
        signals_log=pd.DataFrame(),
        targets_log=pd.DataFrame(),
        strategy_class="tests.Dummy",
    )


def test_save_load_preserves_per_exchange_capital(tmp_path):
    capital = {"BINANCE.UM": 60_000.0, "HYPERLIQUID.F": 40_000.0}
    run_dir = SimulationResultsSaver.save(_result(capital), str(tmp_path))

    loaded = SimulationResultsSaver.load(run_dir)

    assert loaded.capital == capital
    assert loaded.get_total_capital() == 100_000.0
    assert loaded.exchanges == ["BINANCE.UM", "HYPERLIQUID.F"]
    assert loaded.base_currency == "USDT"


def test_save_load_keeps_single_exchange_capital_scalar(tmp_path):
    result = _result(100_000.0)
    result.exchanges = ["BINANCE.UM"]
    run_dir = SimulationResultsSaver.save(result, str(tmp_path))

    loaded = SimulationResultsSaver.load(run_dir)

    assert loaded.capital == 100_000.0
