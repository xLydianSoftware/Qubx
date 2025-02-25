from click.testing import CliRunner

import qubx.pandaz.ta as pta
import tests.qubx.ta.utils_for_testing as test
from qubx.backtester.simulator import simulate
from qubx.cli.commands import release
from qubx.core.series import OHLCV
from qubx.data import loader
from qubx.data.readers import AsOhlcvSeries, CsvStorageDataReader
from tests.strategies.macd_crossover.indicators.macd import macd
from tests.strategies.macd_crossover.models.macd_crossover import MacdCrossoverStrategy


class TestMacdCrossoverLifecycle:
    def test_macd_indicator(self):
        r = CsvStorageDataReader("tests/data/csv/")
        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+5h", transform=AsOhlcvSeries("1Min", "ms"))
        assert isinstance(ohlc, OHLCV)
        _macd = macd(ohlc.close).to_series().dropna()
        expected_macd = pta.macd(ohlc.close.pd()).dropna()
        assert test.N(_macd[-50:]) == expected_macd[-50:]

    def test_macd_crossover_simulation(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)
        test0 = simulate(
            MacdCrossoverStrategy(),
            ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-06-10",
            debug="INFO",
            n_jobs=1,
        )
        sim = test0[0]
        assert len(sim.executions_log) > 1

    def test_release(self):
        # runner = CliRunner()
        # result = runner.invoke(release, ["--strategy", "test_strategy", "--tag", "test_tag", "--comment", "test_comment"])
        pass
