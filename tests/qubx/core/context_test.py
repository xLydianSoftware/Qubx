from unittest.mock import Mock

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import Deal, Instrument, Signal, TargetPosition, TriggerEvent
from qubx.core.interfaces import IPositionGathering, IStrategy, IStrategyContext
from qubx.data.helpers import loader
from qubx.gathering.simplest import SimplePositionGatherer


class Tester1(IStrategy):
    _idx = 0
    _err = False
    _to_test: list[list[Instrument]] = []

    def on_start(self, ctx: IStrategyContext) -> None:
        self._exch = ctx.exchanges[0]
        logger.info(f"Exchange: {self._exch}")

    def on_fit(self, ctx: IStrategyContext):
        instr = [ctx.query_instrument(s, ctx.exchanges[0]) for s in ["BTCUSDT", "ETHUSDT", "BCHUSDT"]]
        logger.info(str(instr))
        # logger.info(f" -> SET NEW UNIVERSE {','.join(i.symbol for i in self._to_test[self._idx])}")
        # self._idx += 1

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        for s in ctx.instruments:
            q = ctx.quote(s)
            if q is None:
                logger.error(f"\n{s.symbol} -> NO QUOTE\n")
                self._err = True

        return []


class TestStrategyContext:
    def test_context_exchanges(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)

        simulate(
            {
                "fail1": (stg := Tester1()),
            },
            {"ohlc(4h)": ld},
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-08-01",
            debug="DEBUG",
            n_jobs=1,
        )

        assert stg._exch == "BINANCE.UM", "Got Errors during the simulation"


class MockCustomGatherer(IPositionGathering):
    """Mock custom gatherer for testing."""

    def __init__(self, custom_param: str = "test"):
        self.custom_param = custom_param
        self.alter_positions_called = False
        self.on_execution_report_called = False

    def alter_positions(
        self, ctx: IStrategyContext, targets: list[TargetPosition] | TargetPosition
    ) -> dict[Instrument, float]:
        self.alter_positions_called = True
        return {}

    def on_execution_report(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal):
        self.on_execution_report_called = True


class TestStrategyWithGatherer(IStrategy):
    """Test strategy that overrides the gatherer method."""

    def __init__(self):
        super().__init__()
        self.gatherer_instance = MockCustomGatherer("custom_value")

    def gatherer(self, ctx: IStrategyContext) -> IPositionGathering | None:
        return self.gatherer_instance


class TestStrategyWithoutGatherer(IStrategy):
    """Test strategy that does not override the gatherer method."""

    pass


class TestStrategyGathererOverride:
    """Test cases for strategy gatherer override functionality."""

    def test_strategy_gatherer_override_method_exists(self):
        """Test that IStrategy has a gatherer method."""
        strategy = TestStrategyWithGatherer()
        assert hasattr(strategy, "gatherer"), "IStrategy should have a gatherer method"

        # Test that the method is callable
        assert callable(strategy.gatherer), "gatherer should be callable"

    def test_strategy_with_gatherer_override(self):
        """Test that strategy can override the gatherer method."""
        strategy = TestStrategyWithGatherer()
        mock_ctx = Mock(spec=IStrategyContext)

        gatherer = strategy.gatherer(mock_ctx)

        assert gatherer is not None, "Strategy should return a gatherer instance"
        assert isinstance(gatherer, MockCustomGatherer), "Should return MockCustomGatherer instance"
        assert gatherer.custom_param == "custom_value", "Custom parameters should be preserved"

    def test_strategy_without_gatherer_override(self):
        """Test that strategy without gatherer override returns None."""
        strategy = TestStrategyWithoutGatherer()
        mock_ctx = Mock(spec=IStrategyContext)

        gatherer = strategy.gatherer(mock_ctx)

        assert gatherer is None, "Strategy without gatherer override should return None"
