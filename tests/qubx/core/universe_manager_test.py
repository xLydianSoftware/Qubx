import pytest
from pytest_mock import MockerFixture

from qubx.core.basics import DataType, Instrument
from qubx.core.mixins.universe import UniverseManager


@pytest.fixture
def mock_dependencies(mocker: MockerFixture):
    # Create a mock delisting detector that by default returns all instruments (no filtering)
    delisting_detector = mocker.Mock()
    delisting_detector.filter_delistings.side_effect = lambda instruments: instruments
    delisting_detector.detect_delistings.return_value = []

    return {
        "context": mocker.Mock(),
        "strategy": mocker.Mock(),
        "cache": mocker.Mock(),
        "logging": mocker.Mock(),
        "subscription_manager": mocker.Mock(),
        "trading_manager": mocker.Mock(),
        "time_provider": mocker.Mock(),
        "account": mocker.Mock(),
        "position_gathering": mocker.Mock(),
        "delisting_detector": delisting_detector,
    }


@pytest.fixture
def universe_manager(mock_dependencies):
    return UniverseManager(
        context=mock_dependencies["context"],
        strategy=mock_dependencies["strategy"],
        cache=mock_dependencies["cache"],
        logging=mock_dependencies["logging"],
        subscription_manager=mock_dependencies["subscription_manager"],
        trading_manager=mock_dependencies["trading_manager"],
        time_provider=mock_dependencies["time_provider"],
        account=mock_dependencies["account"],
        position_gathering=mock_dependencies["position_gathering"],
        delisting_detector=mock_dependencies["delisting_detector"],
    )


def test_set_universe_adds_new_instruments(universe_manager, mock_dependencies, mocker: MockerFixture):
    instruments = list(
        set(
            [
                mocker.Mock(spec=Instrument, symbol="BTCUSDT"),
                mocker.Mock(spec=Instrument, symbol="ETHUSDT"),
            ]
        )
    )
    mock_dependencies["subscription_manager"].auto_subscribe = True

    universe_manager.set_universe(instruments)

    assert set(universe_manager.instruments) == set(instruments)
    mock_dependencies["subscription_manager"].subscribe.assert_called_once()
    mock_dependencies["subscription_manager"].commit.assert_called_once()
    mock_dependencies["strategy"].on_universe_change.assert_called_once_with(
        mock_dependencies["context"], instruments, []
    )


def test_set_universe_removes_instruments(universe_manager, mock_dependencies, mocker: MockerFixture):
    initial_instruments = [
        mocker.Mock(spec=Instrument, symbol="BTCUSDT"),
        mocker.Mock(spec=Instrument, symbol="ETHUSDT"),
    ]
    new_instruments = [initial_instruments[0]]  # Remove second instrument

    universe_manager._instruments = set(initial_instruments)
    mock_dependencies["account"].positions = {}

    universe_manager.set_universe(new_instruments)

    assert universe_manager.instruments == new_instruments
    mock_dependencies["subscription_manager"].unsubscribe.assert_called_with(DataType.ALL, initial_instruments[1])
    mock_dependencies["strategy"].on_universe_change.assert_called_once_with(
        mock_dependencies["context"], [], [initial_instruments[1]]
    )


def test_set_universe_with_skip_callback(universe_manager, mock_dependencies, mocker: MockerFixture):
    instruments = [mocker.Mock(spec=Instrument)]
    universe_manager.set_universe(instruments, skip_callback=True)
    mock_dependencies["strategy"].on_universe_change.assert_not_called()


def test_set_universe_with_position_close(universe_manager, mock_dependencies, mocker: MockerFixture):
    ctx = mock_dependencies["context"]

    sol = (mocker.Mock(spec=Instrument, symbol="SOLUSDT", min_size=0.0),)
    universe_manager.set_universe(
        [
            btc := mocker.Mock(spec=Instrument, symbol="BTCUSDT", min_size=0.0),
            eth := mocker.Mock(spec=Instrument, symbol="ETHUSDT", min_size=0.0),
            ltc := mocker.Mock(spec=Instrument, symbol="LTCUSDT", min_size=0.0),
        ]
    )
    mock_dependencies["account"].positions = {btc: mocker.Mock(quantity=1.0)}

    # - set new universe with close policy
    universe_manager.set_universe([eth, ltc, sol], if_has_position_then="close")

    # - should close position through alter_positions
    mock_dependencies["position_gathering"].alter_positions.assert_called_once_with(ctx, [btc.target(ctx, 0)])

    assert set(universe_manager.instruments) == set([eth, ltc, sol])


def test_set_universe_with_position_wait_for_close(universe_manager, mock_dependencies, mocker: MockerFixture):
    ctx = mock_dependencies["context"]
    account = mock_dependencies["account"]
    strategy = mock_dependencies["strategy"]

    sol = mocker.Mock(spec=Instrument, symbol="SOLUSDT", min_size=0.0)
    universe_manager.set_universe(
        [
            btc := mocker.Mock(spec=Instrument, symbol="BTCUSDT", min_size=0.0),
            eth := mocker.Mock(spec=Instrument, symbol="ETHUSDT", min_size=0.0),
            ltc := mocker.Mock(spec=Instrument, symbol="LTCUSDT", min_size=0.0),
        ]
    )
    # - set position for btc
    account.positions = {btc: mocker.Mock(quantity=1.0)}

    # - set new universe with wait for close policy
    universe_manager.set_universe([eth, ltc, sol], if_has_position_then="wait_for_close")

    assert set(universe_manager.instruments) == set([btc, eth, ltc, sol])
    strategy.on_universe_change.assert_any_call(ctx, [sol], [])

    # - emulate position close - it should remove btc from universe
    account.positions = {btc: mocker.Mock(quantity=0.0)}
    universe_manager.on_alter_position(btc)
    strategy.on_universe_change.assert_any_call(ctx, [], [btc])

    assert set(universe_manager.instruments) == set([eth, ltc, sol])
    assert universe_manager._removal_queue == {}


def test_set_universe_with_position_wait_for_change(universe_manager, mock_dependencies, mocker: MockerFixture):
    ctx = mock_dependencies["context"]
    account = mock_dependencies["account"]
    strategy = mock_dependencies["strategy"]

    sol = mocker.Mock(spec=Instrument, symbol="SOLUSDT", min_size=0.0)
    universe_manager.set_universe(
        [
            btc := mocker.Mock(spec=Instrument, symbol="BTCUSDT", min_size=0.0),
            eth := mocker.Mock(spec=Instrument, symbol="ETHUSDT", min_size=0.0),
            ltc := mocker.Mock(spec=Instrument, symbol="LTCUSDT", min_size=0.0),
        ]
    )
    # - set position for btc
    account.positions = {btc: mocker.Mock(quantity=1.0)}

    # - set new universe with wait for close policy
    universe_manager.set_universe([eth, ltc, sol], if_has_position_then="wait_for_change")

    assert set(universe_manager.instruments) == set([btc, eth, ltc, sol])

    strategy.on_universe_change.assert_any_call(ctx, [sol], [])

    # universe_manager.on_alter_position(btc)
    assert universe_manager.is_trading_allowed(btc) is False

    # - emulate position close - it should remove btc from universe
    account.positions = {btc: mocker.Mock(quantity=0.0)}

    mock_dependencies["position_gathering"].alter_positions.assert_called_once_with(ctx, [btc.target(ctx, 0)])

    strategy.on_universe_change.assert_any_call(ctx, [], [btc])

    assert set(universe_manager.instruments) == set([eth, ltc, sol])
    assert universe_manager._removal_queue == {}


def test_set_universe_filters_delisting_instruments(universe_manager, mock_dependencies, mocker: MockerFixture):
    """Test that delisting filter is applied to instruments being added to universe."""
    btc = mocker.Mock(spec=Instrument, symbol="BTCUSDT", min_size=0.0)
    eth = mocker.Mock(spec=Instrument, symbol="ETHUSDT", min_size=0.0)
    delisting_instrument = mocker.Mock(spec=Instrument, symbol="SOLUSDT", min_size=0.0)

    # Configure the mock detector to filter out the delisting instrument
    mock_dependencies["delisting_detector"].filter_delistings.side_effect = (
        lambda instruments: [i for i in instruments if i != delisting_instrument]
    )

    instruments = [btc, eth, delisting_instrument]
    universe_manager.set_universe(instruments)

    # Delisting detector should have been called
    mock_dependencies["delisting_detector"].filter_delistings.assert_called_once()

    # Only non-delisting instruments should be in universe
    assert set(universe_manager.instruments) == set([btc, eth])
    assert delisting_instrument not in universe_manager.instruments


def test_set_universe_filters_delisting_even_with_skip_callback(
    universe_manager, mock_dependencies, mocker: MockerFixture
):
    """Test that delisting filter applies even when skip_callback=True."""
    btc = mocker.Mock(spec=Instrument, symbol="BTCUSDT", min_size=0.0)
    delisting_instrument = mocker.Mock(spec=Instrument, symbol="ETHUSDT", min_size=0.0)

    # Configure the mock detector to filter out the delisting instrument
    mock_dependencies["delisting_detector"].filter_delistings.side_effect = (
        lambda instruments: [i for i in instruments if i != delisting_instrument]
    )

    instruments = [btc, delisting_instrument]
    universe_manager.set_universe(instruments, skip_callback=True)

    # Delisting detector should have been called even with skip_callback=True
    mock_dependencies["delisting_detector"].filter_delistings.assert_called_once()

    # Only non-delisting instruments should be in universe
    assert set(universe_manager.instruments) == set([btc])
    assert delisting_instrument not in universe_manager.instruments


def test_set_universe_keeps_non_delisting_instruments(universe_manager, mock_dependencies, mocker: MockerFixture):
    """Test that instruments without delist dates pass through the filter."""
    instruments = [
        mocker.Mock(spec=Instrument, symbol="BTCUSDT", min_size=0.0),
        mocker.Mock(spec=Instrument, symbol="ETHUSDT", min_size=0.0),
    ]

    # Default mock behavior is to return all instruments (no filtering)
    universe_manager.set_universe(instruments)

    # All instruments should be in the universe
    assert set(universe_manager.instruments) == set(instruments)
    mock_dependencies["delisting_detector"].filter_delistings.assert_called_once()
