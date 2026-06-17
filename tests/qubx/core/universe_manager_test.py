import numpy as np
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

    # - market_data_manager mock exposes IMarketDataCache via get_market_data_cache()
    cache = mocker.Mock()
    market_data_manager = mocker.Mock()
    market_data_manager.is_instrument_listed.return_value = True
    market_data_manager.get_market_data_cache.return_value = cache

    # - provide a real "now" so _is_market_gone can compare against delist_date
    time_provider = mocker.Mock()
    time_provider.time.return_value = np.datetime64("2026-06-16T00:00:00", "ns")

    from qubx.core.instrument_service import NullInstrumentService

    instrument_service = NullInstrumentService()

    return {
        "context": mocker.Mock(),
        "strategy": mocker.Mock(),
        "cache": cache,
        "market_data_manager": market_data_manager,
        "logging": mocker.Mock(),
        "subscription_manager": mocker.Mock(),
        "trading_manager": mocker.Mock(),
        "time_provider": time_provider,
        "account": mocker.Mock(),
        "position_gathering": mocker.Mock(),
        "delisting_detector": delisting_detector,
        "instrument_service": instrument_service,
    }


@pytest.fixture
def universe_manager(mock_dependencies):
    return UniverseManager(
        context=mock_dependencies["context"],
        strategy=mock_dependencies["strategy"],
        market_data_manager=mock_dependencies["market_data_manager"],
        logging=mock_dependencies["logging"],
        subscription_manager=mock_dependencies["subscription_manager"],
        trading_manager=mock_dependencies["trading_manager"],
        time_provider=mock_dependencies["time_provider"],
        account=mock_dependencies["account"],
        position_gathering=mock_dependencies["position_gathering"],
        delisting_detector=mock_dependencies["delisting_detector"],
        instrument_service=mock_dependencies["instrument_service"],
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

    sol = mocker.Mock(spec=Instrument, symbol="SOLUSDT", min_size=0.0)
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
    mock_dependencies["delisting_detector"].filter_delistings.side_effect = lambda instruments: [
        i for i in instruments if i != delisting_instrument
    ]

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
    mock_dependencies["delisting_detector"].filter_delistings.side_effect = lambda instruments: [
        i for i in instruments if i != delisting_instrument
    ]

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


def _gone_instr(mocker, symbol="TONUSDT"):
    i = mocker.Mock(spec=Instrument, symbol=symbol)
    i.exchange = "OKX.F"
    i.delist_date = None
    i.min_size = 0.001
    return i


def test_set_universe_excludes_gone_instrument(universe_manager, mock_dependencies, mocker):
    mock_dependencies["subscription_manager"].auto_subscribe = True
    live = mocker.Mock(spec=Instrument, symbol="BTCUSDT")
    live.exchange = "OKX.F"
    live.delist_date = None
    live.min_size = 0.001
    gone = _gone_instr(mocker)

    def listed(instr):
        return instr is not gone

    mock_dependencies["market_data_manager"].is_instrument_listed.side_effect = listed
    mock_dependencies["account"].positions = {}

    universe_manager.set_universe([live, gone])

    assert set(universe_manager.instruments) == {live}
    assert gone not in universe_manager.instruments


def test_gone_held_position_is_settled_not_traded(universe_manager, mock_dependencies, mocker):
    gone = _gone_instr(mocker)
    pos = mocker.Mock()
    pos.quantity = 3175.0
    mock_dependencies["account"].positions = {gone: pos}
    mock_dependencies["market_data_manager"].is_instrument_listed.side_effect = lambda i: i is not gone
    mock_dependencies["subscription_manager"].auto_subscribe = True

    universe_manager.set_universe([gone])

    mock_dependencies["account"].settle_position.assert_called_once_with(gone)
    mock_dependencies["position_gathering"].alter_positions.assert_not_called()


def test_remove_instruments_settles_gone_held_position(universe_manager, mock_dependencies, mocker):
    # - gone instrument with a held position, already in the universe (prev_set)
    gone = _gone_instr(mocker)
    pos = mocker.Mock()
    pos.quantity = 3175.0
    universe_manager._instruments = {gone}
    mock_dependencies["account"].positions = {gone: pos}
    mock_dependencies["market_data_manager"].is_instrument_listed.side_effect = lambda i: i is not gone

    # - this is the path the scheduled delisting check uses (NOT set_universe)
    universe_manager.remove_instruments([gone])

    # - gone-branch: settle in place, do NOT trade an exit target
    mock_dependencies["account"].settle_position.assert_called_once_with(gone)
    mock_dependencies["position_gathering"].alter_positions.assert_called_once_with(mock_dependencies["context"], [])
    assert gone not in universe_manager.instruments


def test_future_delist_but_listed_is_not_gone(universe_manager, mock_dependencies, mocker):
    import pandas as pd

    instr = _gone_instr(mocker, symbol="SOLUSDT")
    instr.delist_date = pd.Timestamp("2999-01-01")  # far future
    mock_dependencies["market_data_manager"].is_instrument_listed.return_value = True
    mock_dependencies["account"].positions = {}
    mock_dependencies["subscription_manager"].auto_subscribe = True

    universe_manager.set_universe([instr])

    assert instr in universe_manager.instruments
    mock_dependencies["account"].settle_position.assert_not_called()


def test_gone_held_with_delist_date_settled_before_filter_delistings(universe_manager, mock_dependencies, mocker):
    """A gone instrument that ALSO has a (past) delist_date must still be settled:
    _drop_gone must run before filter_delistings, else the delisting filter strips it
    first and the held position is never settled."""
    import pandas as pd

    gone = _gone_instr(mocker)
    gone.delist_date = pd.Timestamp("2020-01-01")  # already past
    pos = mocker.Mock()
    pos.quantity = 3175.0
    mock_dependencies["account"].positions = {gone: pos}
    mock_dependencies["subscription_manager"].auto_subscribe = True

    # live listing: gone is not listed (authoritative "gone" signal)
    mock_dependencies["market_data_manager"].is_instrument_listed.side_effect = lambda i: i is not gone
    # simulate the REAL DelistingDetector: it strips instruments whose delist_date is set
    mock_dependencies["delisting_detector"].filter_delistings.side_effect = lambda instruments: [
        i for i in instruments if i.delist_date is None
    ]

    universe_manager.set_universe([gone])

    # _drop_gone ran first and settled the gone, still-held instrument
    mock_dependencies["account"].settle_position.assert_called_once_with(gone)
    assert gone not in universe_manager.instruments


def test_filter_blacklisted_noop_with_null_service(universe_manager, mock_dependencies, mocker):
    instruments = [mocker.Mock(spec=Instrument, symbol="BTCUSDT")]
    assert universe_manager._filter_blacklisted(instruments) == instruments


def test_set_universe_drops_blacklisted(mock_dependencies, mocker):
    btc = mocker.Mock(spec=Instrument, symbol="BTCUSDT")
    eth = mocker.Mock(spec=Instrument, symbol="ETHUSDT")
    svc = mocker.Mock()
    svc.is_blacklisted.side_effect = lambda i: i is btc
    deps = dict(mock_dependencies)
    deps["instrument_service"] = svc
    deps["subscription_manager"].auto_subscribe = True
    um = UniverseManager(
        context=deps["context"],
        strategy=deps["strategy"],
        market_data_manager=deps["market_data_manager"],
        logging=deps["logging"],
        subscription_manager=deps["subscription_manager"],
        trading_manager=deps["trading_manager"],
        time_provider=deps["time_provider"],
        account=deps["account"],
        position_gathering=deps["position_gathering"],
        delisting_detector=deps["delisting_detector"],
        instrument_service=svc,
    )
    um.set_universe([btc, eth])
    assert set(um.instruments) == {eth}


def test_add_instruments_drops_blacklisted(mock_dependencies, mocker):
    btc = mocker.Mock(spec=Instrument, symbol="BTCUSDT")
    eth = mocker.Mock(spec=Instrument, symbol="ETHUSDT")
    svc = mocker.Mock()
    svc.is_blacklisted.side_effect = lambda i: i is btc
    deps = dict(mock_dependencies)
    deps["subscription_manager"].auto_subscribe = True
    um = UniverseManager(
        context=deps["context"],
        strategy=deps["strategy"],
        market_data_manager=deps["market_data_manager"],
        logging=deps["logging"],
        subscription_manager=deps["subscription_manager"],
        trading_manager=deps["trading_manager"],
        time_provider=deps["time_provider"],
        account=deps["account"],
        position_gathering=deps["position_gathering"],
        delisting_detector=deps["delisting_detector"],
        instrument_service=svc,
    )
    um.add_instruments([btc, eth])
    assert set(um.instruments) == {eth}


def test_add_instruments_excludes_gone_and_settles(universe_manager, mock_dependencies, mocker):
    mock_dependencies["subscription_manager"].auto_subscribe = True
    live = mocker.Mock(spec=Instrument, symbol="BTCUSDT")
    live.exchange = "OKX.F"
    live.delist_date = None
    live.min_size = 0.001
    gone = _gone_instr(mocker)

    mock_dependencies["market_data_manager"].is_instrument_listed.side_effect = lambda i: i is not gone
    pos = mocker.Mock()
    pos.quantity = 3175.0
    mock_dependencies["account"].positions = {gone: pos}

    universe_manager.add_instruments([live, gone])

    assert live in universe_manager.instruments
    assert gone not in universe_manager.instruments
    mock_dependencies["account"].settle_position.assert_called_once_with(gone)
    mock_dependencies["position_gathering"].alter_positions.assert_not_called()
