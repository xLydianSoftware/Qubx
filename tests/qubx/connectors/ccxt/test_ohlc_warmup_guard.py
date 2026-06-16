import asyncio

from pytest_mock import MockerFixture

from qubx.connectors.ccxt.handlers.ohlc import OhlcDataHandler
from qubx.core.basics import Instrument


def _handler(mocker: MockerFixture, listed_map, fetch_side_effect):
    data_provider = mocker.Mock()
    data_provider.is_instrument_listed.side_effect = lambda i: listed_map[i]
    data_provider._get_exch_timeframe.return_value = "1d"
    data_provider._time_msec_nbars_back.return_value = 0
    # - real warmup calls _update_quote -> _last_quotes[instrument] = ...; needs a real dict
    data_provider._last_quotes = {}

    exchange = mocker.Mock()
    exchange.fetch_ohlcv = mocker.AsyncMock(side_effect=fetch_side_effect)
    exchange_manager = mocker.Mock()
    exchange_manager.exchange = exchange

    return OhlcDataHandler(data_provider=data_provider, exchange_manager=exchange_manager, exchange_id="okx"), exchange


def _instr(mocker, symbol):
    i = mocker.Mock(spec=Instrument)
    i.symbol = symbol
    # - attributes needed by instrument_to_ccxt_symbol (called inside warmup)
    i.base = symbol.replace("USDT", "")
    i.quote = "USDT"
    i.settle = "USDT"
    i.is_futures.return_value = False
    return i


def test_warmup_skips_unlisted_instrument(mocker: MockerFixture):
    listed = _instr(mocker, "BTCUSDT")
    gone = _instr(mocker, "TONUSDT")
    channel = mocker.Mock()

    async def fetch(symbol, timeframe, since, limit):
        return [[0, 1.0, 2.0, 0.5, 1.5, 100.0]]

    handler, exchange = _handler(mocker, {listed: True, gone: False}, fetch)
    mocker.patch.object(handler, "_convert_ohlcv_to_bar", side_effect=lambda oh: oh)
    mocker.patch.object(handler, "_convert_ohlcv_to_quote", return_value=object())

    asyncio.run(handler.warmup({listed, gone}, channel, warmup_period="1d", timeframe="1d"))
    # at least the listed instrument was fetched; gone was skipped (no crash)
    assert exchange.fetch_ohlcv.await_count >= 1


def test_warmup_continues_when_one_instrument_raises(mocker: MockerFixture):
    a = _instr(mocker, "AAAUSDT")
    b = _instr(mocker, "BBBUSDT")
    channel = mocker.Mock()

    from ccxt import BadSymbol

    async def fetch(symbol, timeframe, since, limit):
        if fetch.calls == 0:
            fetch.calls += 1
            raise BadSymbol("okx does not have market symbol AAA/USDT:USDT")
        return [[0, 1.0, 2.0, 0.5, 1.5, 100.0]]

    fetch.calls = 0
    handler, exchange = _handler(mocker, {a: True, b: True}, fetch)
    mocker.patch.object(handler, "_convert_ohlcv_to_bar", side_effect=lambda oh: oh)
    mocker.patch.object(handler, "_convert_ohlcv_to_quote", return_value=object())

    # must NOT raise even though one instrument failed
    asyncio.run(handler.warmup({a, b}, channel, warmup_period="1d", timeframe="1d"))
    assert exchange.fetch_ohlcv.await_count >= 2
