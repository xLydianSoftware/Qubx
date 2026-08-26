"""What the lookup and the venue disagree about must not reach the venue."""

import pytest

from qubx.core.basics import Instrument, MarketType
from qubx.core.instrument_service import reconcile_instruments


def _instr(symbol: str, base: str, market_id: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange="TESTEX",
        base=base,
        quote="USDC",
        settle="USDC",
        exchange_symbol=market_id,
        tick_size=0.01,
        lot_size=0.1,
        min_size=0.1,
    )


def _reconcile(venue, candidates):
    return reconcile_instruments("TESTEX", venue, candidates, key_of=lambda i: int(i.exchange_symbol))


def test_a_market_the_venue_lists_resolves():
    btc = _instr("BTCUSDC", "BTC", "1")
    res = _reconcile({1: "BTC"}, [btc])
    assert res.by_key == {1: btc}
    assert res.dropped == []


def test_a_market_the_venue_does_not_list_is_dropped():
    # the lookup keeps delisted markets; a live connector must not resolve them
    gone = _instr("OLDUSDC", "OLD", "2048")
    res = _reconcile({1: "BTC"}, [gone])
    assert res.by_key == {}
    assert [i for i, _ in res.dropped] == [gone]
    assert "does not list 2048" in res.dropped[0][1]


def test_the_venue_decides_which_of_two_claimants_owns_a_market():
    # market 34 held DATAUSDC and IPUSDC after a rename; the dict kept whichever came last, so
    # fills for one were booked against the other
    data = _instr("DATAUSDC", "DATA", "34")
    ip = _instr("IPUSDC", "IP", "34")
    res = _reconcile({34: "DATA"}, [ip, data])
    assert res.by_key == {34: data}
    assert [i for i, _ in res.dropped] == [ip]
    assert "34 is 'DATA' at the venue, not 'IP'" in res.dropped[0][1]


def test_neither_is_trusted_when_two_claimants_agree_on_the_key_and_the_name():
    # nothing tells them apart, so resolving an inbound event would be a coin flip
    a, b = _instr("XUSDC", "X", "7"), _instr("X2USDC", "X", "7")
    res = _reconcile({7: "X"}, [a, b])
    assert res.by_key == {}
    assert len(res.dropped) == 2


def test_an_instrument_with_no_venue_key_is_dropped():
    res = reconcile_instruments("TESTEX", {1: "BTC"}, [_instr("BTCUSDC", "BTC", "1")], key_of=lambda i: None)
    assert res.by_key == {}
    assert res.dropped[0][1] == "carries no venue key"


def test_an_unreadable_venue_key_drops_that_one_and_keeps_the_rest():
    btc, bad = _instr("BTCUSDC", "BTC", "1"), _instr("BADUSDC", "BAD", "not-a-number")
    res = _reconcile({1: "BTC", 2: "BAD"}, [btc, bad])
    assert res.by_key == {1: btc}
    assert "venue key unreadable" in res.dropped[0][1]


def test_venue_markets_with_no_lookup_entry_are_reported_not_invented():
    res = _reconcile({1: "BTC", 2: "ETH"}, [_instr("BTCUSDC", "BTC", "1")])
    assert res.unmatched_venue_keys == [2]
    assert res.instruments == [_instr("BTCUSDC", "BTC", "1")]


@pytest.mark.parametrize("venue", [{}, {1: "BTC"}])
def test_no_candidates_resolves_nothing(venue):
    res = _reconcile(venue, [])
    assert res.by_key == {} and res.dropped == []
