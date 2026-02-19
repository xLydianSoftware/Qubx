import pytest

from qubx.core.basics import Instrument, MarketType


class TestParseNotation:
    def test_plain_symbol(self):
        e, mt, s = Instrument.parse_notation("BTCUSDT")
        assert e is None
        assert mt is None
        assert s == "BTCUSDT"

    def test_two_part_notation(self):
        e, mt, s = Instrument.parse_notation("BINANCE.UM:BTCUSDT")
        assert e == "BINANCE.UM"
        assert mt is None
        assert s == "BTCUSDT"

    def test_three_part_swap(self):
        e, mt, s = Instrument.parse_notation("BINANCE.UM:SWAP:BTCUSDT")
        assert e == "BINANCE.UM"
        assert mt == MarketType.SWAP
        assert s == "BTCUSDT"

    def test_three_part_future(self):
        e, mt, s = Instrument.parse_notation("BINANCE.UM:FUTURE:BTCUSD.20250914")
        assert e == "BINANCE.UM"
        assert mt == MarketType.FUTURE
        assert s == "BTCUSD.20250914"

    def test_three_part_spot(self):
        e, mt, s = Instrument.parse_notation("BINANCE:SPOT:BTCUSDT")
        assert e == "BINANCE"
        assert mt == MarketType.SPOT
        assert s == "BTCUSDT"

    def test_three_part_option(self):
        e, mt, s = Instrument.parse_notation("IB:OPTION:AAPL230217P00155000")
        assert e == "IB"
        assert mt == MarketType.OPTION
        assert s == "AAPL230217P00155000"

    def test_three_part_cfd(self):
        e, mt, s = Instrument.parse_notation("IB:CFD:AAPL")
        assert e == "IB"
        assert mt == MarketType.CFD
        assert s == "AAPL"

    def test_three_part_index(self):
        e, mt, s = Instrument.parse_notation("IB:INDEX:SPX")
        assert e == "IB"
        assert mt == MarketType.INDEX
        assert s == "SPX"

    def test_market_type_case_insensitive(self):
        e, mt, s = Instrument.parse_notation("BINANCE.UM:swap:BTCUSDT")
        assert mt == MarketType.SWAP

    def test_invalid_market_type_raises(self):
        with pytest.raises(ValueError, match="Invalid market type"):
            Instrument.parse_notation("BINANCE.UM:INVALID:BTCUSDT")

    def test_too_many_parts_raises(self):
        with pytest.raises(ValueError, match="Invalid instrument notation"):
            Instrument.parse_notation("A:B:C:D")

    def test_all_market_types_parseable(self):
        for mt in MarketType:
            e, parsed_mt, s = Instrument.parse_notation(f"EX:{mt.value}:SYM")
            assert parsed_mt == mt
            assert e == "EX"
            assert s == "SYM"
