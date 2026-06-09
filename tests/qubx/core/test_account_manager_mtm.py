import numpy as np

from qubx.core.account_manager import AccountManager, AccountManagerConfig, AccountState
from qubx.core.basics import Instrument, MarketType, Position
from qubx.core.series import Quote


class _T:
    def time(self):
        return np.datetime64("2026-05-28T00:00:00")


def _instrument(symbol="BTCUSDT", exchange="binance") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange=exchange,
        base=symbol.replace("USDT", ""),
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def _am(exchange="binance"):
    am = AccountManager.__new__(AccountManager)
    am._states = {exchange: AccountState(exchange=exchange, base_currency="USDT")}
    am._connectors = {}
    am._cfg = AccountManagerConfig()
    am._time = _T()
    am._liveness_unready_since = {}
    am._applied_funding_buckets = {}
    return am


def _quote(bid, ask):
    return Quote(np.datetime64("2026-05-28T00:00:00").astype("datetime64[ns]").astype(np.int64), bid, ask, 1.0, 1.0)


def test_on_market_quote_updates_position_market_value():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    state.set_position(inst, pos)
    am.on_market_quote(inst, _quote(50_900.0, 51_100.0))
    # mid = 51_000 -> unrealized = 1 * (51000 - 50000) = 1000
    assert pos.last_update_price == 51_000.0
    assert abs(pos.unrealized_pnl() - 1000.0) < 1e-6
    # futures market value tracks unrealized pnl
    assert abs(pos.market_value - 1000.0) < 1e-6


def test_on_market_quote_no_op_when_no_position():
    am = _am()
    inst = _instrument()
    # no position set -> must not raise, must not create a position
    am.on_market_quote(inst, _quote(50_900.0, 51_100.0))
    assert am._states["binance"].get_position(inst) is None


def test_on_market_quote_unknown_exchange_no_op():
    am = _am()

    class _OtherInst:
        exchange = "kraken"

    # state for kraken doesn't exist -> graceful no-op
    am.on_market_quote(_OtherInst(), _quote(1.0, 2.0))
