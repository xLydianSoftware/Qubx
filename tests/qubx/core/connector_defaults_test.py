import pytest

from qubx.core.connector import ChannelEmitter


def test_default_connector_declares_no_moves_and_refuses_to_move():
    emitter = ChannelEmitter()
    assert emitter.wallet_moves() == []
    with pytest.raises(NotImplementedError):
        emitter.move_funds("USDT", "futures_um", "margin")
