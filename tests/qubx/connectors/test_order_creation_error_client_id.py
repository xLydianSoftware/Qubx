from unittest.mock import Mock

import pandas as pd

from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.connectors.xlighter.broker import LighterBroker
from qubx.core.errors import OrderCreationError


def test_ccxt_order_creation_error_includes_client_id():
    channel = Mock()
    time_provider = Mock()
    time_provider.time.return_value = pd.Timestamp("2024-01-01").asm8

    broker = CcxtBroker(
        exchange_manager=Mock(),
        channel=channel,
        time_provider=time_provider,
        account=Mock(),
        data_provider=Mock(),
    )

    instrument = Mock()
    instrument.symbol = "XRPUSDT"

    broker._post_order_error_to_databus(
        Exception("rejected"),
        instrument,
        "SELL",
        "LIMIT",
        100.0,
        1.0,
        "cid_123",
        "gtx",
    )

    assert channel.send.called
    sent = channel.send.call_args[0][0]
    assert isinstance(sent[2], OrderCreationError)
    assert sent[2].client_id == "cid_123"


def test_lighter_order_creation_error_includes_client_id():
    channel = Mock()
    time_provider = Mock()
    time_provider.time.return_value = pd.Timestamp("2024-01-01").asm8

    broker = LighterBroker(
        client=Mock(),
        ws_manager=Mock(),
        channel=channel,
        time_provider=time_provider,
        account=Mock(),
        data_provider=Mock(),
        loop=Mock(),
    )

    instrument = Mock()
    instrument.symbol = "XRPUSDT"

    broker._post_order_error_to_channel(
        Exception("rejected"),
        instrument,
        "SELL",
        "LIMIT",
        100.0,
        1.0,
        "cid_123",
        "gtx",
    )

    assert channel.send.called
    sent = channel.send.call_args[0][0]
    assert isinstance(sent[2], OrderCreationError)
    assert sent[2].client_id == "cid_123"
