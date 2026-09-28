"""Binance-safe client order ids (Qubx#437): BINANCE.UM runs the base CcxtConnector,
BINANCE.PM its subclass — both must emit ids matching Binance's newClientOrderId rule."""

import re
from unittest.mock import MagicMock

import pytest

from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.connectors.ccxt.exchanges.binance.connector import BinancePmCcxtConnector
from qubx.core.basics import FRAMEWORK_CID_PREFIX, OrderOrigin, classify_origin

BINANCE_CID_RE = re.compile(r"^[\.A-Z\:/a-z0-9_-]{1,36}$")


@pytest.fixture(params=[CcxtConnector, BinancePmCcxtConnector], ids=["binance.um", "binance.pm"])
def conn(request) -> CcxtConnector:
    return request.param(
        exchange_name="BINANCE.UM",
        channel=MagicMock(),
        time_provider=MagicMock(),
        exchange_manager=MagicMock(),
        data_provider=MagicMock(),
    )


def _assert_binance_safe(cid: str) -> None:
    assert BINANCE_CID_RE.match(cid), cid
    assert cid.startswith(FRAMEWORK_CID_PREFIX)
    assert classify_origin(cid) == OrderOrigin.RECOVERED  # a framework cid echoed back by the venue


def test_non_ascii_symbol_is_made_conforming(conn: CcxtConnector) -> None:
    cid = conn.make_client_id("qubx_币安人生USDT_17903387279")
    _assert_binance_safe(cid)
    assert cid.endswith("_17903387279")
    assert conn.make_client_id("qubx_币安人生USDT_17903387279") == cid


def test_long_ascii_symbol_is_shortened_keeping_counter(conn: CcxtConnector) -> None:
    suggested = "qubx_1000000BOBUSDTLONGNAMEPERP_17903387279123"
    assert len(suggested) > 36
    cid = conn.make_client_id(suggested)
    _assert_binance_safe(cid)
    assert cid.endswith("_17903387279123")
    assert conn.make_client_id(suggested) == cid


def test_distinct_long_symbols_sharing_a_lead_stay_distinct(conn: CcxtConnector) -> None:
    a = conn.make_client_id("qubx_1000000BOBUSDTLONGNAMEPERPA_17903387279")
    b = conn.make_client_id("qubx_1000000BOBUSDTLONGNAMEPERPB_17903387279")
    assert a != b


def test_distinct_non_ascii_symbols_stay_distinct(conn: CcxtConnector) -> None:
    a = conn.make_client_id("qubx_币安人生USDT_17903387279")
    b = conn.make_client_id("qubx_我踏马来了USDT_17903387279")
    _assert_binance_safe(a)
    _assert_binance_safe(b)
    assert a != b


def test_conforming_ids_pass_through_unchanged(conn: CcxtConnector) -> None:
    assert conn.make_client_id("qubx_BTCUSDT_17903387279") == "qubx_BTCUSDT_17903387279"
    assert conn.make_client_id("qubx_convUSDCUSDT1a2b3c4d") == "qubx_convUSDCUSDT1a2b3c4d"


def test_missing_prefix_is_added_then_made_conforming(conn: CcxtConnector) -> None:
    assert conn.make_client_id("BTCUSDT_1") == "qubx_BTCUSDT_1"
    cid = conn.make_client_id("币安人生USDT_17903387279")
    _assert_binance_safe(cid)
    assert cid.endswith("_17903387279")
    assert cid == conn.make_client_id("qubx_币安人生USDT_17903387279")
