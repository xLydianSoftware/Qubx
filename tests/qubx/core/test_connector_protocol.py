import inspect

from qubx.core.connector import IConnector


def test_iconnector_has_required_methods():
    required = {
        "submit_order", "cancel_order", "update_order",
        "request_order_status", "request_snapshot",
        "connect", "disconnect", "send",
        "is_ws_ready", "force_ws_reconnect_sync",
        "make_client_id",
    }
    actual = {n for n in dir(IConnector) if not n.startswith("_")}
    assert required <= actual, f"missing: {required - actual}"


def test_cancel_order_keyword_only():
    sig = inspect.signature(IConnector.cancel_order)
    assert sig.parameters["client_order_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["venue_order_id"].kind is inspect.Parameter.KEYWORD_ONLY
