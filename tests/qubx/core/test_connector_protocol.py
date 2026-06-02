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


def test_cancel_order_requires_client_order_id():
    # client_order_id is always present (synthesized as ext:<venue_id> for external
    # orders), so it is a required positional; venue_order_id stays optional.
    sig = inspect.signature(IConnector.cancel_order)
    cid = sig.parameters["client_order_id"]
    assert cid.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert cid.default is inspect.Parameter.empty
    assert sig.parameters["venue_order_id"].default is None
