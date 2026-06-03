import inspect

from qubx.core.connector import IConnector


def test_iconnector_has_required_methods():
    required = {
        "submit_order", "cancel_order", "update_order",
        "request_order_status", "request_snapshot",
        "connect", "disconnect",
        "is_ws_ready", "reconnect",
        "make_client_id",
    }
    actual = {n for n in dir(IConnector) if not n.startswith("_")}
    assert required <= actual, f"missing: {required - actual}"


def test_cancel_order_accepts_either_id():
    # cancel/update/request_order_status address an order by EITHER id; both params default
    # to None and the connector requires at least one (enforced at call time, not by the sig).
    for name in ("cancel_order", "update_order", "request_order_status"):
        sig = inspect.signature(getattr(IConnector, name))
        for id_param in ("client_order_id", "venue_order_id"):
            assert sig.parameters[id_param].default is None, f"{name}.{id_param}"
