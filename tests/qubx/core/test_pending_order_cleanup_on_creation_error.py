from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import OrderRequest
from qubx.core.lookups import lookup
from qubx.health.dummy import DummyHealthMonitor
from tests.qubx.core.utils_test import DummyTimeProvider


def test_remove_order_clears_synthetic_pending_order_and_request() -> None:
    time_provider = DummyTimeProvider()
    exchange = "BINANCE.UM"
    instrument = lookup.find_symbol(exchange, "BTCUSDT")
    assert instrument is not None

    account = BasicAccountProcessor(
        account_id="test",
        time_provider=time_provider,
        base_currency="USDT",
        health_monitor=DummyHealthMonitor(),
        exchange=exchange,
    )

    client_id = "client_123"
    req = OrderRequest(
        instrument=instrument,
        quantity=1.0,
        price=100.0,
        order_type="LIMIT",
        side="BUY",
        time_in_force="gtc",
        client_id=client_id,
    )

    account.process_order_request(req)

    orders = account.get_orders(instrument=instrument)
    assert client_id in orders
    assert orders[client_id].status == "PENDING"
    assert client_id in account._pending_order_requests

    # Simulate order creation failure correlation cleanup
    account.remove_order(client_id)

    orders = account.get_orders(instrument=instrument)
    assert client_id not in orders
    assert client_id not in account._pending_order_requests
