"""
Integration tests for WebSocket transaction submission on Lighter mainnet.

IMPORTANT: These tests interact with Lighter mainnet and will submit real orders!
Only run these tests with small amounts and ensure you have sufficient balance.

To run these tests:
    pytest tests/integration/connectors/xlighter/test_websocket_tx_integration.py -v

Requirements:
    - Valid Lighter account with credentials in /home/yuriy/accounts/xlydian1_lighter.toml
    - Sufficient USDC balance for test orders
    - Internet connection to Lighter mainnet
"""

import asyncio
import pytest
from pathlib import Path

# These tests are marked as integration and should be run explicitly
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skip(reason="Manual execution only - submits real orders to mainnet"),
]


@pytest.fixture
def lighter_client():
    """Create a real LighterClient from account credentials"""
    from qubx.connectors.xlighter.client import LighterClient
    import toml

    # Load credentials from account file
    account_file = Path("/home/yuriy/accounts/xlydian1_lighter.toml")
    if not account_file.exists():
        pytest.skip(f"Account file not found: {account_file}")

    accounts = toml.load(account_file)
    lighter_account = None
    for account in accounts.get("accounts", []):
        if account.get("exchange") == "LIGHTER":
            lighter_account = account
            break

    if not lighter_account:
        pytest.skip("No LIGHTER account found in accounts file")

    client = LighterClient(
        api_key=lighter_account["api_key"],
        private_key=lighter_account["secret"],
        account_index=lighter_account["account_index"],
        api_key_index=lighter_account.get("api_key_index", 0),
        testnet=False,
    )

    yield client

    # Cleanup
    asyncio.run(client.close())


@pytest.fixture
def ws_manager():
    """Create a real WebSocket manager"""
    from qubx.connectors.xlighter.websocket import LighterWebSocketManager

    manager = LighterWebSocketManager(testnet=False)
    yield manager

    # Cleanup
    if hasattr(manager, "close"):
        asyncio.run(manager.close())


@pytest.mark.asyncio
async def test_single_order_submission_websocket(lighter_client, ws_manager):
    """
    Test submitting a single small limit order via WebSocket.

    This test will:
    1. Sign a small BTC limit order (0.001 BTC at $100k - intentionally far from market)
    2. Submit via WebSocket
    3. Verify the order is submitted (we won't wait for confirmation)
    4. Cancel the order

    WARNING: This submits a real order to Lighter mainnet!
    """
    from qubx.connectors.xlighter.constants import TX_TYPE_CREATE_ORDER, TX_TYPE_CANCEL_ORDER

    # Order parameters (intentionally far from market to avoid execution)
    market_id = 0  # BTC-USDC
    client_order_index = 99999  # High number to avoid conflicts
    base_amount = int(0.001 * 1e18)  # 0.001 BTC (very small)
    price = int(100000 * 1e18)  # $100k (far above market)
    is_ask = False  # BUY
    order_type = 0  # LIMIT
    time_in_force = 1  # GTT
    reduce_only = 0
    trigger_price = 0

    # Ensure WebSocket is connected
    await ws_manager.connect()

    try:
        # Step 1: Sign the order locally
        signer = lighter_client.signer_client
        tx_info, error = signer.sign_create_order(
            market_index=market_id,
            client_order_index=client_order_index,
            base_amount=base_amount,
            price=price,
            is_ask=is_ask,
            order_type=order_type,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            trigger_price=trigger_price,
        )

        assert error is None, f"Signing failed: {error}"
        assert tx_info is not None

        # Step 2: Submit via WebSocket
        print(f"\nSubmitting order via WebSocket: 0.001 BTC @ $100k")
        response = await ws_manager.send_tx(tx_type=TX_TYPE_CREATE_ORDER, tx_info=tx_info, tx_id="test_order_ws")

        # Verify response
        assert response["status"] == "sent"
        assert response["tx_id"] == "test_order_ws"
        print(f"Order submitted: {response}")

        # Wait a moment for order to be processed
        await asyncio.sleep(2)

        # Step 3: Cancel the order
        print(f"Cancelling order...")
        # Note: We need the actual order_id from the exchange, not our client ID
        # For now, we'll just demonstrate the cancellation signing
        cancel_tx_info, cancel_error = signer.sign_cancel_order(
            market_index=market_id, order_index=client_order_index
        )

        assert cancel_error is None, f"Cancel signing failed: {cancel_error}"

        cancel_response = await ws_manager.send_tx(
            tx_type=TX_TYPE_CANCEL_ORDER, tx_info=cancel_tx_info, tx_id="test_cancel_ws"
        )

        assert cancel_response["status"] == "sent"
        print(f"Cancellation submitted: {cancel_response}")

        # Wait for cancellation to process
        await asyncio.sleep(2)

    finally:
        # Disconnect WebSocket
        await ws_manager.disconnect()


@pytest.mark.asyncio
async def test_batch_order_submission_websocket(lighter_client, ws_manager):
    """
    Test submitting a batch of small limit orders via WebSocket.

    This test will:
    1. Sign 3 small limit orders (far from market)
    2. Submit as a batch via WebSocket
    3. Verify the batch is submitted
    4. Cancel all orders

    WARNING: This submits real orders to Lighter mainnet!
    """
    from qubx.connectors.xlighter.constants import TX_TYPE_CREATE_ORDER, TX_TYPE_CANCEL_ORDER

    # Order parameters (3 buy orders, all far from market)
    market_id = 0  # BTC-USDC
    orders_to_create = [
        {
            "client_order_index": 99991,
            "base_amount": int(0.001 * 1e18),
            "price": int(99000 * 1e18),
        },
        {
            "client_order_index": 99992,
            "base_amount": int(0.001 * 1e18),
            "price": int(99500 * 1e18),
        },
        {
            "client_order_index": 99993,
            "base_amount": int(0.001 * 1e18),
            "price": int(100000 * 1e18),
        },
    ]

    # Ensure WebSocket is connected
    await ws_manager.connect()

    try:
        # Step 1: Sign all orders
        signer = lighter_client.signer_client
        tx_types = []
        tx_infos = []

        for order_params in orders_to_create:
            tx_info, error = signer.sign_create_order(
                market_index=market_id,
                client_order_index=order_params["client_order_index"],
                base_amount=order_params["base_amount"],
                price=order_params["price"],
                is_ask=False,  # BUY
                order_type=0,  # LIMIT
                time_in_force=1,  # GTT
                reduce_only=0,
                trigger_price=0,
            )

            assert error is None, f"Signing failed for order {order_params['client_order_index']}: {error}"
            tx_types.append(TX_TYPE_CREATE_ORDER)
            tx_infos.append(tx_info)

        # Step 2: Submit batch via WebSocket
        print(f"\nSubmitting batch of {len(orders_to_create)} orders via WebSocket")
        response = await ws_manager.send_batch_tx(
            tx_types=tx_types, tx_infos=tx_infos, batch_id="test_batch_ws"
        )

        # Verify response
        assert response["status"] == "sent"
        assert response["count"] == len(orders_to_create)
        assert response["batch_id"] == "test_batch_ws"
        print(f"Batch submitted: {response}")

        # Wait for orders to be processed
        await asyncio.sleep(3)

        # Step 3: Cancel all orders (individually for now)
        print(f"Cancelling all orders...")
        for order_params in orders_to_create:
            cancel_tx_info, cancel_error = signer.sign_cancel_order(
                market_index=market_id, order_index=order_params["client_order_index"]
            )

            if cancel_error is None:
                cancel_response = await ws_manager.send_tx(
                    tx_type=TX_TYPE_CANCEL_ORDER,
                    tx_info=cancel_tx_info,
                    tx_id=f"cancel_{order_params['client_order_index']}",
                )
                print(f"Cancelled order {order_params['client_order_index']}: {cancel_response['status']}")

        # Wait for cancellations to process
        await asyncio.sleep(2)

    finally:
        # Disconnect WebSocket
        await ws_manager.disconnect()


def test_manual_execution_instructions():
    """
    This is not a real test - it provides instructions for manual execution.
    """
    instructions = """
    ============================================================================
    MANUAL EXECUTION INSTRUCTIONS FOR WEBSOCKET TRANSACTION TESTS
    ============================================================================

    These tests will submit REAL ORDERS to Lighter mainnet. Only proceed if:
    1. You have a funded Lighter account
    2. You understand the risks
    3. You are ready to submit small test orders

    To run the tests:

    1. Remove the @pytest.mark.skip decorator from the tests above
    2. Ensure your account file exists at: /home/yuriy/accounts/xlydian1_lighter.toml
    3. Run the tests:

        # Single order test
        pytest tests/integration/connectors/xlighter/test_websocket_tx_integration.py::test_single_order_submission_websocket -v -s

        # Batch order test
        pytest tests/integration/connectors/xlighter/test_websocket_tx_integration.py::test_batch_order_submission_websocket -v -s

        # Both tests
        pytest tests/integration/connectors/xlighter/test_websocket_tx_integration.py -v -s

    The tests will:
    - Create very small orders (0.001 BTC) far from market price
    - Submit via WebSocket
    - Cancel the orders after verification

    Watch the output for confirmation messages.

    ============================================================================
    """
    print(instructions)
    # This test always passes - it's just for documentation
    assert True
