from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from qubx.transfers import xchanges as xchanges_module
from qubx.transfers.xchanges import (
    _EXECUTE_GRACE_DELAYS_S,
    DEFAULT_WALLET_MAPPING,
    TransferServiceError,
    XChangesTransferService,
)

BASE = "http://svc/api"
TS_MS = 1750000000000


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(xchanges_module.time, "sleep", lambda s: recorded.append(s))
    return recorded


def _response(
    json_body: dict, status_code: int = 200, method: str = "POST", url: str = "http://test"
) -> httpx.Response:
    # request= is required: without it raise_for_status() raises RuntimeError, not HTTPStatusError
    return httpx.Response(status_code, json=json_body, request=httpx.Request(method, url))


def _service(client, **kwargs) -> XChangesTransferService:
    defaults: dict = {"base_url": BASE, "provider": "XLYDIAN", "user": "bohdan"}
    defaults.update(kwargs)
    return XChangesTransferService(client=client, **defaults)


def _progress(status: str, transfer_id: str = "T1", **extra) -> dict:
    payload = {
        "transferId": transfer_id,
        "status": status,
        "timestamp": TS_MS,
        "balanceUnit": {"amount": 1000.0, "assetId": "USDC"},
        "direction": "BINANCE-SUB → HYPERLIQUID",
    }
    payload.update(extra)
    return payload


class TestTransferFunds:
    def test_transfer_funds_quotes_then_executes(self):
        client = MagicMock()
        client.post.side_effect = [
            _response({"values": []}),
            _response({"transferId": "T1", "status": "PENDING_CONFIRMATION"}),
            _response({}),
        ]
        svc = _service(client)

        tid = svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 1000.0)

        assert tid == "T1"
        calls = client.post.call_args_list
        assert calls[1].args[0] == f"{BASE}/transfers/quote"
        assert calls[1].kwargs["json"] == {
            "type": "BOT",
            "userId": {"provider": "XLYDIAN", "username": "bohdan"},
            "srcWalletId": "BINANCE-SUB",
            "srcWalletCustodian": "BINANCE",
            "dstWalletId": "HYPERLIQUID",
            "dstWalletCustodian": "HYPERLIQUID",
            "assetId": "USDC",
            "policy": {"type": "BOT", "chains": ["ARBITRUM"]},
            "amount": 1000.0,
            "startIndex": 0,
        }
        assert calls[2].args[0] == f"{BASE}/transfers/execute"
        assert calls[2].kwargs["json"] == {"transferId": "T1", "confirm": True}

    def test_amount_rounded_to_two_decimals(self):
        client = MagicMock()
        client.post.side_effect = [_response({"transferId": "T1"}), _response({})]
        svc = _service(client, single_flight=False)

        svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 12.00000001)

        quote_payload = client.post.call_args_list[0].kwargs["json"]
        assert quote_payload["amount"] == 12.0

    def test_non_positive_amount_raises_value_error(self):
        client = MagicMock()
        svc = _service(client)
        with pytest.raises(ValueError, match="must be positive"):
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 0)
        with pytest.raises(ValueError, match="must be positive"):
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", -5)
        client.post.assert_not_called()

    def test_sub_cent_amount_rounds_to_zero_and_raises(self):
        client = MagicMock()
        svc = _service(client)
        with pytest.raises(ValueError, match="must be positive"):
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 0.004)
        client.post.assert_not_called()

    @pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_amount_raises_value_error(self, amount):
        client = MagicMock()
        svc = _service(client, max_amount=100.0)
        with pytest.raises(ValueError, match="must be finite"):
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", amount)
        client.post.assert_not_called()

    def test_half_cent_amount_rounds_up_and_passes(self):
        client = MagicMock()
        client.post.side_effect = [_response({"transferId": "T1"}), _response({})]
        svc = _service(client, single_flight=False)

        svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 0.006)

        quote_payload = client.post.call_args_list[0].kwargs["json"]
        assert quote_payload["amount"] == 0.01

    def test_unsupported_currency_raises_value_error(self):
        client = MagicMock()
        svc = _service(client)
        with pytest.raises(ValueError, match="Unsupported currency 'BTC'"):
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "BTC", 10.0)
        client.post.assert_not_called()

    def test_lowercase_currency_accepted(self):
        client = MagicMock()
        client.post.side_effect = [_response({"transferId": "T1"}), _response({})]
        svc = _service(client, single_flight=False)

        svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "usdc", 10.0)

        quote_payload = client.post.call_args_list[0].kwargs["json"]
        assert quote_payload["assetId"] == "USDC"

    def test_max_amount_cap_raises_value_error(self):
        client = MagicMock()
        svc = _service(client, max_amount=100.0)
        with pytest.raises(ValueError, match="exceeds configured max_amount"):
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 150.0)
        client.post.assert_not_called()

    def test_unmapped_exchange_raises_value_error(self):
        client = MagicMock()
        svc = _service(client)
        with pytest.raises(ValueError, match="BINANCE.UM"):
            svc.transfer_funds("KRAKEN", "HYPERLIQUID", "USDC", 10.0)
        client.post.assert_not_called()

    def test_simulation_guard_raises(self):
        client = MagicMock()
        svc = _service(client, is_simulation=lambda: True)
        with pytest.raises(RuntimeError, match="simulation"):
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0)
        client.post.assert_not_called()
        client.get.assert_not_called()


class TestSingleFlight:
    @pytest.mark.parametrize("blocking_status", ["CONFIRMED", "IN_PROGRESS"])
    def test_single_flight_blocks_inflight_transfer(self, blocking_status):
        client = MagicMock()
        client.post.return_value = _response({"values": [{"transferId": "OLD", "status": blocking_status}]})
        svc = _service(client)

        with pytest.raises(TransferServiceError, match="already in flight") as exc_info:
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0)

        assert exc_info.value.transfer_id == "OLD"
        assert client.post.call_count == 1  # quote never reached
        assert client.post.call_args.args[0] == f"{BASE}/transfers/latest"

    def test_single_flight_allows_terminal_and_orphaned_quotes(self):
        client = MagicMock()
        client.post.side_effect = [
            _response(
                {
                    "values": [
                        {"transferId": "A", "status": "COMPLETED"},
                        {"transferId": "B", "status": "FAILED"},
                        {"transferId": "C", "status": "PENDING_CONFIRMATION"},
                    ]
                }
            ),
            _response({"transferId": "T1"}),
            _response({}),
        ]
        svc = _service(client)

        assert svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0) == "T1"
        assert client.post.call_count == 3

    def test_single_flight_disabled_skips_latest(self):
        client = MagicMock()
        client.post.side_effect = [_response({"transferId": "T1"}), _response({})]
        svc = _service(client, single_flight=False)

        svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0)

        urls = [c.args[0] for c in client.post.call_args_list]
        assert f"{BASE}/transfers/latest" not in urls
        assert urls == [f"{BASE}/transfers/quote", f"{BASE}/transfers/execute"]

    def _svc_with_tracked_pending(self) -> tuple[MagicMock, XChangesTransferService]:
        client = MagicMock()
        client.post.side_effect = [_response({"values": []}), _response({"transferId": "T1"}), _response({})]
        svc = _service(client)
        svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0)
        return client, svc

    def test_single_flight_probes_tracked_pending_paged_out_of_latest(self):
        client, svc = self._svc_with_tracked_pending()
        # T1 pushed out of the 10-row /transfers/latest page but still in flight server-side
        client.post.side_effect = [_response({"values": [{"transferId": "NEW", "status": "COMPLETED"}]})]
        client.get.return_value = _response(_progress("IN_PROGRESS"), method="GET")

        with pytest.raises(TransferServiceError, match="already in flight") as exc_info:
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 20.0)

        assert exc_info.value.transfer_id == "T1"

    def test_single_flight_marks_not_found_pending_as_failed_and_proceeds(self):
        client, svc = self._svc_with_tracked_pending()
        client.post.side_effect = [_response({"values": []}), _response({"transferId": "T2"}), _response({})]
        client.get.return_value = _response(
            {"jbstMessageOnClient": "Invalid UUID string: T1"}, status_code=500, method="GET"
        )

        assert svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 20.0) == "T2"
        assert svc._transfers["T1"]["status"] == "failed"

    def test_single_flight_pending_probe_hiccup_does_not_block(self):
        client, svc = self._svc_with_tracked_pending()
        client.post.side_effect = [_response({"values": []}), _response({"transferId": "T2"}), _response({})]
        client.get.return_value = _response({"jbstMessageOnClient": "oops"}, status_code=500, method="GET")

        assert svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 20.0) == "T2"
        assert svc._transfers["T1"]["status"] == "pending"


class TestWalletMapping:
    def test_valid_spec_parses_and_uppercases_exchange(self):
        svc = _service(MagicMock(), wallets={"binance.um": "BINANCE-SUB:BINANCE"})
        assert svc._wallets == {"BINANCE.UM": ("BINANCE-SUB", "BINANCE")}

    def test_venue_alias_canonicalized(self):
        svc = _service(MagicMock(), wallets={"BINANCE.PM": "BINANCE-SUB:BINANCE"})
        assert svc._wallets == {"BINANCE.UM": ("BINANCE-SUB", "BINANCE")}

    def test_identical_specs_for_aliases_collapse(self):
        svc = _service(MagicMock(), wallets={"BINANCE.PM": "BINANCE-SUB:BINANCE", "BINANCE.UM": "BINANCE-SUB:BINANCE"})
        assert svc._wallets == {"BINANCE.UM": ("BINANCE-SUB", "BINANCE")}

    def test_conflicting_specs_for_same_canonical_exchange_raise(self):
        with pytest.raises(ValueError, match="Conflicting wallet specs"):
            _service(MagicMock(), wallets={"BINANCE.PM": "SUB-A:BINANCE", "BINANCE.UM": "SUB-B:BINANCE"})

    def test_missing_custodian_raises(self):
        with pytest.raises(ValueError, match="WALLET_ID:CUSTODIAN"):
            _service(MagicMock(), wallets={"BINANCE.UM": "BINANCE-SUB"})

    def test_none_uses_default_mapping(self):
        svc = _service(MagicMock())
        assert svc._wallets == DEFAULT_WALLET_MAPPING
        assert svc._wallets is not DEFAULT_WALLET_MAPPING

    def test_duplicate_wallet_id_across_exchanges_raises(self):
        with pytest.raises(ValueError, match="Duplicate wallet id"):
            _service(MagicMock(), wallets={"BINANCE.UM": "SHARED:BINANCE", "HYPERLIQUID": "SHARED:HYPERLIQUID"})


class TestQuoteExecuteFailures:
    def test_quote_failure_raises_no_retry(self):
        client = MagicMock()
        client.post.return_value = _response({"jbstMessageOnClient": "insufficient balance"}, status_code=500)
        svc = _service(client, single_flight=False)

        with pytest.raises(TransferServiceError, match="insufficient balance"):
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0)

        assert client.post.call_count == 1

    def test_execute_timeout_reconciles_to_success(self, sleeps):
        client = MagicMock()
        client.post.side_effect = [_response({"transferId": "T1"}), httpx.TimeoutException("timed out")]
        client.get.return_value = _response(_progress("CONFIRMED"), method="GET")
        svc = _service(client, single_flight=False)

        assert svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0) == "T1"
        assert sleeps == list(_EXECUTE_GRACE_DELAYS_S[:1])
        assert client.get.call_count == 1

    def test_execute_timeout_grace_polls_until_landed(self, sleeps):
        client = MagicMock()
        client.post.side_effect = [_response({"transferId": "T1"}), httpx.TimeoutException("timed out")]
        client.get.side_effect = [
            _response(_progress("PENDING_CONFIRMATION"), method="GET"),
            _response(_progress("IN_PROGRESS"), method="GET"),
        ]
        svc = _service(client, single_flight=False)

        assert svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0) == "T1"
        assert sleeps == list(_EXECUTE_GRACE_DELAYS_S[:2])
        urls = [c.args[0] for c in client.post.call_args_list]
        assert f"{BASE}/transfers/T1/stop" not in urls
        assert "T1" in svc._transfers

    def test_execute_timeout_still_pending_stops_quote_and_raises(self, sleeps):
        client = MagicMock()
        client.post.side_effect = [
            _response({"transferId": "T1"}),
            httpx.TimeoutException("timed out"),
            _response({}),  # stop
        ]
        client.get.return_value = _response(_progress("PENDING_CONFIRMATION"), method="GET")
        svc = _service(client, single_flight=False)

        with pytest.raises(TransferServiceError, match="quote abandoned") as exc_info:
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0)

        assert exc_info.value.transfer_id == "T1"
        assert sleeps == list(_EXECUTE_GRACE_DELAYS_S)
        assert client.post.call_args_list[-1].args[0] == f"{BASE}/transfers/T1/stop"
        assert client.get.call_count == len(_EXECUTE_GRACE_DELAYS_S) + 1  # grace probes + final post-stop probe
        assert "T1" not in svc._transfers

    def test_execute_timeout_stop_loses_race_returns_success(self, sleeps):
        client = MagicMock()
        client.post.side_effect = [
            _response({"transferId": "T1"}),
            httpx.TimeoutException("timed out"),
            _response({"jbstMessageOnClient": "no worker"}, status_code=500),  # stop errors are swallowed
        ]
        pending = _response(_progress("PENDING_CONFIRMATION"), method="GET")
        client.get.side_effect = [pending] * len(_EXECUTE_GRACE_DELAYS_S) + [
            _response(_progress("CONFIRMED"), method="GET")
        ]
        svc = _service(client, single_flight=False)

        assert svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0) == "T1"
        assert "T1" in svc._transfers

    def test_execute_timeout_failed_probe_raises_with_reason(self, sleeps):
        client = MagicMock()
        client.post.side_effect = [_response({"transferId": "T1"}), httpx.TimeoutException("timed out")]
        steps = {"values": [{"action": "WITHDRAW", "trace": "insufficient gas"}]}
        client.get.return_value = _response(_progress("FAILED", steps=steps), method="GET")
        svc = _service(client, single_flight=False)

        with pytest.raises(TransferServiceError, match="insufficient gas") as exc_info:
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0)

        assert exc_info.value.transfer_id == "T1"

    def test_execute_timeout_unknown_outcome_records_pending_and_raises(self, sleeps):
        client = MagicMock()
        client.post.side_effect = [_response({"transferId": "T1"}), httpx.TimeoutException("timed out")]
        client.get.side_effect = httpx.ConnectError("refused")
        svc = _service(client, single_flight=False)

        with pytest.raises(TransferServiceError, match="UNKNOWN") as exc_info:
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0)

        assert exc_info.value.transfer_id == "T1"
        assert svc._transfers["T1"]["status"] == "pending"
        assert sleeps == list(_EXECUTE_GRACE_DELAYS_S)
        urls = [c.args[0] for c in client.post.call_args_list]
        assert f"{BASE}/transfers/T1/stop" not in urls

    def test_execute_rejection_reconciles_once_without_grace(self, sleeps):
        client = MagicMock()
        client.post.side_effect = [
            _response({"transferId": "T1"}),
            _response({"jbstMessageOnClient": "boom"}, status_code=500),
        ]
        client.get.return_value = _response(_progress("PENDING_CONFIRMATION"), method="GET")
        svc = _service(client, single_flight=False)

        with pytest.raises(TransferServiceError, match="quote abandoned") as exc_info:
            svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0)

        assert exc_info.value.transfer_id == "T1"
        assert sleeps == []
        assert client.get.call_count == 1
        urls = [c.args[0] for c in client.post.call_args_list]
        assert f"{BASE}/transfers/T1/stop" not in urls


class TestGetTransferStatus:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("PENDING_CONFIRMATION", "pending"),
            ("CONFIRMED", "pending"),
            ("IN_PROGRESS", "pending"),
            ("COMPLETED", "completed"),
            ("FAILED", "failed"),
        ],
    )
    def test_get_transfer_status_status_mapping(self, raw, expected):
        client = MagicMock()
        client.get.return_value = _response(_progress(raw), method="GET")
        svc = _service(client)

        record = svc.get_transfer_status("T1")

        assert record["status"] == expected
        assert record["raw_status"] == raw

    def test_get_transfer_status_returns_sim_schema(self):
        client = MagicMock()
        client.get.return_value = _response(_progress("COMPLETED"), method="GET")
        svc = _service(client)

        record = svc.get_transfer_status("T1")

        sim_keys = {"transaction_id", "timestamp", "from_exchange", "to_exchange", "currency", "amount", "status"}
        assert sim_keys | {"raw_status", "failure_reason"} <= set(record)
        assert record["transaction_id"] == "T1"
        assert record["timestamp"] == np.datetime64(TS_MS, "ms")
        assert record["currency"] == "USDC"
        assert record["amount"] == 1000.0

    @pytest.mark.parametrize(
        "message",
        [
            # what the service actually emits: Hibernate proxy miss and UUID parse failure
            "Unable to find tech1.xchanges.server.transfers.domain.postgres.TransfersPostgresTransfer"
            " with id 7ef0b2f6-8c1d-4c6a-9d3e-000000000000",
            "Invalid UUID string: sim_1",
            "No row with the given identifier exists",
            "Transfer not found",  # forward-compat should the service fix its lookup
        ],
    )
    def test_get_transfer_status_unknown_id_raises_value_error(self, message):
        client = MagicMock()
        client.get.return_value = _response({"jbstMessageOnClient": message}, status_code=500, method="GET")
        svc = _service(client)

        with pytest.raises(ValueError, match="Transfer not found: nope"):
            svc.get_transfer_status("nope")

    def test_get_transfer_status_reverse_maps_direction(self):
        client = MagicMock()
        client.get.return_value = _response(_progress("IN_PROGRESS"), method="GET")
        svc = _service(client)

        record = svc.get_transfer_status("T1")  # no local record
        assert (record["from_exchange"], record["to_exchange"]) == ("BINANCE.UM", "HYPERLIQUID")

        client.get.return_value = _response(_progress("IN_PROGRESS", direction="FOO → BAR"), method="GET")
        record = svc.get_transfer_status("T1")
        assert (record["from_exchange"], record["to_exchange"]) == ("FOO", "BAR")

    def test_get_transfer_status_failure_reason_from_steps(self):
        client = MagicMock()
        steps = {"values": [{"action": "WITHDRAW", "trace": "insufficient gas", "stepStatus": "FAILED"}]}
        client.get.return_value = _response(_progress("FAILED", steps=steps), method="GET")
        svc = _service(client)

        record = svc.get_transfer_status("T1")

        assert record["status"] == "failed"
        assert "WITHDRAW" in record["failure_reason"]
        assert "insufficient gas" in record["failure_reason"]

    def test_status_get_retries_once_on_connect_error(self):
        client = MagicMock()
        client.get.side_effect = [httpx.ConnectError("refused"), _response(_progress("COMPLETED"), method="GET")]
        svc = _service(client)

        record = svc.get_transfer_status("T1")

        assert record["status"] == "completed"
        assert client.get.call_count == 2


class TestGetTransfers:
    def _svc_with_pending(self) -> tuple[MagicMock, XChangesTransferService]:
        client = MagicMock()
        client.post.side_effect = [_response({"transferId": "T1"}), _response({})]
        svc = _service(client, single_flight=False)
        svc.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 1000.0)
        return client, svc

    def test_get_transfers_refreshes_pending(self):
        client, svc = self._svc_with_pending()
        client.get.return_value = _response(_progress("COMPLETED"), method="GET")

        transfers = svc.get_transfers()

        assert transfers["T1"]["status"] == "completed"
        assert transfers["T1"] is not svc._transfers["T1"]  # copies returned

    def test_get_transfers_swallows_refresh_errors(self):
        client, svc = self._svc_with_pending()
        # two errors: _get_progress retries once before giving up
        client.get.side_effect = [httpx.ConnectError("refused"), httpx.ConnectError("refused")]

        transfers = svc.get_transfers()

        assert transfers["T1"]["status"] == "pending"
