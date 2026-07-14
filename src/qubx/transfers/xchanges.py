"""Live transfer manager: HTTP client for the tech1-xchanges transfer service.

Strategy usage (call, persist, poll)::

    txid = ctx.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 1000.0)
    # persist txid in strategy state immediately, then poll:
    status = ctx.get_transfer_status(txid)
    if status["status"] in ("completed", "failed"):
        ...  # terminal; on "failed" see status["failure_reason"]

Status vocabulary is exactly {"pending", "completed", "failed"}; ``raw_status`` carries the
service enum (PENDING_CONFIRMATION/CONFIRMED/IN_PROGRESS/COMPLETED/FAILED).

Error contract: ``ValueError`` = caller mistake (bad amount, over cap, unmapped exchange,
unknown txid). ``TransferServiceError`` without ``transfer_id`` = nothing was submitted, safe
to retry later; with ``transfer_id`` = a transfer may exist server-side — resolve it via
``get_transfer_status(e.transfer_id)`` before any retry.

Mode behavior: paper runs get a SimulationTransferManager rehearsal against simulated balances;
``read_only`` disables transfers entirely (RuntimeError on call). The service transfers stablecoins
only — which is why amounts are rounded to 2dp — and the transferred amount must be strictly below
the source's available USDC+USDT sum: the service rejects a full-balance sweep.
"""

import math
import time
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np

from qubx import logger
from qubx.core.interfaces import ITransferManager
from qubx.core.mixins.utils import canonical_exchange

# canonical exchange -> (walletId, custodian); BINANCE.PM canonicalizes to BINANCE.UM upstream
DEFAULT_WALLET_MAPPING: dict[str, tuple[str, str]] = {
    "BINANCE.UM": ("BINANCE-SUB", "BINANCE"),
    "HYPERLIQUID": ("HYPERLIQUID", "HYPERLIQUID"),
}

# per-direction canonical assets in the service's TransferDirections; stablecoins only
_SUPPORTED_CURRENCIES = {"USDC", "USDT"}

_STATUS_MAP = {"COMPLETED": "completed", "FAILED": "failed"}  # everything else -> "pending"
# PENDING_CONFIRMATION excluded: unexecuted quotes never move funds and never expire
_BLOCKING_STATUSES = {"CONFIRMED", "IN_PROGRESS"}
_LANDED_STATUSES = ("CONFIRMED", "IN_PROGRESS", "COMPLETED")
# unknown-txid 500 messages the service actually emits (Hibernate proxy miss / UUID parse);
# "not found" kept for forward-compatibility should the service fix its lookup
_NOT_FOUND_MARKERS = ("not found", "unable to find", "invalid uuid", "no row with the given identifier")
# grace-poll schedule after an ambiguous execute failure (server may still be processing)
_EXECUTE_GRACE_DELAYS_S: tuple[float, ...] = (2.0, 5.0, 15.0, 30.0, 60.0)


def _parse_wallet_specs(wallets: dict[str, str]) -> dict[str, tuple[str, str]]:
    parsed: dict[str, tuple[str, str]] = {}
    for exchange, spec in wallets.items():
        wallet_id, _, custodian = spec.partition(":")
        if not wallet_id or not custodian:
            raise ValueError(f"Invalid wallet spec for '{exchange}': {spec!r} (expected 'WALLET_ID:CUSTODIAN')")
        # strategies pass canonical names (BINANCE.PM -> BINANCE.UM), so key by canonical exchange
        key = canonical_exchange(exchange)
        entry = (wallet_id, custodian)
        if key in parsed and parsed[key] != entry:
            raise ValueError(
                f"Conflicting wallet specs for exchange '{key}' (from '{exchange}'): {parsed[key]} vs {entry}"
            )
        parsed[key] = entry
    return parsed


class TransferServiceError(RuntimeError):
    """Transfer service call failed. Carries transfer_id when a quote already exists;
    ambiguous=True means no response was received and the server may still process the request."""

    def __init__(self, message: str, transfer_id: str | None = None, ambiguous: bool = False):
        super().__init__(message)
        self.transfer_id = transfer_id
        self.ambiguous = ambiguous


class XChangesTransferService(ITransferManager):
    """Executes real transfers through tech1-xchanges-server-transfers.

    Quote + execute happen in one synchronous call (quotes don't survive service
    restarts); the transfer itself runs async server-side (2.5-30+ min) and the
    strategy polls get_transfer_status() to a terminal state.
    """

    def __init__(
        self,
        base_url: str,
        user: str,
        provider: str = "XLYDIAN",
        wallets: dict[str, str] | None = None,  # exchange -> "WALLET_ID:CUSTODIAN"; None -> DEFAULT_WALLET_MAPPING
        max_amount: float | None = None,
        single_flight: bool = True,
        timeout_s: float = 10.0,
        is_simulation: Callable[[], bool] | None = None,
        client: httpx.Client | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._user_id = {"provider": provider, "username": user}
        self._wallets = _parse_wallet_specs(wallets) if wallets is not None else dict(DEFAULT_WALLET_MAPPING)
        self._max_amount = max_amount
        self._single_flight = single_flight
        self._is_simulation = is_simulation
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)
        self._transfers: dict[str, dict[str, Any]] = {}
        # walletId -> exchange, for reverse-mapping service responses
        self._wallet_to_exchange: dict[str, str] = {}
        for exch, (w_id, _) in self._wallets.items():
            if w_id in self._wallet_to_exchange:
                raise ValueError(
                    f"Duplicate wallet id '{w_id}' mapped to both '{self._wallet_to_exchange[w_id]}' and '{exch}'"
                )
            self._wallet_to_exchange[w_id] = exch

    def transfer_funds(self, from_exchange: str, to_exchange: str, currency: str, amount: float) -> str:
        # hard backstop: framework layers should make this unreachable; firing = bug alarm
        if self._is_simulation is not None and self._is_simulation():
            raise RuntimeError("live transfer attempted in simulation/warmup context")
        if currency.upper() not in _SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency '{currency}'; supported: {sorted(_SUPPORTED_CURRENCIES)}")
        if not math.isfinite(amount):
            raise ValueError(f"Transfer amount must be finite, got {amount}")
        amount = round(amount, 2)  # service payloads carry 2dp amounts
        if amount <= 0:
            raise ValueError(f"Transfer amount must be positive after rounding to 2dp, got {amount}")
        if self._max_amount is not None and amount > self._max_amount:
            raise ValueError(f"Transfer amount {amount} exceeds configured max_amount {self._max_amount}")
        src_wallet, src_custodian = self._resolve_wallet(from_exchange)
        dst_wallet, dst_custodian = self._resolve_wallet(to_exchange)

        if self._single_flight:
            self._assert_no_transfer_in_flight()

        transfer_id = self._quote(src_wallet, src_custodian, dst_wallet, dst_custodian, currency, amount)
        record = {
            "transaction_id": transfer_id,
            "timestamp": np.datetime64("now", "ms"),
            "from_exchange": from_exchange,
            "to_exchange": to_exchange,
            "currency": currency.upper(),
            "amount": amount,
            "status": "pending",
        }
        self._execute(transfer_id, record)

        self._transfers[transfer_id] = record
        logger.info(
            f"[XChangesTransferService] :: {amount:.2f} {currency} {from_exchange} -> {to_exchange} ({transfer_id})"
        )
        return transfer_id

    def get_transfer_status(self, transaction_id: str) -> dict[str, Any]:
        payload = self._get_progress(transaction_id)
        raw_status = payload.get("status", "")
        balance_unit = payload.get("balanceUnit") or {}
        local = self._transfers.get(transaction_id)
        from_exchange, to_exchange = self._resolve_direction(payload, local)
        record = {
            "transaction_id": transaction_id,
            "timestamp": np.datetime64(int(payload["timestamp"]), "ms"),
            "from_exchange": from_exchange,
            "to_exchange": to_exchange,
            "currency": str(balance_unit.get("assetId", local["currency"] if local else "")),
            "amount": float(balance_unit.get("amount", local["amount"] if local else 0.0)),
            "status": _STATUS_MAP.get(raw_status, "pending"),
            "raw_status": raw_status,
            "failure_reason": self._extract_failure_reason(payload) if raw_status == "FAILED" else None,
        }
        if local is not None:
            local.update(record)
        return record

    def get_transfers(self) -> dict[str, dict[str, Any]]:
        # this process's transfers only; the service DB is the ledger
        # snapshot: the control-server thread may call this while the strategy thread inserts
        for tid, rec in list(self._transfers.items()):
            if rec.get("status") == "pending":
                try:
                    self.get_transfer_status(tid)
                except Exception as e:
                    logger.warning(f"[XChangesTransferService] :: refresh of {tid} failed: {e}")
        return {tid: dict(rec) for tid, rec in list(self._transfers.items())}

    def _resolve_wallet(self, exchange: str) -> tuple[str, str]:
        wallet = self._wallets.get(exchange)
        if wallet is None:
            raise ValueError(f"No wallet mapping for exchange '{exchange}'; supported: {sorted(self._wallets)}")
        return wallet

    def _resolve_direction(self, payload: dict[str, Any], local: dict[str, Any] | None) -> tuple[str, str]:
        if local is not None:
            return local["from_exchange"], local["to_exchange"]
        # service serializes direction as "SRC-WALLET → DST-WALLET"
        direction = str(payload.get("direction", ""))
        src, _, dst = direction.partition("→")
        src, dst = src.strip(), dst.strip()
        return self._wallet_to_exchange.get(src, src), self._wallet_to_exchange.get(dst, dst)

    @staticmethod
    def _extract_failure_reason(payload: dict[str, Any]) -> str | None:
        steps = (payload.get("steps") or {}).get("values") or []
        for step in reversed(steps):
            if step.get("trace"):
                return f"{step.get('action')}: {step['trace']}"
        return None

    def _assert_no_transfer_in_flight(self) -> None:
        payload = self._post("/transfers/latest", {"userId": self._user_id})
        for progress in payload.get("values", []):
            if progress.get("status") in _BLOCKING_STATUSES:
                raise TransferServiceError(
                    f"transfer already in flight: {progress.get('transferId')}",
                    transfer_id=str(progress.get("transferId")),
                )
        # /transfers/latest caps at the 10 newest rows; probe tracked pending transfers by id
        for tid, rec in list(self._transfers.items()):
            if rec.get("status") != "pending":
                continue
            try:
                status = self._get_progress(tid).get("status")
            except ValueError:
                rec["status"] = "failed"  # unknown to the service; stop tracking as pending
                continue
            except TransferServiceError:
                continue  # probe hiccup must not block; the page scan already passed
            rec["status"] = _STATUS_MAP.get(status, "pending")
            if status in _BLOCKING_STATUSES:
                raise TransferServiceError(f"transfer already in flight: {tid}", transfer_id=tid)

    def _quote(
        self, src_wallet: str, src_custodian: str, dst_wallet: str, dst_custodian: str, currency: str, amount: float
    ) -> str:
        payload = {
            "type": "BOT",
            "userId": self._user_id,
            "srcWalletId": src_wallet,
            "srcWalletCustodian": src_custodian,
            "dstWalletId": dst_wallet,
            "dstWalletCustodian": dst_custodian,
            "assetId": currency.upper(),
            "policy": {"type": "BOT", "chains": ["ARBITRUM"]},
            "amount": amount,
            "startIndex": 0,
        }
        # no retry on the money path; orphaned quotes are inert
        response = self._post("/transfers/quote", payload)
        transfer_id = response.get("transferId")
        if not transfer_id:
            raise TransferServiceError(f"quote returned no transferId: {response}")
        return str(transfer_id)

    def _execute(self, transfer_id: str, record: dict[str, Any]) -> None:
        try:
            self._post("/transfers/execute", {"transferId": transfer_id, "confirm": True})
        except TransferServiceError as e:
            if e.ambiguous:
                self._reconcile_ambiguous_execute(transfer_id, record)
            else:
                self._reconcile_rejected_execute(transfer_id)

    def _reconcile_rejected_execute(self, transfer_id: str) -> None:
        # server responded: processing finished, one probe settles the outcome
        try:
            status = self._get_progress(transfer_id).get("status")
        except Exception:
            status = None
        if status in _LANDED_STATUSES:
            logger.warning(f"[XChangesTransferService] :: execute of {transfer_id} errored but landed ({status})")
            return
        raise TransferServiceError(
            f"execute failed for transfer {transfer_id} (status={status}); quote abandoned",
            transfer_id=transfer_id,
        )

    def _reconcile_ambiguous_execute(self, transfer_id: str, record: dict[str, Any]) -> None:
        # no response: the stalled server thread may still commit the execute after our timeout
        status = None
        for delay in _EXECUTE_GRACE_DELAYS_S:
            time.sleep(delay)
            try:
                payload = self._get_progress(transfer_id)
            except TransferServiceError:
                continue
            status = payload.get("status")
            if status in _LANDED_STATUSES:
                logger.warning(f"[XChangesTransferService] :: execute of {transfer_id} errored but landed ({status})")
                return
            if status == "FAILED":
                raise TransferServiceError(
                    f"transfer {transfer_id} failed: {self._extract_failure_reason(payload)}",
                    transfer_id=transfer_id,
                )
        if status is None:
            # every probe failed: outcome unknowable; keep tracking so the caller can resolve it
            self._transfers[transfer_id] = record
            raise TransferServiceError(
                f"execute outcome for transfer {transfer_id} is UNKNOWN (service unreachable); "
                f"poll get_transfer_status('{transfer_id}') to a settled state before retrying",
                transfer_id=transfer_id,
            )
        # still PENDING_CONFIRMATION: abandon the quote so the stalled execute can't land later
        try:
            self._post(f"/transfers/{transfer_id}/stop", {})
        except Exception:
            pass  # service 500s on missing/finished workers
        try:
            status = self._get_progress(transfer_id).get("status")
        except Exception:
            status = None
        if status in _LANDED_STATUSES:
            logger.warning(f"[XChangesTransferService] :: stop of {transfer_id} lost the race; landed ({status})")
            return
        raise TransferServiceError(
            f"execute failed for transfer {transfer_id} (status={status}); quote abandoned",
            transfer_id=transfer_id,
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            resp = self._client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except httpx.HTTPStatusError as e:
            raise TransferServiceError(f"POST {path} -> {e.response.status_code}: {_service_message(e.response)}")
        except httpx.HTTPError as e:
            # timeout/transport error: no response, the server may still process the request
            raise TransferServiceError(f"POST {path} failed: {e}", ambiguous=True)

    def _get_progress(self, transfer_id: str, _retried: bool = False) -> dict[str, Any]:
        url = f"{self._base_url}/transfers/{transfer_id}"
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            message = _service_message(e.response)
            # service 500s unknown ids; match sim's ValueError
            if any(m in message.lower() for m in _NOT_FOUND_MARKERS):
                raise ValueError(f"Transfer not found: {transfer_id}")
            raise TransferServiceError(f"GET /transfers/{transfer_id} -> {e.response.status_code}: {message}")
        except httpx.HTTPError as e:
            if not _retried:  # status reads (not money-path) may retry once
                return self._get_progress(transfer_id, _retried=True)
            raise TransferServiceError(f"GET /transfers/{transfer_id} failed: {e}")


def _service_message(response: httpx.Response) -> str:
    # JbstExceptionResponse carries the human message in "jbstMessageOnClient"
    try:
        body = response.json()
        return str(body.get("jbstMessageOnClient") or body.get("message") or response.text)
    except Exception:
        return response.text
