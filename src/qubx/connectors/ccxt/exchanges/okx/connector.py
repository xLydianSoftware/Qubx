"""OKX CcxtConnector subclass.

Adds the OKX-specific behavior on top of the generic ``CcxtConnector``:

- **Split orders/fills streams** (via ``_TwoStreamCcxtConnector``): OKX's
  ``watch_orders`` carries only status, not fills; the fills come on a separate
  ``watch_my_trades`` stream. The base runs both concurrently — status events ride
  with ``fill=None`` and each trade arrives as a ``DealEvent``; the AccountManager
  correlates them by trade id (see the two-stream base docstring).
- **Balance extraction**: ccxt's OKX balance mapping is wrong for the framework — see
  ``_convert_balances``.
- **Venue account figures**: OKX's trading-balance payload carries account-level
  figures (``totalEq`` / ``mgnRatio`` / ``adjEq`` / ``imr`` in ``info.data[0]``) — see
  ``_extract_venue_figures``; AM prefers them per metric over its derived values.
- **make_client_id / cid_framework_prefix**: OKX clOrdId is case-sensitive
  alphanumeric only, 1-32 chars — the underscore in ``qubx_`` is stripped, so origin
  classification keys on the sanitized prefix (``qubx``), derived from the same
  regex the producer uses.

There is no real-time ``watch_balance`` stream — AM's snapshot cadence covers
balance refresh.
"""

import re
from typing import Any

from qubx.core.basics import FRAMEWORK_CID_PREFIX, Balance

from ...utils import info_float
from .._two_stream import _TwoStreamCcxtConnector

_OKX_CLIENT_ID_RE = re.compile(r"[^a-zA-Z0-9]")
_OKX_CLIENT_ID_MAX_LEN = 32


class OkxCcxtConnector(_TwoStreamCcxtConnector):
    """OKX connector: split orders/fills streams + OKX balance/clOrdId rules."""

    # OKX strips "_" from cids (see make_client_id), so framework orders echo back as
    # "qubx..." — classify with the prefix produced by the SAME sanitizing regex, so
    # producer and classifier can never drift. Residual caveat: an external cid that
    # happens to start with "qubx" reads as RECOVERED (unavoidable given the charset).
    cid_framework_prefix = _OKX_CLIENT_ID_RE.sub("", FRAMEWORK_CID_PREFIX)

    def _convert_balances(self, raw_balance: dict[str, Any]) -> list[Balance]:
        """Use OKX ``cashBal``/``frozenBal`` per currency from the raw response.

        ccxt maps OKX's ``eq`` (equity = cashBal + unrealizedPnL) to balance ``total``;
        we want the cash leg, so we read ``cashBal`` (total) and ``frozenBal`` (locked)
        straight from ``info.data[0].details``. Currencies with a zero cash balance are
        skipped.
        """
        details = raw_balance.get("info", {}).get("data", [{}])[0].get("details", [])
        balances: list[Balance] = []
        for detail in details:
            cash_bal = float(detail.get("cashBal", 0) or 0)
            if not cash_bal:
                continue
            frozen_bal = float(detail.get("frozenBal", 0) or 0)
            balances.append(
                Balance(
                    exchange=self.exchange_name,
                    currency=detail["ccy"],
                    free=cash_bal - frozen_bal,
                    locked=frozen_bal,
                    total=cash_bal,
                )
            )
        return balances

    def _extract_venue_figures(self, raw_balance: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
        """OKX account-level figures from ``info.data[0]`` of the trading-balance payload.

        - equity: ``totalEq`` — total account equity. USD-denominated; reported as-is
          against the USDT base (USD≈USDT, a bp-level basis difference).
        - available_margin: ``adjEq − imr`` (adjusted equity minus initial margin
          requirement) — both populated only in multi-currency/portfolio margin modes.
        - margin_ratio: ``mgnRatio`` — same coverage-multiple convention as the derived
          ``AccountState.margin_ratio``, but the venue value is not capped at 100.

        Not-applicable fields arrive as ``""`` → None → AM derives that metric.
        """
        acct = raw_balance.get("info", {}).get("data", [{}])[0]
        equity = info_float(acct, "totalEq")
        margin_ratio = info_float(acct, "mgnRatio")
        adj_eq = info_float(acct, "adjEq")
        imr = info_float(acct, "imr")
        available_margin = adj_eq - imr if adj_eq is not None and imr is not None else None
        return equity, available_margin, margin_ratio

    def make_client_id(self, suggested: str) -> str:
        """OKX clOrdId: case-sensitive alphanumeric only, 1-32 chars.

        Enforce the base ``qubx_`` prefix first, then strip the underscore (and any
        other non-alphanumeric character) and truncate to 32. The ``qubx`` lead
        survives the strip (alphanumeric), and origin classification keys on that
        sanitized form via ``cid_framework_prefix``.
        """
        prefixed = super().make_client_id(suggested)
        sanitized = _OKX_CLIENT_ID_RE.sub("", prefixed)
        sanitized = sanitized[:_OKX_CLIENT_ID_MAX_LEN]
        return sanitized if sanitized else prefixed[:_OKX_CLIENT_ID_MAX_LEN]
