"""OKX CcxtConnector subclass (commit 4b).

Ports the OKX-specific behavior the old ``OkxAccountProcessor`` / ``OkxCcxtBroker``
carried, on top of the generic ``CcxtConnector``:

- **Split orders/fills streams** (via ``_TwoStreamExecutionsMixin``): OKX's
  ``watch_orders`` carries only status, not fills; the fills come on a separate
  ``watch_my_trades`` stream. The mixin runs both concurrently and promotes to
  FILLED carrying the last deal (dedup-safe — see the mixin docstring).
- **Balance extraction**: ccxt maps OKX's ``eq`` (= cashBal + unrealizedPnL) onto
  the canonical ``total``; the framework wants the cash leg, so we read per-currency
  ``cashBal`` / ``frozenBal`` from the raw ``info.data[0].details`` instead.
- **make_client_id**: OKX clOrdId is case-sensitive alphanumeric only, 1-32 chars.

Note on the old ``totalEq`` / ``mgnRatio`` extraction: those fed the OLD account
processor's own equity/margin-ratio computation. In the new architecture Account
Manager derives total capital and margin ratio itself from per-currency balances +
position PnL (``_total_capital_for`` / ``get_margin_ratio``), so there is no snapshot
slot for an account-level equity figure — supplying the correct per-currency
``cashBal`` balances is what AM needs. The real-time ``watch_balance`` stream that
fed ``totalEq`` is therefore not ported; AM's snapshot cadence covers balance
refresh.
"""

import re
from typing import Any

from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.core.basics import Balance

from .._two_stream import _TwoStreamExecutionsMixin

_OKX_CLIENT_ID_RE = re.compile(r"[^a-zA-Z0-9]")
_OKX_CLIENT_ID_MAX_LEN = 32


class OkxCcxtConnector(_TwoStreamExecutionsMixin, CcxtConnector):
    """OKX connector: split orders/fills streams + OKX balance/clOrdId rules."""

    def _convert_balances(self, raw_balance: dict[str, Any]) -> list[Balance]:
        """Use OKX ``cashBal``/``frozenBal`` per currency from the raw response.

        ccxt maps OKX's ``eq`` (equity = cashBal + unrealizedPnL) to balance ``total``;
        we want the cash leg, so we read ``cashBal`` (total) and ``frozenBal`` (locked)
        straight from ``info.data[0].details``. Currencies with a zero cash balance are
        skipped (ports ``OkxAccountProcessor._update_balance``).
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

    def make_client_id(self, suggested: str) -> str:
        """OKX clOrdId: case-sensitive alphanumeric only, 1-32 chars.

        Enforce the base ``qubx_`` prefix first, then strip the underscore (and any
        other non-alphanumeric character) and truncate to 32. The ``qubx`` lead
        survives the strip (alphanumeric), preserving the origin marker as far as the
        venue allows. Caveat (ported as-is from the old ``OkxCcxtBroker``): the
        downstream origin check (`ccxt_convert_order_info`) keys on the literal
        ``qubx_`` *with* the underscore, which OKX bans — so OKX framework orders are
        mis-detected as EXTERNAL. That is a pre-existing venue limitation, not changed
        here.
        """
        prefixed = super().make_client_id(suggested)
        sanitized = _OKX_CLIENT_ID_RE.sub("", prefixed)
        sanitized = sanitized[:_OKX_CLIENT_ID_MAX_LEN]
        return sanitized if sanitized else prefixed[:_OKX_CLIENT_ID_MAX_LEN]
