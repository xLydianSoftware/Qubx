import re

from qubx.connectors.ccxt.broker import CcxtBroker

_OKX_CLIENT_ID_RE = re.compile(r"[^a-zA-Z0-9]")
_OKX_CLIENT_ID_MAX_LEN = 32


class OkxCcxtBroker(CcxtBroker):
    """OKX-specific broker.

    OKX clOrdId format: case-sensitive alphanumeric only, 1-32 characters.
    """

    def make_client_id(self, client_id: str) -> str:
        # Strip any character that isn't alphanumeric; truncate to 32 characters
        sanitized = _OKX_CLIENT_ID_RE.sub("", client_id)
        return sanitized[:_OKX_CLIENT_ID_MAX_LEN] if sanitized else client_id[:_OKX_CLIENT_ID_MAX_LEN]
