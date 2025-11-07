"""Constants and enums for Lighter exchange connector"""

from enum import IntEnum, StrEnum

from lighter.signer_client import SignerClient

# Import constants from SignerClient for consistency
# Transaction types
TX_TYPE_CHANGE_PUB_KEY = SignerClient.TX_TYPE_CHANGE_PUB_KEY
TX_TYPE_CREATE_SUB_ACCOUNT = SignerClient.TX_TYPE_CREATE_SUB_ACCOUNT
TX_TYPE_CREATE_PUBLIC_POOL = SignerClient.TX_TYPE_CREATE_PUBLIC_POOL
TX_TYPE_UPDATE_PUBLIC_POOL = SignerClient.TX_TYPE_UPDATE_PUBLIC_POOL
TX_TYPE_TRANSFER = SignerClient.TX_TYPE_TRANSFER
TX_TYPE_WITHDRAW = SignerClient.TX_TYPE_WITHDRAW
TX_TYPE_CREATE_ORDER = SignerClient.TX_TYPE_CREATE_ORDER
TX_TYPE_CANCEL_ORDER = SignerClient.TX_TYPE_CANCEL_ORDER
TX_TYPE_CANCEL_ALL_ORDERS = SignerClient.TX_TYPE_CANCEL_ALL_ORDERS
TX_TYPE_MODIFY_ORDER = SignerClient.TX_TYPE_MODIFY_ORDER
TX_TYPE_MINT_SHARES = SignerClient.TX_TYPE_MINT_SHARES
TX_TYPE_BURN_SHARES = SignerClient.TX_TYPE_BURN_SHARES
TX_TYPE_UPDATE_LEVERAGE = SignerClient.TX_TYPE_UPDATE_LEVERAGE

# Order types
ORDER_TYPE_LIMIT = SignerClient.ORDER_TYPE_LIMIT
ORDER_TYPE_MARKET = SignerClient.ORDER_TYPE_MARKET
ORDER_TYPE_STOP_LOSS = SignerClient.ORDER_TYPE_STOP_LOSS
ORDER_TYPE_STOP_LOSS_LIMIT = SignerClient.ORDER_TYPE_STOP_LOSS_LIMIT
ORDER_TYPE_TAKE_PROFIT = SignerClient.ORDER_TYPE_TAKE_PROFIT
ORDER_TYPE_TAKE_PROFIT_LIMIT = SignerClient.ORDER_TYPE_TAKE_PROFIT_LIMIT
ORDER_TYPE_TWAP = SignerClient.ORDER_TYPE_TWAP

# Time in force (using SignerClient's naming convention)
ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL = SignerClient.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL
ORDER_TIME_IN_FORCE_GOOD_TILL_TIME = SignerClient.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME
ORDER_TIME_IN_FORCE_POST_ONLY = SignerClient.ORDER_TIME_IN_FORCE_POST_ONLY

# Order expiry defaults
DEFAULT_28_DAY_ORDER_EXPIRY = SignerClient.DEFAULT_28_DAY_ORDER_EXPIRY
DEFAULT_IOC_EXPIRY = SignerClient.DEFAULT_IOC_EXPIRY

# Margin modes
CROSS_MARGIN_MODE = SignerClient.CROSS_MARGIN_MODE
ISOLATED_MARGIN_MODE = SignerClient.ISOLATED_MARGIN_MODE

# Other constants
USDC_SCALE = SignerClient.USDC_TICKER_SCALE

# Qubx-specific WebSocket constants
WS_CHANNEL_ORDER_BOOK = "order_book"
WS_CHANNEL_TRADE = "trade"
WS_CHANNEL_MARKET_STATS = "market_stats"
WS_CHANNEL_ACCOUNT_ALL = "account_all"
WS_CHANNEL_USER_STATS = "user_stats"
WS_CHANNEL_EXECUTED_TX = "executed_transaction"

WS_RESUBSCRIBE_DELAY = 5.0  # seconds to wait before resubscribing

# WebSocket message types
WS_MSG_TYPE_CONNECTED = "connected"
WS_MSG_TYPE_SUBSCRIBE = "subscribe"
WS_MSG_TYPE_UNSUBSCRIBE = "unsubscribe"
WS_MSG_TYPE_SUBSCRIBED_ORDER_BOOK = "subscribed/order_book"
WS_MSG_TYPE_UPDATE_ORDER_BOOK = "update/order_book"
WS_MSG_TYPE_SUBSCRIBED_ACCOUNT = "subscribed/account_all"
WS_MSG_TYPE_UPDATE_ACCOUNT = "update/account_all"
WS_MSG_TYPE_TRADE = "trade"

# API endpoints
API_BASE_MAINNET = "https://mainnet.zklighter.elliot.ai"
API_BASE_TESTNET = "https://testnet.zklighter.elliot.ai"

WS_BASE_MAINNET = "wss://mainnet.zklighter.elliot.ai/stream"
WS_BASE_TESTNET = "wss://testnet.zklighter.elliot.ai/stream"

DEFAULT_PING_INTERVAL = None
DEFAULT_PING_TIMEOUT = None
DEFAULT_MAX_RETRIES = 999999  # Effectively infinite retries for websocket reconnection
DEFAULT_MAX_SIZE = None
DEFAULT_MAX_QUEUE = 5000


# Enums for type safety (kept for backward compatibility)
class LighterOrderType(IntEnum):
    """Lighter order types"""

    LIMIT = 0
    MARKET = 1
    STOP_LOSS = 2
    STOP_LOSS_LIMIT = 3
    TAKE_PROFIT = 4
    TAKE_PROFIT_LIMIT = 5
    TWAP = 6


class LighterTimeInForce(IntEnum):
    """Lighter time in force values"""

    IOC = 0  # Immediate or Cancel
    GTT = 1  # Good Till Time
    POST_ONLY = 2  # Post Only


class LighterOrderSide(StrEnum):
    """Lighter order side values"""

    BUY = "B"
    SELL = "S"


class LighterMarginMode(IntEnum):
    """Lighter margin modes"""

    CROSS = 0
    ISOLATED = 1


# Channel types that support multiplexing (multiple symbols on one connection)
MULTIPLEX_CHANNELS = {
    WS_CHANNEL_ORDER_BOOK,
    WS_CHANNEL_TRADE,
    WS_CHANNEL_MARKET_STATS,
}

# Channel types that need dedicated connections
DEDICATED_CHANNELS = {
    WS_CHANNEL_ACCOUNT_ALL,
    WS_CHANNEL_USER_STATS,
    WS_CHANNEL_EXECUTED_TX,
}
