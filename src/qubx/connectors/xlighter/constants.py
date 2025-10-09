"""Constants and enums for Lighter exchange connector"""

from enum import IntEnum, StrEnum


# WebSocket channels
WS_CHANNEL_ORDER_BOOK = "order_book"
WS_CHANNEL_TRADE = "trade"
WS_CHANNEL_MARKET_STATS = "market_stats"
WS_CHANNEL_ACCOUNT_ALL = "account_all"
WS_CHANNEL_USER_STATS = "user_stats"
WS_CHANNEL_EXECUTED_TX = "executed_transaction"

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

# Default values
DEFAULT_PING_INTERVAL = 20.0
DEFAULT_PING_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 10

# Lighter-specific
USDC_SCALE = 1e6  # Lighter uses 6 decimals for USDC


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
