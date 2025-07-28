from typing import Dict, List

import ccxt.pro as cxp
from ccxt.async_support.base.ws.cache import ArrayCache, ArrayCacheByTimestamp
from ccxt.async_support.base.ws.client import Client
from ccxt.base.errors import ArgumentsRequired, BadRequest, InsufficientFunds, NotSupported
from ccxt.base.precise import Precise
from ccxt.base.types import (
    Any,
    Balances,
    Num,
    Order,
    OrderSide,
    OrderType,
    Strings,
    Tickers,
)


class BinanceQV(cxp.binance):
    """
    Extended binance exchange to provide quote asset volumes support
    """

    def describe(self):
        """
        Overriding watchTrades to use aggTrade instead of trade.
        """
        return self.deep_extend(
            super().describe(),
            {
                "options": {
                    "watchTrades": {
                        "name": "aggTrade",
                    },
                    "localOrderBookLimit": 10_000,  # set a large limit to avoid cutting off the orderbook
                },
                "exceptions": {
                    "exact": {
                        "-2019": InsufficientFunds,  # ccxt doesn't have this code for some weird reason !!
                    },
                },
            },
        )

    async def un_watch_bids_asks(self, symbols: Strings = None, params: dict = {}) -> Any:
        """
        unwatches best bid & ask for symbols

        https://developers.binance.com/docs/binance-spot-api-docs/web-socket-api#symbol-order-book-ticker
        https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/All-Book-Tickers-Stream
        https://developers.binance.com/docs/derivatives/coin-margined-futures/websocket-market-streams/All-Book-Tickers-Stream

        :param str[] symbols: unified symbol of the market to fetch the ticker for
        :param dict [params]: extra parameters specific to the exchange API endpoint
        :returns dict: a `ticker structure <https://docs.ccxt.com/#/?id=ticker-structure>`
        """
        await self.load_markets()
        methodName = "watchBidsAsks"
        channelName = "bookTicker"
        symbols = self.market_symbols(symbols, None, True, False, True)
        firstMarket = None
        marketType = None
        symbolsDefined = symbols is not None
        if symbolsDefined:
            firstMarket = self.market(symbols[0])
        marketType, params = self.handle_market_type_and_params(methodName, firstMarket, params)
        subType = None
        subType, params = self.handle_sub_type_and_params(methodName, firstMarket, params)
        rawMarketType = None
        if self.isLinear(marketType, subType):
            rawMarketType = "future"
        elif self.isInverse(marketType, subType):
            rawMarketType = "delivery"
        elif marketType == "spot":
            rawMarketType = marketType
        else:
            raise NotSupported(str(self.id) + " " + methodName + "() does not support options markets")
        isBidAsk = True
        subscriptionArgs = []
        subMessageHashes = []
        messageHashes = []
        if symbolsDefined:
            for i in range(0, len(symbols)):
                symbol = symbols[i]
                market = self.market(symbol)
                subscriptionArgs.append(market["lowercaseId"] + "@" + channelName)
                subMessageHashes.append(self.get_message_hash(channelName, market["symbol"], isBidAsk))
                messageHashes.append("unsubscribe:bidsasks:" + symbol)
        else:
            if marketType == "spot":
                raise ArgumentsRequired(
                    str(self.id) + " " + methodName + "() requires symbols for this channel for spot markets"
                )
            subscriptionArgs.append("!" + channelName)
            subMessageHashes.append(self.get_message_hash(channelName, None, isBidAsk))
            messageHashes.append("unsubscribe:bidsasks")
        streamHash = channelName
        if symbolsDefined:
            streamHash = channelName + "::" + ",".join(symbols)
        url = self.urls["api"]["ws"][rawMarketType] + "/" + self.stream(rawMarketType, streamHash)
        requestId = self.request_id(url)
        request: dict = {
            "method": "UNSUBSCRIBE",
            "params": subscriptionArgs,
            "id": requestId,
        }
        subscription: dict = {
            "unsubscribe": True,
            "id": str(requestId),
            "subMessageHashes": subMessageHashes,
            "messageHashes": subMessageHashes,
            "symbols": symbols,
            "topic": "bidsasks",
        }
        return await self.watch_multiple(
            url, subMessageHashes, self.extend(request, params), subMessageHashes, subscription
        )

    def parse_ohlcv(self, ohlcv, market=None):
        """
        [
            1499040000000,      // Kline open time                   0
            "0.01634790",       // Open price                        1
            "0.80000000",       // High price                        2
            "0.01575800",       // Low price                         3
            "0.01577100",       // Close price                       4
            "148976.11427815",  // Volume                            5
            1499644799999,      // Kline Close time                  6
            "2434.19055334",    // Quote asset volume                7
            308,                // Number of trades                  8
            "1756.87402397",    // Taker buy base asset volume       9
            "28.46694368",      // Taker buy quote asset volume     10
            "0"                 // Unused field, ignore.
        ]
        """
        return [
            self.safe_integer(ohlcv, 0),
            self.safe_number(ohlcv, 1),
            self.safe_number(ohlcv, 2),
            self.safe_number(ohlcv, 3),
            self.safe_number(ohlcv, 4),
            self.safe_number(ohlcv, 5),
            self.safe_number(ohlcv, 7),  # Quote asset volume
            self.safe_integer(ohlcv, 8),  # Number of trades
            self.safe_number(ohlcv, 9),  # Taker buy base asset volume
            self.safe_number(ohlcv, 10),  # Taker buy quote asset volume
        ]

    def handle_ohlcv(self, client: Client, message):
        event = self.safe_string(message, "e")
        eventMap: dict = {
            "indexPrice_kline": "indexPriceKline",
            "markPrice_kline": "markPriceKline",
        }
        event = self.safe_string(eventMap, event, event)
        kline = self.safe_value(message, "k")
        marketId = self.safe_string_2(kline, "s", "ps")
        if event == "indexPriceKline":
            # indexPriceKline doesn't have the _PERP suffix
            marketId = self.safe_string(message, "ps")
        interval = self.safe_string(kline, "i")
        # use a reverse lookup in a static map instead
        unifiedTimeframe = self.find_timeframe(interval)
        parsed = [
            self.safe_integer(kline, "t"),
            self.safe_float(kline, "o"),
            self.safe_float(kline, "h"),
            self.safe_float(kline, "l"),
            self.safe_float(kline, "c"),
            self.safe_float(kline, "v"),
            # - additional fields
            self.safe_float(kline, "q"),  # - quote asset volume
            self.safe_integer(kline, "n"),  # - number of trades
            self.safe_float(kline, "V"),  # - taker buy base asset volume
            self.safe_float(kline, "Q"),  # - taker buy quote asset volume
        ]
        isSpot = (client.url.find("/stream") > -1) or (client.url.find("/testnet.binance") > -1)
        marketType = "spot" if (isSpot) else "contract"
        symbol = self.safe_symbol(marketId, None, None, marketType)
        messageHash = "ohlcv::" + symbol + "::" + unifiedTimeframe
        self.ohlcvs[symbol] = self.safe_value(self.ohlcvs, symbol, {})
        stored = self.safe_value(self.ohlcvs[symbol], unifiedTimeframe)
        if stored is None:
            limit = self.safe_integer(self.options, "OHLCVLimit", 1000)
            stored = ArrayCacheByTimestamp(limit)
            self.ohlcvs[symbol][unifiedTimeframe] = stored
        stored.append(parsed)
        resolveData = [symbol, unifiedTimeframe, stored]
        client.resolve(resolveData, messageHash)

    def handle_trade(self, client: Client, message):
        """
        There is a custom trade handler implementation, because Binance sends
        some trades marked with "X" field, which is "MARKET" for market trades
        and "INSURANCE_FUND" for insurance fund trades. We are interested only
        in market trades, so we filter the rest out.

        Update 07072024: Apparently insurance fund trades not aggregated so
        we don't need to filter via "X" field, but let's keep it just in case.
        """
        # the trade streams push raw trade information in real-time
        # each trade has a unique buyer and seller
        isSpot = (client.url.find("wss://stream.binance.com") > -1) or (client.url.find("/testnet.binance") > -1)
        marketType = "spot" if (isSpot) else "contract"
        marketId = self.safe_string(message, "s")
        market = self.safe_market(marketId, None, None, marketType)
        symbol = market["symbol"]
        messageHash = "trade::" + symbol
        executionType = self.safe_string(message, "X")
        if executionType == "INSURANCE_FUND":
            return

        # - fix 2025-04-16: filter out trades with zero price
        trade = self.parse_ws_trade(message, market)
        if trade["price"] == 0.0:
            return

        tradesArray = self.safe_value(self.trades, symbol)
        if tradesArray is None:
            limit = self.safe_integer(self.options, "tradesLimit", 1000)
            tradesArray = ArrayCache(limit)
        tradesArray.append(trade)
        self.trades[symbol] = tradesArray
        client.resolve(tradesArray, messageHash)

    def handle_order_book_subscription(self, client: Client, message, subscription):
        defaultLimit = self.safe_integer(self.options, "localOrderBookLimit", 4000)
        # messageHash = self.safe_string(subscription, 'messageHash')
        symbolOfSubscription = self.safe_string(subscription, "symbol")  # watchOrderBook
        symbols = self.safe_value(subscription, "symbols", [symbolOfSubscription])  # watchOrderBookForSymbols
        limit = self.safe_integer(subscription, "limit", defaultLimit)
        # handle list of symbols
        for i in range(0, len(symbols)):
            symbol = symbols[i]
            if symbol in self.orderbooks:
                del self.orderbooks[symbol]
            self.orderbooks[symbol] = self.order_book({}, limit)
            subscription = self.extend(subscription, {"symbol": symbol})
            # fetch the snapshot in a separate async call
            self.spawn(self.fetch_order_book_snapshot, client, message, subscription)


class BinanceQVUSDM(cxp.binanceusdm, BinanceQV):
    """
    The order of inheritance is important here, because we want
    binanceusdm to take precedence over binanceqv. And this is how MRO is defined
    in Python.

    Describe method needs to be overriden, because of the way super is called in binanceusdm.
    """

    _funding_intervals: Dict[str, str]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._funding_intervals = {}

    def describe(self):
        """
        Overriding watchTrades to use aggTrade instead of trade.
        """
        return self.deep_extend(
            super().describe(),
            {
                "options": {
                    "watchTrades": {
                        "name": "aggTrade",
                    }
                }
            },
        )

    async def watch_funding_rates(self, symbols: List[str] | None = None):
        symbol_count = len(symbols) if symbols else 0
        
        try:
            await self.load_markets()
            await self._update_funding_intervals()
            
            # Use watch_mark_prices which streams one symbol per WebSocket message
            # This is normal behavior - WebSocket messages contain one symbol at a time
            mark_prices = await self.watch_mark_prices(symbols)
            
            if not mark_prices:
                raise Exception("No mark price data received")
            
            # Process whatever symbol(s) we received (usually 1 per WebSocket message)
            funding_rates = {}
            processed_count = 0
            
            for symbol, info in mark_prices.items():
                try:
                    interval = self._funding_intervals.get(symbol, "8h")
                    
                    # Ensure we have the required fields for funding rate
                    if "info" not in info or "r" not in info["info"]:
                        continue
                        
                    funding_rates[symbol] = {
                        "timestamp": info["timestamp"],
                        "interval": interval,
                        "fundingRate": float(info["info"]["r"]),
                        "nextFundingTime": info["info"]["T"],
                        "markPrice": info["markPrice"],
                        "indexPrice": info["indexPrice"],
                    }
                    processed_count += 1
                    
                except Exception as e:
                    continue
            
            if processed_count == 0:
                raise Exception("No funding rates could be processed from mark price data")
            
            return funding_rates
            
        except Exception as e:
            raise

    async def _update_funding_intervals(self):
        if self._funding_intervals:
            return
        symbol_to_info = await self.fetch_funding_intervals()
        self._funding_intervals = {str(s): str(info["interval"]) for s, info in symbol_to_info.items()}

    async def create_order_ws(
        self, symbol: str, type: OrderType, side: OrderSide, amount: float, price: Num = None, params={}
    ) -> Order:
        """
        create a trade order
        :see: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-api#place-new-order-trade
        :see: https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/websocket-api/New-Order
        :param str symbol: unified symbol of the market to create an order in
        :param str type: 'market' or 'limit'
        :param str side: 'buy' or 'sell'
        :param float amount: how much of currency you want to trade in units of base currency
        :param float|None [price]: the price at which the order is to be fulfilled, in units of the quote currency, ignored in market orders
        :param dict [params]: extra parameters specific to the exchange API endpoint
        :param boolean params['test']: test order, default False
        :param boolean params['returnRateLimits']: set to True to return rate limit information, default False
        :returns dict: an `order structure <https://docs.ccxt.com/#/?id=order-structure>`
        """
        await self.load_markets()
        market = self.market(symbol)
        marketType = self.get_market_type("createOrderWs", market, params)
        if marketType != "spot" and marketType != "future":
            raise BadRequest(self.id + " createOrderWs only supports spot or swap markets")
        url = self.urls["api"]["ws"]["ws-api"][marketType]
        requestId = self.request_id(url)
        messageHash = str(requestId)
        sor = self.safe_bool_2(params, "sor", "SOR", False)
        params = self.omit(params, "sor", "SOR")
        payload = self.create_order_request(symbol, type, side, amount, price, params)
        returnRateLimits = False
        returnRateLimits, params = self.handle_option_and_params(params, "createOrderWs", "returnRateLimits", False)
        payload["returnRateLimits"] = returnRateLimits
        test = self.safe_bool(params, "test", False)
        params = self.omit(params, "test")
        # Here the ccxt code does an extend of payload with params which breaks the type parameter
        message: dict = {
            "id": messageHash,
            "method": "order.place",
            "params": self.sign_params(payload),
        }
        if test:
            if sor:
                message["method"] = "sor.order.test"
            else:
                message["method"] = "order.test"
        subscription: dict = {
            "method": self.handle_order_ws,
        }
        return await self.watch(url, messageHash, message, messageHash, subscription)

    async def cancel_order_ws(self, id: str, symbol: str | None = None, params={}) -> Order:
        """
        cancel multiple orders

        https://developers.binance.com/docs/binance-spot-api-docs/web-socket-api#cancel-order-trade
        https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/websocket-api/Cancel-Order
        https://developers.binance.com/docs/derivatives/coin-margined-futures/trade/websocket-api/Cancel-Order

        :param str id: order id
        :param str [symbol]: unified market symbol, default is None
        :param dict [params]: extra parameters specific to the exchange API endpoint
        :param str|None [params.cancelRestrictions]: Supported values: ONLY_NEW - Cancel will succeed if the order status is NEW. ONLY_PARTIALLY_FILLED - Cancel will succeed if order status is PARTIALLY_FILLED.
        :returns dict: an list of `order structures <https://docs.ccxt.com/#/?id=order-structure>`
        """
        await self.load_markets()
        if symbol is None:
            raise BadRequest(self.id + " cancelOrderWs requires a symbol")
        market = self.market(symbol)
        type = self.get_market_type("cancelOrderWs", market, params)
        url = self.urls["api"]["ws"]["ws-api"][type]
        requestId = self.request_id(url)
        messageHash = str(requestId)
        returnRateLimits = False
        returnRateLimits, params = self.handle_option_and_params(params, "cancelOrderWs", "returnRateLimits", False)
        payload: dict = {
            "symbol": self.market_id(symbol),
            "returnRateLimits": returnRateLimits,
        }
        clientOrderId = self.safe_string_2(params, "origClientOrderId", "clientOrderId")
        if clientOrderId is not None:
            payload["origClientOrderId"] = clientOrderId
        else:
            payload["orderId"] = self.parse_to_int(id)
        params = self.omit(params, ["origClientOrderId", "clientOrderId"])
        # Same as in create_order_ws
        message: dict = {
            "id": messageHash,
            "method": "order.cancel",
            "params": self.sign_params(payload),
        }
        subscription: dict = {
            "method": self.handle_order_ws,
        }
        return await self.watch(url, messageHash, message, messageHash, subscription)

    async def un_watch_funding_rates(self):
        """Unwatch funding rates to ensure fresh connections"""
        from qubx import logger
        logger.debug("un_watch_funding_rates called - resetting connection")
        
        # Try to unwatch mark prices if possible
        if hasattr(self, 'un_watch_mark_prices'):
            try:
                await self.un_watch_mark_prices()
            except Exception as e:
                logger.debug(f"Error unwatching mark prices: {e}")
        
        # Clear any internal caches that might exist
        if hasattr(self, 'markPrices') and self.markPrices:
            self.markPrices.clear()
            logger.debug("Cleared mark prices cache")
            
        return None


class BinancePortfolioMargin(BinanceQVUSDM):
    def describe(self):
        return self.deep_extend(
            super().describe(),
            {
                "options": {
                    "defaultType": "margin",
                    "portfolioMargin": True,
                    "defaultSubType": None,
                    "fetchMarkets": ["spot", "linear", "inverse"],
                }
            },
        )

    # this fixes the PM total balance calculation
    def parse_balance_custom(self, response, type=None, marginMode=None, isPortfolioMargin=False) -> Balances:
        result = {
            "info": response,
        }
        timestamp = None
        isolated = marginMode == "isolated"
        cross = (type == "margin") or (marginMode == "cross")
        if isPortfolioMargin:
            for i in range(0, len(response)):
                entry = response[i]
                account = self.account()
                currencyId = self.safe_string(entry, "asset")
                code = self.safe_currency_code(currencyId)
                if type == "linear":
                    account["free"] = self.safe_string(entry, "umWalletBalance")
                    account["used"] = self.safe_string(entry, "umUnrealizedPNL")
                elif type == "inverse":
                    account["free"] = self.safe_string(entry, "cmWalletBalance")
                    account["used"] = self.safe_string(entry, "cmUnrealizedPNL")
                elif cross:
                    borrowed = self.safe_string(entry, "crossMarginBorrowed")
                    interest = self.safe_string(entry, "crossMarginInterest")
                    account["debt"] = Precise.string_add(borrowed, interest)
                    account["free"] = self.safe_string(entry, "crossMarginFree")
                    account["used"] = self.safe_string(entry, "crossMarginLocked")
                    # account['total'] = self.safe_string(entry, 'crossMarginAsset')
                    account["total"] = self.safe_string(entry, "totalWalletBalance")
                else:
                    usedLinear = self.safe_string(entry, "umUnrealizedPNL")
                    usedInverse = self.safe_string(entry, "cmUnrealizedPNL")
                    totalUsed = Precise.string_add(usedLinear, usedInverse)
                    totalWalletBalance = self.safe_string(entry, "totalWalletBalance")
                    account["total"] = Precise.string_add(totalUsed, totalWalletBalance)
                result[code] = account
        elif not isolated and ((type == "spot") or cross):
            timestamp = self.safe_integer(response, "updateTime")
            balances = self.safe_list_2(response, "balances", "userAssets", [])
            for i in range(0, len(balances)):
                balance = balances[i]
                currencyId = self.safe_string(balance, "asset")
                code = self.safe_currency_code(currencyId)
                account = self.account()
                account["free"] = self.safe_string(balance, "free")
                account["used"] = self.safe_string(balance, "locked")
                if cross:
                    debt = self.safe_string(balance, "borrowed")
                    interest = self.safe_string(balance, "interest")
                    account["debt"] = Precise.string_add(debt, interest)
                result[code] = account
        elif isolated:
            assets = self.safe_list(response, "assets")
            for i in range(0, len(assets)):
                asset = assets[i]
                marketId = self.safe_string(asset, "symbol")
                symbol = self.safe_symbol(marketId, None, None, "spot")
                base = self.safe_dict(asset, "baseAsset", {})
                quote = self.safe_dict(asset, "quoteAsset", {})
                baseCode = self.safe_currency_code(self.safe_string(base, "asset"))
                quoteCode = self.safe_currency_code(self.safe_string(quote, "asset"))
                subResult: dict = {}
                subResult[baseCode] = self.parse_balance_helper(base)
                subResult[quoteCode] = self.parse_balance_helper(quote)
                result[symbol] = self.safe_balance(subResult)
        elif type == "savings":
            positionAmountVos = self.safe_list(response, "positionAmountVos", [])
            for i in range(0, len(positionAmountVos)):
                entry = positionAmountVos[i]
                currencyId = self.safe_string(entry, "asset")
                code = self.safe_currency_code(currencyId)
                account = self.account()
                usedAndTotal = self.safe_string(entry, "amount")
                account["total"] = usedAndTotal
                account["used"] = usedAndTotal
                result[code] = account
        elif type == "funding":
            for i in range(0, len(response)):
                entry = response[i]
                account = self.account()
                currencyId = self.safe_string(entry, "asset")
                code = self.safe_currency_code(currencyId)
                account["free"] = self.safe_string(entry, "free")
                frozen = self.safe_string(entry, "freeze")
                withdrawing = self.safe_string(entry, "withdrawing")
                locked = self.safe_string(entry, "locked")
                account["used"] = Precise.string_add(frozen, Precise.string_add(locked, withdrawing))
                result[code] = account
        else:
            balances = response
            if not isinstance(response, list):
                balances = self.safe_list(response, "assets", [])
            for i in range(0, len(balances)):
                balance = balances[i]
                currencyId = self.safe_string(balance, "asset")
                code = self.safe_currency_code(currencyId)
                account = self.account()
                account["free"] = self.safe_string(balance, "availableBalance")
                account["used"] = self.safe_string(balance, "initialMargin")
                account["total"] = self.safe_string_2(balance, "marginBalance", "balance")
                result[code] = account
        result["timestamp"] = timestamp
        result["datetime"] = self.iso8601(timestamp)
        return result if isolated else self.safe_balance(result)


class BINANCE_UM_MM(BinanceQVUSDM):
    def describe(self):
        return self.deep_extend(
            super().describe(),
            {
                "urls": {
                    "api": {
                        "fapiPublic": "https://fapi-mm.binance.com/fapi/v1",
                        "fapiPublicV2": "https://fapi-mm.binance.com/fapi/v2",
                        "fapiPublicV3": "https://fapi-mm.binance.com/fapi/v3",
                        "fapiPrivate": "https://fapi-mm.binance.com/fapi/v1",
                        "fapiPrivateV2": "https://fapi-mm.binance.com/fapi/v2",
                        "fapiPrivateV3": "https://fapi-mm.binance.com/fapi/v3",
                        "future": "wss://fstream-mm.binance.com/ws",
                        "ws-api": {
                            "future": "wss://ws-fapi-mm.binance.com/ws-fapi/v1",
                        },
                    }
                }
            },
        )
