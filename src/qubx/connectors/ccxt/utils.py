import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import ccxt.pro as cxp
from ccxt import BadSymbol
from qubx import logger
from qubx.core.basics import (
    AssetBalance,
    Deal,
    FundingRate,
    Instrument,
    Liquidation,
    Order,
    Position,
)
from qubx.core.series import OrderBook, Quote, Trade, time_as_nsec
from qubx.utils.marketdata.ccxt import (
    ccxt_symbol_to_instrument,
)
from qubx.utils.orderbook import build_orderbook_snapshots

from .exceptions import (
    CcxtLiquidationParsingError,
    CcxtSymbolNotRecognized,
)

EXCH_SYMBOL_PATTERN = re.compile(r"(?P<base>[^/]+)/(?P<quote>[^:]+)(?::(?P<margin>.+))?")


def ccxt_convert_order_info(instrument: Instrument, raw: dict[str, Any]) -> Order:
    """
    Convert CCXT excution record to Order object
    """
    ri = raw["info"]
    amnt = float(ri.get("origQty", raw.get("amount")))
    price = raw["price"]
    status = raw["status"]
    side = raw["side"].upper()
    _type = ri.get("type", raw.get("type")).upper()
    if status == "open":
        status = ri.get("status", status)  # for filled / part_filled ?

    return Order(
        id=raw["id"],
        type=_type,
        instrument=instrument,
        time=pd.Timestamp(raw["timestamp"], unit="ms"),  # type: ignore
        quantity=amnt,
        price=float(price) if price is not None else 0.0,
        side=side,
        status=status.upper(),
        time_in_force=raw["timeInForce"],
        client_id=raw["clientOrderId"],
        cost=float(raw["cost"] or 0),  # cost can be None
    )


def ccxt_convert_deal_info(raw: Dict[str, Any]) -> Deal:
    fee_amount = None
    fee_currency = None
    if "fee" in raw:
        fee_amount = float(raw["fee"]["cost"])
        fee_currency = raw["fee"]["currency"]
    return Deal(
        id=raw["id"],
        order_id=raw["order"],
        time=pd.Timestamp(raw["timestamp"], unit="ms"),  # type: ignore
        amount=float(raw["amount"]) * (-1 if raw["side"] == "sell" else +1),
        price=float(raw["price"]),
        aggressive=raw["takerOrMaker"] == "taker",
        fee_amount=fee_amount,
        fee_currency=fee_currency,
    )


def ccxt_extract_deals_from_exec(report: Dict[str, Any]) -> List[Deal]:
    """
    Small helper for extracting deals (trades) from CCXT execution report
    """
    deals = list()
    if trades := report.get("trades"):
        for t in trades:
            deals.append(ccxt_convert_deal_info(t))
    return deals


def ccxt_restore_position_from_deals(
    pos: Position, current_volume: float, deals: List[Deal], reserved_amount: float = 0.0
) -> Position:
    if current_volume != 0:
        instr = pos.instrument
        _last_deals = []

        # - try to find last deals that led to this position
        for d in sorted(deals, key=lambda x: x.time, reverse=True):
            current_volume -= d.amount
            # - spot case when fees may be deducted from the base coin
            #   that may decrease total amount
            if d.fee_amount is not None:
                if instr.base == d.fee_currency:
                    current_volume += d.fee_amount
            # print(d.amount, current_volume)
            _last_deals.insert(0, d)

            # - take in account reserves
            if abs(current_volume) - abs(reserved_amount) < instr.lot_size:
                break

        # - reset to 0
        pos.reset()

        if abs(current_volume) - abs(reserved_amount) > instr.lot_size:
            # - - - TODO - - - !!!!
            logger.warning(
                f"Couldn't restore full deals history for {instr.symbol} symbol. Qubx will use zero position !"
            )
        else:
            fees_in_base = 0.0
            for d in _last_deals:
                pos.update_position_by_deal(d)
                if d.fee_amount is not None:
                    if instr.base == d.fee_currency:
                        fees_in_base += d.fee_amount
            # - we round fees up in case of fees in base currency
            pos.quantity -= pos.instrument.round_size_up(fees_in_base)
    return pos


def ccxt_convert_trade(trade: dict[str, Any]) -> Trade:
    t_ns = trade["timestamp"] * 1_000_000  # this is trade time
    info, price, amnt = trade["info"], trade["price"], trade["amount"]
    m = info["m"]
    return Trade(t_ns, price, amnt, int(not m), int(trade["id"]))


def ccxt_convert_positions(
    pos_infos: list[dict], ccxt_exchange_name: str, markets: dict[str, dict[str, Any]]
) -> list[Position]:
    positions = []
    for info in pos_infos:
        symbol = info["symbol"]
        if symbol not in markets:
            logger.warning(f"Could not find symbol {symbol}, skipping position...")
            continue
        instr = ccxt_symbol_to_instrument(
            ccxt_exchange_name,
            markets[symbol],
        )
        pos = Position(
            instrument=instr,
            quantity=info["contracts"] * (-1 if info["side"] == "short" else 1),
            pos_average_price=info["entryPrice"],
        )
        pos.update_market_price(pd.Timestamp(info["timestamp"], unit="ms").asm8, info["markPrice"], 1)
        positions.append(pos)
    return positions


def ccxt_convert_orderbook(
    ob: dict, instr: Instrument, levels: int = 50, tick_size_pct: float = 0.01, sizes_in_quoted: bool = False
) -> OrderBook | None:
    """
    Convert a ccxt order book to an OrderBook object with a fixed tick size percentage.
    Parameters:
        ob (dict): The order book dictionary from ccxt.
        instr (Instrument): The instrument object containing market-specific details.
        levels (int, optional): The number of levels to include in the order book. Default is 50.
        tick_size_pct (float, optional): The tick size percentage. Default is 0.01%.
        sizes_in_quoted (bool, optional): Whether the size is in the quoted currency. Default is False.
    Returns:
        OrderBook: The converted OrderBook object.
    """
    _dt = pd.Timestamp(ob["datetime"]).replace(tzinfo=None).asm8
    _prev_dt = _dt - pd.Timedelta("1ms").asm8

    updates = [
        *[(_prev_dt, update[0], update[1], True) for update in ob["bids"]],
        *[(_prev_dt, update[0], update[1], False) for update in ob["asks"]],
    ]
    # add an artificial update to trigger the snapshot building
    updates.append((_dt, 0, 0, True))

    try:
        snapshots = build_orderbook_snapshots(
            updates,
            levels=levels,
            tick_size_pct=tick_size_pct,
            min_tick_size=instr.tick_size,
            min_size_step=instr.lot_size,
            sizes_in_quoted=sizes_in_quoted,
        )
    except Exception as e:
        logger.error(f"Failed to build order book snapshots: {e}")
        snapshots = None

    if not snapshots:
        return None

    (dt, _bids, _asks, top_bid, top_ask, tick_size) = snapshots[-1]
    bids = np.array([s for _, s in _bids[::-1]])
    asks = np.array([s for _, s in _asks])

    return OrderBook(
        time=time_as_nsec(dt),
        top_bid=top_bid,
        top_ask=top_ask,
        tick_size=tick_size,
        bids=bids,
        asks=asks,
    )


def ccxt_convert_liquidation(liq: dict[str, Any]) -> Liquidation:
    try:
        _dt = pd.Timestamp(liq["datetime"]).replace(tzinfo=None).asm8
        return Liquidation(
            time=_dt,
            price=liq["price"],
            quantity=liq["contracts"],
            side=(1 if liq["info"]["S"] == "BUY" else -1),
        )
    except Exception as e:
        raise CcxtLiquidationParsingError(f"Failed to parse liquidation: {e}")


def ccxt_convert_ticker(ticker: dict[str, Any]) -> Quote:
    """
    Convert a ccxt ticker to a Quote object.
    Parameters:
        ticker (dict): The ticker dictionary from ccxt.
        instr (Instrument): The instrument object containing market-specific details.
    Returns:
        Quote: The converted Quote object.
    """
    return Quote(
        time=pd.Timestamp(ticker["datetime"]).replace(tzinfo=None).asm8,
        bid=ticker["bid"],
        ask=ticker["ask"],
        bid_size=ticker["bidVolume"],
        ask_size=ticker["askVolume"],
    )


def ccxt_convert_funding_rate(info: dict[str, Any]) -> FundingRate:
    return FundingRate(
        time=pd.Timestamp(info["timestamp"], unit="ms").asm8,
        rate=info["fundingRate"],
        interval=info["interval"],
        next_funding_time=pd.Timestamp(info["nextFundingTime"], unit="ms").asm8,
        mark_price=info.get("markPrice"),
        index_price=info.get("indexPrice"),
    )


def ccxt_convert_balance(d: dict[str, Any]) -> dict[str, AssetBalance]:
    balances = {}
    for currency, data in d["total"].items():
        if not data:
            continue
        total = float(d["total"].get(currency, 0) or 0)
        locked = float(d["used"].get(currency, 0) or 0)
        balances[currency] = AssetBalance(free=total - locked, locked=locked, total=total)
    return balances


def find_instrument_for_exch_symbol(exch_symbol: str, symbol_to_instrument: Dict[str, Instrument]) -> Instrument:
    match = EXCH_SYMBOL_PATTERN.match(exch_symbol)
    if not match:
        raise CcxtSymbolNotRecognized(f"Invalid exchange symbol {exch_symbol}")
    base = match.group("base")
    quote = match.group("quote")
    symbol = f"{base}{quote}"
    if symbol not in symbol_to_instrument:
        raise CcxtSymbolNotRecognized(f"Unknown symbol {symbol}")
    return symbol_to_instrument[symbol]


def instrument_to_ccxt_symbol(instr: Instrument) -> str:
    return f"{instr.base}/{instr.quote}:{instr.settle}" if instr.is_futures() else f"{instr.base}/{instr.quote}"


def ccxt_find_instrument(
    symbol: str, exchange: cxp.Exchange, symbol_to_instrument: Dict[str, Instrument] | None = None
) -> Instrument:
    instrument = None
    if symbol_to_instrument is not None:
        instrument = symbol_to_instrument.get(symbol)
        if instrument is not None:
            return instrument
        try:
            instrument = find_instrument_for_exch_symbol(symbol, symbol_to_instrument)
        except CcxtSymbolNotRecognized:
            pass
    if instrument is None:
        try:
            symbol_info = exchange.market(symbol)
        except BadSymbol:
            raise CcxtSymbolNotRecognized(f"Unknown symbol {symbol}")
        exchange_name = exchange.name
        assert exchange_name is not None
        instrument = ccxt_symbol_to_instrument(exchange_name, symbol_info)
    if symbol_to_instrument is not None and symbol not in symbol_to_instrument:
        symbol_to_instrument[symbol] = instrument
    return instrument
