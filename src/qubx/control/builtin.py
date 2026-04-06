"""Built-in control actions available on every bot."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable

import pandas as pd

from qubx.core.basics import Instrument, MarketType, Signal
from qubx.core.lookups import lookup
from qubx.utils.time import to_timedelta, to_timestamp

from .decorator import collect_state, collect_state_schema
from .types import ActionDef, ActionParam, ActionResult

if TYPE_CHECKING:
    from qubx.core.interfaces import IStrategyContext


# --- Helpers ---


def _resolve(ctx: IStrategyContext, symbol: str, exchange: str | None = None) -> Instrument | None:
    """Resolve a symbol to an Instrument, returning None if not found."""
    try:
        return ctx.query_instrument(symbol, exchange=exchange)
    except Exception:
        return None


# --- Rounding helpers ---


def _rm(v: float) -> float:
    """Round money/dollar amounts to 2 decimals."""
    return round(v, 2)


def _rl(v: float) -> float:
    """Round leverage to 4 decimals."""
    return round(v, 4)


def _rms(v: float) -> float:
    """Round latency (ms) to 1 decimal."""
    return round(v, 1)


# --- Universe actions ---


def _get_universe(ctx: IStrategyContext, **kwargs) -> ActionResult:
    instruments = ctx.instruments
    return ActionResult(
        status="ok",
        data={"instruments": [str(i) for i in instruments], "count": len(instruments)},
    )


def _add_instruments(ctx: IStrategyContext, symbols: list[str], exchange: str | None = None, **kwargs) -> ActionResult:
    instruments = []
    errors = []
    for s in symbols:
        instr = _resolve(ctx, s, exchange)
        if instr is not None:
            instruments.append(instr)
        else:
            errors.append(s)

    if errors:
        return ActionResult(status="error", error=f"Unknown symbols: {errors}")

    ctx.add_instruments(instruments)
    return ActionResult(
        status="ok",
        data={"added": [str(i) for i in instruments], "universe_size": len(ctx.instruments)},
    )


def _remove_instruments(ctx: IStrategyContext, symbols: list[str], exchange: str | None = None, if_has_position: str = "close", **kwargs) -> ActionResult:
    instruments = []
    errors = []
    for s in symbols:
        instr = _resolve(ctx, s, exchange)
        if instr is not None:
            instruments.append(instr)
        else:
            errors.append(s)

    if errors:
        return ActionResult(status="error", error=f"Unknown symbols: {errors}")

    ctx.remove_instruments(instruments, if_has_position_then=if_has_position)
    return ActionResult(
        status="ok",
        data={"removed": [str(i) for i in instruments], "universe_size": len(ctx.instruments)},
    )


def _set_universe(ctx: IStrategyContext, symbols: list[str], exchange: str | None = None, if_has_position: str = "close", **kwargs) -> ActionResult:
    instruments = []
    errors = []
    for s in symbols:
        instr = _resolve(ctx, s, exchange)
        if instr is not None:
            instruments.append(instr)
        else:
            errors.append(s)

    if errors:
        return ActionResult(status="error", error=f"Unknown symbols: {errors}")

    ctx.set_universe(instruments, if_has_position_then=if_has_position)
    return ActionResult(
        status="ok",
        data={"instruments": [str(i) for i in instruments], "count": len(instruments)},
    )


# --- Instrument discovery actions ---


def _get_available_instruments(ctx: IStrategyContext, exchange: str, quote: str = "USDT", market_type: str = "SWAP", **kwargs) -> ActionResult:
    mt = MarketType(market_type) if market_type else None
    as_of = to_timestamp(ctx.time())
    instruments = lookup.find_instruments(exchange, quote=quote, market_type=mt, as_of=as_of)
    if instruments is None:
        instruments = []
    symbols = [i.symbol for i in instruments]
    return ActionResult(
        status="ok",
        data={
            "exchange": exchange,
            "market_type": market_type,
            "quote": quote,
            "count": len(symbols),
            "instruments": symbols,
        },
    )


def _get_instrument_details(ctx: IStrategyContext, symbols: list[str], exchange: str | None = None, **kwargs) -> ActionResult:
    details = {}
    errors = []
    for s in symbols:
        instr = _resolve(ctx, s, exchange)
        if instr is None:
            errors.append(s)
            continue
        details[str(instr)] = {
            "symbol": instr.symbol,
            "exchange": instr.exchange,
            "market_type": instr.market_type,
            "base": instr.base,
            "quote": instr.quote,
            "tick_size": instr.tick_size,
            "lot_size": instr.lot_size,
            "min_size": instr.min_size,
            "min_notional": instr.min_notional,
            "contract_size": instr.contract_size,
        }
    result: dict = {"instruments": details}
    if errors:
        result["not_found"] = errors
    return ActionResult(status="ok", data=result)


def _get_top_instruments(
    ctx: IStrategyContext,
    exchange: str,
    count: int = 20,
    sort_by: str = "turnover",
    period: str = "3d",
    timeframe: str = "1d",
    quote: str = "USDT",
    market_type: str = "SWAP",
    **kwargs,
) -> ActionResult:
    stop = to_timestamp(ctx.time())
    start = stop - to_timedelta(period)

    if sort_by == "market_cap":
        return _rank_by_market_cap(ctx, exchange, market_type, quote, count, start, stop)
    elif sort_by == "turnover":
        return _rank_by_turnover(ctx, exchange, market_type, quote, count, timeframe, start, stop)
    elif sort_by == "funding":
        return _rank_by_funding(ctx, exchange, market_type, quote, count, start, stop)
    else:
        return ActionResult(status="error", error=f"Unknown sort_by: {sort_by}. Use 'turnover', 'market_cap', or 'funding'.")


def _rank_by_turnover(
    ctx: IStrategyContext, exchange: str, market_type: str, quote: str,
    count: int, timeframe: str, start: pd.Timestamp, stop: pd.Timestamp,
) -> ActionResult:
    try:
        reader = ctx.get_aux_reader(exchange, market_type)
    except Exception as e:
        return ActionResult(status="error", error=f"Aux reader for {exchange}:{market_type} not available: {e}")

    try:
        mt = MarketType(market_type) if market_type else None
        instruments = lookup.find_instruments(exchange, quote=quote, market_type=mt) or []
        if not instruments:
            return ActionResult(status="ok", data={"instruments": [], "count": 0, "sort_by": "turnover"})

        symbols = [i.symbol for i in instruments]
        candles = reader.read(symbols, f"ohlc({timeframe})", start=str(start), stop=str(stop), skip_rate_limited=True).to_pd(True)

        if candles is None or candles.empty:
            return ActionResult(status="ok", data={"instruments": [], "count": 0, "sort_by": "turnover"})

        candles = candles.reset_index("symbol")
        volumes = candles.groupby("symbol")["quote_volume"].mean()
        ranked = volumes.sort_values(ascending=False).head(count)

        result = [{"symbol": sym, "avg_turnover": _rm(vol)} for sym, vol in ranked.items()]
        return ActionResult(status="ok", data={"instruments": result, "count": len(result), "sort_by": "turnover"})
    except Exception as e:
        return ActionResult(status="error", error=f"Failed to rank by turnover: {e}")


def _rank_by_market_cap(
    ctx: IStrategyContext, exchange: str, market_type: str, quote: str,
    count: int, start: pd.Timestamp, stop: pd.Timestamp,
) -> ActionResult:
    try:
        reader = ctx.get_aux_reader("COINGECKO", "FUNDAMENTAL")
    except Exception as e:
        return ActionResult(status="error", error=f"CoinGecko aux reader not available: {e}. Configure aux data with COINGECKO:FUNDAMENTAL.")

    try:
        coins = reader.get_data_id()
        raw = reader.read(coins, "fundamental", start=str(start), stop=str(stop), skip_rate_limited=True).to_pd(True)

        if raw is None or raw.empty:
            return ActionResult(status="ok", data={"instruments": [], "count": 0, "sort_by": "market_cap"})

        df = raw.pivot_table(index=["timestamp", "asset"], columns="metric", values="value")
        if "market_cap" not in df.columns:
            return ActionResult(status="error", error="market_cap metric not available in fundamental data")

        latest = df.groupby("asset")["market_cap"].last()
        ranked = latest.sort_values(ascending=False).head(count)

        result = [{"symbol": f"{asset}{quote}", "market_cap": _rm(cap)} for asset, cap in ranked.items()]
        return ActionResult(status="ok", data={"instruments": result, "count": len(result), "sort_by": "market_cap"})
    except Exception as e:
        return ActionResult(status="error", error=f"Failed to rank by market cap: {e}")


def _rank_by_funding(
    ctx: IStrategyContext, exchange: str, market_type: str, quote: str,
    count: int, start: pd.Timestamp, stop: pd.Timestamp,
) -> ActionResult:
    try:
        reader = ctx.get_aux_reader(exchange, market_type)
    except Exception as e:
        return ActionResult(status="error", error=f"Aux reader for {exchange}:{market_type} not available: {e}")

    try:
        mt = MarketType(market_type) if market_type else None
        instruments = lookup.find_instruments(exchange, quote=quote, market_type=mt) or []
        if not instruments:
            return ActionResult(status="ok", data={"instruments": [], "count": 0, "sort_by": "funding"})

        symbols = [i.symbol for i in instruments]
        fp = reader.read(symbols, "funding_payment", start=str(start), stop=str(stop), skip_rate_limited=True).to_pd(True)

        if fp is None or fp.empty:
            return ActionResult(status="ok", data={"instruments": [], "count": 0, "sort_by": "funding"})

        fp = fp.reset_index("symbol")
        hours = max((stop - start).total_seconds() / 3600, 1)
        avg_rate = fp.groupby("symbol")["funding_rate"].sum() / hours
        apr = avg_rate * 24 * 365 * 100  # annualized percentage
        ranked = apr.abs().sort_values(ascending=False).head(count)

        result = [{"symbol": sym, "funding_apr": _rm(apr.loc[sym])} for sym in ranked.index]
        return ActionResult(status="ok", data={"instruments": result, "count": len(result), "sort_by": "funding"})
    except Exception as e:
        return ActionResult(status="error", error=f"Failed to rank by funding: {e}")


# --- Diagnostics actions ---


def _get_positions(ctx: IStrategyContext, **kwargs) -> ActionResult:
    positions = {}
    for instr, pos in ctx.get_positions().items():
        positions[str(instr)] = {
            "quantity": pos.quantity,
            "avg_price": pos.position_avg_price,
            "market_price": pos.last_update_price,
            "unrealized_pnl": _rm(pos.unrealized_pnl()),
            "realized_pnl": _rm(pos.r_pnl),
            "market_value": _rm(pos.market_value_funds),
        }
    return ActionResult(status="ok", data={"positions": positions})


def _get_balances(ctx: IStrategyContext, **kwargs) -> ActionResult:
    balances = {}
    for bal in ctx.get_balances():
        key = f"{bal.exchange}:{bal.currency}"
        balances[key] = {"total": _rm(bal.total), "free": _rm(bal.free), "locked": _rm(bal.locked)}
    return ActionResult(status="ok", data={"balances": balances})


def _get_orders(ctx: IStrategyContext, symbol: str | None = None, **kwargs) -> ActionResult:
    orders_data = []
    if symbol:
        instr = _resolve(ctx, symbol)
        if instr is None:
            return ActionResult(status="error", error=f"Unknown symbol: {symbol}")
        orders = ctx.get_orders(instrument=instr)
    else:
        orders = ctx.get_orders()
    for order_id, order in orders.items():
        orders_data.append(
            {
                "id": order.id,
                "instrument": str(order.instrument),
                "type": order.type,
                "side": order.side,
                "quantity": order.quantity,
                "price": order.price,
                "status": order.status,
                "client_id": order.client_id,
            }
        )
    return ActionResult(status="ok", data={"orders": orders_data})


def _get_quote(ctx: IStrategyContext, symbol: str, **kwargs) -> ActionResult:
    instr = _resolve(ctx, symbol)
    if instr is None:
        return ActionResult(status="error", error=f"Unknown symbol: {symbol}")

    quote = ctx.quote(instr)
    if quote is None:
        return ActionResult(status="error", error=f"No quote available for {symbol}")

    return ActionResult(
        status="ok",
        data={
            "bid": quote.bid,
            "ask": quote.ask,
            "bid_size": quote.bid_size,
            "ask_size": quote.ask_size,
            "time": str(to_timestamp(quote.time)),
        },
    )


def _get_ohlc(ctx: IStrategyContext, symbol: str, timeframe: str = "1h", length: int = 20, **kwargs) -> ActionResult:
    instr = _resolve(ctx, symbol)
    if instr is None:
        return ActionResult(status="error", error=f"Unknown symbol: {symbol}")

    try:
        df = ctx.ohlc_pd(instr, timeframe=timeframe, length=length)
        if df is None or df.empty:
            return ActionResult(status="ok", data={"bars": [], "count": 0})

        bars = []
        for ts, row in df.iterrows():
            bars.append(
                {
                    "time": str(ts),
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                }
            )
        return ActionResult(status="ok", data={"bars": bars, "count": len(bars)})
    except Exception as e:
        return ActionResult(status="error", error=str(e))


def _get_state(ctx: IStrategyContext, **kwargs) -> ActionResult:
    account = ctx.account
    exchanges = ctx.exchanges

    exchanges_snapshot: dict[str, dict] = {}
    for exchange in exchanges:
        positions = account.get_positions(exchange)
        orders = account.get_orders(exchange=exchange)

        # Group orders by instrument symbol
        orders_by_symbol: dict[str, list[dict]] = defaultdict(list)
        for order in orders.values():
            orders_by_symbol[order.instrument.symbol].append(
                {
                    "id": order.id,
                    "type": order.type,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.price,
                    "status": order.status,
                }
            )

        # Build positions snapshot
        positions_snapshot: dict[str, dict] = {}
        open_positions = 0
        for instr, pos in positions.items():
            if pos.is_open():
                open_positions += 1
            positions_snapshot[instr.symbol] = {
                "quantity": pos.quantity,
                "avg_price": pos.position_avg_price,
                "market_price": pos.last_update_price,
                "unrealized_pnl": _rm(pos.unrealized_pnl()),
                "market_value": _rm(pos.market_value_funds),
                "leverage": _rl(account.get_leverage(instr)),
            }

        # Build balances snapshot
        balances_snapshot: dict[str, dict] = {}
        for bal in account.get_balances(exchange):
            balances_snapshot[bal.currency] = {
                "total": _rm(bal.total),
                "free": _rm(bal.free),
                "locked": _rm(bal.locked),
            }

        exchanges_snapshot[exchange] = {
            "base_currency": account.get_base_currency(exchange),
            "capital": {
                "total": _rm(account.get_total_capital(exchange)),
                "available": _rm(account.get_capital(exchange)),
            },
            "net_leverage": _rl(account.get_net_leverage(exchange)),
            "gross_leverage": _rl(account.get_gross_leverage(exchange)),
            "open_positions": open_positions,
            "positions": positions_snapshot,
            "orders": dict(orders_by_symbol),
            "balances": balances_snapshot,
        }

    data: dict = {
        "timestamp": str(ctx.time()),
        "total_capital": _rm(ctx.get_total_capital()),
        "exchanges": exchanges_snapshot,
        "instruments": [str(i) for i in ctx.instruments],
        "is_warmup": ctx.is_warmup_in_progress,
        "is_simulation": ctx.is_simulation,
    }

    # Include custom state from @state-decorated methods
    custom = collect_state(ctx.strategy, ctx)
    if custom:
        data["custom"] = custom

    return ActionResult(status="ok", data=data)


def _get_total_capital(ctx: IStrategyContext, **kwargs) -> ActionResult:
    return ActionResult(
        status="ok",
        data={"total_capital": _rm(ctx.get_total_capital())},
    )


def _get_leverages(ctx: IStrategyContext, **kwargs) -> ActionResult:
    leverages = {}
    for instr, lev in ctx.get_leverages().items():
        leverages[str(instr)] = _rl(lev)
    return ActionResult(
        status="ok",
        data={
            "leverages": leverages,
            "net": _rl(ctx.get_net_leverage()),
            "gross": _rl(ctx.get_gross_leverage()),
        },
    )


def _get_subscriptions(ctx: IStrategyContext, symbol: str | None = None, **kwargs) -> ActionResult:
    if symbol:
        instr = _resolve(ctx, symbol)
        if instr is None:
            return ActionResult(status="error", error=f"Unknown symbol: {symbol}")
        subs = ctx.get_subscriptions(instr)
        return ActionResult(status="ok", data={"subscriptions": {str(instr): subs}})

    result = {}
    for instr in ctx.instruments:
        result[str(instr)] = ctx.get_subscriptions(instr)
    return ActionResult(status="ok", data={"subscriptions": result})


def _get_health(ctx: IStrategyContext, **kwargs) -> ActionResult:
    health = ctx.health
    exchanges = ctx.exchanges
    connected = {}
    latencies = {}
    for exch in exchanges:
        connected[exch] = health.is_connected(exch)
        latencies[exch] = {k: _rms(v) for k, v in health.get_data_latencies(exch).items()}
    return ActionResult(
        status="ok",
        data={
            "connected": connected,
            "queue_size": health.get_queue_size(),
            "data_latencies_ms": latencies,
        },
    )


def _get_state_schema(ctx: IStrategyContext, **kwargs) -> ActionResult:
    schema = collect_state_schema(ctx.strategy)
    return ActionResult(status="ok", data={"fields": schema})


# --- Trading actions ---


def _trade(ctx: IStrategyContext, symbol: str, amount: float, price: float | None = None, time_in_force: str = "gtc", **kwargs) -> ActionResult:
    instr = _resolve(ctx, symbol)
    if instr is None:
        return ActionResult(status="error", error=f"Unknown symbol: {symbol}")

    try:
        order = ctx.trade(instr, amount, price=price, time_in_force=time_in_force)
        return ActionResult(
            status="ok",
            data={"order_id": order.id if order else None, "instrument": str(instr), "amount": amount},
        )
    except Exception as e:
        return ActionResult(status="error", error=str(e))


def _set_target_position(ctx: IStrategyContext, symbol: str, target: float, price: float | None = None, **kwargs) -> ActionResult:
    instr = _resolve(ctx, symbol)
    if instr is None:
        return ActionResult(status="error", error=f"Unknown symbol: {symbol}")

    positions = ctx.get_positions()
    previous = positions[instr].quantity if instr in positions else 0.0

    try:
        ctx.set_target_position(instr, target, price=price)
        return ActionResult(status="ok", data={"previous": previous, "new": target, "instrument": str(instr)})
    except Exception as e:
        return ActionResult(status="error", error=str(e))


def _set_target_leverage(ctx: IStrategyContext, symbol: str, leverage: float, price: float | None = None, **kwargs) -> ActionResult:
    instr = _resolve(ctx, symbol)
    if instr is None:
        return ActionResult(status="error", error=f"Unknown symbol: {symbol}")

    try:
        ctx.set_target_leverage(instr, leverage, price=price)
        return ActionResult(status="ok", data={"instrument": str(instr), "leverage": leverage})
    except Exception as e:
        return ActionResult(status="error", error=str(e))


def _close_position(ctx: IStrategyContext, symbol: str, **kwargs) -> ActionResult:
    instr = _resolve(ctx, symbol)
    if instr is None:
        return ActionResult(status="error", error=f"Unknown symbol: {symbol}")

    positions = ctx.get_positions()
    amount = positions[instr].quantity if instr in positions else 0.0

    try:
        ctx.close_position(instr)
        return ActionResult(status="ok", data={"closed": True, "amount": amount, "instrument": str(instr)})
    except Exception as e:
        return ActionResult(status="error", error=str(e))


def _close_positions(ctx: IStrategyContext, **kwargs) -> ActionResult:
    try:
        positions = {str(i): p.quantity for i, p in ctx.get_positions().items() if p.quantity != 0}
        ctx.close_positions()
        return ActionResult(status="ok", data={"closed": positions})
    except Exception as e:
        return ActionResult(status="error", error=str(e))


def _cancel_orders(ctx: IStrategyContext, symbol: str | None = None, **kwargs) -> ActionResult:
    try:
        if symbol:
            instr = _resolve(ctx, symbol)
            if instr is None:
                return ActionResult(status="error", error=f"Unknown symbol: {symbol}")
            ctx.cancel_orders(instr)
        else:
            for instr in ctx.instruments:
                ctx.cancel_orders(instr)
        return ActionResult(status="ok", data={"cancelled": True})
    except Exception as e:
        return ActionResult(status="error", error=str(e))


def _emit_signal(ctx: IStrategyContext, symbol: str, signal_value: float, price: float | None = None, group: str = "", **kwargs) -> ActionResult:
    instr = _resolve(ctx, symbol)
    if instr is None:
        return ActionResult(status="error", error=f"Unknown symbol: {symbol}")

    try:
        sig = Signal(
            time=ctx.time(),
            instrument=instr,
            signal=signal_value,
            price=price,
            group=group,
        )
        ctx.emit_signal(sig)
        return ActionResult(status="ok", data={"emitted": True, "instrument": str(instr), "signal": signal_value})
    except Exception as e:
        return ActionResult(status="error", error=str(e))


# --- Action registry ---

_MARKET_TYPE_DOC = "Market type: SPOT, SWAP (perpetual futures), FUTURE (dated futures), OPTION, MARGIN"

BUILTIN_ACTIONS: dict[str, tuple[ActionDef, Callable]] = {
    # Universe
    "get_universe": (
        ActionDef(name="get_universe", description="Get current trading universe", category="universe", read_only=True),
        _get_universe,
    ),
    "add_instruments": (
        ActionDef(
            name="add_instruments",
            description="Add instruments to the trading universe. Symbols can include exchange prefix (e.g., BINANCE.UM:BTCUSDT) or use the exchange parameter.",
            category="universe",
            params=[
                ActionParam(name="symbols", type="array", description="List of symbol strings to add", items_type="string"),
                ActionParam(name="exchange", type="string", description="Exchange for symbol resolution (optional if symbols include exchange prefix)", required=False, default=None),
            ],
        ),
        _add_instruments,
    ),
    "remove_instruments": (
        ActionDef(
            name="remove_instruments",
            description="Remove instruments from the universe",
            category="universe",
            dangerous=True,
            params=[
                ActionParam(name="symbols", type="array", description="List of symbol strings to remove", items_type="string"),
                ActionParam(name="exchange", type="string", description="Exchange for symbol resolution (optional if symbols include exchange prefix)", required=False, default=None),
                ActionParam(name="if_has_position", type="string", description="Policy if instrument has position: close, wait_for_close, wait_for_change", required=False, default="close"),
            ],
        ),
        _remove_instruments,
    ),
    "set_universe": (
        ActionDef(
            name="set_universe",
            description="Replace the entire trading universe",
            category="universe",
            dangerous=True,
            params=[
                ActionParam(name="symbols", type="array", description="List of symbol strings for the new universe", items_type="string"),
                ActionParam(name="exchange", type="string", description="Exchange for symbol resolution (optional if symbols include exchange prefix)", required=False, default=None),
                ActionParam(name="if_has_position", type="string", description="Policy for removed instruments with positions: close, wait_for_close, wait_for_change", required=False, default="close"),
            ],
        ),
        _set_universe,
    ),
    # Instrument discovery
    "get_available_instruments": (
        ActionDef(
            name="get_available_instruments",
            description="Get all tradable instruments on an exchange. " + _MARKET_TYPE_DOC,
            category="discovery",
            read_only=True,
            params=[
                ActionParam(name="exchange", type="string", description="Exchange name (e.g., BINANCE.UM, KRAKEN, BYBIT)"),
                ActionParam(name="quote", type="string", description="Quote currency filter", required=False, default="USDT"),
                ActionParam(name="market_type", type="string", description=_MARKET_TYPE_DOC, required=False, default="SWAP"),
            ],
        ),
        _get_available_instruments,
    ),
    "get_instrument_details": (
        ActionDef(
            name="get_instrument_details",
            description="Get trading details (tick size, lot size, min notional, etc.) for a list of instruments",
            category="discovery",
            read_only=True,
            params=[
                ActionParam(name="symbols", type="array", description="List of symbol strings to get details for", items_type="string"),
                ActionParam(name="exchange", type="string", description="Exchange for symbol resolution (optional if symbols include exchange prefix)", required=False, default=None),
            ],
        ),
        _get_instrument_details,
    ),
    "get_top_instruments": (
        ActionDef(
            name="get_top_instruments",
            description="Get top N instruments ranked by turnover, market cap, or funding rate. Requires aux data configured.",
            category="discovery",
            read_only=True,
            params=[
                ActionParam(name="exchange", type="string", description="Exchange name (e.g., BINANCE.UM)"),
                ActionParam(name="count", type="integer", description="Number of top instruments to return", required=False, default=20),
                ActionParam(name="sort_by", type="string", description="Ranking metric: turnover, market_cap, or funding", required=False, default="turnover"),
                ActionParam(name="period", type="string", description="Lookback period (e.g., 3d, 7d, 1h)", required=False, default="3d"),
                ActionParam(name="timeframe", type="string", description="Candle timeframe for turnover (e.g., 1d, 1h)", required=False, default="1d"),
                ActionParam(name="quote", type="string", description="Quote currency filter", required=False, default="USDT"),
                ActionParam(name="market_type", type="string", description=_MARKET_TYPE_DOC, required=False, default="SWAP"),
            ],
        ),
        _get_top_instruments,
    ),
    # Diagnostics
    "get_positions": (
        ActionDef(name="get_positions", description="Get all current positions with PnL", category="diagnostics", read_only=True),
        _get_positions,
    ),
    "get_balances": (
        ActionDef(name="get_balances", description="Get account balances", category="diagnostics", read_only=True),
        _get_balances,
    ),
    "get_orders": (
        ActionDef(
            name="get_orders",
            description="Get open orders",
            category="diagnostics",
            read_only=True,
            params=[ActionParam(name="symbol", type="string", description="Filter by symbol", required=False, default=None)],
        ),
        _get_orders,
    ),
    "get_quote": (
        ActionDef(
            name="get_quote",
            description="Get latest quote for an instrument",
            category="diagnostics",
            read_only=True,
            params=[ActionParam(name="symbol", type="string", description="Trading instrument symbol")],
        ),
        _get_quote,
    ),
    "get_ohlc": (
        ActionDef(
            name="get_ohlc",
            description="Get recent OHLC bars",
            category="diagnostics",
            read_only=True,
            params=[
                ActionParam(name="symbol", type="string", description="Trading instrument symbol"),
                ActionParam(name="timeframe", type="string", description="Bar timeframe", required=False, default="1h"),
                ActionParam(name="length", type="integer", description="Number of bars", required=False, default=20),
            ],
        ),
        _get_ohlc,
    ),
    "get_state": (
        ActionDef(name="get_state", description="Get full bot state: per-exchange capital, leverage, positions, orders, balances, and custom strategy state", category="diagnostics", read_only=True),
        _get_state,
    ),
    "get_health": (
        ActionDef(name="get_health", description="Get health metrics: connectivity, queue size, data latencies", category="diagnostics", read_only=True),
        _get_health,
    ),
    "get_state_schema": (
        ActionDef(name="get_state_schema", description="Get descriptions of custom state fields returned by get_state", category="diagnostics", read_only=True),
        _get_state_schema,
    ),
    "get_total_capital": (
        ActionDef(name="get_total_capital", description="Get total capital across all exchanges", category="diagnostics", read_only=True),
        _get_total_capital,
    ),
    "get_leverages": (
        ActionDef(name="get_leverages", description="Get per-instrument and portfolio leverage", category="diagnostics", read_only=True),
        _get_leverages,
    ),
    "get_subscriptions": (
        ActionDef(
            name="get_subscriptions",
            description="Get active data subscriptions",
            category="diagnostics",
            read_only=True,
            params=[ActionParam(name="symbol", type="string", description="Filter by symbol", required=False, default=None)],
        ),
        _get_subscriptions,
    ),
    # Trading
    "trade": (
        ActionDef(
            name="trade",
            description="Place an order",
            category="trading",
            dangerous=True,
            params=[
                ActionParam(name="symbol", type="string", description="Trading instrument symbol"),
                ActionParam(name="amount", type="number", description="Order amount (positive=buy, negative=sell)"),
                ActionParam(name="price", type="number", description="Limit price", required=False, default=None),
                ActionParam(name="time_in_force", type="string", description="Time in force", required=False, default="gtc"),
            ],
        ),
        _trade,
    ),
    "set_target_position": (
        ActionDef(
            name="set_target_position",
            description="Set target position for an instrument",
            category="trading",
            dangerous=True,
            params=[
                ActionParam(name="symbol", type="string", description="Trading instrument symbol"),
                ActionParam(name="target", type="number", description="Target position size"),
                ActionParam(name="price", type="number", description="Limit price", required=False, default=None),
            ],
        ),
        _set_target_position,
    ),
    "set_target_leverage": (
        ActionDef(
            name="set_target_leverage",
            description="Set target leverage for an instrument",
            category="trading",
            dangerous=True,
            params=[
                ActionParam(name="symbol", type="string", description="Trading instrument symbol"),
                ActionParam(name="leverage", type="number", description="Target leverage"),
                ActionParam(name="price", type="number", description="Limit price", required=False, default=None),
            ],
        ),
        _set_target_leverage,
    ),
    "close_position": (
        ActionDef(
            name="close_position",
            description="Close position for an instrument",
            category="trading",
            dangerous=True,
            params=[ActionParam(name="symbol", type="string", description="Trading instrument symbol")],
        ),
        _close_position,
    ),
    "close_positions": (
        ActionDef(name="close_positions", description="Close all open positions", category="trading", dangerous=True),
        _close_positions,
    ),
    "cancel_orders": (
        ActionDef(
            name="cancel_orders",
            description="Cancel open orders",
            category="trading",
            params=[ActionParam(name="symbol", type="string", description="Filter by symbol", required=False, default=None)],
        ),
        _cancel_orders,
    ),
    "emit_signal": (
        ActionDef(
            name="emit_signal",
            description="Emit a trading signal",
            category="trading",
            dangerous=True,
            params=[
                ActionParam(name="symbol", type="string", description="Trading instrument symbol"),
                ActionParam(name="signal_value", type="number", description="Signal value (target position)"),
                ActionParam(name="price", type="number", description="Limit price", required=False, default=None),
                ActionParam(name="group", type="string", description="Signal group", required=False, default=""),
            ],
        ),
        _emit_signal,
    ),
}
