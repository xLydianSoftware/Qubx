"""Built-in control actions available on every bot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from qubx.core.basics import Signal

from .types import ActionDef, ActionParam, ActionResult

if TYPE_CHECKING:
    from qubx.core.interfaces import IStrategyContext


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


def _get_universe(ctx: IStrategyContext, **kwargs) -> ActionResult:
    instruments = ctx.instruments
    return ActionResult(
        status="ok",
        data={"instruments": [str(i) for i in instruments], "count": len(instruments)},
    )


def _add_instruments(ctx: IStrategyContext, symbols: list[str], **kwargs) -> ActionResult:
    instruments = []
    errors = []
    for s in symbols:
        instr = ctx.query_instrument(s)
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


def _remove_instruments(ctx: IStrategyContext, symbols: list[str], if_has_position: str = "close", **kwargs) -> ActionResult:
    instruments = []
    errors = []
    for s in symbols:
        instr = ctx.query_instrument(s)
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


def _get_positions(ctx: IStrategyContext, **kwargs) -> ActionResult:
    positions = {}
    for instr, pos in ctx.get_positions().items():
        positions[str(instr)] = {
            "quantity": pos.quantity,
            "avg_price": pos.position_avg_price,
            "market_price": pos.last_update_price,
            "pnl": _rm(pos.pnl),
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
        instr = ctx.query_instrument(symbol)
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
    instr = ctx.query_instrument(symbol)
    if instr is None:
        return ActionResult(status="error", error=f"Unknown symbol: {symbol}")

    quote = ctx.quote(instr)
    if quote is None:
        return ActionResult(status="error", error=f"No quote available for {symbol}")

    import pandas as pd

    return ActionResult(
        status="ok",
        data={
            "bid": quote.bid,
            "ask": quote.ask,
            "bid_size": quote.bid_size,
            "ask_size": quote.ask_size,
            "time": str(pd.Timestamp(quote.time, unit="ns")),
        },
    )


def _get_ohlc(ctx: IStrategyContext, symbol: str, timeframe: str = "1h", length: int = 20, **kwargs) -> ActionResult:
    instr = ctx.query_instrument(symbol)
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


def _trade(ctx: IStrategyContext, symbol: str, amount: float, price: float | None = None, time_in_force: str = "gtc", **kwargs) -> ActionResult:
    instr = ctx.query_instrument(symbol)
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
    instr = ctx.query_instrument(symbol)
    if instr is None:
        return ActionResult(status="error", error=f"Unknown symbol: {symbol}")

    positions = ctx.get_positions()
    previous = positions[instr].quantity if instr in positions else 0.0

    try:
        ctx.set_target_position(instr, target, price=price)
        return ActionResult(status="ok", data={"previous": previous, "new": target, "instrument": str(instr)})
    except Exception as e:
        return ActionResult(status="error", error=str(e))


def _close_position(ctx: IStrategyContext, symbol: str, **kwargs) -> ActionResult:
    instr = ctx.query_instrument(symbol)
    if instr is None:
        return ActionResult(status="error", error=f"Unknown symbol: {symbol}")

    positions = ctx.get_positions()
    amount = positions[instr].quantity if instr in positions else 0.0

    try:
        ctx.close_position(instr)
        return ActionResult(status="ok", data={"closed": True, "amount": amount, "instrument": str(instr)})
    except Exception as e:
        return ActionResult(status="error", error=str(e))


def _cancel_orders(ctx: IStrategyContext, symbol: str | None = None, **kwargs) -> ActionResult:
    try:
        if symbol:
            instr = ctx.query_instrument(symbol)
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
    instr = ctx.query_instrument(symbol)
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


def _get_state(ctx: IStrategyContext, **kwargs) -> ActionResult:
    positions = {}
    for instr, pos in ctx.get_positions().items():
        positions[str(instr)] = {
            "quantity": pos.quantity,
            "avg_price": pos.position_avg_price,
            "market_price": pos.last_update_price,
            "pnl": _rm(pos.pnl),
        }

    return ActionResult(
        status="ok",
        data={
            "total_capital": _rm(ctx.get_total_capital()),
            "net_leverage": _rl(ctx.get_net_leverage()),
            "gross_leverage": _rl(ctx.get_gross_leverage()),
            "positions": positions,
            "instruments": [str(i) for i in ctx.instruments],
            "is_warmup": ctx.is_warmup_in_progress,
            "is_simulation": ctx.is_simulation,
        },
    )


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
        instr = ctx.query_instrument(symbol)
        if instr is None:
            return ActionResult(status="error", error=f"Unknown symbol: {symbol}")
        subs = ctx.get_subscriptions(instr)
        return ActionResult(status="ok", data={"subscriptions": {str(instr): subs}})

    result = {}
    for instr in ctx.instruments:
        result[str(instr)] = ctx.get_subscriptions(instr)
    return ActionResult(status="ok", data={"subscriptions": result})


def _set_universe(ctx: IStrategyContext, symbols: list[str], if_has_position: str = "close", **kwargs) -> ActionResult:
    instruments = []
    errors = []
    for s in symbols:
        instr = ctx.query_instrument(s)
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


def _set_target_leverage(ctx: IStrategyContext, symbol: str, leverage: float, price: float | None = None, **kwargs) -> ActionResult:
    instr = ctx.query_instrument(symbol)
    if instr is None:
        return ActionResult(status="error", error=f"Unknown symbol: {symbol}")

    try:
        ctx.set_target_leverage(instr, leverage, price=price)
        return ActionResult(status="ok", data={"instrument": str(instr), "leverage": leverage})
    except Exception as e:
        return ActionResult(status="error", error=str(e))


def _close_positions(ctx: IStrategyContext, **kwargs) -> ActionResult:
    try:
        positions = {str(i): p.quantity for i, p in ctx.get_positions().items() if p.quantity != 0}
        ctx.close_positions()
        return ActionResult(status="ok", data={"closed": positions})
    except Exception as e:
        return ActionResult(status="error", error=str(e))


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


# --- Action registry ---

BUILTIN_ACTIONS: dict[str, tuple[ActionDef, Callable]] = {
    "get_universe": (
        ActionDef(name="get_universe", description="Get current trading universe", category="universe", read_only=True),
        _get_universe,
    ),
    "add_instruments": (
        ActionDef(
            name="add_instruments",
            description="Add instruments to the trading universe",
            category="universe",
            params=[ActionParam(name="symbols", type="array", description="List of symbol strings to add", items_type="string")],
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
                ActionParam(name="if_has_position", type="string", description="Policy if instrument has position: close, wait_for_close, wait_for_change", required=False, default="close"),
            ],
        ),
        _remove_instruments,
    ),
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
    "get_state": (
        ActionDef(name="get_state", description="Get full bot state dump", category="diagnostics", read_only=True),
        _get_state,
    ),
    "get_health": (
        ActionDef(name="get_health", description="Get health metrics", category="diagnostics", read_only=True),
        _get_health,
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
    "set_universe": (
        ActionDef(
            name="set_universe",
            description="Replace the entire trading universe",
            category="universe",
            dangerous=True,
            params=[
                ActionParam(name="symbols", type="array", description="List of symbol strings for the new universe", items_type="string"),
                ActionParam(name="if_has_position", type="string", description="Policy for removed instruments with positions: close, wait_for_close, wait_for_change", required=False, default="close"),
            ],
        ),
        _set_universe,
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
    "close_positions": (
        ActionDef(name="close_positions", description="Close all open positions", category="trading", dangerous=True),
        _close_positions,
    ),
}
