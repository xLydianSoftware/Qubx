import asyncio

from qubx import logger
from qubx.connectors.ccxt.account import CcxtAccountProcessor
from qubx.connectors.ccxt.utils import (
    ccxt_convert_order_info,
    ccxt_extract_deals_from_exec,
    ccxt_find_instrument,
)
from qubx.core.basics import AssetBalance, CtrlChannel, Instrument


class OkxAccountProcessor(CcxtAccountProcessor):
    """
    OKX-specific account processor.

    OKX's watch_orders does NOT include trades/fills in the response,
    so we must subscribe to both channels separately:
    - orders channel: for order status updates (watch_orders)
    - trades channel: for trade/fill updates (watch_my_trades)
    """

    async def _subscribe_executions(self, name: str, channel: CtrlChannel):
        logger.info("<yellow>[OKX]</yellow> Subscribing to executions (orders + trades)")
        _symbol_to_instrument = {}

        async def _watch_orders():
            orders = await self.exchange_manager.exchange.watch_orders()
            for order in orders:
                instrument = ccxt_find_instrument(
                    order["symbol"], self.exchange_manager.exchange, _symbol_to_instrument
                )
                order = ccxt_convert_order_info(instrument, order)
                channel.send((instrument, "order", order, False))

        async def _watch_my_trades():
            trades = await self.exchange_manager.exchange.watch_my_trades()
            for trade in trades:
                instrument = ccxt_find_instrument(
                    trade["symbol"], self.exchange_manager.exchange, _symbol_to_instrument
                )
                deals = ccxt_extract_deals_from_exec({"trades": [trade]})
                channel.send((instrument, "deals", deals, False))

        await asyncio.gather(
            self._listen_to_stream(
                subscriber=_watch_orders,
                exchange=self.exchange_manager.exchange,
                channel=channel,
                name=f"{name}_orders",
            ),
            self._listen_to_stream(
                subscriber=_watch_my_trades,
                exchange=self.exchange_manager.exchange,
                channel=channel,
                name=f"{name}_trades",
            ),
        )

    async def _update_balance(self) -> None:
        """OKX balance override: use cashBal instead of eq (equity) to avoid double-counting unrealized PnL.

        CCXT maps OKX's `eq` (equity = cashBal + unrealizedPnL) to balance `total`.
        Since get_total_capital() adds position market values (= unrealized PnL) on top,
        using `eq` would double-count unrealized PnL. We use `cashBal` instead.
        """
        balances_raw = await self.exchange_manager.exchange.fetch_balance()

        # Extract cashBal from raw OKX response instead of CCXT-parsed eq
        balances: list[AssetBalance] = []
        raw_details = balances_raw.get("info", {}).get("data", [{}])[0].get("details", [])
        for detail in raw_details:
            cash_bal = float(detail.get("cashBal", 0) or 0)
            if not cash_bal:
                continue
            frozen_bal = float(detail.get("frozenBal", 0) or 0)
            currency = detail["ccy"]
            balances.append(
                AssetBalance(
                    exchange=self.exchange,
                    currency=currency,
                    free=cash_bal - frozen_bal,
                    locked=frozen_bal,
                    total=cash_bal,
                )
            )

        current_balances = self.get_balances()

        # remove balances that are not there anymore
        current_currencies = {b.currency for b in current_balances}
        new_currencies = {b.currency for b in balances}
        for currency in current_currencies - new_currencies:
            self.update_balance(currency, 0, 0)

        # update current balances
        for balance in balances:
            self.update_balance(currency=balance.currency, total=balance.total, locked=balance.locked)

        # update required instruments that we need to subscribe to
        currencies = [b.currency for b in self.get_balances()]
        instruments = [
            self._get_instrument_for_currency(c) for c in currencies if c.upper() != self.base_currency.upper()
        ]
        self._required_instruments.update(instruments)

        # fetch tickers for instruments that don't have recent price updates
        await self._fetch_missing_tickers(instruments)

    async def _subscribe_instruments(self, instruments: list) -> None:
        """Skip SPOT instruments on OKX.F - futures connector only handles swap contracts."""
        await super()._subscribe_instruments([i for i in instruments if i.is_futures()])

    async def _init_spot_positions(self) -> None:
        logger.debug("Skipping spot position initialization for OKX")

    async def _fetch_missing_tickers(self, instruments: list[Instrument]) -> None:
        # skip spot
        instruments = [i for i in instruments if i.is_futures()]
        await super()._fetch_missing_tickers(instruments)
