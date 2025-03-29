from qubx.backtester.simulated_exchange import ISimulatedExchange
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import (
    CtrlChannel,
    Instrument,
    Order,
    Position,
    Timestamped,
    dt_64,
)
from qubx.core.series import OrderBook, Quote, Trade, TradeArray
from qubx.restorers import RestoredState


class SimulatedAccountProcessor(BasicAccountProcessor):
    _channel: CtrlChannel
    _exchange: ISimulatedExchange

    def __init__(
        self,
        account_id: str,
        exchange: ISimulatedExchange,
        channel: CtrlChannel,
        base_currency: str,
        initial_capital: float,
        restored_state: RestoredState | None = None,
    ) -> None:
        super().__init__(
            account_id=account_id,
            time_provider=exchange.get_time_provider(),
            base_currency=base_currency,
            tcc=exchange.get_transaction_costs_calculator(),
            initial_capital=initial_capital,
        )

        self._exchange = exchange
        self._channel = channel

        if restored_state is not None:
            self._balances.update(restored_state.balances)
            for instrument, position in restored_state.positions.items():
                _pos = self.get_position(instrument)
                _pos.reset_by_position(position)

    def get_orders(self, instrument: Instrument | None = None) -> dict[str, Order]:
        return self._exchange.get_open_orders(instrument)

    def get_position(self, instrument: Instrument) -> Position:
        if instrument in self.positions:
            return self.positions[instrument]

        # - initialize empty position
        position = Position(instrument)  # type: ignore
        self.attach_positions(position)
        return self.positions[instrument]

    def update_position_price(self, time: dt_64, instrument: Instrument, update: float | Timestamped) -> None:
        self.get_position(instrument)

        super().update_position_price(time, instrument, update)

        quote = (
            update if isinstance(update, Quote) else self._exchange.emulate_quote_from_data(instrument, time, update)
        )
        if quote is None:
            return

        # - process new data
        self._process_new_data(instrument, quote)

    def process_market_data(self, time: dt_64, instrument: Instrument, update: Timestamped) -> None:
        if isinstance(update, (TradeArray, Quote, Trade, OrderBook)):
            # - process new data
            self._process_new_data(instrument, update)

        super().process_market_data(time, instrument, update)

    def _process_new_data(self, instrument: Instrument, data: Quote | OrderBook | Trade | TradeArray) -> None:
        for r in self._exchange.process_market_data(instrument, data):
            if r.exec is not None:
                # - process methods will be called from stg context
                self._channel.send((instrument, "order", r.order, False))
                self._channel.send((instrument, "deals", [r.exec], False))
