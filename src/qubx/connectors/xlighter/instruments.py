import time
from threading import Lock, Thread

import pandas as pd

from qubx import logger
from qubx.core.basics import Instrument, MarketType
from qubx.core.lookups import lookup


class LighterInstrumentLoader:
    """
    Loads and manages Lighter market instruments.
    """

    def __init__(self, update_interval: str = "1h"):
        self._lock = Lock()
        self._update_interval = pd.Timedelta(update_interval)
        self.market_id_to_instrument: dict[int, Instrument] = {}
        self.symbol_to_market_id: dict[str, int] = {}
        self._update_instruments()
        self._thread = Thread(target=self._update_instruments_loop, daemon=True)
        self._thread.start()

    def _update_instruments_loop(self):
        while True:
            time.sleep(self._update_interval.total_seconds())
            self._update_instruments()

    def _update_instruments(self):
        try:
            instruments = lookup.find_instruments(exchange="LIGHTER", market_type=MarketType.SWAP)
        except Exception as e:
            logger.error(f"Error updating instruments: {e}")
            return

        with self._lock:
            self.instruments = instruments
            self.market_id_to_instrument = {int(instrument.exchange_symbol): instrument for instrument in instruments}
            self.symbol_to_market_id = {
                instrument.symbol: int(instrument.exchange_symbol) for instrument in instruments
            }

    def get_instrument_by_market_id(self, market_id: int) -> Instrument | None:
        with self._lock:
            return self.market_id_to_instrument.get(market_id)

    def get_instrument_by_symbol(self, symbol: str) -> Instrument | None:
        with self._lock:
            market_id = self.symbol_to_market_id.get(symbol)
            if not market_id:
                return None
            instrument = self.market_id_to_instrument.get(market_id)
            if not instrument:
                return None
            return instrument

    def get_market_id(self, symbol: str) -> int | None:
        with self._lock:
            return self.symbol_to_market_id.get(symbol)
