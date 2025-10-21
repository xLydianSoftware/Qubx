import gzip
import os
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from os.path import exists, join
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pandas as pd
from numba import njit, types
from numba.typed import Dict
from sortedcontainers import SortedDict
from tqdm.auto import tqdm

from qubx import logger
from qubx.core.basics import Instrument, dt_64
from qubx.core.lookups import lookup
from qubx.core.series import OrderBook, time_as_nsec
from qubx.pandaz.utils import scols, srows
from qubx.utils.numbers_utils import count_decimal_places


@njit
def prec_floor(a: float, precision: int) -> float:
    return np.sign(a) * np.true_divide(np.floor(round(abs(a) * 10**precision, precision)), 10**precision)


@njit
def prec_ceil(a: float, precision: int):
    return np.sign(a) * np.true_divide(np.ceil(round(abs(a) * 10**precision, precision)), 10**precision)


@njit
def get_tick(price: float, is_bid: bool, tick_size: float):
    if is_bid:
        return int(np.floor(round(price / tick_size, 1)))
    else:
        return int(np.ceil(round(price / tick_size, 1)))


@njit
def tick_to_price(tick: int, tick_size: float, decimals: int):
    return round(tick * tick_size, decimals)


@njit
def get_tick_price(price: float, is_bid: bool, tick_size: float, decimals: int):
    return tick_to_price(get_tick(price, is_bid, tick_size), tick_size, decimals)


@njit
def _interpolate_levels(
    levels: list[tuple[float, float]],
    is_bid: bool,
    tick_count: int,
    tick_size: float,
    decimals: int,
    size_decimals: int,
    sizes_in_quoted: bool,
):
    # TODO: asks are not interpolated correctly
    prices = []
    for price, size in levels:
        prices.append(price)

    if is_bid:
        max_tick = get_tick(max(prices), is_bid, tick_size)
        min_tick = max_tick - tick_count + 1
        start_tick = max_tick
    else:
        min_tick = get_tick(min(prices), is_bid, tick_size)
        max_tick = min_tick + tick_count - 1
        start_tick = min_tick

    # Initialize a dictionary to hold the aggregated sizes
    interp_levels = Dict.empty(key_type=types.float64, value_type=types.float64)

    # Iterate through each bid and aggregate the sizes based on the tick size
    for price, size in levels:
        tick = get_tick(price, is_bid, tick_size)
        if tick >= min_tick and tick <= max_tick:
            _size = (price * size) if sizes_in_quoted else size
            if tick in interp_levels:
                interp_levels[tick] += _size
            else:
                interp_levels[tick] = _size

    # Create the final list including zero sizes where necessary
    result = []
    for tick in range(min_tick, max_tick + 1):
        size = round(interp_levels[tick], size_decimals) if tick in interp_levels else 0.0
        idx = tick - start_tick
        result.append((-idx if is_bid else idx, size))

    return result, tick_to_price(max_tick if is_bid else min_tick, tick_size, decimals)


@njit
def __build_orderbook_snapshots(
    dates: np.ndarray,
    prices: np.ndarray,
    sizes: np.ndarray,
    is_bids: np.ndarray,
    levels: int,
    tick_size_fraction: float,
    price_decimals: int,
    size_decimals: int,
    sizes_in_quoted: bool,
    init_bid_ticks: np.ndarray,
    init_bid_sizes: np.ndarray,
    init_ask_ticks: np.ndarray,
    init_ask_sizes: np.ndarray,
    init_top_bid: float,
    init_top_ask: float,
    init_tick_size: float,
) -> list[tuple[np.datetime64, list[tuple[float, float]], list[tuple[float, float]], float, float, float]]:
    """
    Build order book snapshots from given market data.

    Parameters:
        dates (np.ndarray): Array of datetime64 timestamps.
        prices (np.ndarray): Array of price points.
        sizes (np.ndarray): Array of sizes corresponding to the prices.
        is_bids (np.ndarray): Array indicating if the price is a bid (True) or ask (False).
        levels (int): Number of levels to interpolate for bids and asks.
        tick_size_fraction (float): Fraction to determine the tick size dynamically based on mid-price.
        price_decimals (int): Number of decimal places for price rounding.
        size_decimals (int): Number of decimal places for size rounding.
        sizes_in_quoted (bool): Flag indicating if sizes are in quoted currency.
        init_bid_ticks (np.ndarray): Initial bid ticks.
        init_bid_sizes (np.ndarray): Initial bid sizes.
        init_ask_ticks (np.ndarray): Initial ask ticks.
        init_ask_sizes (np.ndarray): Initial ask sizes.
        init_top_bid (float): Initial top bid price.
        init_top_ask (float): Initial top ask price.
        init_tick_size (float): Initial tick size.

    Returns:
    list[tuple[np.datetime64, list[tuple[float, float]], list[tuple[float, float]], float, float, float]]:
        A list of tuples where each tuple contains:
        - Timestamp of the snapshot.
        - List of interpolated bid levels (price, size).
        - List of interpolated ask levels (price, size).
        - Top bid price.
        - Top ask price.
        - Tick size.
    """
    price_to_size = Dict.empty(key_type=types.float64, value_type=types.float64)
    price_to_bid_ask = Dict.empty(key_type=types.float64, value_type=types.boolean)

    for i in range(init_bid_ticks.shape[0]):
        bp = init_top_bid - init_tick_size * init_bid_ticks[i]
        price_to_size[bp] = init_bid_sizes[i]
        price_to_bid_ask[bp] = True

    for i in range(init_ask_ticks.shape[0]):
        ap = init_top_ask + init_tick_size * init_ask_ticks[i]
        price_to_size[ap] = init_ask_sizes[i]
        price_to_bid_ask[ap] = False

    snapshots = []
    prev_timestamp = dates[0]
    for i in range(dates.shape[0]):
        date = dates[i]
        if date > prev_timestamp:
            # emit snapshot
            bids, asks = [], []
            top_a, top_b = np.inf, 0
            for price, size in price_to_size.items():
                if price_to_bid_ask[price]:
                    bids.append((price, size))
                    top_b = max(top_b, price)
                else:
                    asks.append((price, size))
                    top_a = min(top_a, price)

            if len(bids) > 0 and len(asks) > 0:
                # - find tick_size dynamically based on mid_price
                tick_size = prec_ceil(0.5 * (top_b + top_a) * tick_size_fraction, price_decimals)
                interp_bids, top_bid_price = _interpolate_levels(
                    bids,
                    True,
                    levels,
                    tick_size,
                    price_decimals,
                    size_decimals,
                    sizes_in_quoted,
                )
                interp_asks, top_ask_price = _interpolate_levels(
                    asks,
                    False,
                    levels,
                    tick_size,
                    price_decimals,
                    size_decimals,
                    sizes_in_quoted,
                )
                if len(interp_bids) >= levels and len(interp_asks) >= levels:
                    if top_bid_price <= top_ask_price:
                        snapshots.append(
                            (
                                prev_timestamp,
                                interp_bids[-levels:],
                                interp_asks[:levels],
                                # - also store top bid, ask prices and tick_size
                                top_b,
                                top_a,
                                tick_size,
                            )
                        )
                    else:
                        # something went wrong, bids can't be above asks
                        # clean up the local state and hope for the best
                        price_to_size.clear()
                        price_to_bid_ask.clear()

        price = prices[i]
        size = sizes[i]
        is_bid = is_bids[i]
        if size == 0:
            if price in price_to_size:
                del price_to_size[price]
            if price in price_to_bid_ask:
                del price_to_bid_ask[price]
        else:
            price_to_size[price] = size
            price_to_bid_ask[price] = is_bid

        prev_timestamp = date

    return snapshots


def build_orderbook_snapshots(
    updates: list[tuple[np.datetime64, float, float, bool]],
    levels: int,
    tick_size_pct: float,
    min_tick_size: float,
    min_size_step: float,
    sizes_in_quoted: bool = False,
    initial_snapshot: (
        tuple[
            np.datetime64,  # timestamp   [0]
            list[tuple[float, float]],  # bids levels [1]
            list[tuple[float, float]],  # asks levels [2]
            float,
            float,
            float,  # top bid, top ask prices, tick_size [3, 4, 5]
        ]
        | None
    ) = None,
):
    dates, prices, sizes, is_bids = zip(*updates)
    dates = np.array(dates, dtype=np.datetime64)
    prices = np.array(prices)
    sizes = np.array(sizes)
    is_bids = np.array(is_bids)

    price_decimals = max(count_decimal_places(min_tick_size), 1)
    size_decimals = max(count_decimal_places(min_size_step), 1)

    if initial_snapshot is not None and dates[0] > initial_snapshot[0]:
        init_bid_ticks, init_bid_sizes = zip(*initial_snapshot[1])
        init_ask_ticks, init_ask_sizes = zip(*initial_snapshot[2])
        init_bid_ticks = np.array(init_bid_ticks, dtype=np.float64)
        init_bid_sizes = np.array(init_bid_sizes, dtype=np.float64)
        init_ask_ticks = np.array(init_ask_ticks, dtype=np.float64)
        init_ask_sizes = np.array(init_ask_sizes, dtype=np.float64)
        init_top_bid = initial_snapshot[3]
        init_top_ask = initial_snapshot[4]
        init_tick_size = initial_snapshot[5]
    else:
        init_bid_ticks = np.array([], dtype=np.float64)
        init_bid_sizes = np.array([], dtype=np.float64)
        init_ask_ticks = np.array([], dtype=np.float64)
        init_ask_sizes = np.array([], dtype=np.float64)
        init_top_bid, init_top_ask, init_tick_size = 0, 0, 0

    snapshots = __build_orderbook_snapshots(
        dates,
        prices,
        sizes,
        is_bids,
        levels,
        tick_size_pct / 100,
        price_decimals,
        size_decimals,
        sizes_in_quoted,
        init_bid_ticks,
        init_bid_sizes,
        init_ask_ticks,
        init_ask_sizes,
        init_top_bid,
        init_top_ask,
        init_tick_size,
    )
    return snapshots


def snapshots_to_frame(snaps: list) -> pd.DataFrame:
    """
    Convert snapshots to dataframe
    """
    reindx = lambda s, d: {f"{s}{k}": v for k, v in d.items()}  # noqa: E731
    data = {
        snaps[i][0]: (
            reindx("b", dict(snaps[i][1]))
            | reindx("a", dict(snaps[i][2]))
            | {"top_bid": snaps[i][3], "top_ask": snaps[i][4], "tick_size": snaps[i][5]}
        )
        for i in range(len(snaps))
    }
    return pd.DataFrame.from_dict(data).T


def read_and_process_orderbook_updates(
    exchange: str,
    path: str,
    price_bin_pct: float,
    n_levels: int,
    sizes_in_quoted=False,
    symbols: list[str] | None = None,
    dates: slice | None = None,
    path_to_store: str | None = None,
    collect_snapshots: bool = True,
) -> dict[str, dict[datetime, pd.DataFrame]]:
    # QubxLogConfig.set_log_level("INFO")

    # - preprocess ranges
    dates_start = pd.Timestamp(dates.start if dates and dates.start else "1970-01-01")
    dates_stop = pd.Timestamp(dates.stop if dates and dates.stop else "2170-01-01")
    dates_start, dates_stop = min(dates_start, dates_stop), max(dates_start, dates_stop)

    def __process_updates_record(line: str):
        data = msgspec.json.decode(line)
        # - we need only full depth here !
        if (s_d := data.get("stream")) is not None and s_d[-6:] == "@depth":
            update = data["data"]
            if update.get("e") == "depthUpdate":
                ts = datetime.fromtimestamp(update["E"] / 1000)
                for is_bid, key in [(True, "b"), (False, "a")]:
                    for price, size in update[key]:
                        yield (ts, float(price), float(size), is_bid)

    symb_snapshots = defaultdict(dict)
    for s in Path(path).glob("*"):
        symbol = s.name.upper()

        # - skip if list is defined but symbol not in it
        if symbols and symbol not in symbols:
            continue

        instr = lookup.find_symbol(exchange.upper(), symbol)
        if not isinstance(instr, Instrument):
            logger.error(f"Instrument not found for {symbol} !")
            continue

        _latest_snapshot = None
        for d in sorted(s.glob("raw/*")):
            _d_ts = pd.Timestamp(d.name)
            if _d_ts < dates_start or _d_ts > dates_stop:
                continue

            if path_to_store and exists(_f := get_path_to_snapshots_file(path_to_store, symbol, _d_ts)):
                logger.info(f"File {_f} already exists, skipping.")
                continue

            day_updates = []
            logger.info(f"Loading {symbol} : {d.name} ... ")
            for file in sorted(d.glob("*.txt.gz")):
                try:
                    with gzip.open(file, "rt") as f:
                        try:
                            while line := f.readline():
                                for upd in __process_updates_record(line):
                                    day_updates.append(upd)
                        except Exception as exc:
                            logger.warning(f">>> Exception in processing {file.name} : {exc}")
                            # logger.opt(colors=False).error(traceback.format_exc())
                except EOFError as exc:
                    logger.error(f">>> Exception in reading {exc}")
                    logger.opt(colors=False).error(traceback.format_exc())

            if len(day_updates) == 0:
                logger.info(f"No data for {symbol} at {d.name}")
                continue

            logger.info(f"loaded {len(day_updates)} updates")

            snaps = build_orderbook_snapshots(
                day_updates,
                n_levels,
                price_bin_pct,
                instr.tick_size,
                instr.lot_size,
                sizes_in_quoted=sizes_in_quoted,
                initial_snapshot=_latest_snapshot,
            )
            _latest_snapshot = snaps[-1]

            processed_snap = snapshots_to_frame(snaps)
            t_key = pd.Timestamp(d.name).strftime("%Y-%m-%d")

            # - collect snapshots
            if collect_snapshots:
                symb_snapshots[symbol][t_key] = processed_snap

            # - save data
            if path_to_store:
                store_snapshots_to_h5(path_to_store, {symbol: {t_key: processed_snap}}, price_bin_pct, n_levels)

    return symb_snapshots


def get_combined_cumulative_snapshot(data: dict[str, dict[datetime, pd.DataFrame]], max_levs=1000000) -> pd.DataFrame:
    frms = []
    for s, dv in data.items():
        _f = {}
        for d, v in dv.items():
            ca = v.mean(axis=0).filter(regex="^a.*")[:max_levs].cumsum(axis=0)
            cb = v.mean(axis=0).filter(regex="^b.*")[::-1][:max_levs].cumsum(axis=0)
            _f[pd.Timestamp(d)] = srows(ca[::-1], cb, sort=False).to_dict()
        frms.append(pd.DataFrame.from_dict(_f, orient="index"))
    return scols(*frms, keys=data.keys())


def get_path_to_snapshots_file(path: str, symbol: str, date: str) -> str:
    _s_path = join(path, symbol.upper())
    if not os.path.exists(_s_path):
        os.makedirs(_s_path)
    return join(_s_path, pd.Timestamp(date).strftime("%Y-%m-%d")) + ".h5"


def store_snapshots_to_h5(path: str, data: dict[str, dict[str, pd.DataFrame]], p, nl):
    """
    Store orderbook data to HDF5 files
    """
    for s, v in data.items():
        for t, vd in v.items():
            logger.info(f"Storing {s} : {t}")
            vd.to_hdf(
                get_path_to_snapshots_file(path, s, t), key=f"orderbook_{str(p).replace('.', '_')}_{nl}", complevel=9
            )


def load_snapshots_from_h5(path: str, symbol: str, dates: slice | str, p: float, nl: int) -> dict[str, pd.DataFrame]:
    symbol = symbol.upper()
    if isinstance(dates, slice):
        dates_start = pd.Timestamp(dates.start if dates and dates.start else "1970-01-01")
        dates_stop = pd.Timestamp(dates.stop if dates and dates.stop else "2170-01-01")
    else:
        dates_start = pd.Timestamp(dates)
        dates_stop = pd.Timestamp(dates)
    dates_start, dates_stop = min(dates_start, dates_stop), max(dates_start, dates_stop)
    rs = {symbol: {}}
    for d in tqdm(sorted((Path(path) / symbol).glob("*.h*"))):
        _d_ts = pd.Timestamp(d.name.split(".")[0])
        if _d_ts < dates_start or _d_ts > dates_stop:
            continue
        rs[symbol][_d_ts] = pd.read_hdf(d, f"orderbook_{str(p).replace('.', '_')}_{nl}")
    return rs


def aggregate_symbol(path: str, symbol: str, p: float, nl: int, reload=False) -> pd.DataFrame | None:
    """
    Aggregate orderbook data for a symbol on a daily basis and save to HDF5 file
    """
    symbol = symbol.upper()
    result = None
    with pd.HDFStore(f"{path}/aggregated.h5", "a", complevel=9) as store:
        if reload or (f"/{symbol}" not in store.keys()):
            _f = {}
            for d in tqdm(sorted((Path(path) / symbol).glob("*.h*")), leave=False, desc=symbol):
                date = d.name.split(".")[0]
                rs = pd.read_hdf(d, f"orderbook_{str(p).replace('.', '_')}_{nl}")
                rs = rs.loc[date]
                if not rs.empty:
                    ca = rs.mean(axis=0).filter(regex="^a.*").cumsum(axis=0)
                    cb = rs.mean(axis=0).filter(regex="^b.*")[::-1].cumsum(axis=0)
                    _f[pd.Timestamp(date)] = srows(ca[::-1], cb, sort=False).to_dict()
            result = pd.DataFrame.from_dict(_f, orient="index")
            store.put(symbol, result)
    return result


def aggregate_symbols_from_list(path: str, symbols: list[str] | dict[str, Any], p: float, nl: int, reload=False):
    """
    Aggregate orderbook data for a list of symbols on a daily basis and save to HDF5 file
    """
    for s in tqdm(symbols):
        aggregate_symbol(path, s, p, nl, reload)


@njit
def accumulate_orderbook_levels(
    raw_levels: np.ndarray, buffer: np.ndarray, tick_size: float, is_bid: bool, levels: int, sizes_in_quoted: bool
) -> tuple[float, np.ndarray]:
    """
    Accumulate order book levels into price buckets based on tick size.

    Parameters:
        raw_levels (list): List of [price, size] pairs from the raw order book
        buffer (np.ndarray): Pre-allocated buffer to store accumulated sizes
        tick_size (float): The tick size to use for price bucketing
        is_bid (bool): Whether these are bid levels (True) or ask levels (False)
        levels (int): Number of price levels to include
        sizes_in_quoted (bool): Whether sizes are in quoted currency

    Returns:
        tuple: (top_price, accumulated_sizes)
    """
    if len(raw_levels) == 0:
        return 0.0, buffer

    # Find the top price (highest bid or lowest ask)
    top_price = raw_levels[0][0]

    # Calculate price buckets and accumulate sizes
    for price, size in raw_levels:
        if is_bid:
            # For bids, we floor the price to the nearest tick
            idx = int(np.floor((top_price - price) / tick_size))
        else:
            # For asks, we ceil the price to the nearest tick
            idx = int(np.floor((price - top_price) / tick_size))

        # Only accumulate if within our desired number of levels
        if 0 <= idx < levels:
            # Convert size to quoted currency if needed
            size_to_add = price * size if sizes_in_quoted else size
            buffer[idx] += size_to_add

    return top_price, buffer


@dataclass
class OrderBookState:
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]

    @property
    def bid_price(self) -> float:
        if not self.bids:
            raise ValueError("No bids available")
        return max(self.bids, key=lambda x: x[0])[0]

    @property
    def ask_price(self) -> float:
        if not self.asks:
            raise ValueError("No asks available")
        return min(self.asks, key=lambda x: x[0])[0]

    @property
    def bid_size(self) -> float:
        if not self.bids:
            raise ValueError("No bids available")
        return max(self.bids, key=lambda x: x[0])[1]

    @property
    def ask_size(self) -> float:
        if not self.asks:
            raise ValueError("No asks available")
        return min(self.asks, key=lambda x: x[0])[1]

    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            raise ValueError("No bids or asks available")
        return (self.bid_price + self.ask_price) / 2


class OrderBookStateManager:
    """
    Manages orderbook state with efficient updates and lookups.

    Uses SortedDict for maintaining price levels in sorted order without
    explicit sorting on each access. Pre-allocates numpy buffers to avoid
    allocation overhead on frequent orderbook generation.

    Args:
        max_levels: Maximum number of orderbook levels to support (default: 1000)
                   Used for buffer pre-allocation to avoid repeated allocations
    """

    def __init__(self, max_levels: int = 1000):
        self.time = None
        self.bids: SortedDict = SortedDict()  # Price -> Size (maintained in sorted order)
        self.asks: SortedDict = SortedDict()  # Price -> Size (maintained in sorted order)
        self.max_levels = max_levels

        # Pre-allocate buffers to avoid allocation overhead
        # These are reused across get_orderbook() calls
        self._bids_buffer = np.zeros(max_levels, dtype=np.float64)
        self._asks_buffer = np.zeros(max_levels, dtype=np.float64)

    def reset(self):
        """
        Reset orderbook state to initial empty state.

        This method clears all bid/ask price levels and resets the timestamp.
        Should be called when the WebSocket connection is reestablished to ensure
        clean state before processing new updates.
        """
        # Recreate SortedDict instances to ensure clean state
        self.bids = SortedDict()
        self.asks = SortedDict()
        self.time = None

    def get_state(self):
        """
        Get current orderbook state as OrderBookState object.

        Returns sorted price levels from SortedDict (no explicit sorting needed).
        """
        # SortedDict maintains sorted order, just convert to list
        bids = list(self.bids.items())
        asks = list(self.asks.items())
        return OrderBookState(bids=bids, asks=asks)

    def get_orderbook(self, tick_size: float, levels: int) -> OrderBook | None:
        """
        Generate OrderBook from current state with aggregation.

        Args:
            tick_size: Price tick size for aggregation
            levels: Number of price levels to include

        Returns:
            OrderBook object, or None if state is empty or invalid

        Note:
            Detects crossed orderbook (bid >= ask) which indicates corrupted state
            from missed or out-of-order updates. Automatically resets state on detection.
        """
        # Check if we have bids and asks
        if not self.bids or not self.asks:
            return None

        # Validate levels parameter
        if levels > self.max_levels:
            logger.warning(
                f"Requested levels ({levels}) exceeds max_levels ({self.max_levels}). "
                f"Using max_levels={self.max_levels}"
            )
            levels = self.max_levels

        # Get best bid/ask for crossed orderbook detection
        # SortedDict: keys are in ascending order
        best_bid = self.bids.keys()[-1]  # Highest bid (last key)
        best_ask = self.asks.keys()[0]  # Lowest ask (first key)

        # Detect crossed orderbook (invalid state)
        # This should rarely happen now with proactive cleaning in update_state()
        if best_bid >= best_ask:
            logger.error(
                f"Crossed orderbook detected AFTER proactive cleaning: "
                f"best_bid={best_bid:.2f} >= best_ask={best_ask:.2f}. "
                f"This indicates extreme message disorder or logic bug. Resetting state."
            )
            self.reset()
            return None

        # Get bids descending (highest first) and asks ascending (lowest first)
        # SortedDict maintains ascending order, so reverse bids
        sorted_bids = list(reversed(self.bids.items()))
        sorted_asks = list(self.asks.items())

        # Clear pre-allocated buffers for requested levels
        self._bids_buffer[:levels].fill(0.0)
        self._asks_buffer[:levels].fill(0.0)

        # Apply accumulation to aggregate into uniform grid
        # Use slices of pre-allocated buffers to avoid allocation
        top_bid_agg, bids_accumulated = accumulate_orderbook_levels(
            np.array(sorted_bids, dtype=np.float64),
            self._bids_buffer[:levels],
            tick_size,
            True,
            levels,
            False,  # is_bid=True, sizes_in_quoted=False
        )
        top_ask_agg, asks_accumulated = accumulate_orderbook_levels(
            np.array(sorted_asks, dtype=np.float64),
            self._asks_buffer[:levels],
            tick_size,
            False,
            levels,
            False,  # is_bid=False, sizes_in_quoted=False
        )

        return OrderBook(
            time=time_as_nsec(self.time),
            top_bid=top_bid_agg,
            top_ask=top_ask_agg,
            tick_size=tick_size,
            bids=bids_accumulated,
            asks=asks_accumulated,
        )

    def update_state(self, time: dt_64, bids: list[tuple[float, float]], asks: list[tuple[float, float]]):
        """
        Update orderbook state and proactively clean crossed levels.

        Uses extreme prices from the UPDATE data (not state) for cross-checking:
        - Highest bid in updates is the fresh price to check against asks
        - Lowest ask in updates is the fresh price to check against bids

        This ensures we only clean based on fresh data, not potentially stale state.

        Args:
            time: Timestamp of update
            bids: List of (price, size) tuples for bid updates
            asks: List of (price, size) tuples for ask updates
        """
        self.time = time

        # Find highest bid price in updates (excluding zero sizes)
        highest_bid_update = None
        if bids:
            non_zero_bids = [price for price, size in bids if size > 0]
            if non_zero_bids:
                highest_bid_update = max(non_zero_bids)

        # Apply all bid updates
        for price, size in bids:
            if size == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size

        # Clean crossed asks using highest bid from UPDATE (not state)
        if highest_bid_update is not None:
            self._clean_crossed_asks(highest_bid_update)

        # Find lowest ask price in updates (excluding zero sizes)
        lowest_ask_update = None
        if asks:
            non_zero_asks = [price for price, size in asks if size > 0]
            if non_zero_asks:
                lowest_ask_update = min(non_zero_asks)

        # Apply all ask updates
        for price, size in asks:
            if size == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size

        # Clean crossed bids using lowest ask from UPDATE (not state)
        if lowest_ask_update is not None:
            self._clean_crossed_bids(lowest_ask_update)

    def _clean_crossed_asks(self, bid_price: float) -> int:
        """
        Remove all asks at or below the given bid price.

        Args:
            bid_price: Bid price from fresh update that may cross asks

        Returns:
            Number of crossed asks removed
        """
        if not self.asks:
            return 0

        # Use irange() to efficiently find asks <= bid_price
        crossed_keys = list(self.asks.irange(maximum=bid_price, inclusive=(True, True)))

        if crossed_keys:
            logger.debug(
                f"Cleaning {len(crossed_keys)} crossed asks at/below bid_price={bid_price:.2f}: " f"{crossed_keys}"
            )
            for key in crossed_keys:
                del self.asks[key]

        return len(crossed_keys)

    def _clean_crossed_bids(self, ask_price: float) -> int:
        """
        Remove all bids at or above the given ask price.

        Args:
            ask_price: Ask price from fresh update that may cross bids

        Returns:
            Number of crossed bids removed
        """
        if not self.bids:
            return 0

        # Use irange() to efficiently find bids >= ask_price
        crossed_keys = list(self.bids.irange(minimum=ask_price, inclusive=(True, True)))

        if crossed_keys:
            logger.debug(
                f"Cleaning {len(crossed_keys)} crossed bids at/above ask_price={ask_price:.2f}: " f"{crossed_keys}"
            )
            for key in crossed_keys:
                del self.bids[key]

        return len(crossed_keys)
