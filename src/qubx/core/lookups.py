import configparser
import dataclasses
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path

import stackprinter

from qubx import logger
from qubx.core.basics import ZERO_COSTS, AssetType, Instrument, MarketType, TransactionCostsCalculator
from qubx.utils.marketdata.dukas import SAMPLE_INSTRUMENTS
from qubx.utils.misc import get_local_qubx_folder, load_qubx_resources_as_json, makedirs

_DEF_INSTRUMENTS_FOLDER = "instruments"
_DEF_FEES_FOLDER = "fees"


class _InstrumentEncoder(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return {k: v for k, v in dataclasses.asdict(obj).items() if not k.startswith("_")}
        if isinstance(obj, (datetime)):
            return obj.isoformat()
        return super().default(obj)


class _InstrumentDecoder(json.JSONDecoder):
    def decode(self, json_string):
        obj = super(_InstrumentDecoder, self).decode(json_string)
        if isinstance(obj, dict):
            # Convert delivery_date and onboard_date strings to datetime
            delivery_date = obj.get("delivery_date")
            onboard_date = obj.get("onboard_date")
            if delivery_date:
                obj["delivery_date"] = datetime.strptime(delivery_date.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            if onboard_date:
                obj["onboard_date"] = datetime.strptime(onboard_date.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            return Instrument(
                symbol=obj["symbol"],
                asset_type=AssetType[obj["asset_type"]],
                market_type=MarketType[obj["market_type"]],
                exchange=obj["exchange"],
                base=obj["base"],
                quote=obj["quote"],
                settle=obj["settle"],
                exchange_symbol=obj.get("exchange_symbol", obj["symbol"]),
                tick_size=float(obj["tick_size"]),
                lot_size=float(obj["lot_size"]),
                min_size=float(obj["min_size"]),
                min_notional=float(obj.get("min_notional", 0.0)),
                initial_margin=float(obj.get("initial_margin", 0.0)),
                maint_margin=float(obj.get("maint_margin", 0.0)),
                liquidation_fee=float(obj.get("liquidation_fee", 0.0)),
                contract_size=float(obj.get("contract_size", 1.0)),
                onboard_date=obj.get("onboard_date"),
                delivery_date=obj.get("delivery_date"),
            )
        elif isinstance(obj, list):
            return [self.decode(json.dumps(item)) for item in obj]
        return obj


class InstrumentsLookup:
    _lookup: dict[str, Instrument]
    _path: str

    def __init__(self, path: str = makedirs(get_local_qubx_folder(), _DEF_INSTRUMENTS_FOLDER)) -> None:
        self._path = path
        if not self.load():
            self.refresh()
        self.load()

    def load(self) -> bool:
        self._lookup = {}
        data_exists = False
        for fs in glob.glob(self._path + "/*.json"):
            try:
                with open(fs, "r") as f:
                    instrs: list[Instrument] = json.load(f, cls=_InstrumentDecoder)
                    for i in instrs:
                        self._lookup[f"{i.exchange}:{i.symbol}"] = i
                    data_exists = True
            except Exception as ex:
                stackprinter.show_current_exception()
                logger.warning(ex)

        return data_exists

    def find(self, exchange: str, base: str, quote: str, settle: str | None = None) -> Instrument | None:
        for i in self._lookup.values():
            if i.exchange == exchange and (
                (i.base == base and i.quote == quote) or (i.base == quote and i.quote == base)
            ):
                if settle is not None and i.settle is not None:
                    if i.settle == settle:
                        return i
                else:
                    return i
        return None

    def find_symbol(self, exchange: str, symbol: str) -> Instrument | None:
        for i in self._lookup.values():
            if (i.exchange == exchange) and (i.symbol == symbol):
                return i
        return None

    def find_instruments(self, exchange: str, quote: str | None = None) -> list[Instrument]:
        return [i for i in self._lookup.values() if i.exchange == exchange and (quote is None or i.quote == quote)]

    def _save_to_json(self, path, instruments: list[Instrument]):
        with open(path, "w") as f:
            json.dump(instruments, f, cls=_InstrumentEncoder, indent=4)
        logger.info(f"Saved {len(instruments)} to {path}")

    def find_aux_instrument_for(self, instrument: Instrument, base_currency: str) -> Instrument | None:
        """
        Tries to find aux instrument (for conversions to funded currency)
        for example:
            ETHBTC -> BTCUSDT for base_currency USDT
            EURGBP -> GBPUSD for base_currency USD
            ...
        """
        base_currency = base_currency.upper()
        if instrument.quote != base_currency:
            return self.find(instrument.exchange, instrument.quote, base_currency)
        return None

    def __getitem__(self, spath: str) -> list[Instrument]:
        res = []
        c = re.compile(spath)
        for k, v in self._lookup.items():
            if re.match(c, k):
                res.append(v)
        return res

    def refresh(self, query_exchanges: bool = False):
        for mn in dir(self):
            if mn.startswith("_update_"):
                getattr(self, mn)(self._path, query_exchanges)

    def _ccxt_update(
        self,
        path: str,
        file_name: str,
        exchange_to_ccxt_name: dict[str, str],
        keep_types: list[MarketType] | None = None,
        query_exchanges: bool = False,
    ):
        import ccxt as cx

        from qubx.utils.marketdata.ccxt import ccxt_symbol_to_instrument

        # - first we try to load packed data from QUBX resources
        instruments = {}
        try:
            _packed_data = load_qubx_resources_as_json(f"instruments/symbols-{file_name}")
            if _packed_data:
                for i in _convert_instruments_metadata_to_qubx(_packed_data):
                    instruments[i] = i
        except Exception as e:
            logger.warning(f"Can't load resource file from instruments/symbols-{file_name} - {str(e)}")

        if query_exchanges:
            # - replace defaults with data from CCXT
            for exch, ccxt_name in exchange_to_ccxt_name.items():
                exch = exch.upper()
                ccxt_name = ccxt_name.lower()
                ex: cx.Exchange = getattr(cx, ccxt_name)()
                mkts = ex.load_markets()
                for v in mkts.values():
                    if v["index"]:
                        continue
                    instr = ccxt_symbol_to_instrument(exch, v)
                    if not keep_types or instr.market_type in keep_types:
                        instruments[instr] = instr

        # - drop to file
        self._save_to_json(os.path.join(path, f"{file_name}.json"), list(instruments.values()))

    def _update_kraken(self, path: str, query_exchanges: bool = False):
        self._ccxt_update(path, "kraken.f", {"kraken.f": "krakenfutures"}, query_exchanges=query_exchanges)
        self._ccxt_update(path, "kraken", {"kraken": "kraken"}, query_exchanges=query_exchanges)

    def _update_hyperliquid(self, path: str, query_exchanges: bool = False):
        self._ccxt_update(
            path,
            "hyperliquid",
            {"hyperliquid": "hyperliquid"},
            keep_types=[MarketType.SPOT],
            query_exchanges=query_exchanges,
        )
        self._ccxt_update(
            path,
            "hyperliquid.f",
            {"hyperliquid.f": "hyperliquid"},
            keep_types=[MarketType.SWAP],
            query_exchanges=query_exchanges,
        )

    def _update_binance(self, path: str, query_exchanges: bool = False):
        self._ccxt_update(
            path,
            "binance",
            {"binance": "binance"},
            keep_types=[MarketType.SPOT, MarketType.MARGIN],
            query_exchanges=query_exchanges,
        )
        self._ccxt_update(path, "binance.um", {"binance.um": "binanceusdm"}, query_exchanges=query_exchanges)
        self._ccxt_update(path, "binance.cm", {"binance.cm": "binancecoinm"}, query_exchanges=query_exchanges)

    def _update_bitfinex(self, path: str, query_exchanges: bool = False):
        self._ccxt_update(
            path,
            "bitfinex.f",
            {"bitfinex.f": "bitfinex"},
            keep_types=[MarketType.SWAP],
            query_exchanges=query_exchanges,
        )

    def _update_dukas(self, path: str, query_exchanges: bool = False):
        self._save_to_json(os.path.join(path, "dukas.json"), SAMPLE_INSTRUMENTS)


# - TODO: need to find better way to extract actual data !!
_DEFAULT_FEES = """
[binance]
# SPOT (maker, taker)
vip0_usdt = 0.1000,0.1000
vip1_usdt = 0.0900,0.1000
vip2_usdt = 0.0800,0.1000
vip3_usdt = 0.0420,0.0600
vip4_usdt = 0.0420,0.0540
vip5_usdt = 0.0360,0.0480
vip6_usdt = 0.0300,0.0420
vip7_usdt = 0.0240,0.0360
vip8_usdt = 0.0180,0.0300
vip9_usdt = 0.0120,0.0240

# SPOT (maker, taker)
vip0_bnb = 0.0750,0.0750
vip1_bnb = 0.0675,0.0750
vip2_bnb = 0.0600,0.0750
vip3_bnb = 0.0315,0.0450
vip4_bnb = 0.0315,0.0405
vip5_bnb = 0.0270,0.0360
vip6_bnb = 0.0225,0.0315
vip7_bnb = 0.0180,0.0270
vip8_bnb = 0.0135,0.0225
vip9_bnb = 0.0090,0.0180

# UM futures (maker, taker)
[binance.um]
vip0_usdt = 0.0200,0.0500
vip1_usdt = 0.0160,0.0400
vip2_usdt = 0.0140,0.0350
vip3_usdt = 0.0120,0.0320
vip4_usdt = 0.0100,0.0300
vip5_usdt = 0.0080,0.0270
vip6_usdt = 0.0060,0.0250
vip7_usdt = 0.0040,0.0220
vip8_usdt = 0.0020,0.0200
vip9_usdt = 0.0000,0.0170

# CM futures (maker, taker)
[binance.cm]
vip0 = 0.0200,0.0500
vip1 = 0.0160,0.0400
vip2 = 0.0140,0.0350
vip3 = 0.0120,0.0320
vip4 = 0.0100,0.0300
vip5 = 0.0080,0.0270
vip6 = 0.0060,0.0250
vip7 = 0.0040,0.0220
vip8 = 0.0020,0.0200
vip9 = 0.0000,0.0170

[bitmex]
tierb_xbt=0.02,0.075
tierb_usdt=-0.015,0.075
tieri_xbt=0.01,0.05
tieri_usdt=-0.015,0.05
tiert_xbt=0.0,0.04
tiert_usdt=-0.015,0.04
tierm_xbt=0.0,0.035
tierm_usdt=-0.015,0.035
tiere_xbt=0.0,0.03
tiere_usdt=-0.015,0.03
tierx_xbt=0.0,0.025
tierx_usdt=-0.015,0.025
tierd_xbt=-0.003,0.024
tierd_usdt=-0.015,0.024
tierw_xbt=-0.005,0.023
tierw_usdt=-0.015,0.023
tierk_xbt=-0.008,0.022
tierk_usdt=-0.015,0.022
tiers_xbt=-0.01,0.0175
tiers_usdt=-0.015,0.02

[dukas]
regular=0.0035,0.0035
premium=0.0017,0.0017

[kraken]
K0=0.25,0.40
K10=0.20,0.35
K50=0.14,0.24
K100=0.12,0.22
K250=0.10,0.20
K500=0.08,0.18
M1=0.06,0.16
M2.5=0.04,0.14
M5=0.02,0.12
M10=0.0,0.10

[kraken.f]
K0=0.0200,0.0500
K100=0.0150,0.0400
M1=0.0125,0.0300
M5=0.0100,0.0250
M10=0.0075,0.0200
M20=0.0050,0.0150
M50=0.0025,0.0125
M100=0.0000,0.0100
"""


class FeesLookup:
    """
    Fees lookup
    """

    _lookup: dict[str, tuple[float, float]]
    _path: str

    def __init__(self, path: str = makedirs(get_local_qubx_folder(), _DEF_FEES_FOLDER)) -> None:
        self._path = path
        if not self.load():
            self.refresh()
        self.load()

    def load(self) -> bool:
        self._lookup = {}
        data_exists = False
        parser = configparser.ConfigParser()
        # - load all avaliable configs
        for fs in glob.glob(self._path + "/*.ini"):
            parser.read(fs)
            data_exists = True

        for exch in parser.sections():
            for spec, info in parser[exch].items():
                try:
                    maker, taker = info.split(",")
                    self._lookup[f"{exch}_{spec}"] = (float(maker), float(taker))
                except (ValueError, TypeError) as e:
                    logger.warning(f'Wrong spec format for {exch}: "{info}". Should be spec=maker,taker. Error: {e}')

        return data_exists

    def __getitem__(self, spath: str) -> list[Instrument]:
        res = []
        c = re.compile(spath)
        for k, v in self._lookup.items():
            if re.match(c, k):
                res.append((k, v))
        return res

    def refresh(self):
        with open(os.path.join(self._path, "default.ini"), "w") as f:
            f.write(_DEFAULT_FEES)

    def find(self, exchange: str, spec: str | None) -> TransactionCostsCalculator:
        if spec is None:
            return ZERO_COSTS

        key = f"{exchange}_{spec}"

        # - check if spec is of type maker=...,taker=...
        # Check if spec is in the format maker=X,taker=Y
        maker_taker_pattern = re.compile(r"maker=(-?[0-9.]+)[,\ ]taker=(-?[0-9.]+)")
        match = maker_taker_pattern.match(spec)
        if match:
            maker_rate, taker_rate = float(match.group(1)), float(match.group(2))
            return TransactionCostsCalculator(key, maker_rate, taker_rate)

        # - otherwise lookup in lookup table
        vals = self._lookup.get(key)
        if vals is None:
            raise ValueError(f"No fees found for {key}")

        assert isinstance(vals, tuple)
        return TransactionCostsCalculator(key, vals[0], vals[1])

    def __repr__(self) -> str:
        s = "Name:\t\t\t(maker, taker)\n"
        for k, v in self._lookup.items():
            s += f"{k.ljust(25)}: {v}\n"
        return s


@dataclasses.dataclass(frozen=True)
class GlobalLookup:
    instruments: InstrumentsLookup
    fees: FeesLookup

    def find_fees(self, exchange: str, spec: str | None) -> TransactionCostsCalculator | None:
        return self.fees.find(exchange, spec)

    def find_aux_instrument_for(self, instrument: Instrument, base_currency: str) -> Instrument | None:
        return self.instruments.find_aux_instrument_for(instrument, base_currency)

    def find_instrument(self, exchange: str, base: str, quote: str) -> Instrument | None:
        return self.instruments.find(exchange, base, quote)

    def find_instruments(self, exchange: str, quote: str | None = None) -> list[Instrument]:
        return self.instruments.find_instruments(exchange, quote)

    def find_symbol(self, exchange: str, symbol: str) -> Instrument | None:
        return self.instruments.find_symbol(exchange, symbol)


def _convert_instruments_metadata_to_qubx(data: list[dict]):
    """
    Converting tardis symbols meta-data to Qubx instruments
    """
    _excs = {
        "binance": "BINANCE",
        "binance-delivery": "BINANCE.CM",
        "binance-futures": "BINANCE.UM",
        "kraken": "KRAKEN",
        "cryptofacilities": "KRAKEN.F",
        "bitfinex": "BITFINEX",
        "bitfinex-derivatives": "BITFINEX.F",
        "hyperliquid": "HYPERLIQUID",
    }
    r = []
    for s in data:
        match s["type"]:
            case "perpetual":
                _type = MarketType.SWAP
            case "spot":
                _type = MarketType.SPOT
            case "future":
                _type = MarketType.FUTURE
            case _:
                raise ValueError(f" -> Unknown type {s['type']}")
        r.append(
            Instrument(
                s["baseCurrency"] + s["quoteCurrency"],
                AssetType.CRYPTO,
                _type,
                _excs.get(s["exchange"], s["exchange"].upper()),
                s["baseCurrency"],
                s["quoteCurrency"],
                s["quoteCurrency"],
                s["id"],
                tick_size=s["priceIncrement"],
                lot_size=s["minTradeAmount"],
                min_size=s["amountIncrement"],
                min_notional=0,  # we don't have this info from tardis
                contract_size=s.get("contractMultiplier", 1.0),
                onboard_date=s.get("availableSince", None),
                delivery_date=s.get("availableTo", None),
            )
        )
    return r


# - global lookup helper
lookup = GlobalLookup(InstrumentsLookup(), FeesLookup())
