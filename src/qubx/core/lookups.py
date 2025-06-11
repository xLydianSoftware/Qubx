import configparser
import dataclasses
import glob
import json
import os
import re
import stackprinter

from datetime import datetime
import pandas as pd
from pathlib import Path
from qubx import logger

from qubx.core.basics import (
    AssetType,
    FeesLookup,
    Instrument,
    InstrumentsLookup,
    MarketType,
    TransactionCostsCalculator,
    ZERO_COSTS,
)
from qubx.utils.marketdata.dukas import SAMPLE_INSTRUMENTS
from qubx.utils.misc import get_local_qubx_folder, load_qubx_resources_as_json, makedirs


_DEF_INSTRUMENTS_FOLDER = "instruments"
_DEF_FEES_FOLDER = "fees"

_INI_FILE = "settings.ini"
_INI_SECTION_INSTRUMENTS = "instrument-lookup"
_INI_SECTION_FEES = "fees-lookup"


EXCHANGE_TO_DEFAULT_MARKET_TYPE = {
    "BINANCE": MarketType.SPOT,
    "BINANCE.UM": MarketType.SWAP,
    "BINANCE.CM": MarketType.SWAP,
    "DUKAS": MarketType.SPOT,
    "KRAKEN": MarketType.SPOT,
    "KRAKEN.F": MarketType.SWAP,
    "BITFINEX": MarketType.SPOT,
    "BITFINEX.F": MarketType.SWAP,
    "BITMEX": MarketType.SWAP,
    "DERIBIT": MarketType.SWAP,
    "BYBIT": MarketType.SWAP,
    "OKX.F": MarketType.SWAP,
    "HYPERLIQUID": MarketType.SPOT,
    "HYPERLIQUID.F": MarketType.SWAP,
}


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
            if (delivery_date := obj.get("delivery_date")) and delivery_date != "NaT":
                obj["delivery_date"] = pd.Timestamp(delivery_date)

            if (onboard_date := obj.get("onboard_date")) and onboard_date != "NaT":
                obj["onboard_date"] = pd.Timestamp(onboard_date)

            if (delist_date := obj.get("delist_date")) and delist_date != "NaT":
                obj["delist_date"] = pd.Timestamp(delist_date)

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
                onboard_date=obj.get("onboard_date", None),
                delivery_date=obj.get("delivery_date", None),
                inverse=obj.get("inverse", False),
                delist_date=obj.get("delist_date", None),
            )
        elif isinstance(obj, list):
            return [self.decode(json.dumps(item)) for item in obj]
        return obj


class FileInstrumentsLookupWithCCXT(InstrumentsLookup):
    _lookup: dict[str, Instrument]
    _path: str

    def __init__(
        self, path: str = makedirs(get_local_qubx_folder(), _DEF_INSTRUMENTS_FOLDER), query_exchanges=False
    ) -> None:
        self._path = path
        if not self.load():
            self.refresh(query_exchanges)
        self.load()

    def load(self) -> bool:
        self._lookup = {}
        data_exists = False
        for fs in glob.glob(self._path + "/*.json"):
            try:
                with open(fs, "r") as f:
                    for i in json.load(f, cls=_InstrumentDecoder):
                        self._lookup[f"{i.exchange}:{i.market_type}:{i.symbol}"] = i
                    data_exists = True
            except Exception as ex:
                stackprinter.show_current_exception()
                logger.warning(ex)

        return data_exists

    def get_market_type(self, exchange: str) -> MarketType | None:
        return EXCHANGE_TO_DEFAULT_MARKET_TYPE.get(exchange)

    def get_lookup(self) -> dict[str, Instrument]:
        return self._lookup

    def _save_to_json(self, path, instruments: list[Instrument]):
        with open(path, "w") as f:
            json.dump(instruments, f, cls=_InstrumentEncoder, indent=4)
        logger.info(f"Saved {len(instruments)} to {path}")

    def refresh(self, query_exchanges: bool = False):
        for mn in dir(self):
            if mn.startswith("_update_"):
                getattr(self, mn)(self._path, query_exchanges)

    def _copy_instruments_and_update_from_ccxt(
        self,
        path: str,
        file_name: str,
        exchange_to_ccxt_name: dict[str, str],
        keep_types: list[MarketType] | None = None,
        query_exchanges: bool = False,
    ):
        from qubx.utils.marketdata.ccxt import ccxt_fetch_instruments

        # - first we try to load packed data from QUBX resources
        instruments = {}
        try:
            _package_data = load_qubx_resources_as_json(f"instruments/symbols-{file_name}")
            if _package_data:
                for i in _convert_instruments_metadata_to_qubx(_package_data):
                    instruments[i] = i
        except Exception as e:
            logger.warning(f"Can't load resource file from instruments/symbols-{file_name} - {str(e)}")

        if query_exchanges:
            # - replace defaults with data from CCXT
            instruments = ccxt_fetch_instruments(exchange_to_ccxt_name, keep_types, instruments)

        # - save to file
        self._save_to_json(os.path.join(path, f"{file_name}.json"), list(instruments.values()))

    def _update_kraken(self, path: str, query_exchanges: bool = False):
        self._copy_instruments_and_update_from_ccxt(
            path, "kraken-spot", {"kraken": "kraken"}, keep_types=[MarketType.SPOT], query_exchanges=query_exchanges
        )
        self._copy_instruments_and_update_from_ccxt(
            path,
            "kraken.f-perpetual",
            {"kraken.f": "krakenfutures"},
            keep_types=[MarketType.SWAP],
            query_exchanges=query_exchanges,
        )
        self._copy_instruments_and_update_from_ccxt(
            path,
            "kraken.f-future",
            {"kraken.f": "krakenfutures"},
            keep_types=[MarketType.FUTURE],
            query_exchanges=query_exchanges,
        )

    def _update_hyperliquid(self, path: str, query_exchanges: bool = False):
        self._copy_instruments_and_update_from_ccxt(
            path,
            "hyperliquid-spot",
            {"hyperliquid": "hyperliquid"},
            keep_types=[MarketType.SPOT],
            query_exchanges=query_exchanges,
        )
        self._copy_instruments_and_update_from_ccxt(
            path,
            "hyperliquid.f-perpetual",
            {"hyperliquid.f": "hyperliquid"},
            keep_types=[MarketType.SWAP],
            query_exchanges=query_exchanges,
        )

    def _update_binance(self, path: str, query_exchanges: bool = False):
        self._copy_instruments_and_update_from_ccxt(
            path,
            "binance-spot",
            {"binance": "binance"},
            keep_types=[MarketType.SPOT, MarketType.MARGIN],
            query_exchanges=query_exchanges,
        )
        self._copy_instruments_and_update_from_ccxt(
            path,
            "binance.um-perpetual",
            {"binance.um": "binanceusdm"},
            keep_types=[MarketType.SWAP],
            query_exchanges=query_exchanges,
        )

        self._copy_instruments_and_update_from_ccxt(
            path,
            "binance.um-future",
            {"binance.um": "binanceusdm"},
            keep_types=[MarketType.FUTURE],
            query_exchanges=query_exchanges,
        )
        self._copy_instruments_and_update_from_ccxt(
            path,
            "binance.cm-perpetual",
            {"binance.cm": "binancecoinm"},
            keep_types=[MarketType.SWAP],
            query_exchanges=query_exchanges,
        )
        self._copy_instruments_and_update_from_ccxt(
            path,
            "binance.cm-future",
            {"binance.cm": "binancecoinm"},
            keep_types=[MarketType.FUTURE],
            query_exchanges=query_exchanges,
        )

    # todo: temporaty disabled ccxt call to exchange, due to conectivity issues. Revert for bitfinex live usage
    def _update_bitfinex(self, path: str, query_exchanges: bool = False):
        self._copy_instruments_and_update_from_ccxt(
            path,
            "bitfinex.f-perpetual",
            {"bitfinex.f": "bitfinex"},
            keep_types=[MarketType.SWAP],
            query_exchanges=False,
        )

    def _update_bitmex(self, path: str, query_exchanges: bool = False):
        self._copy_instruments_and_update_from_ccxt(
            path,
            "bitmex",
            {"bitmex": "bitmex"},
            query_exchanges=query_exchanges,
        )

    def _update_deribit(self, path: str, query_exchanges: bool = False):
        self._copy_instruments_and_update_from_ccxt(
            path,
            "deribit",
            {"deribit": "deribit"},
            keep_types=[MarketType.SWAP],
            query_exchanges=query_exchanges,
        )

    def _update_bybit(self, path: str, query_exchanges: bool = False):
        self._copy_instruments_and_update_from_ccxt(
            path,
            "bybit.f",
            {"bybit.f": "bybit"},
            keep_types=[MarketType.SWAP],
            query_exchanges=query_exchanges,
        )

    def _update_okx(self, path: str, query_exchanges: bool = False):
        self._copy_instruments_and_update_from_ccxt(
            path,
            "okx.f",
            {"okx.f": "okx"},
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


class FeesLookupFile(FeesLookup):
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

    def refresh(self):
        with open(os.path.join(self._path, "default.ini"), "w") as f:
            f.write(_DEFAULT_FEES)

    def find_fees(self, exchange: str, spec: str | None) -> TransactionCostsCalculator:
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


def _convert_instruments_metadata_to_qubx(data: list[dict]) -> list[Instrument]:
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
        _pfx = ""
        if _delist_date := s.get("availableTo", None):
            _delist_date = pd.Timestamp(_delist_date)

        if _delivery_date := s.get("expiry", None):
            _delivery_date = pd.Timestamp(_delivery_date)

        match s["type"]:
            case "perpetual":
                _type = MarketType.SWAP
            case "spot":
                _type = MarketType.SPOT
            case "future":
                _type = MarketType.FUTURE
                if _delivery_date:
                    _pfx = "." + _delivery_date.strftime("%Y%m%d")
            case _:
                raise ValueError(f" -> Unsupported type {s['type']}")
        r.append(
            Instrument(
                s["baseCurrency"] + s["quoteCurrency"] + _pfx,
                AssetType.CRYPTO,
                _type,
                _excs.get(s["exchange"], s["exchange"].upper()),
                s["baseCurrency"],
                s["quoteCurrency"],
                s["quoteCurrency"],
                s["datasetId"],
                tick_size=s["priceIncrement"],
                lot_size=s["minTradeAmount"],
                min_size=s["amountIncrement"],
                min_notional=0,  # we don't have this info from tardis
                contract_size=s.get("contractMultiplier", 1.0),
                onboard_date=s.get("availableSince", None),
                delivery_date=_delivery_date,
                inverse=s.get("inverse", False),
                delist_date=_delist_date,
            )
        )
    return r


class InstrumentsLookupMongo(InstrumentsLookup):
    _MONGO_DB_BASE_NAME = "metadata"
    _MONGO_DB_TABLE_NAME = "instruments"

    _lookup: dict[str, Instrument]
    _mongo_url: str

    def __init__(self, mongo_url: str = "mongodb://localhost:27017/"):
        self._mongo_url = mongo_url
        self.load()

    def load(self):
        from pymongo import MongoClient

        self._lookup = {}
        with MongoClient(self._mongo_url) as client:
            db = client[self._MONGO_DB_BASE_NAME]
            collection = db[self._MONGO_DB_TABLE_NAME]
            for i in collection.find():
                i.pop("_id")
                instr = Instrument(**i)
                self._lookup[f"{instr.exchange}:{instr.market_type}:{instr.symbol}"] = instr

    def get_lookup(self) -> dict[str, Instrument]:
        return self._lookup

    def get_market_type(self, exchange: str) -> MarketType | None:
        return EXCHANGE_TO_DEFAULT_MARKET_TYPE.get(exchange)


class LookupsManager(InstrumentsLookup, FeesLookup):
    _i_lookup: InstrumentsLookup
    _t_lookup: FeesLookup

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(LookupsManager, cls).__new__(cls)

            # - try to load settings
            parser = configparser.ConfigParser()
            parser.read(Path(get_local_qubx_folder()) / _INI_FILE)

            if _INI_SECTION_INSTRUMENTS in parser:
                cls.instance._i_lookup = LookupsManager._get_instrument_lookup(**dict(parser[_INI_SECTION_INSTRUMENTS]))
            else:
                cls.instance._i_lookup = FileInstrumentsLookupWithCCXT()

            if _INI_SECTION_FEES in parser:
                cls.instance._t_lookup = LookupsManager._get_fees_lookup(**dict(parser[_INI_SECTION_FEES]))
            else:
                cls.instance._t_lookup = FeesLookupFile()

        return cls.instance

    @staticmethod
    def _get_instrument_lookup(type: str, **kwargs) -> InstrumentsLookup:
        match type.lower():
            case "file":
                return FileInstrumentsLookupWithCCXT(**kwargs)
            case "mongo":
                return InstrumentsLookupMongo(**kwargs)
            case _:
                raise ValueError(f"Invalid lookup type: {type}")

    @staticmethod
    def _get_fees_lookup(type: str, **kwargs) -> FeesLookup:
        match type.lower():
            case "file":
                return FeesLookupFile(**kwargs)
            case _:
                raise ValueError(f"Invalid lookup type: {type}")

    def find_symbol(self, exchange: str, symbol: str, market_type: MarketType | None = None) -> Instrument | None:
        return self._i_lookup.find_symbol(exchange, symbol, market_type)

    def find_instruments(
        self, exchange: str, quote: str | None = None, market_type: MarketType | None = None
    ) -> list[Instrument]:
        return self._i_lookup.find_instruments(exchange, quote, market_type)

    def find_aux_instrument_for(
        self, instrument: Instrument, base_currency: str, market_type: MarketType | None = None
    ) -> Instrument | None:
        return self._i_lookup.find_aux_instrument_for(instrument, base_currency, market_type)

    def find_fees(self, exchange: str, spec: str | None) -> TransactionCostsCalculator:
        return self._t_lookup.find_fees(exchange, spec)

    def __getitem__(self, spath: str) -> list[Instrument]:
        return self._i_lookup[spath]


# - global lookup helper
lookup = LookupsManager()
