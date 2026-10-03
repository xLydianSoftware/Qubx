import configparser
import contextlib
import dataclasses
import glob
import json
import os
import re
import shutil
import tempfile
import threading
from datetime import datetime

import httpx
import pandas as pd
import stackprinter

from qubx import logger
from qubx.core.basics import (
    ZERO_COSTS,
    AccountsLookup,
    AssetKind,
    FeesLookup,
    Instrument,
    InstrumentsLookup,
    MarketType,
    TransactionCostsCalculator,
    Underlying,
)
from qubx.utils.marketdata.dukas import SAMPLE_INSTRUMENTS
from qubx.utils.misc import get_local_qubx_folder, load_qubx_resources_as_json, load_qubx_resources_as_text, makedirs
from qubx.utils.time import to_timedelta

_DEF_INSTRUMENTS_FOLDER = "instruments"
_DEF_FEES_FOLDER = "fees"

_PACKAGED_FEES_FILE = "crypto-fees.ini"


class _InstrumentEncoder(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return {k: v for k, v in dataclasses.asdict(obj).items() if not k.startswith("_")}
        if isinstance(obj, (datetime)):
            return obj.isoformat()
        return super().default(obj)


def _ts(value) -> pd.Timestamp | None:
    if value is None or value == "NaT":
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def _instrument_from_dict(obj: dict) -> Instrument:
    """Reads both the current shape and files written before the Instrument redesign
    (onboard_date/delist_date/delivery_date, contract_multiplier, margin fields)."""
    _u = obj.get("underlying")
    return Instrument(
        symbol=obj["symbol"],
        market_type=MarketType[obj["market_type"]],
        exchange=obj["exchange"],
        base=obj["base"],
        quote=obj["quote"],
        settle=obj["settle"],
        exchange_symbol=obj.get("exchange_symbol") or obj["symbol"],
        tick_size=float(obj["tick_size"]),
        lot_size=float(obj["lot_size"]),
        min_size=float(obj["min_size"]),
        min_notional=float(obj.get("min_notional") or 0.0),
        contract_size=float(obj.get("contract_size") or 1.0) * float(obj.get("contract_multiplier") or 1.0),
        inverse=bool(obj.get("inverse") or False),
        listing_id=obj.get("listing_id"),
        underlying=Underlying(AssetKind(_u["kind"]), _u["code"]) if _u else None,
        calendar=obj.get("calendar"),
        expiry=_ts(obj.get("expiry", obj.get("delivery_date"))),
        strike=obj.get("strike"),
        option_right=obj.get("option_right"),
        listed_at=_ts(obj.get("listed_at", obj.get("onboard_date"))),
        delisted_at=_ts(obj.get("delisted_at", obj.get("delist_date"))),
        margin_tradable=bool(obj.get("margin_tradable") or False),
        venue_attributes=obj.get("venue_attributes") or {},
    )


class _InstrumentDecoder(json.JSONDecoder):
    def decode(self, json_string):
        obj = super(_InstrumentDecoder, self).decode(json_string)
        if isinstance(obj, dict):
            return _instrument_from_dict(obj)
        elif isinstance(obj, list):
            return [_instrument_from_dict(item) for item in obj]
        return obj


class FileInstrumentsLookupWithCCXT(InstrumentsLookup):
    _lookup: dict[str, Instrument]
    _path: str

    def __init__(
        self, path: str = makedirs(get_local_qubx_folder(), _DEF_INSTRUMENTS_FOLDER), query_exchanges=False
    ) -> None:
        self._path = path
        if not self.load():
            self._build_cache(query_exchanges)
        self.load()

    def _build_cache(self, query_exchanges: bool) -> None:
        """Build a cold cache in a sibling temp folder and publish it with one rename: processes
        sharing the folder (xdist workers, bots) see either no cache or all of it, never a
        half-written one. A process that loses the publish race drops its copy and uses the winner's."""
        parent = os.path.dirname(os.path.abspath(self._path))
        staging = tempfile.mkdtemp(prefix=f".{os.path.basename(self._path)}-", dir=parent)
        try:
            self.refresh(query_exchanges, path=staging)
            try:
                # - an empty folder may stand in the way (it is created up front); a filled one is the winner's
                with contextlib.suppress(FileNotFoundError):
                    os.rmdir(self._path)
                os.rename(staging, self._path)
            except OSError:
                pass
        finally:
            shutil.rmtree(staging, ignore_errors=True)

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

    def get_lookup(self) -> dict[str, Instrument]:
        return self._lookup

    def _save_to_json(self, path, instruments: list[Instrument]):
        # - written aside and swapped in, so a concurrent reader never sees a truncated file
        fd, staging = tempfile.mkstemp(prefix=f".{os.path.basename(path)}-", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(instruments, f, cls=_InstrumentEncoder, indent=4)
            os.replace(staging, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.remove(staging)
            raise
        logger.info(f"Saved {len(instruments)} to {path}")

    def refresh(self, query_exchanges: bool = False, path: str | None = None):
        for mn in dir(self):
            if mn.startswith("_update_"):
                getattr(self, mn)(path or self._path, query_exchanges)

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

    def _update_gateio(self, path: str, query_exchanges: bool = False):
        self._copy_instruments_and_update_from_ccxt(
            path,
            "gateio.f",
            {"gateio.f": "gate"},
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
        try:
            _packaged_fees = load_qubx_resources_as_text(_PACKAGED_FEES_FILE)
            with open(os.path.join(self._path, "default.ini"), "w") as f:
                f.write(_packaged_fees)
        except Exception as e:
            logger.error(f"Can't load resource file from {_PACKAGED_FEES_FILE} - {str(e)}")

    def find_fees(self, exchange: str, spec: str | None) -> TransactionCostsCalculator:
        if spec is None:
            return ZERO_COSTS

        key = f"{exchange.lower()}_{spec}"

        # - check if spec is of type maker=...,taker=...
        # Check if spec is in the format maker=X,taker=Y
        maker_taker_pattern = re.compile(r"maker=(-?[0-9.]+)[,\s]taker=(-?[0-9.]+)")
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
        _delist_date = _ts(s.get("availableTo", None))
        _delivery_date = _ts(s.get("expiry", None))

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
                _type,
                _excs.get(s["exchange"], s["exchange"].upper()),
                s["baseCurrency"],
                s["quoteCurrency"],
                s["quoteCurrency"],
                s["datasetId"],
                tick_size=s["priceIncrement"],
                # lot_size is the quantity step everywhere it is read (rounding, add_in_lots,
                # half-lot tolerances); minTradeAmount is a floor, not a step
                lot_size=s["amountIncrement"],
                min_size=s["amountIncrement"],
                min_notional=0,  # we don't have this info from tardis
                contract_size=s.get("contractMultiplier", 1.0),
                listed_at=_ts(s.get("availableSince", None)),
                expiry=_delivery_date,
                inverse=s.get("inverse", False),
                delisted_at=_delist_date,
            )
        )
    return r


_EPOCH = pd.Timestamp(0)


def listing_to_instrument(listing: dict) -> Instrument:
    """Instrument from an instrument-service /snapshot listing, built from its current version."""
    versions = listing["versions"]
    v = next((x for x in reversed(versions) if x.get("valid_to") is None), versions[-1])
    u = listing["underlying"]
    return Instrument(
        symbol=v["symbol"],
        market_type=MarketType(listing["market_type"]),
        exchange=listing["exchange"],
        base=v["base"],
        quote=listing["quote"],
        settle=listing["settle"],
        exchange_symbol=v["exchange_symbol"] or v["symbol"],
        tick_size=float(v["tick_size"]),
        lot_size=float(v["lot_size"]),
        min_size=float(v["min_size"]),
        min_notional=float(v["min_notional"] or 0.0),
        contract_size=float(v["contract_size"] or 1.0),
        inverse=bool(listing["inverse"]),
        listing_id=listing["id"],
        underlying=Underlying(AssetKind(u["kind"]), u["code"]),
        calendar=listing.get("calendar"),
        expiry=_ts(listing.get("expiry")),
        strike=listing.get("strike"),
        option_right=listing.get("option_right"),
        listed_at=_ts(listing.get("listed_at")),
        delisted_at=_ts(listing.get("delisted_at")),
        margin_tradable=bool(v.get("margin_tradable")),
        venue_attributes=v.get("venue_attributes") or {},
    )


class InstrumentsLookupService(InstrumentsLookup):
    """Instruments from the platform instrument service ({url}/snapshot).

    The first load must succeed. Afterwards the snapshot is re-read every reload_interval
    with If-None-Match; a failed refresh keeps the in-memory copy. There is no fallback lookup.
    """

    _lookup: dict[str, Instrument]
    _by_symbol: dict[tuple[str, str], list[Instrument]]
    _by_alias: dict[tuple[str, str], list[Instrument]]

    def __init__(
        self,
        url: str,
        token: str | None = None,
        reload_interval: str | None = None,
        exchanges: list[str] | None = None,
        timeout: float = 60.0,
    ):
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._params = [("exchange", e) for e in exchanges or []]
        self._reload_interval = to_timedelta(reload_interval) if reload_interval else None
        self._timeout = timeout
        self._etag: str | None = None
        self._refresh_lock = threading.Lock()
        try:
            self._fetch()
        except Exception as e:
            raise RuntimeError(f"[lookup] instrument service at {self._url} is unavailable: {e}") from e
        self._last_refresh = pd.Timestamp.now()

    def _fetch(self) -> bool:
        headers = dict(self._headers)
        if self._etag:
            headers["If-None-Match"] = self._etag
        r = httpx.get(f"{self._url}/snapshot", params=self._params, headers=headers, timeout=self._timeout)
        if r.status_code == 304:
            return False
        r.raise_for_status()
        self._build(r.json()["listings"])
        self._etag = r.headers.get("ETag")
        return True

    def _build(self, listings: list[dict]) -> None:
        built: list[tuple[Instrument, list[str]]] = []
        for listing in listings:
            try:
                built.append((listing_to_instrument(listing), listing.get("aliases") or []))
            except (ValueError, KeyError) as e:
                logger.warning(f"[lookup] skipping listing {listing.get('id')} ({listing.get('exchange')}): {e}")

        # - a relisted symbol is a new listing next to the delisted one: the active, then the newest, wins
        built.sort(key=lambda b: (b[0].delisted_at is not None, -(b[0].listed_at or _EPOCH).value))
        lookup, by_symbol, by_alias = {}, {}, {}
        for i, aliases in built:
            lookup.setdefault(f"{i.exchange}:{i.market_type}:{i.symbol}", i)
            by_symbol.setdefault((i.exchange, i.symbol), []).append(i)
            for alias in aliases:
                if alias != i.symbol:
                    by_alias.setdefault((i.exchange, alias), []).append(i)
        self._lookup, self._by_symbol, self._by_alias = lookup, by_symbol, by_alias
        logger.info(f"[lookup] loaded {len(lookup)} instruments from the instrument service")

    def refresh(self) -> None:
        if not self._refresh_lock.acquire(blocking=False):
            return
        try:
            self._fetch()
        except Exception as e:
            logger.warning(f"[lookup] instrument service refresh failed, keeping {len(self._lookup)} instruments: {e}")
        finally:
            self._last_refresh = pd.Timestamp.now()
            self._refresh_lock.release()

    def _maybe_refresh(self) -> None:
        if self._reload_interval and pd.Timestamp.now() - self._last_refresh > self._reload_interval:
            self.refresh()

    def get_lookup(self) -> dict[str, Instrument]:
        self._maybe_refresh()
        return self._lookup

    def find_symbol(self, exchange: str, symbol: str, market_type: MarketType | None = None) -> Instrument | None:
        """Current instrument for a symbol, resolving former symbols of renamed listings."""
        self._maybe_refresh()
        for index in (self._by_symbol, self._by_alias):
            for i in index.get((exchange, symbol), ()):
                if market_type is None or i.market_type == market_type:
                    return i
        return None


class AccountsLookupFromManager(AccountsLookup):
    """Concrete AccountsLookup that delegates to an AccountConfigurationManager."""

    _manager = None

    def register(self, manager) -> None:
        """Register account manager. Called once at startup by the runner."""
        self._manager = manager

    def get_credentials(self, exchange: str):
        if self._manager is None:
            raise RuntimeError("No account manager registered — call lookup.register_accounts() at startup")
        return self._manager.get_exchange_credentials(exchange)

    def get_settings(self, exchange: str):
        if self._manager is None:
            raise RuntimeError("No account manager registered — call lookup.register_accounts() at startup")
        return self._manager.get_exchange_settings(exchange)


class LookupsManager(InstrumentsLookup, FeesLookup, AccountsLookup):
    _i_lookup: InstrumentsLookup
    _t_lookup: FeesLookup
    _a_lookup: AccountsLookupFromManager

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(LookupsManager, cls).__new__(cls)

            from qubx.config import settings

            i_cfg = settings.instrument_lookup
            f_cfg = settings.fees_lookup

            i_kwargs = {}
            if i_cfg.url:
                i_kwargs["url"] = i_cfg.url
            if i_cfg.token:
                i_kwargs["token"] = i_cfg.token
            if i_cfg.reload_interval:
                i_kwargs["reload_interval"] = i_cfg.reload_interval
            if i_cfg.path:
                i_kwargs["path"] = i_cfg.path
            cls.instance._i_lookup = LookupsManager._get_instrument_lookup(type=i_cfg.type, **i_kwargs)

            f_kwargs = {}
            if f_cfg.path:
                f_kwargs["path"] = f_cfg.path
            cls.instance._t_lookup = LookupsManager._get_fees_lookup(type=f_cfg.type, **f_kwargs)
            cls.instance._a_lookup = AccountsLookupFromManager()

        return cls.instance

    @staticmethod
    def _get_instrument_lookup(type: str, **kwargs) -> InstrumentsLookup:
        match type.lower():
            case "file":
                return FileInstrumentsLookupWithCCXT(**{k: v for k, v in kwargs.items() if k == "path"})
            case "service":
                if not kwargs.get("url"):
                    raise ValueError("Instrument lookup type 'service' requires a url")
                kwargs.pop("path", None)
                return InstrumentsLookupService(**kwargs)
            case "mongo":
                raise ValueError("The mongo instrument lookup was removed: use type 'service' with a url")
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
        self,
        exchange: str,
        base: str | None = None,
        quote: str | None = None,
        market_type: MarketType | None = None,
        as_of: str | pd.Timestamp | None = None,
    ) -> list[Instrument]:
        return self._i_lookup.find_instruments(exchange, base, quote, market_type, as_of=as_of)

    def find_aux_instrument_for(
        self, instrument: Instrument, base_currency: str, market_type: MarketType | None = None
    ) -> Instrument | None:
        return self._i_lookup.find_aux_instrument_for(instrument, base_currency, market_type)

    def find_fees(self, exchange: str, spec: str | None) -> TransactionCostsCalculator:
        return self._t_lookup.find_fees(exchange, spec)

    def __getitem__(self, spath: str) -> list[Instrument]:
        return self._i_lookup[spath]

    def get_credentials(self, exchange: str):
        return self._a_lookup.get_credentials(exchange)

    def get_settings(self, exchange: str):
        return self._a_lookup.get_settings(exchange)


# - global lookup helper (lazy-loaded to avoid slow import)
_lookup = None


def __getattr__(name):
    global _lookup
    if name == "lookup":
        if _lookup is None:
            _lookup = LookupsManager()
        return _lookup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_accounts(manager) -> None:
    """Register account manager in the global lookup. Called once at startup by the runner."""
    global _lookup
    if _lookup is None:
        _lookup = LookupsManager()
    _lookup._a_lookup.register(manager)
