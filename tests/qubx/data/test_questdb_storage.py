import re
from datetime import timedelta

import pandas as pd
import pytest

from qubx.core.basics import DataType
from qubx.data.storages.questdb import QuestDBReader, SymbolsManifestManager, xLTableMetaInfo

FUNDING_TABLE = "binance.umswap.funding_payment"


class FakePGC:
    """Answers exactly the SQL QuestDBReader/SymbolsManifestManager issue, honoring symbol filters.

    ``funding_rows``: (timestamp, symbol, funding_rate, funding_interval_hours) tuples — the table.
    ``manifest_rows``: (table_name, symbol, last_updated) tuples — pre-existing manifest state.
    """

    def __init__(self, funding_rows=(), manifest_rows=()):
        self.funding_rows = sorted(funding_rows)
        self.manifest_rows = list(manifest_rows)
        self.queries: list[str] = []

    def _symbol_filter(self, ql: str) -> set[str] | None:
        m = re.search(r"symbol\s*(?:=\s*'([^']+)'|in \(([^)]*)\))", ql)
        if m is None:
            return None
        if m.group(1):
            return {m.group(1).upper()}
        return {s.strip().strip("'").upper() for s in m.group(2).split(",")}

    def execute(self, query: str):
        self.queries.append(query)
        ql = " ".join(query.split()).lower()
        if "from tables()" in ql:
            return ["table_name"], [["_qubx_symbols_manifest"], [FUNDING_TABLE]]
        if "_qubx_symbols_manifest" in ql:
            return ["table_name", "symbol", "last_updated"], [list(r) for r in self.manifest_rows]
        if "distinct" in ql:
            m = re.search(r"where timestamp > '([^']+)'", ql)
            rows = self.funding_rows
            if m:
                floor = pd.Timestamp(m.group(1))
                rows = [r for r in rows if pd.Timestamp(r[0]) > floor]
            return ["symbol"], [[s] for s in sorted({r[1] for r in rows})]
        if f'from "{FUNDING_TABLE}"' in ql:
            wanted = self._symbol_filter(ql)
            rows = [r for r in self.funding_rows if wanted is None or r[1].upper() in wanted]
            if "limit 1" in ql:  # get_time_range's min/max UNION probe (UNION dedups equal rows)
                if not rows:
                    return ["timestamp"], []
                ts = sorted(r[0] for r in rows)
                return ["timestamp"], [[ts[0]]] if ts[0] == ts[-1] else [[ts[0]], [ts[-1]]]
            return ["timestamp", "symbol", "funding_rate", "funding_interval_hours"], [list(r) for r in rows]
        raise AssertionError(f"unexpected query: {query}")

    def execute_no_result(self, query: str) -> None:
        self.queries.append(query)


def _reader(pgc: FakePGC, manifest_symbols: set[str]) -> QuestDBReader:
    meta = xLTableMetaInfo.decode_table_metadata(FUNDING_TABLE)
    assert meta is not None
    return QuestDBReader(
        "BINANCE.UM",
        "SWAP",
        [meta],
        pgc,
        synthetic_ohlc_timeframes_types=False,
        min_symbols_for_all_data_request=50,
        symbols_by_table={FUNDING_TABLE: set(manifest_symbols)},
        manifest_manager=None,  # never consulted: refresh interval far exceeds the test
        lookup_refresh_interval=timedelta(days=365),
    )


_ROWS = [
    (pd.Timestamp("2025-11-01 00:00"), "BTCUSDT", 0.0001, 8.0),
    (pd.Timestamp("2025-11-01 08:00"), "BTCUSDT", 0.0002, 8.0),
    (pd.Timestamp("2025-11-01 00:00"), "BNBUSDT", 0.0003, 8.0),
]


class TestManifestIsNotAuthority:
    """The symbols manifest is a lookup cache; a stale one must never hide rows the table holds."""

    def test_read_returns_symbols_missing_from_manifest(self):
        pgc = FakePGC(funding_rows=_ROWS)
        reader = _reader(pgc, manifest_symbols={"BNBUSDT"})  # stale: BTCUSDT not registered
        res = reader.read(["BTCUSDT", "BNBUSDT"], "funding_payment", start="2025-11-01", stop="2025-11-02")
        assert {k: len(v) for k, v in res.raws.items()} == {"BTCUSDT": 2, "BNBUSDT": 1}
        data_queries = [q for q in pgc.queries if "funding_rate" in q]
        assert any("BTCUSDT" in q and "BNBUSDT" in q for q in data_queries), "both symbols must be in the WHERE"

    def test_read_of_unknown_symbol_stays_filtered(self):
        # pre-fix an empty manifest intersection dropped the WHERE clause and scanned the whole table
        pgc = FakePGC(funding_rows=_ROWS)
        reader = _reader(pgc, manifest_symbols={"BNBUSDT"})
        res = reader.read(["BTCUSDT"], "funding_payment", start="2025-11-01", stop="2025-11-02")
        assert set(res.raws) == {"BTCUSDT"}
        data_queries = [q for q in pgc.queries if "funding_rate" in q]
        assert all("BNBUSDT" not in q for q in data_queries), "unfiltered all-data scan leaked other symbols"

    def test_get_time_range_probes_table_not_manifest(self):
        reader = _reader(FakePGC(funding_rows=_ROWS), manifest_symbols={"BNBUSDT"})
        t0, t1 = reader.get_time_range("BTCUSDT", "funding_payment")
        assert pd.Timestamp(t0) == pd.Timestamp("2025-11-01 00:00")
        assert pd.Timestamp(t1) == pd.Timestamp("2025-11-01 08:00")

    def test_get_time_range_single_row_table(self):
        reader = _reader(FakePGC(funding_rows=_ROWS[2:]), manifest_symbols={"BNBUSDT"})
        t0, t1 = reader.get_time_range("BNBUSDT", "funding_payment")
        assert pd.Timestamp(t0) == pd.Timestamp(t1) == pd.Timestamp("2025-11-01 00:00")

    def test_get_time_range_absent_symbol_raises(self):
        reader = _reader(FakePGC(funding_rows=_ROWS), manifest_symbols={"BNBUSDT"})
        with pytest.raises(ValueError, match="NOPEUSDT"):
            reader.get_time_range("NOPEUSDT", "funding_payment")


class TestManifestRefreshSeesBackfills:
    def test_stale_refresh_discovers_backfilled_symbols(self):
        # BTCUSDT rows were backfilled: their DATA timestamps predate the manifest watermark,
        # so the old incremental refresh (WHERE timestamp > last_updated) could never see them.
        watermark = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=3)
        rows = [
            (watermark - pd.Timedelta(days=300), "BTCUSDT", 0.0001, 8.0),
            (watermark - pd.Timedelta(days=1), "ETHUSDT", 0.0001, 8.0),
        ]
        pgc = FakePGC(funding_rows=rows, manifest_rows=[(FUNDING_TABLE, "ETHUSDT", watermark.to_pydatetime())])
        mgr = SymbolsManifestManager(pgc, cache_ttl=timedelta(hours=24))
        meta = xLTableMetaInfo.decode_table_metadata(FUNDING_TABLE)
        assert meta is not None
        out = mgr.get_symbols_for_tables([meta])
        assert out[FUNDING_TABLE] == {"BTCUSDT", "ETHUSDT"}
        inserts = [q for q in pgc.queries if q.startswith("INSERT")]
        assert any("BTCUSDT" in q for q in inserts), "discovered symbol must be persisted to the manifest"


class TestQuestDbStorages:
    def test_xldecoding(self):
        x1 = xLTableMetaInfo.decode_table_metadata("gateio.swap.candles_1m")
        assert x1 is not None
        assert x1.exchange == "GATEIO.F"
        assert x1.market_type == "SWAP"
        assert x1.dtype == DataType.OHLC
        assert x1.data_timeframe == "1m"

        x2 = xLTableMetaInfo.decode_table_metadata("any_custom_table")
        assert x2 is None

        x3 = xLTableMetaInfo.decode_table_metadata("binance.umswap.funding_payment")
        assert x3 is not None
        assert x3.exchange == "BINANCE.UM"
        assert x3.market_type == "SWAP"
        assert x3.dtype == DataType.FUNDING_PAYMENT
        assert x3.data_timeframe is None

        x4 = xLTableMetaInfo.decode_table_metadata("test1.universe.factors")
        assert x4 is not None
        assert x4.exchange == "TEST1"
        assert x4.market_type == "UNIVERSE"
        assert x4.dtype == DataType.RECORD
        assert x4.alias_for_record_type == "factors"
