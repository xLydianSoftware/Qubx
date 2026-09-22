import datetime as dt
from functools import partial
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.catalog.sql import SqlCatalog  # noqa: E402
from pyiceberg.table import DataScan, Table  # noqa: E402
from pyiceberg.transforms import DayTransform, MonthTransform  # noqa: E402

from qubx.core.basics import DataType  # noqa: E402
from qubx.data.containers import RawData, RawMultiData  # noqa: E402
from qubx.data.storages.iceberg import (  # noqa: E402
    IcebergLakeReader,
    IcebergLakeStorage,
    _resolve_namespace_prefix,
    _task_order,
    aggregate,
    decode_table,
)

# - the lake layout datavault writes, rebuilt with pyiceberg alone (Qubx cannot import the datavault writers)
FLOW_AGGS = {"taker_buy_volume": "sum", "taker_sell_volume": "sum", "flow_toxicity_score": "avg"}
FLOW_SCHEMA = pa.schema(
    [pa.field("timestamp", pa.timestamp("us"), nullable=False), pa.field("symbol", pa.string(), nullable=False)]
    + [pa.field(c, pa.float64()) for c in FLOW_AGGS]
)
QUOTES_SCHEMA = pa.schema(
    [
        pa.field("ts_event", pa.timestamp("us"), nullable=False),
        pa.field("ts_recv", pa.timestamp("us"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("bid_price", pa.float64()),
        pa.field("bid_amount", pa.float64()),
        pa.field("ask_price", pa.float64()),
        pa.field("ask_amount", pa.float64()),
        pa.field("src_month", pa.string()),
    ]
)


def _create(catalog, identifier, schema, properties, partition=None):
    """`partition` is (column, transform) or (column, None) for identity."""
    if (identifier[0],) not in catalog.list_namespaces():
        catalog.create_namespace(identifier[0])
    table = catalog.create_table(identifier, schema=schema, properties=properties)
    if partition is not None:
        column, transform = partition
        with table.update_spec() as update:
            if transform is None:
                update.add_identity(column)
            else:
                update.add_field(column, transform, f"{column}_{transform}")
    return table


def _feature_props(kernel, interval, kind="feature", rollup_of=None):
    props = {"dvault.kind": kind, "dvault.kernel": kernel, "dvault.interval": interval, "dvault.provider": "tardis"}
    props |= {f"dvault.agg.{c}": op for c, op in FLOW_AGGS.items()}
    if rollup_of:
        props["dvault.rollup_of"] = rollup_of
    return props


def _feature_day(day, symbols, minutes=120):
    rows = {n: [] for n in FLOW_SCHEMA.names}
    t0 = dt.datetime(day.year, day.month, day.day)
    for s in symbols:
        for m in range(minutes):
            rows["timestamp"].append(t0 + dt.timedelta(minutes=m))
            rows["symbol"].append(s)
            for c in FLOW_AGGS:
                rows[c].append(float(m))
    return pa.Table.from_pydict(rows, schema=FLOW_SCHEMA)


def _flow_table(catalog, namespace, days_symbols):
    table = _create(
        catalog,
        (namespace, "trade_flow_1m"),
        FLOW_SCHEMA,
        _feature_props("trade_flow", "1m"),
        ("timestamp", DayTransform()),
    )
    for day, symbols in days_symbols:
        table.append(_feature_day(day, symbols))
    return table


def _rollup(catalog, namespace, interval):
    base = catalog.load_table((namespace, "trade_flow_1m")).scan().to_arrow()
    rolled = aggregate(base, FLOW_AGGS, interval).cast(FLOW_SCHEMA)
    table = _create(
        catalog,
        (namespace, f"trade_flow_{interval}"),
        FLOW_SCHEMA,
        _feature_props("trade_flow", interval, kind="rollup", rollup_of=f"{namespace}.trade_flow_1m"),
        ("timestamp", MonthTransform()),
    )
    table.append(rolled)


def _quotes(t0, rows, symbol, price):
    stamps = [t0 + dt.timedelta(seconds=i) for i in range(rows)]
    return pa.Table.from_pydict(
        {
            "ts_event": stamps,
            "ts_recv": stamps,
            "symbol": [symbol] * rows,
            "bid_price": [price + i for i in range(rows)],
            "bid_amount": [1.0] * rows,
            "ask_price": [price + i + 0.5 for i in range(rows)],
            "ask_amount": [2.0] * rows,
            "src_month": ["2026-08"] * rows,
        },
        schema=QUOTES_SCHEMA,
    )


def _quotes_table(catalog):
    return _create(
        catalog,
        ("binance_perp", "quotes"),
        QUOTES_SCHEMA,
        {"dvault.kind": "raw", "dvault.provider": "tardis"},
        ("symbol", None),
    )


@pytest.fixture
def catalog(tmp_path):
    return SqlCatalog("test", **{"uri": f"sqlite:///{tmp_path}/cat.db", "warehouse": f"file://{tmp_path}"})


DAYS = (dt.date(2026, 8, 10), dt.date(2026, 8, 11))


@pytest.fixture
def lake(catalog):
    _flow_table(catalog, "binance_perp", [(d, ["BTCUSDT", "ETHUSDT"]) for d in DAYS])
    _quotes_table(catalog).append(_quotes(dt.datetime(2026, 8, 10), 10, "BTCUSDT", 1.0))
    return IcebergLakeStorage.from_catalog(catalog)


QUOTE_FILES = ((dt.datetime(2026, 8, 10), 4000), (dt.datetime(2026, 8, 10, 12), 4000), (dt.datetime(2026, 8, 11), 2000))
QUOTE_SYMBOLS = {"BTCUSDT": 100.0, "ETHUSDT": 200.0}


def _quotes_lake(catalog, order):
    table = _quotes_table(catalog)
    for symbol, price in QUOTE_SYMBOLS.items():
        for i in order:
            t0, rows = QUOTE_FILES[i]
            table.append(_quotes(t0, rows, symbol, price))
    return IcebergLakeStorage.from_catalog(catalog)


@pytest.fixture
def raw_lake(catalog):
    """One quotes table with three data files per symbol, appended in time order."""
    return _quotes_lake(catalog, (0, 1, 2))


@pytest.fixture
def shuffled_raw_lake(catalog):
    """The same three files per symbol, the newest appended first."""
    return _quotes_lake(catalog, (2, 0, 1))


@pytest.fixture
def daily_lake(catalog, lake):
    """The `lake` feature table plus the `_1d` rollup discovery reads."""
    _rollup(catalog, "binance_perp", "1d")
    return IcebergLakeStorage.from_catalog(catalog)


@pytest.fixture
def two_venue_lake(catalog, lake):
    _flow_table(catalog, "bybit_perp", [(dt.date(2026, 8, 10), ["BTCUSDT"])])
    return IcebergLakeStorage.from_catalog(catalog)


@pytest.fixture
def scratch_lake(catalog):
    """A `trade_flow_1m` table under a `scratch__` namespace prefix: the plain reader must not see it."""
    _flow_table(catalog, "scratch__binance_perp", [(dt.date(2026, 8, 10), ["BTCUSDT"])])
    return catalog


def test_decode_feature_and_raw_names():
    f = decode_table(
        ("binance_perp", "trade_flow_1m"), {"dvault.kind": "feature", "dvault.agg.taker_buy_volume": "sum"}
    )
    assert (f.kind, f.provider, f.dtype, f.alias, f.timeframe) == ("feature", "", DataType.RECORD, "trade_flow", "1m")
    assert f.aggs == {"taker_buy_volume": "sum"}
    r = decode_table(("binance_perp", "quotes"), {"dvault.kind": "raw", "dvault.provider": "tardis"})
    assert (r.kind, r.provider, r.dtype, r.alias) == ("raw", "tardis", DataType.QUOTE, None)
    c = decode_table(("binance_perp", "candles_1m"), {"dvault.kind": "feature"})
    assert (c.dtype, c.timeframe) == (DataType.OHLC, "1m")


def test_decode_skips_a_table_without_kind():
    """Every lake table carries dvault.kind (census 2026-09-22); one without it is not the lake's."""
    assert decode_table(("binance_perp", "orderbook_updates_l2"), {}) is None
    assert decode_table(("binance_perp", "trade_flow_1m"), {"dvault.agg.x": "sum"}) is None


def test_decode_reads_a_rollup_as_derived():
    t = decode_table(("binance_perp", "trade_flow_1h"), {"dvault.kind": "rollup", "dvault.rollup_of": "x"})
    assert (t.kind, t.alias, t.timeframe, t.time_column) == ("rollup", "trade_flow", "1h", "timestamp")


def test_decode_skips_scratch_and_ops_and_every_other_arity():
    kind = {"dvault.kind": "feature"}
    assert decode_table(("scratch", "rg20k"), kind) is None
    assert decode_table(("ops", "parity"), kind) is None
    assert decode_table(("binance_perp",), kind) is None
    assert decode_table(("features", "binance", "perp", "trade_flow_1m"), kind) is None
    assert decode_table(("binance", "perp", "trades"), kind) is None


def test_exchanges_and_readers(lake):
    assert lake.get_exchanges() == ["BINANCE.UM"]
    assert lake.get_market_types("BINANCE.UM") == ["SWAP"]
    reader = lake["BINANCE.UM", "SWAP"]
    assert set(reader.get_data_id("trade_flow")) == {"BTCUSDT", "ETHUSDT"}
    assert reader.get_data_id("quotes") == ["BTCUSDT"]


def test_get_reader_defers_discovery(lake, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("handing out a reader must not touch the catalog")

    monkeypatch.setattr(SqlCatalog, "list_namespaces", refuse)
    monkeypatch.setattr(SqlCatalog, "load_table", refuse)
    monkeypatch.setattr(Table, "scan", refuse)
    reader = lake["BINANCE.UM", "SWAP"]
    monkeypatch.undo()

    assert set(reader.get_data_id("trade_flow")) == {"BTCUSDT", "ETHUSDT"}
    assert reader.read("BTCUSDT", "trade_flow", "2026-08-10", "2026-08-11").data.num_rows == 120


def test_discovery_publishes_its_index_before_it_looks_done():
    tables = [
        decode_table(("binance_perp", "trade_flow_1m"), {"dvault.kind": "feature"}),
        decode_table(("binance_perp", "quotes"), {"dvault.kind": "raw"}),
    ]
    calls, published = [], []

    class Spy(IcebergLakeReader):
        def __setattr__(self, name, value):
            if name in ("_index", "_resolved") and value:
                published.append(name)
            super().__setattr__(name, value)

    def source():
        calls.append((reader._resolved, dict(reader._index)))
        return tables

    reader = Spy(None, "BINANCE.UM", "SWAP", source)
    assert reader.tables == tables
    assert calls == [(None, {})]
    assert published == ["_index", "_resolved"]
    assert set(reader._lookup) >= {"trade_flow", "quotes", "quote"}
    reader._discover()
    assert len(calls) == 1


def test_read_single_symbol_feature(lake):
    raw = lake["BINANCE.UM", "SWAP"].read("BTCUSDT", "trade_flow", "2026-08-10", "2026-08-11")
    assert isinstance(raw, RawData)
    df = raw.to_pd()
    assert len(df) == 120
    assert "symbol" not in df.columns
    assert "taker_buy_volume" in df.columns


def test_read_all_symbols_and_resample(lake):
    multi = lake["BINANCE.UM", "SWAP"].read([], "trade_flow(1h)", "2026-08-10", "2026-08-12")
    assert isinstance(multi, RawMultiData)
    df = multi.to_pd(id_in_index=True)
    assert len(df) == 2 * 2 * 2  # 2 symbols x 2 days x 2 hours of 120 minutes
    assert df.index.get_level_values("symbol").nunique() == 2


def test_resample_applies_the_table_aggregations(lake):
    df = lake["BINANCE.UM", "SWAP"].read("BTCUSDT", "trade_flow(1h)", "2026-08-10", "2026-08-11").to_pd()
    first_hour = df.iloc[0]
    assert first_hour["taker_buy_volume"] == sum(range(60))  # sum
    assert first_hour["flow_toxicity_score"] == pytest.approx(sum(range(60)) / 60)  # avg


def test_read_raw_quotes_renames_ts_event(lake):
    df = lake["BINANCE.UM", "SWAP"].read("BTCUSDT", "quotes", "2026-08-10", "2026-08-11").to_pd()
    assert len(df) == 10
    assert list(df.columns)[:4] == ["bid_price", "bid_amount", "ask_price", "ask_amount"]
    assert df.index.name == "timestamp"


def test_time_range(lake):
    s, e = lake["BINANCE.UM", "SWAP"].get_time_range("BTCUSDT", "trade_flow")
    assert str(s)[:10] == "2026-08-10" and str(e)[:10] == "2026-08-11"


def test_time_range_raw_comes_from_file_bounds(lake):
    s, e = lake["BINANCE.UM", "SWAP"].get_time_range("BTCUSDT", "quotes")
    assert (str(s), str(e)) == ("2026-08-10T00:00:00.000000", "2026-08-10T00:00:09.000000")


def test_prefixed_table_hidden_without_the_prefix(scratch_lake):
    assert IcebergLakeStorage.from_catalog(scratch_lake).get_exchanges() == []


def test_prefixed_table_discovered_with_the_prefix(scratch_lake):
    store = IcebergLakeStorage.from_catalog(scratch_lake, namespace_prefix="scratch__")
    assert store.get_exchanges() == ["BINANCE.UM"]
    assert store["BINANCE.UM", "SWAP"].get_data_id("trade_flow") == ["BTCUSDT"]


def test_namespace_prefix_env_fallback_is_honoured(monkeypatch):
    monkeypatch.setattr("qubx.data.storages.iceberg._qubx_account", lambda account: {})
    monkeypatch.setenv("DVAULT_ICEBERG_NAMESPACE_PREFIX", "scratch__")
    assert _resolve_namespace_prefix("r2", "") == "scratch__"


def test_namespace_prefix_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setattr("qubx.data.storages.iceberg._qubx_account", lambda account: {})
    monkeypatch.setenv("DVAULT_ICEBERG_NAMESPACE_PREFIX", "scratch__")
    assert _resolve_namespace_prefix("r2", "explicit__") == "explicit__"


def test_namespace_prefix_defaults_to_empty_with_nothing_configured(monkeypatch):
    monkeypatch.setattr("qubx.data.storages.iceberg._qubx_account", lambda account: {})
    monkeypatch.delenv("DVAULT_ICEBERG_NAMESPACE_PREFIX", raising=False)
    assert _resolve_namespace_prefix("r2", "") == ""


def test_from_catalog_honours_the_env_var_when_the_kwarg_is_omitted(scratch_lake, monkeypatch):
    monkeypatch.setenv("DVAULT_ICEBERG_NAMESPACE_PREFIX", "scratch__")
    assert IcebergLakeStorage.from_catalog(scratch_lake).get_exchanges() == ["BINANCE.UM"]


def test_from_catalog_explicit_kwarg_overrides_the_env_var(scratch_lake, monkeypatch):
    monkeypatch.setenv("DVAULT_ICEBERG_NAMESPACE_PREFIX", "scratch__")
    assert IcebergLakeStorage.from_catalog(scratch_lake, namespace_prefix="").get_exchanges() == []


def test_chunked_read_yields_one_block_per_window(lake):
    reader = lake["BINANCE.UM", "SWAP"]
    chunks = list(reader.read("BTCUSDT", "trade_flow", "2026-08-10", "2026-08-10T04:00", chunksize=60))
    assert [len(c) for c in chunks] == [60, 60, 0, 0]


def test_chunked_raw_read_walks_the_files_in_time_order(raw_lake):
    reader = raw_lake["BINANCE.UM", "SWAP"]
    chunks = list(reader.read("BTCUSDT", "quotes", "2026-08-10", "2026-08-12", chunksize=4000))
    assert [len(c) for c in chunks] == [4000, 4000, 2000]
    whole = reader.read("BTCUSDT", "quotes", "2026-08-10", "2026-08-12")
    pd.testing.assert_frame_equal(pd.concat([c.to_pd() for c in chunks]), whole.to_pd())


def test_chunked_raw_read_orders_files_the_writer_appended_out_of_order(shuffled_raw_lake):
    reader = shuffled_raw_lake["BINANCE.UM", "SWAP"]
    chunks = list(reader.read("BTCUSDT", "quotes", "2026-08-10", "2026-08-12", chunksize=4000))
    frame = pd.concat([c.to_pd() for c in chunks])
    assert frame.index.is_monotonic_increasing
    whole = reader.read("BTCUSDT", "quotes", "2026-08-10", "2026-08-12")
    pd.testing.assert_frame_equal(frame, whole.to_pd().sort_index())


def test_task_order_keys_on_the_lower_bound_and_sorts_a_missing_one_last():
    micros = int(dt.datetime(2026, 8, 10, tzinfo=dt.UTC).timestamp() * 1_000_000)
    bounded = SimpleNamespace(file=SimpleNamespace(lower_bounds={1: micros.to_bytes(8, "little", signed=True)}))
    unbounded = SimpleNamespace(file=SimpleNamespace(lower_bounds={}))
    no_stats = SimpleNamespace(file=SimpleNamespace(lower_bounds=None))
    assert _task_order(bounded, 1) == (0, micros)
    assert _task_order(unbounded, 1) == (1, 0)
    assert _task_order(no_stats, 1) == (1, 0)
    assert _task_order(bounded, 2) == (1, 0)
    key = partial(_task_order, field_id=1)
    assert sorted([unbounded, no_stats, bounded], key=key) == [bounded, unbounded, no_stats]


def test_chunked_raw_read_never_materializes_the_range(raw_lake, monkeypatch):
    reader = raw_lake["BINANCE.UM", "SWAP"]

    def refuse(*args, **kwargs):
        raise AssertionError("a chunked raw read must not materialize the whole range")

    monkeypatch.setattr(DataScan, "to_arrow", refuse)
    chunks = list(reader.read("BTCUSDT", "quotes", "2026-08-10", "2026-08-12", chunksize=3000))
    assert [len(c) for c in chunks] == [3000, 3000, 3000, 1000]


def test_chunked_raw_read_of_several_symbols_yields_multi_data(raw_lake):
    reader = raw_lake["BINANCE.UM", "SWAP"]
    chunks = list(reader.read(["BTCUSDT", "ETHUSDT"], "quotes", "2026-08-10", "2026-08-12", chunksize=6000))
    assert all(isinstance(c, RawMultiData) for c in chunks)
    assert [sum(len(r) for r in c) for c in chunks] == [6000, 6000, 6000, 2000]
    assert {s for c in chunks for s in c.raws} == set(QUOTE_SYMBOLS)


def test_chunked_raw_read_of_an_empty_range_yields_nothing(raw_lake):
    assert (
        list(raw_lake["BINANCE.UM", "SWAP"].read("BTCUSDT", "quotes", "2026-09-01", "2026-09-02", chunksize=4000)) == []
    )


def test_rollup_table_answers_the_coarse_request(catalog, lake):
    _rollup(catalog, "binance_perp", "1h")
    reader = IcebergLakeStorage.from_catalog(catalog)["BINANCE.UM", "SWAP"]
    assert set(reader.get_data_types("BTCUSDT")) == {"trade_flow(1m)", "quote"}
    multi = reader.read([], "trade_flow(1h)", "2026-08-10", "2026-08-12")
    assert {s: len(r) for s, r in multi.raws.items()} == {"BTCUSDT": 4, "ETHUSDT": 4}


def test_unsupported_timeframe_is_rejected(lake):
    with pytest.raises(ValueError, match="15m"):
        lake["BINANCE.UM", "SWAP"].read("BTCUSDT", "trade_flow(15m)", "2026-08-10", "2026-08-11")


def test_read_venues_concatenates_in_the_given_order(two_venue_lake):
    df = two_venue_lake.read_venues(
        ["bybit.f", "BINANCE.UM"], "SWAP", "BTCUSDT", "trade_flow", "2026-08-10", "2026-08-11"
    )
    assert list(df.columns)[0] == "venue"
    assert list(df["venue"].unique()) == ["BYBIT.F", "BINANCE.UM"]
    assert len(df) == 120 + 120


def test_read_venues_over_all_symbols_keeps_the_symbol_index(two_venue_lake):
    df = two_venue_lake.read_venues(["BINANCE.UM", "BYBIT.F"], "SWAP", [], "trade_flow", "2026-08-10", "2026-08-11")
    assert list(df.index.names) == ["timestamp", "symbol"]
    assert df.groupby("venue").size().to_dict() == {"BINANCE.UM": 240, "BYBIT.F": 120}


def test_read_venues_skips_a_venue_with_no_rows(two_venue_lake):
    df = two_venue_lake.read_venues(
        ["BINANCE.UM", "BYBIT.F"], "SWAP", "ETHUSDT", "trade_flow", "2026-08-10", "2026-08-11"
    )
    assert list(df["venue"].unique()) == ["BINANCE.UM"]
    assert len(df) == 120


def test_read_venues_with_no_rows_anywhere_is_an_empty_frame(two_venue_lake):
    df = two_venue_lake.read_venues(
        ["BINANCE.UM", "BYBIT.F"], "SWAP", "SOLUSDT", "trade_flow", "2026-08-10", "2026-08-11"
    )
    assert df.empty
    assert list(df.columns) == ["venue"]


def _scan_spy(monkeypatch) -> list[tuple[str, ...]]:
    seen: list[tuple[str, ...]] = []
    original = Table.scan

    def record(self, *args, **kwargs):
        seen.append(tuple(self.name()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Table, "scan", record)
    return seen


def test_feature_discovery_comes_from_the_daily_rollup(daily_lake):
    reader = daily_lake["BINANCE.UM", "SWAP"]
    assert reader.get_data_id("trade_flow") == ["BTCUSDT", "ETHUSDT"]
    s, e = reader.get_time_range("BTCUSDT", "trade_flow")
    assert (str(s), str(e)) == ("2026-08-10T00:00:00.000000", "2026-08-11T23:59:00.000000")


def test_coarse_requests_discover_through_the_same_daily_rollup(daily_lake, monkeypatch):
    reader = daily_lake["BINANCE.UM", "SWAP"]
    expected = reader.get_time_range("BTCUSDT", "trade_flow")
    seen = _scan_spy(monkeypatch)
    for request in ("trade_flow(1h)", "trade_flow(1d)"):
        assert reader.get_data_id(request) == ["BTCUSDT", "ETHUSDT"]
        assert reader.get_time_range("ETHUSDT", request) == expected
    assert seen == []


def test_feature_discovery_never_scans_the_minute_table(daily_lake, monkeypatch):
    reader = daily_lake["BINANCE.UM", "SWAP"]
    seen = _scan_spy(monkeypatch)
    assert reader.get_data_id("trade_flow") == ["BTCUSDT", "ETHUSDT"]
    reader.get_time_range("BTCUSDT", "trade_flow")
    reader.get_time_range("ETHUSDT", "trade_flow")
    assert set(reader.get_data_types("BTCUSDT")) == {"trade_flow(1m)", "quote"}
    assert seen == [("binance_perp", "trade_flow_1d")]


def test_a_symbol_absent_from_the_daily_rollup_is_an_error(daily_lake):
    with pytest.raises(ValueError, match="trade_flow_1d has no data for SOLUSDT"):
        daily_lake["BINANCE.UM", "SWAP"].get_time_range("SOLUSDT", "trade_flow")


def test_feature_discovery_falls_back_to_a_data_scan_without_a_daily_rollup(lake, monkeypatch):
    reader = lake["BINANCE.UM", "SWAP"]
    seen = _scan_spy(monkeypatch)
    assert reader.get_data_id("trade_flow") == ["BTCUSDT", "ETHUSDT"]
    s, e = reader.get_time_range("BTCUSDT", "trade_flow")
    assert (str(s), str(e)) == ("2026-08-10T00:00:00.000000", "2026-08-11T01:59:00.000000")
    assert seen == [("binance_perp", "trade_flow_1m")] * 2


def test_columns_project_a_direct_read(lake):
    df = (
        lake["BINANCE.UM", "SWAP"]
        .read("BTCUSDT", "trade_flow", "2026-08-10", "2026-08-11", columns=["taker_buy_volume"])
        .to_pd()
    )
    assert list(df.columns) == ["taker_buy_volume"]
    assert len(df) == 120


def test_columns_project_a_rollup_read(catalog, lake):
    _rollup(catalog, "binance_perp", "1d")
    reader = IcebergLakeStorage.from_catalog(catalog)["BINANCE.UM", "SWAP"]
    df = reader.read("BTCUSDT", "trade_flow(1d)", "2026-08-10", "2026-08-12", columns=["flow_toxicity_score"]).to_pd()
    assert list(df.columns) == ["flow_toxicity_score"]
    assert len(df) == 2


def test_columns_project_a_resample_and_aggregate_only_what_was_read(lake):
    df = (
        lake["BINANCE.UM", "SWAP"]
        .read([], "trade_flow(1h)", "2026-08-10", "2026-08-12", columns=["taker_sell_volume"])
        .to_pd(id_in_index=True)
    )
    assert list(df.columns) == ["taker_sell_volume"]
    assert len(df) == 8


def test_columns_project_a_streamed_raw_read(raw_lake):
    chunks = list(
        raw_lake["BINANCE.UM", "SWAP"].read(
            "BTCUSDT", "quotes", "2026-08-10", "2026-08-12", chunksize=4000, columns=["bid_price"]
        )
    )
    frames = [c.to_pd() for c in chunks]
    assert all(list(f.columns) == ["bid_price"] for f in frames)
    assert pd.concat(frames).index.is_monotonic_increasing
    assert [len(f) for f in frames] == [4000, 4000, 2000]


def test_an_unknown_column_is_named_in_the_error(lake):
    with pytest.raises(ValueError, match=r"binance_perp.trade_flow_1m has no column\(s\) \['nope'\]"):
        lake["BINANCE.UM", "SWAP"].read("BTCUSDT", "trade_flow", "2026-08-10", "2026-08-11", columns=["nope"])


def test_no_columns_argument_reads_every_data_column(lake):
    df = lake["BINANCE.UM", "SWAP"].read("BTCUSDT", "trade_flow", "2026-08-10", "2026-08-11").to_pd()
    assert list(df.columns) == list(FLOW_AGGS)


FUNDAMENTAL_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("us"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("asset", pa.string()),
        pa.field("metric", pa.string()),
        pa.field("value", pa.float64()),
    ]
)


@pytest.fixture
def coingecko_lake(catalog):
    table = _create(
        catalog,
        ("global_crypto", "fundamental"),
        FUNDAMENTAL_SCHEMA,
        {"dvault.kind": "event", "dvault.kernel": "fundamental", "dvault.provider": "coingecko"},
        ("timestamp", MonthTransform()),
    )
    rows = {n: [] for n in FUNDAMENTAL_SCHEMA.names}
    for day in (dt.datetime(2026, 7, 31), dt.datetime(2026, 8, 1)):
        for asset, cap in (("BTC", 1.5e12), ("ETH", 4.0e11)):
            for metric, value in (("market_cap", cap), ("price", cap / 1e7), ("total_volume", cap / 50)):
                rows["timestamp"].append(day)
                rows["symbol"].append(asset)
                rows["asset"].append(asset)
                rows["metric"].append(metric)
                rows["value"].append(value)
    table.append(pa.Table.from_pydict(rows, schema=FUNDAMENTAL_SCHEMA))
    return IcebergLakeStorage.from_catalog(catalog)


def test_coingecko_is_addressed_like_the_questdb_storage_addresses_it(coingecko_lake):
    assert coingecko_lake.get_exchanges() == ["COINGECKO"]
    assert coingecko_lake.get_market_types("COINGECKO") == ["FUNDAMENTAL"]


def test_fundamental_request_resolves_through_the_data_type(coingecko_lake):
    reader = coingecko_lake["COINGECKO", "FUNDAMENTAL"]
    assert reader.get_data_id("fundamental") == ["BTC", "ETH"]
    multi = reader.read(["BTC", "ETH"], "fundamental", "2026-07-31", "2026-08-02", columns=["metric", "value"])
    df = multi.to_pd(id_in_index=True)
    assert list(df.columns) == ["metric", "value"]
    caps = df[df["metric"] == "market_cap"]["value"]
    assert len(caps) == 4
    assert caps.xs("BTC", level="symbol").iloc[0] == 1.5e12


def test_fundamental_time_range_crosses_a_month_partition(coingecko_lake):
    s, e = coingecko_lake["COINGECKO", "FUNDAMENTAL"].get_time_range("ETH", "fundamental")
    assert (str(s)[:10], str(e)[:10]) == ("2026-07-31", "2026-08-01")


def test_two_tables_answering_one_request_are_refused(catalog):
    props = {"dvault.kind": "feature", "dvault.kernel": "open_interest", "dvault.interval": "1m"}
    oi_schema = pa.schema(
        [
            pa.field("timestamp", pa.timestamp("us"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("open_interest", pa.float64()),
        ]
    )
    _create(catalog, ("binance_perp", "open_interest"), oi_schema, props)
    _create(catalog, ("binance_perp", "open_interest_1m"), oi_schema, props)
    reader = IcebergLakeStorage.from_catalog(catalog)["BINANCE.UM", "SWAP"]
    with pytest.raises(ValueError, match=r"both answer open_interest\(1m\)"):
        reader.get_data_id("open_interest")
