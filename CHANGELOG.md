# Changelog

All notable changes to Qubx will be documented in this file.

## [1.0.0.dev1] - 2026-03-09

### Features

- **runner:** Pass account_manager to storage constructors ([#193](https://github.com/xLydianSoftware/Qubx/issues/193))


### Miscellaneous

- Fix git cliff

## [1.0.0.dev4] - 2026-03-06

### Bug Fixes

- Remove unnecessary imports

- Add logs when failed, update binance live example notebook


### Features

- Speed up simulation loading

- Add unified config management


### Miscellaneous

- Update changelog for 1.0.0.dev4

## [1.0.0.dev3] - 2026-03-04

### Bug Fixes

- Adds theme for tui

- Improve release versioning to consider all tag types


### Miscellaneous

- Update changelog for 1.0.0.dev3

## [0.7.40.dev14] - 2026-03-04

### Miscellaneous

- Update changelog for 0.7.40.dev14

## [0.7.40.dev13] - 2026-03-04

### Bug Fixes

- Change focus to table

- Tui fix

- Many small fixes to backtest results storage manager. Also improves loading speed

- Emitter data / log copying

- **ccxt:** Ccxt reader market type override

- Refactoring results saving with attachments

- Move stale data detector to a separate scheduled task

- Add support for quote(1m)

- Trading session result from file with simulation time

- Simulator prefetch fix (not yet fully handled proper re-reading)

- Rounding bug in pnl accounting


### Documentation

- Update release process with changelog and version details


### Features

- Add equity curve plotting to tui

- Add log cloud storing, add storage close to interface

- Add session slicing

- Issues #170, #171 first implementation of market data cache and its use in indicators


### Miscellaneous

- Update changelog for 0.7.40.dev13

- Add changelog and update release flow

- Update release md

## [1.0.0.dev2] - 2026-02-25

### Bug Fixes

- Cached storage fixed

- Fix funding payment storage timestamp and processing

- Added gateio swap to questdb storage mapping

- Stale detector argument fix

- Fixed CCXT storage and integration tests config

- Removed old tests and renamed composite to iteratedstream

- TestMultiStorage test fixed for fallback

- Fixed integration test for quaestdb emitter

- Ta small fix data

- Refactor ome tests - use new storage API, remove RestoreTicksFromOHLC, add date range test

- Warmup config and runner to support multiple storages and required main data storage #181, #178

- Added tolerance logic to MutliStorage

- Removed deprecated tests

- Gathering tests

- Context_test

- Context initializer

- Account processor tests

- Refactor indicators tests

- Performance issue to_pd() transformer for big number of raw containers

- Cached storage eviction logic fix

- Cached reader cutoff time fix

- Csv reader / tests / guard

- Add docstrings

- Add close interface method to IReader and implementation for QuestDBReader

- Simulated data subscription

- Make test more flexible

- Fixed issue #168

- Fix types

- Fix test

- Issue-157

- Fixed performance of questdb reader

- Handle string columns in _extract_column

- Questd

- CsvReader fixed

- Transformers tests fixed

- Fix build_snapshots and related code

- Another approach for creating ohlc series


### Features

- Add release command to justfile

- Issue #175 (trading sessions times)

- TimeGuardedStorage/Reader implementation

- **questdb:** Add auto manifest table

- Add bundled series

- Implement columnar series

- Issue 156 implemented


### Miscellaneous

- **data:** Fix columnar series transformer

- Rename tests

- Redo _tol_mask_nb to numba

- Add test_emulated_updates_subscription_with_trading_session, fix some timeframe issues in handy storage

- Migration of aux in backtester / fixes of some series tests

- Marked outdated tests

- Transformer's redesign to provide more clean interface

- Add storage / aux tests

- Guards tests have same behaviour on different storages now

- Imports fix

- Fix bundled series

- Add bundled series to interface

- Add lazy loading, add back find minimal timeframe


### Refactoring

- Postwarmup init tests

- Removed direct referencing to CachedMarketDataHolder's instance as much as possible and now IMarketManager provides interface for accessing cache.

- Removed unnecessary reference to scheduler object in SimulatedDataProvider / CcxtDataProvider

- Redo SimulatedDataProvider and get_ohlc implementation is moved to SimulatedDataIterator

- Remove old code / refactor new one

- Pandas and ohlc transformers

- Refactor Raw container (issue 152)


### Wip

- Ccxt storage initial implementation

- Removed hft.py and deps

- Fix integration tests

- Implemented first version of MultiStorage/MultiReader (replacemnt of CompositeReader)

- Add trading session to simulator's config

- Fix / cleanup tests

- Tests refactoring

- All simulator tests fixed

- Simulator 2'nd bunch of tests refactored

- Simulator tests (1'st stage) fixed

- Simulators tests in progress

- Fix simulator tests

- Parametrization for cache

- Initial implementation of cached storages issues: #128/#127/#126

- Aux data storages implementation

- Add ohlc test

- Simulator runner refactoring

- Issue #163

- Cleanup code

- Refactoring IterableSimulationData for IStorage

- Issue 159 first version

- Move emulating transformers to backtest package

- IStorage issue 135 and lot of small fixes in transformers / readers

- Docs

- Notebook

- TickSeries transformer

- Before some tests

- Add new instrument types to InstrumentType enum

- RawData refactoring

## [0.7.40.dev12] - 2026-02-22

### Features

- **ccxt:** Add Gate.io futures exchange support ([#179](https://github.com/xLydianSoftware/Qubx/issues/179))

- **cli:** Add backward-compatible poetry support to deploy command ([#174](https://github.com/xLydianSoftware/Qubx/issues/174))


### Miscellaneous

- Add pid log

## [0.7.40.dev10] - 2026-02-17

### Bug Fixes

- **cli:** Clean uv-specific sections from pyproject.toml during release ([#172](https://github.com/xLydianSoftware/Qubx/issues/172))

## [0.7.40.dev9] - 2026-02-11

### Bug Fixes

- **live:** Np.integer handling for recognize_time

- Value bug in portfolio construction for spot


### Documentation

- Update release documentation to reflect current CI pipeline


### Features

- Enhance reader construction with account manager support

- Introduce plugin system for loading external connectors and readers


### Miscellaneous

- Minor fixes

- Remove test for order creation error handling with client ID

- Remove unnecessary import

## [0.7.40.dev8] - 2026-01-25

### Bug Fixes

- **ci:** Only create GitHub Releases for stable versions

## [0.7.40.dev7] - 2026-01-25

### Features

- **ci:** Add GitHub Release creation with artifacts and release notes

## [0.7.40.dev6] - 2026-01-25

### Features

- **ci:** Enable PyPI publishing after TestPyPI

## [0.7.40.dev5] - 2026-01-25

### Bug Fixes

- Remove numpy upper bound to allow numpy 2.x for Python 3.13 wheels

## [0.7.40.dev4] - 2026-01-25

### Bug Fixes

- **ci:** Install uv before cibuildwheel for macOS/Windows

## [0.7.40.dev3] - 2026-01-25

### Features

- **ci:** Parallelize Python version builds (3.12, 3.13)

## [0.7.40.dev2] - 2026-01-25

### Features

- **ci:** Expand build matrix to all platforms and Python versions

## [0.7.40.dev1] - 2026-01-25

### Bug Fixes

- Upgrade to cibuildwheel v3 and make uvloop platform-specific

- Update pyarrow constraint and version scheme for Python 3.13


### Features

- **release:** Enhance version management and CI workflows


### Miscellaneous

- **ci:** Simplify build matrix to use ubuntu-latest

- **ci:** Update build matrix for macOS 15 and Ubuntu 22.04/24.04

- Comment out PyPI publishing steps in CI workflow

- Remove CHANGELOG.md file and update justfile for changelog generation

## [0.7.33] - 2026-01-16

### Features

- Add streaming vwma indicator to Qubx by @dmarienko

## [0.7.32] - 2026-01-15

### Bug Fixes

- Fix data transforming when None comes in raw by @dmarienko

## [0.7.31] - 2026-01-14

### Bug Fixes

- Fix https://github.com/xLydianSoftware/Qubx/issues/114 by @dmarienko


### Miscellaneous

- Added xicorr to pandaz.stats by @dmarienko

## [0.7.30] - 2026-01-09

### Bug Fixes

- Ccxt reaader for sub hour timeframes by @dmarienko

- Fix MarketManager.ohlc unnecessary data provider requests by @dmarienko

## [0.7.28] - 2026-01-05

### Bug Fixes

- Remove pandas warning message by @dmarienko


### Lighter

- Fix empty positions by @yuriy-arabskyy


### Rust

- Add structure by @yuriy-arabskyy

## [0.7.29] - 2026-01-06

### Bug Fixes

- Cleanup moved some old stuff into pandaz module by @dmarienko


### Miscellaneous

- Removed unnecessary imports by @dmarienko

- Small cleanup by @dmarienko


### Lighter

- Use position updates in account by @yuriy-arabskyy

- Api migrations by @yuriy-arabskyy

- Fix orderbook handler by @yuriy-arabskyy


### Wip

- Add stack trace output on simulation failure by @dmarienko

## [0.7.27] - 2025-12-23

### Bug Fixes

- Extends OHLC handler to support chunked data retrieval by @dmarienko


### Refactoring

- Fix constant name by @dmarienko

## [0.7.21] - 2025-12-13

### Bug Fixes

- Fix OME recreation issue with emulation order/trade id by @dmarienko

## [0.7.19] - 2025-12-08

### Features

- Add custom path for simulation reports (.md) by @dmarienko

- Add Qr to meta by @dmarienko

- Refined .md backtest reports and simulator now stores reports by @dmarienko

## [0.7.18] - 2025-12-05

### Bug Fixes

- Sanity timestamp conversion reverted by @dmarienko

- Fix strategy warmups processing by @dmarienko

- Minutes intervals cron parsing fix: "5Min -1s" now generate correct cron schedule by @dmarienko


### Miscellaneous

- Removed deprecated code / tests by @dmarienko

- Test for IterableSimulationData on subscribe/resubscibe by @dmarienko

- Test by @dmarienko


### Refactoring

- Refactor fix by @dmarienko


### Account

- Restore accounting by @yuriy-arabskyy


### Live

- Fix ccxt total capital, fix lighter/ccxt margin ratio by @yuriy-arabskyy


### Wip

- Warmup / cache clean fixes by @dmarienko

- Stale quote bug fixing in progress ... by @dmarienko


### Xlighter

- Added sync send order operation

## [0.7.17] - 2025-12-02

### Bug Fixes

- QuestDBStorage fixed for multithreading environment by @dmarienko

## [0.7.14] - 2025-12-01

### Bug Fixes

- Get_base_currency fixed by @dmarienko

- BacktestsResultsManager small fix by @dmarienko


### Miscellaneous

- Storage open questions by @dmarienko


### Health

- Fix tests by @yuriy-arabskyy

- Track active subscriptions only by @yuriy-arabskyy

- Add is connected by @yuriy-arabskyy


### Lighter

- Minor instrument loader fixes by @yuriy-arabskyy

- Refactor instrument loader by @yuriy-arabskyy

- Fix deal/position processing by @yuriy-arabskyy

## [0.7.13] - 2025-11-12

### Bug Fixes

- Fixed cusum indicator on interbar evemts updates by @dmarienko


### Miscellaneous

- Decomposed tests by @dmarienko

## [0.7.12] - 2025-11-11

### Bug Fixes

- Streaming indicators fixes by @dmarienko

## [0.7.9] - 2025-11-04

### Bug Fixes

- SuperTrend fixed for smaller timeframe updates by @dmarienko

- Add signatures by @dmarienko


### Features

- SuperTrend streaming version by @dmarienko

## [0.7.8] - 2025-11-04

### Lighter

- Make retry infinite by @yuriy-arabskyy

## [0.7.6] - 2025-11-02

### Bug Fixes

- Refactored cusum filter indicator by @dmarienko

- Fix OHLCV.from_dataframe() by @dmarienko


### Wip

- Series chached values helper by @dmarienko

## [0.7.5] - 2025-10-31

### Bug Fixes

- Fix pxd file by @dmarienko

- Improve cusum speed by @dmarienko

- Resample returns cached resamplers now to avoid recaclulations by @dmarienko


### Features

- Add macd indicator by @dmarienko

- Streaming version of cusum indicator by @dmarienko


### Miscellaneous

- Description in pyi by @dmarienko


### Slack

- Change to blocks by @yuriy-arabskyy

## [0.7.4] - 2025-10-30

### Bug Fixes

- Fixed OHLC resample logic for correct indicators updates by @dmarienko


### Features

- Refactor stdema streaming indicator by @dmarienko


### Miscellaneous

- Add streaming data tests for stdema by @dmarienko


### Refactoring

- Resample method by @dmarienko

- Moved cusum indicator here by @dmarienko


### Indicators

- Add differencing, improve min size handling by @yuriy-arabskyy


### Lighter

- Improve broker by @yuriy-arabskyy

- Add automatic reduce only by @yuriy-arabskyy

- Loop fixes by @yuriy-arabskyy

## [0.7.3] - 2025-10-28

### Features

- Streamed implementation of RSI indicator by @dmarienko

- Add support for negative shifts in schedule intervals: "1h -1s" etc by @dmarienko


### Lighter

- Update dependency to forked version by @yuriy-arabskyy

- Add orderbook resubscribe and offset tracking by @yuriy-arabskyy


### Readers

- Set default market type to swap by @yuriy-arabskyy


### Textual

- Change rich log to text area by @yuriy-arabskyy

- Update data widgets properly by @yuriy-arabskyy


### Wip

- Volatility ema (std on ema) indicator by @dmarienko

## [0.6.95] - 2025-10-24

### Runner

- Add base currency option by @yuriy-arabskyy

## [0.6.94] - 2025-10-20

### Features

- Add DataType.FUNDAMENTAL with parametrization. It's possible to specify what we excatly want to load from DB. by @dmarienko


### Lighter

- Add market stats handler by @yuriy-arabskyy

- Fix orderbook processing by @yuriy-arabskyy

- Improve orderbook handler by @yuriy-arabskyy


### Transfer

- Change info to debug by @yuriy-arabskyy

## [0.6.92] - 2025-10-15

### Bug Fixes

- Add expanduser on path in BacktestsResultsManager by @dmarienko

- Aggregation for aggregated_liquidations uses right boundary timestamping now by @dmarienko

- Timeframe detection issue fixed by @dmarienko

- Fix .pd() in GenericSeries and timeframe detection in TypedGenericSeries transformer by @dmarienko


### Features

- ;wimplementation for GenericSeries and IndicatorGeneric classes by @dmarienko


### Miscellaneous

- Add definitions to .pxd by @dmarienko


### Lighter

- Fix order update by @yuriy-arabskyy

- Fix account processing by @yuriy-arabskyy

## [0.6.91] - 2025-10-09

### Bug Fixes

- Fix test by @dmarienko

- Lowercase fix by @dmarienko

- Implementation for new orderbook data format conversion by @dmarienko

- Fi possible sorting bag in get_time_range in QuestDBStorage by @dmarienko

- Add AggregatedLiquidations to Timestamped type alias by @dmarienko

- Fix wrong clause by @dmarienko

- Liquidations by @dmarienko

- Removed trailing commas in sql by @dmarienko

- Add get_time_range implementation by @dmarienko

- Fixed splitting utility by @dmarienko

- Fix _find_time_idx in CSV reader by @dmarienko

- Types control fixed by @dmarienko

- Fix small issues by @dmarienko

- Fix TradingSessionResult::from_file datetime formats reading by @dmarienko


### Features

- QuestDB storage by @dmarienko

- Add multi symbol reading support. Also handles chanksize reading by @dmarienko

- Add multi raw data processing and combining by @dmarienko


### Miscellaneous

- Removes old code by @dmarienko

- Small fix by @dmarienko

- Add docstrings to data contaiers by @dmarienko


### Refactoring

- Refactor storages, add storage registry and add transformable helpers by @dmarienko

- Refactor names by @dmarienko


### Hyperliquid

- Improve execution by @yuriy-arabskyy


### Wip

- New orderbook data by @dmarienko

- New orderbook data by @dmarienko

- Add initial idea for external SQL builders in QuestDB by @dmarienko

- Reading refactoring by @dmarienko

- Quest DB in progress by @dmarienko

- QuestDB in progress by @dmarienko

- Add simulated ticks transformation implementation by @dmarienko

- Transformers with tests by @dmarienko

- Transformers in progress by @dmarienko

- Add tests for IStorage by @dmarienko

- IStorage approach by @dmarienko

## [0.6.87] - 2025-10-03

### Cli

- Add config validation command and task by @yuriy-arabskyy

## [0.6.85] - 2025-09-29

### Features

- Add ohlcv["close"] etc accessor for handy getting columns by @dmarienko


### Trades

- Add individual subscription support, support live trade delays, process bought volume, quote volume, trade count by @yuriy-arabskyy

## [0.6.84] - 2025-09-24

### Features

- Add markdown export by @dmarienko


### Ccxt

- Added auto reduce only flag for reducing positions orders

## [0.6.77] - 2025-09-09

### Jupyter

- Sort by notional value by @yuriy-arabskyy

## [0.6.74] - 2025-09-02

### Features

- Add dots marker support to matplotlib LookingGlass by @dmarienko

- Add swing indicator based stop loss tracker by @dmarienko


### Miscellaneous

- Move training stops on indicator to trackers folder by @dmarienko


### Series

- Fix update last by @yuriy-arabskyy


### Wip

- Add trailing impl and test for broker side by @dmarienko

- Trailing stop for client side by @dmarienko

- Add trailing stop position tracker by @dmarienko

## [0.6.69] - 2025-08-04

### Bug Fixes

- Show_latency_report in runner force INFO log level by @dmarienko


### Features

- Add tolerance-based deduplication to CompositeReader by @yuriy-arabskyy


### Miscellaneous

- Add wvf indicator by @dmarienko

- Add wvf indicator by @dmarienko

## [0.6.68] - 2025-07-30

### Bug Fixes

- Standardize Bar constructor to use keyword arguments by @yuriy-arabskyy

- Fix update_by_bars Series method for take in account volumes / trade counts by @dmarienko

- Fix volumes restoring in simulator by @dmarienko

- Hover width in mpl backend by @dmarienko

- Trick to return empty plot in mpl backend by @dmarienko

- Makes LookingGlass mpl to be more in line with plotly by @dmarienko

- Includes pivot's times by @dmarienko


### Features

- Add vector version of pivots_highs_lows indicator by @dmarienko


### Miscellaneous

- Fix handle_start_stop in chart_signals by @dmarienko

- Add chart_signals to TradingSessionResult by @dmarienko

- Cleanup some pyx code by @dmarienko

- Adds pivot's indexes by @dmarienko


### Wip

- Ohlc updates by @dmarienko

- Ohlc updates by @dmarienko

## [0.6.64] - 2025-07-14

### Bug Fixes

- F-string double quotes in get_fundamental_data by @dmarienko

- Ensure accurate leverage calculations by handling capital as a float or sum of values by @yuriy-arabskyy

- Resolve peek_historical_data issue in SimulatedDataProvider for filtered funding payment subscriptions by @yuriy-arabskyy

- Fix aux data extension in InMemoryCachedReader by @dmarienko

- Fix travel_in_time method in InMemoryCachedReader by @dmarienko


### Features

- Add get_funding_payment method to QuestDBConnector for retrieving funding payments with optional filtering by @yuriy-arabskyy


### Refactoring

- Streamline portfolio data dumping and enhance funding handling for SWAP instruments by @yuriy-arabskyy


### Bugfix

- Fix consolidated ohlc pd without explicit timeframe by @yuriy-arabskyy

## [0.6.63] - 2025-07-02

### Features

- Add OhlcDict.__getitem__ for list/tuple keys by @dmarienko

## [0.6.60] - 2025-06-30

### Bug Fixes

- Fix target position processing by @dmarienko

- Remove unused tracker map by @dmarienko

- Fix jupyter runner by @dmarienko

- Processing class didn't intialize collections by @dmarienko

- Fix test runner: weird behavior of pytest fixtures ! by @dmarienko

- Fix test runner by @dmarienko

- Fix initialization stage processing - skip initializing signal if active target is present by @dmarienko

- Fix test typo by @dmarienko

- Fix test runner by @dmarienko

- Adds log mesage by @dmarienko

- Fixed tracker routing by @dmarienko

- Add targets to TradingSessionResults class by @dmarienko

- Fix atr tests by @dmarienko

- Fix tests in progress by @dmarienko


### Features

- Active position implementation. SignalRiskPositionTracker is added by @dmarienko


### Miscellaneous

- Add todo in trade method by @dmarienko

- Restore test by @dmarienko


### Refactoring

- Refactor test scenario by @dmarienko

- Add tests for active position by @dmarienko

- Remove unnecessary checks by @dmarienko

- Processing by @dmarienko


### Wip

- Fix tests by @dmarienko

- Processing init signal by @dmarienko

- Post-warmup initialization by @dmarienko

- All tests are passing after splitting targets and signals by @dmarienko

- Removing signal from target position (still issues in tests) by @dmarienko

## [0.6.57] - 2025-06-20

### Bug Fixes

- Fix finalization for only subscribed instruments in live mode to prevent altering OHLC data for other instruments by @dmarienko

## [0.6.56] - 2025-06-17

### Bug Fixes

- Finalize OHLC data for all instruments in the universe on trigger event by @dmarienko

## [0.6.55] - 2025-06-17

### Bug Fixes

- Test_queue_monitoring_with_channel increased sleep time by @dmarienko


### Miscellaneous

- Removed some old code by @dmarienko


### Refactoring

- Added as_of parameter to find_instruments by @dmarienko

- Added reload interval to instruments mongo lookup by @dmarienko

- Refactors instruments lookup and many small changes by @dmarienko


### Wip

- New lookup design by @dmarienko

## [0.6.54] - 2025-06-08

### Bug Fixes

- Raise an error if no variable parameters were found for a simulation by @dmarienko


### Features

- Add variation_plot method to BacktestsResultsManager by @dmarienko

## [0.6.50] - 2025-06-06

### Bug Fixes

- Fix ohlc data retrieval bug when receiving only one record by @dmarienko


### Miscellaneous

- Loggers code cleanup by @dmarienko

## [0.6.48] - 2025-06-05

### Bug Fixes

- Imports fix by @dmarienko

- OME add support for deferred execution reports for stop market orders by @dmarienko

- Stop order price validation by @dmarienko

## [0.6.46] - 2025-05-29

### Bug Fixes

- Add subscription's warmup in on_init stage by @dmarienko

## [0.6.43] - 2025-05-27

### Bug Fixes

- Choppiness indicator : parameter with_raw_indicator by @dmarienko


### Features

- Add rolling_zscore to ta.py by @dmarienko

## [0.6.42] - 2025-05-15

### Exporters

- Added leverages initialization for redis exporter ([#46](https://github.com/xLydianSoftware/Qubx/issues/46)) by @bogdanKaftanatiy ([#46](https://github.com/xLydianSoftware/Qubx/pull/46))

## [0.6.41] - 2025-05-06

### Bug Fixes

- Moved instrument updates info to market data processor and fix on_fit check by @dmarienko


### Miscellaneous

- Add import of extend_trading_results by @dmarienko

- Add extend_trading_results() method by @dmarienko

## [0.6.40] - 2025-04-28

### Bug Fixes

- Fix run_paper.sh script generation by @dmarienko

## [0.6.39] - 2025-04-28

### Features

- Add run_paper.sh script generation by @dmarienko

## [0.6.38] - 2025-04-25

### Bug Fixes

- Fix simulations data readers configs in line with live runner by @dmarienko

## [0.6.37] - 2025-04-24

### Bug Fixes

- Error handling for order creation and cancellation by @dmarienko

- Backtest results loading and saving fix by @dmarienko

- Feature manager prevents multiple features with the same signatures by @dmarienko


### Features

- Notify error if error level is medium or higher by @dmarienko

- Add is_warmup_in_progress and is_paper_trading properties to IStrategyContext by @dmarienko

- Start and stop timestamps are processed properly in HFT data reader by @dmarienko


### Miscellaneous

- Fix type hints by @dmarienko


### Binance.um

- Enable ws for order cancelation by @yuriy-arabskyy

## [0.6.31] - 2025-04-16

### Bug Fixes

- Ccxt binance trade with zero price by @dmarienko


### Wip

- Add market state to execution reports in OME by @dmarienko

## [0.6.29] - 2025-04-11

### Bug Fixes

- Check if data is newer than previous update by @dmarienko


### Miscellaneous

- Bump version by @dmarienko

## [0.6.23] - 2025-03-29

### Bug Fixes

- Typo fix by @dmarienko

## [0.6.20] - 2025-03-28

### Bug Fixes

- Update version by @dmarienko


### Features

- Redesign of simulated exchange by @dmarienko


### Refactoring

- Rename OmeReport to SimulatedExecutionReport by @dmarienko

- Introduce interface for simulated exchange by @dmarienko

## [0.6.19] - 2025-03-26

### Bug Fixes

- Do not raise exception when canceling order that is not found in OME by @dmarienko

## [0.6.15] - 2025-03-24

### Bug Fixes

- Account.py by @dmarienko

- Small fix by @dmarienko


### Features

- Add traded range from time tests by @dmarienko


### Refactoring

- Tests refactoring by @dmarienko

## [0.6.13] - 2025-03-20

### Bug Fixes

- Part fills processing correctly by @dmarienko

## [0.6.6] - 2025-03-18

### Bug Fixes

- Small fixes in broker and riskctrl by @dmarienko

- Small fix in broker (wrong signature) by @dmarienko

## [0.6.5] - 2025-03-13

### Bug Fixes

- Add quote method to ActiveInstrument by @dmarienko

- Print new universe by @dmarienko

- Fix on_universe_change interceptor in jupyter runner and some other issues by @dmarienko

- Fix tests for release and deploy commands by @dmarienko

- Indicator small fix by @dmarienko


### Features

- Fix release and deploy commands for support composite strategies and pyx files by @dmarienko

## [0.6.4] - 2025-03-07

### Bug Fixes

- Broker and client executed at correct stop levels now by @dmarienko

- OME with exact fill at signal price for market orders by @dmarienko

- Fix signals log symbol column name by @dmarienko

## [0.6.3] - 2025-03-07

### Bug Fixes

- Support composite strategies in runner by @dmarienko

- Small fix by @dmarienko

- Restored import of loader by @dmarienko

- Fix test by @dmarienko

- Small fixe by @dmarienko

- HFT data processing in OME by @dmarienko

- Drop unnecessary quote -> mid_price -> restored quote conversion by @dmarienko

- Small fix in notebook by @dmarienko

- Fixes utils imports by @dmarienko


### Features

- Version update to 0.6.2 by @dmarienko

- Jupyter tools by @dmarienko

- Handy jupyter tools by @dmarienko

- Add special case for execution price at signal price for market orders by @dmarienko

- Adds test for real quotes by @dmarienko

- Multi data reader with examples by @dmarienko

- Add time expiration tracker by @dmarienko


### Wip

- Hft data reader by @dmarienko

- Processing single trade and array of trades in OME by @dmarienko

- Extends inside spread OME test by @dmarienko

- Add tests for inside spread execution by @dmarienko

- Fix OME by @dmarienko

## [0.5.7] - 2025-02-24

### Bug Fixes

- Allow scale position size by signal's value (flag in FixedRiskSizer) by @dmarienko

- Add processing of sort_by=None parameter to backtest management by @dmarienko

- Add tearsheet method to TradingSessionResult by @dmarienko

- Backtest managment by @dmarienko

- Tearsheet's title overflow by @dmarienko

- Add choppy identification method by @dmarienko

- Fix typo by @dmarienko

- Add volatility calculation and fix choppiness index calculation by @dmarienko

- Handling when historical ohlcv is empty by @dmarienko

- Fix pretty print for list method by @dmarienko

- Fix risk manager orders cancelling processing by @dmarienko

- Fix remove_instruments logic by @dmarienko

- Adds comments by @dmarienko


### Documentation

- Add docs to interfaces by @dmarienko


### Features

- Add variations support to backtests results manager by @dmarienko

- Add load_config method to backtester management by @dmarienko

- Fix list method to backtester management by @dmarienko

- Wip - set_unverse with position wait_for_change policy by @dmarienko

- Wip - set_unverse with position close policies by @dmarienko

- Add support for aux_data parameter in simulation config by @dmarienko

- Add support for cron expressions with custom format ("Q @ 23:59" - every quarter at 23:59 etc) by @dmarienko


### Testing

- Add tests for set_universe with different policies by @dmarienko


### Wip

- Tests for set_universe with position wait_for_change policy by @dmarienko

## [stable-0.5.5] - 2025-01-30

### Bug Fixes

- Fix description in list command by @dmarienko

- Fix color of sell orders by @dmarienko

- Fix fixed risk sizer entry price detection by @dmarienko

- F by @dmarienko

- Set Qubx default log level to WARNING by @dmarienko

- Hurst function docstring by @dmarienko

- Show_portfolio flag by @dmarienko

- Hover by @dmarienko

- Make dark plotly looks like mpl by @dmarienko

- None title is not displayed in LG. chart_signals can use plugins by @dmarienko

- Extracting strategy parameters before simualtion - strategy may alter intial parameters. Also it scans startegy mixings to get all parameters. by @dmarienko

- Typo by @dmarienko

- Fixes composite trackers issue by @dmarienko

- Variate now accepts function by @dmarienko

- Adds decorator on jma method by @dmarienko

- Makes tqdm progress smaller by @dmarienko

- When subscribing to new symbols it's also need to submit last quote to OME in simulator by @dmarienko

- Adds all spot symbols for Binance and Bitfinex by @dmarienko

- Temp fix of ccxt_integration_test by @dmarienko

- Strategy simulation doesn't stop after max number of failures in a row by @dmarienko

- Removes BatchEvent from processing by @dmarienko

- Removes leverage parameter from simulation method by @dmarienko

- Refactors simulator and adds applying default warmups if it's not specified in strategy by @dmarienko

- Small fixe for py < 3.12 by @dmarienko

- Correct sentinel in simulator by @dmarienko

- Additional test is added by @dmarienko

- Fixes exchange name for paper trading by @dmarienko

- Adds exchange into paper broker by @dmarienko

- More explainable warning by @dmarienko

- Tests by @dmarienko

- Refactors of update method with BatchEvent support by @dmarienko

- Update by @dmarienko

- Merged incoming by @dmarienko

- Adds check for preferred data type when probing data from reader by @dmarienko

- Fixes simulator tests and removed quotes by @dmarienko

- Small tests by @dmarienko

- Data sniffer now knows how to use get_ranges from the reader by @dmarienko

- Csv candles reading error by @dmarienko

- Some fixes of signals by @dmarienko

- Generated signals series might have non str names by @dmarienko

- Fixes search by naming in Series or DataFrame generated signals by @dmarienko

- Fixes non strategy config repr in TradingSessionResult by @dmarienko

- Trackers test fixed by @dmarienko

- Loggers fix (adds exchange in execution report) by @dmarienko

- Fixes progress bar and add dark ipywidgets background workaround trick by @dmarienko

- Fixes test for loader by @dmarienko

- Adds get_subscriptions_for_instrument method and it's test by @dmarienko

- Adds test on case for > 2 streams in slicer by @dmarienko

- Refactors typing by @dmarienko

- Small refactoring by @dmarienko

- Removes unnecessary rebuilding by @dmarienko

- Small cosmetic changes by @dmarienko


### Features

- Add performance export for TradingSessionResults.to_file() by @dmarienko

- Add simulation config file to results by @dmarienko

- Add support for conditions in variate by @dmarienko

- Add variations to the simulation results by @dmarienko

- Constant capital rtisk sizer by @dmarienko

- Add description to strategy config by @dmarienko

- Backtester management tool - add description to results by @dmarienko

- Add simulate command to Qubx CLI by @dmarienko

- Add quantity to execution log by @dmarienko

- Add TradingSessionResult.from_file() method for loading backtest results from zip file by @dmarienko

- Adds OHLCV.from_dataframe(pd.DataFrame) static method. Resets watchdog before simulation. by @dmarienko

- Dynamic mixin of strategies by @dmarienko

- Adds exchange() method to IMarketManager interface by @dmarienko

- Adds advanced trackers with entry improvements by @dmarienko

- Customize dataframes look in notebooks by @dmarienko

- Adds packed binance symbols meta-data by @dmarienko

- Last version of hist + last quote in simulator by @dmarienko

- Adds tests for historical data retrieval from simulted data provider by @dmarienko

- Adds exchange method to context by @dmarienko

- To_html and to_file in TradingSessionResult class by @dmarienko

- Removes "hist_" prefix conversion by @dmarienko

- Simulator uses default schedule if detected by @dmarienko

- Adds tests on simulation data recognition by @dmarienko

- Refactors recognize_simulation_setups methos and adds test by @dmarienko

- Adds RestoredBarsFromOHLC transformer by @dmarienko


### Miscellaneous

- Calls StrategyContext.stop() after simulation is finished. Adds latency report printing. by @dmarienko

- Some debug adjustment by @dmarienko

- Changed debug logging format by @dmarienko

- Adds hyperliquid symbols loading by @dmarienko

- Adds some helpers: this_project_root() method by @dmarienko

- Adjusting open_close_time_indent by @dmarienko

- Small refactoring by @dmarienko

- Version update by @dmarienko

- Small refactoring by @dmarienko

- Removes some unnecessary files by @dmarienko

- Provide exchange name in StrategyContext by @dmarienko

- Small changes by @dmarienko

- Before renaming Subtype -> DataType by @dmarienko

- Adds tests for IterableSimulatorData by @dmarienko


### Refactoring

- Moves plotting to utils module by @dmarienko

- Refactors backtesting utilities by @dmarienko

- Redone recognizer by @dmarienko

- Simulator utils by @dmarienko

- Cleanup quotes / bars emulating code by @dmarienko

- Refactoring IteratedDataStreamsSlicer class by @dmarienko


### Acc

- Finish position retrieval by @yuriy-arabskyy


### Wip

- Refactoring during composite tracker tests by @dmarienko

- Adds composite tracker's test (still failing ...) by @dmarienko

- Prettier logs in simulation by @dmarienko

- Tests by @dmarienko

- Crontab based on timedelta by @dmarienko

- ... by @dmarienko

- Refactoring of reader by @dmarienko

- ... by @dmarienko

- Adds some helpers to TradingSessionResult by @dmarienko

- Unsubscription is called by @dmarienko

- Checking tests by @dmarienko

- Removed unused tests by @dmarienko

- Simulated data provider by @dmarienko

- Datafetchers redesigned by @dmarienko

- Subscriptions by @dmarienko

- Full redo of iterators slicer by @dmarienko

- Timed queue by @dmarienko

- Tests added by @dmarienko

- Simulation data reader by @dmarienko

## [stable-0.4.3] - 2024-12-05

### Bug Fixes

- Fixes test by @dmarienko

- Reformats file by @dmarienko

- Fixes MarketDataProvider::ohlc logic by @dmarienko

- Changes info to debug messages in PortfolioRebalancerTracker by @dmarienko

- Fixes OhlcDict processing empty dataframes and removed annoying error message by @dmarienko

- OhlcDict key requirements led to simulation failure on instrument names started with number by @dmarienko

- Handling timeseries in TimeGuardedWrapper by @dmarienko

- Fixes justfile by @dmarienko

- Small fixes by @dmarienko

- Fixes test by @dmarienko

- Removed old load_data method by @dmarienko

- Add candles reading interface to csv reader, so csv reader can be used in loader by @dmarienko

- Notebook by @dmarienko

- Typo in cached reader by @dmarienko

- Simulator with aux_data by @dmarienko

- Refactoring by @dmarienko

- Position test fixed by @dmarienko

- Multiple small fixes and additional helpers by @dmarienko

- Fixes equal intervals splitting function by @dmarienko

- Adds some fields / methods interfaces into Bar class by @dmarienko

- Fixes BrokerSideRiskController stop order issue by @dmarienko

- Simulator stops backtesting if strategy fails more than N times a row by @dmarienko

- Simplest Gathering can handle entry at specified price by limit or stop orders by @dmarienko

- Fixes test by @dmarienko

- Fixes broker mode in risk controller by @dmarienko

- Client side risk controller fix by @dmarienko

- Locator logic is fixed and added ability to process complex indicators by @dmarienko

- Adds rounding for limit order price by @dmarienko

- Fixes test by @dmarienko

- Removes cell by @dmarienko

- Fixes tests by @dmarienko

- Typo fixed by @dmarienko

- Fixed states in stoptake basic class by @dmarienko

- Small fixes in tests by @dmarienko

- Version increasign by @dmarienko

- Testing taregt position by @dmarienko

- Fixes short execution marker color by @dmarienko

- Add support for signals in signals viewer by @dmarienko

- Increase version by @dmarienko

- Typo in position update by @dmarienko

- Upper case for debug level by @dmarienko

- Adds debug level in simulate method by @dmarienko

- Adds docs by @dmarienko

- Removes duplicated method for rounding average price by @dmarienko

- Fixes accuracy in position class by @dmarienko

- Version increase by @dmarienko

- Fixed risk sizer by @dmarienko

- Small typo in definitions by @dmarienko

- Fix queue tests by @yuriy-arabskyy

- Choppyness index indicator fixed type by @dmarienko

- Typo in signal processing by @dmarienko

- Adds tests by @dmarienko

- Refactors code a bit by @dmarienko

- Fixes test by @dmarienko

- Fixes sizer test by @dmarienko

- Fixes datareader test by @dmarienko

- Small fix by @dmarienko

- Negative balance fix by @dmarienko

- Fixes test by @dmarienko

- Fixes fixed risk calculations by @dmarienko

- Refactors trackers/sizers methods arguments and fixes Portfolio balancer by @dmarienko

- Position adjuster by @dmarienko

- Fixes scheduler test by @dmarienko

- Fixes closing positions by zero signal by @dmarienko

- Fixes stupid typo by @dmarienko

- Small fixes by @dmarienko

- Adds in memory data reader by @dmarienko

- Bookmarks file by @dmarienko

- Removed bookmarks file by @dmarienko

- Crone simulation is fixed by @dmarienko

- Removes _get_ohlc_data method from IExchangeServiceProvider by @dmarienko

- Adds notification on order's canceling and execution by @dmarienko

- Small fix in handler by @dmarienko

- Version update by @dmarienko

- Tries to fix pickling cython classes issue by @dmarienko

- Fixes pewma on streaming data by @dmarienko

- Fixes pewma_outliers_detector indicator in Qubx by @dmarienko

- Adds control on attemptin to update series by past data by @dmarienko

- Adds readme by @dmarienko

- Fixes math imports by @dmarienko

- Small one by @dmarienko

- Fixes issue with swings and psar series renamed by @dmarienko

- Adds test for swings by @dmarienko

- Removed file by @dmarienko

- Use 1min resample if not specified for candle builder by @dmarienko

- Fixes tables naming for QuestDB connector by @dmarienko

- Version update by @dmarienko

- Temporary disabled import imp module by @dmarienko

- Small one by @dmarienko

- Removes unnecessary notebook by @dmarienko

- Requirements and version by @dmarienko

- Small fix by @dmarienko

- Use apply_async for loogers by @dmarienko

- Balance logging save fix by @dmarienko

- Refactors by @dmarienko

- Reserves and position restoring by @dmarienko

- Adds abitlity to reserve desired amount of assets from trading by @dmarienko

- Temporary sync ccxt trading connector (WIP) by @dmarienko

- Balance calculations and part fills processing by @dmarienko

- Adds client order id by @dmarienko

- Fixes datareader, adds correct implementation of position average price by @dmarienko

- Fixes PnL calc in positions by @dmarienko

- Removes logs by @dmarienko

- Some experiments by @dmarienko

- Commented some init code by @dmarienko

- Typo by @dmarienko

- Tests loop by @dmarienko

- Removes unnecessary Self import by @dmarienko

- Bar trigger to be processed only once by @dmarienko

- Renamed methods and adds additional checks by @dmarienko

- Fixes default scheme for fees data by @dmarienko

- Add flag to recalculate indicators on closed bar by @dmarienko

- Removes unnecessary files by @dmarienko

- Small names fixes by @dmarienko

- Adds tests for ticks simulated data by @dmarienko

- Adds pyarrow dep in poetry by @dmarienko

- New logo test by @dmarienko

- Now indicators are being updated in correct order by @dmarienko

- Compare indicator fixed by @dmarienko

- Fixes Kama indcator and adds unit tests by @dmarienko

- Adds small refactoring for Kama by @dmarienko


### Features

- Adds example doc in loader by @dmarienko

- Adds OhlcDict wrapper to loader's dict output by @dmarienko

- Adds accurate_stop_orders_execution option to simulate method by @dmarienko

- New tracker implementation by @dmarienko

- LookinGlass accepts Qubx series as arguments by @dmarienko

- LookinGlass can show Qubx signals by @dmarienko

- Variate can process strategy with tracker by @dmarienko

- Adds signals logging by @dmarienko

- Atr based risk manager - first version by @dmarienko

- Refactored trackers / sizers / gathering by @dmarienko

- Trackers, sizers and gathering alpha version by @dmarienko

- Executions viewer by @dmarienko

- Adds ability to variate parameters by @dmarienko

- Adds get_historical_ohlc to simulated broker by @dmarienko

- Adds ability to convert Qubx portfolio to Qube presentation by @dmarienko

- Adds locator so it's possible to do slicing and searching in TimeSeries by @dmarienko

- Adds actual timestamps of spotted pivots in swings indicator by @dmarienko

- Adds loc method to TimeSeries class by @dmarienko

- Swings detector in cython by @dmarienko

- ATR indicator is added by @dmarienko

- Adds tests for psar OHLC based indicator by @dmarienko

- OHLC based indicators and PSAR indicator by @dmarienko

- Pewma_outliers_detector is added by @dmarienko

- Adds test for pewma indicator by @dmarienko

- Pewma indicator on streaming data is added by @dmarienko

- Adds positions updates by current prices by @dmarienko

- Loading positions and active orders from exchange during startegy starting by @dmarienko

- First version of StrategyCtx by @dmarienko

- Transfers some useful utils from Qube1 by @dmarienko

- BAR triggers - very first impl tested by @dmarienko

- Adds ohlc data subscriber by @dmarienko

- Adds ohlc data subscriber by @dmarienko

- Adds Kraken symbols info getting for lookup by @dmarienko

- Adds list of coins in the update_binance_data_storage by @dmarienko

- Adds binance data loader by @dmarienko

- Adds fees lookup with configuration etc by @dmarienko

- Adds a test by @dmarienko

- Flag that indicates whether calculate indicators on closed bars by @dmarienko

- Tests for on_formed_bar indicators calculations by @dmarienko

- Adds pandas csv reader by @dmarienko

- Emulating quotes / trades from ohlc data by @dmarienko

- First version of data readers with pyarrow as engine by @dmarienko

- Finishes Position and Transaction costs classes by @dmarienko

- Adds method for finding aux instrument by @dmarienko

- Adds lowest and highest indicators by @dmarienko

- Adds Kama indicator by @dmarienko

- Qube2 ideas by @dmarienko


### Miscellaneous

- Add qubx console utility by @dmarienko

- Adds abitlity to generate mpl tearsheet by @dmarienko

- Adds test on ohlc history in context by @dmarienko

- Version increased by @dmarienko

- Version 0.3.0 by @dmarienko

- Version increment by @dmarienko

- Version increment by @dmarienko

- Version increase by @dmarienko

- Version increase by @dmarienko

- Adds additional test by @dmarienko

- Merging main by @dmarienko

- Refactors IndicatorOHLC by @dmarienko

- Version increased by @dmarienko

- Refactors risk manager logic by @dmarienko

- Before refactoring by @dmarienko

- Renames options argument in trade method by @dmarienko

- Tests for stop orders by @dmarienko

- Move pnl to bottom in chart signals by @yuriy-arabskyy

- Version increment by @dmarienko

- Small renamings by @dmarienko

- Metric fixes, add indicator typing, swings middles by @yuriy-arabskyy

- Adds definitions for indicators by @dmarienko

- Add type hints for kama, atr, etc by @yuriy-arabskyy

- Adds pyi definitions for series methods and increases version by @dmarienko

- Add equity method, variate fixes by @yuriy-arabskyy

- Add historical quote processing by @yuriy-arabskyy

- Implement universe updates by @yuriy-arabskyy

- Version increment by @dmarienko

- Logger prints simulated timestamps when backtesting running by @dmarienko

- Changes version and update ignoring by @dmarienko

- Adds tests for trackers by @dmarienko

- Adds test for tracking and gathering by @dmarienko

- Validation of simulator by @dmarienko

- Adds metrics and tearcheets for Qubx by @dmarienko

- Adds portfolio performance metrics by @dmarienko

- New backtester initial version by @dmarienko

- Store working version before refactoring by @dmarienko

- Refactoring in progress by @dmarienko

- First working version of simulated broker by @dmarienko

- Big refactoring needed for simulator by @dmarienko

- Fixes printing exceptions by @dmarienko

- OME functionality is done by @dmarienko

- Adds tests on locator by @dmarienko

- Version increasing by @dmarienko

- Increment version for MultiQuestDB connector by @dmarienko

- Some generalization of QuestDB reader class by @dmarienko

- Adds suffix for QuestDB reader by @dmarienko

- Idea on OrderBook custom connector by @dmarienko

- Adds pandas ta tests from Qube by @dmarienko

- Adds implementation for QuestDBReader::get_names by @dmarienko

- New impl for readers and transformers by @dmarienko

- First QuestDB implementation by @dmarienko

- Adds ohlc_plot method by @dmarienko

- Cleanup runner config structure by @dmarienko

- Adds impl for get_historical_ohlc method by @dmarienko

- Implements on_fit processing when all data is ready by @dmarienko

- Small refactoring by @dmarienko

- Adds more logs on latency by @dmarienko

- Refactors portfolio logging subsystem by @dmarienko

- Adds balance logger by @dmarienko

- Adds run_id to log writer by @dmarienko

- Adds market value and it's sum to PnL report by @dmarienko

- Adds more details in exception by @dmarienko

- Adds README memo by @dmarienko

- Adds example of account config by @dmarienko

- Update version by @dmarienko

- Adds jupyter as option to run strategy by @dmarienko

- Adds new runner by @dmarienko

- Adds more detailed logs on errors by @dmarienko

- Test for account processor by @dmarienko

- Wip for orders processing by @dmarienko

- Adds test for position average price by @dmarienko

- Adds order processing by @dmarienko

- Version increment by @dmarienko

- More convenient parameters log by @dmarienko

- Adds tests for OHLCV listening by @dmarienko

- Moved some pandas utils from Qube1 by @dmarienko

- Renames method in Indicator class by @dmarienko

- Adds deps and useful utility in pandas by @dmarienko

- Renamed to Qubx by @dmarienko

- Fixes poetry building system by @dmarienko

- Test for BFS by @dmarienko

- Renamed repo and cleaned up code by @dmarienko

- Refactors by @dmarienko

- Small refactoring by @dmarienko

- Strats design by @dmarienko

- Compare indicator by @dmarienko

- Adds lag indicator, fixes indicator calculation on already formed series by @dmarienko

- More clear way to represent indicators as function by @dmarienko

- Fixes names presentation for indicators by @dmarienko

- Fixes tests by @dmarienko

- Refactors indicator by @dmarienko


### Refactoring

- TimeGuard class as decorator on DataReader by @dmarienko


### Testing

- Adds some memory tests by @dmarienko


### Bug

- Fix time in on start by @yuriy-arabskyy


### Bugfix

- Fix rounding, add tests by @yuriy-arabskyy


### Bugs

- Fix typos in series declarations by @yuriy-arabskyy


### Cjore

- Version increase by @dmarienko


### Core

- Pandas TA Indicators by @dmarienko


### Experiments

- Cythom implementation tests by @dmarienko

- Test TimeSeries approach by @dmarienko


### Kama

- Fix er division by zero by @yuriy-arabskyy


### Qdb

- Create per request cursors by @yuriy-arabskyy


### Wip

- Fix in progress by @dmarienko

- Test for loader by @dmarienko

- Adds OHLC tests by @dmarienko

- Ohlc data issues by @dmarienko

- Adds test on loader by @dmarienko

- Reader by @dmarienko

- Handy data loader by @dmarienko

- Indexing is added by @dmarienko

- Data reader and helper class by @dmarienko

- Service target positions by @dmarienko

- Testing of stop / takes in advance by @dmarienko

- Test for rebalancer by @dmarienko

- Trackers by @dmarienko

- Preparation for signals pricessing etc by @dmarienko

- Simulating pre-generated signals is done by @dmarienko

- Subscription and trigger schemes are provided by @dmarienko

- Simulator in progress ... by @dmarienko

- Simuation utility in progress ... by @dmarienko

- Simulator by @dmarienko

- Refactors core classes for backtester by @dmarienko

- In progress ... by @dmarienko

- Simulator by @dmarienko

- Ome by @dmarienko

- Before refactoring DataReader / processors by @dmarienko

- Trigger/fit schedules by @dmarienko

- Scheduler tests added by @dmarienko

- Scheduler in progress by @dmarienko

- Scheduler spec formats by @dmarienko

- Scheduler in progress by @dmarienko

- Fixes by @dmarienko

- Loggers etc in progress ... by @dmarienko

- Correct handling async methods by @dmarienko

- Before async refactoring by @dmarienko

- Small refactoring by @dmarienko

- Separated trading and data providers for CCXT connector by @dmarienko

- Orders processing by @dmarienko

- Fixes positions restoring from deals by @dmarienko

- Positions in progress by @dmarienko

- Accounts auth is added by @dmarienko

- Renames cached market data holder to proper name by @dmarienko

- Async issues by @dmarienko

- Ohlc series holder first version by @dmarienko

- Small fix by @dmarienko

- Triggers sub-system in development by @dmarienko

- In progress by @dmarienko

- DataProvider for CCXT in progress by @dmarienko

- In progress by @dmarienko

- CCXT market data reder by @dmarienko

- CCXT receiving market data experiments by @dmarienko

- Disruptors tests by @dmarienko

- Disruptor experiments by @dmarienko

- Disruptor experiments by @dmarienko

- Before renaming to qubx by @dmarienko

- Adds timeframe detection by @dmarienko

- Data readers and processors by @dmarienko

- Position for spot + tests by @dmarienko

- Position in progress by @dmarienko

- Positions calculations in progress by @dmarienko

- Strategy impl by @dmarienko

- Bsf experiments by @dmarienko

- Adds OHLCV implementation by @dmarienko

- Adds functionality for calculating indicators from already existing series data by @dmarienko

- Fixed inicator on inicator calculations by @dmarienko

- Fixed EMA, optimized SMA by @dmarienko

- Ema fixed by @dmarienko

- Ema in new cython approach by @dmarienko


