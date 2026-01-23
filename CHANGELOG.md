# CHANGELOG


## v0.7.39-rc.1 (2026-01-23)


## v0.7.37 (2026-01-22)


## v0.7.36 (2026-01-22)


## v0.7.35 (2026-01-22)


## v0.7.34 (2026-01-21)


## v0.7.33 (2026-01-16)

### Features

- Add streaming vwma indicator to Qubx
  ([`2d0bff6`](https://github.com/xLydianSoftware/Qubx/commit/2d0bff6709fe49d8c8e25996d3aeb994aac1729c))


## v0.7.32 (2026-01-15)

### Bug Fixes

- Fix data transforming when None comes in raw
  ([`9fa0450`](https://github.com/xLydianSoftware/Qubx/commit/9fa0450e2cdde2af3452d9c6d9de6ed226506352))


## v0.7.31 (2026-01-14)

### Bug Fixes

- Fix https://github.com/xLydianSoftware/Qubx/issues/114
  ([`42eaf73`](https://github.com/xLydianSoftware/Qubx/commit/42eaf7378bd32e6a5452ffa20bf95acfbf8aeeb9))

### Chores

- Added xicorr to pandaz.stats
  ([`508c48a`](https://github.com/xLydianSoftware/Qubx/commit/508c48a5b7582b6c2526034a54fa57a87e5c600f))


## v0.7.30 (2026-01-09)

### Bug Fixes

- Ccxt reaader for sub hour timeframes
  ([`39a04d5`](https://github.com/xLydianSoftware/Qubx/commit/39a04d53fc1150dda55ba1f9bc8fcc5e956e01df))

- Fix MarketManager.ohlc unnecessary data provider requests
  ([`fd5a0a7`](https://github.com/xLydianSoftware/Qubx/commit/fd5a0a767ae9fb639d87882c13d9725fe0c2e8b1))


## v0.7.28 (2026-01-05)

### Bug Fixes

- Remove pandas warning message
  ([`d8097f1`](https://github.com/xLydianSoftware/Qubx/commit/d8097f167c6c2bc9f8b2c71b148c61c7e078ace5))


## v0.7.29 (2026-01-06)

### Bug Fixes

- Cleanup moved some old stuff into pandaz module
  ([`f917c3d`](https://github.com/xLydianSoftware/Qubx/commit/f917c3dcdc56c3c00b1281f9773c18efbd494700))

### Chores

- Removed unnecessary imports
  ([`43dae1c`](https://github.com/xLydianSoftware/Qubx/commit/43dae1c13f3963bdb0494808909a1d8f2fbb0410))

- Small cleanup
  ([`310a7de`](https://github.com/xLydianSoftware/Qubx/commit/310a7deae3cc329f0e951e9707773bde86b64505))


## v0.7.27 (2025-12-23)

### Bug Fixes

- Extends OHLC handler to support chunked data retrieval
  ([`66b765d`](https://github.com/xLydianSoftware/Qubx/commit/66b765d2282f2e76e2411ca70468660758bb9de6))

### Refactoring

- Fix constant name
  ([`ec57c75`](https://github.com/xLydianSoftware/Qubx/commit/ec57c757665f709ed0975f3317074b6580c66c2e))


## v0.7.26 (2025-12-19)


## v0.7.25 (2025-12-18)


## v0.7.24 (2025-12-18)


## v0.7.23 (2025-12-17)


## v0.7.22 (2025-12-13)


## v0.7.21 (2025-12-13)

### Bug Fixes

- Fix OME recreation issue with emulation order/trade id
  ([`d337ba2`](https://github.com/xLydianSoftware/Qubx/commit/d337ba290c90300c3d1bcac32e59bf0c485cda7c))


## v0.7.20 (2025-12-08)


## v0.7.19 (2025-12-08)

### Features

- Add custom path for simulation reports (.md)
  ([`954aba2`](https://github.com/xLydianSoftware/Qubx/commit/954aba2cfa07d3b8bec5b1d04184895306ea5b45))

- Add Qr to meta
  ([`54deb78`](https://github.com/xLydianSoftware/Qubx/commit/54deb781a7e1e6fb11aa262fdc24a59e98e9551d))

- Refined .md backtest reports and simulator now stores reports
  ([`05f7a67`](https://github.com/xLydianSoftware/Qubx/commit/05f7a67278bc66a300b94a86c6fa05d1cee1c79e))


## v0.7.18 (2025-12-05)

### Bug Fixes

- Fix strategy warmups processing
  ([`394b16d`](https://github.com/xLydianSoftware/Qubx/commit/394b16da934cc2564811827317617e8fce20d902))

- Minutes intervals cron parsing fix: "5Min -1s" now generate correct cron schedule
  ([`47b1cdb`](https://github.com/xLydianSoftware/Qubx/commit/47b1cdb465af1fa351146db6a7ae0c092ca0e6cc))

- Sanity timestamp conversion reverted
  ([`cc87b99`](https://github.com/xLydianSoftware/Qubx/commit/cc87b990a089f7f0d3082bd1195bda081ab3f6f5))

### Chores

- Removed deprecated code / tests
  ([`64630fa`](https://github.com/xLydianSoftware/Qubx/commit/64630facb7491572e04039711e22b384472d632d))

- Test
  ([`95eb100`](https://github.com/xLydianSoftware/Qubx/commit/95eb100aff18d2d0aa0a85510bbd21d94421fb84))

- Test for IterableSimulationData on subscribe/resubscibe
  ([`e61f7e0`](https://github.com/xLydianSoftware/Qubx/commit/e61f7e04d41a9b87b16467d840389d4043e2f255))

### Refactoring

- Refactor fix
  ([`5d9a130`](https://github.com/xLydianSoftware/Qubx/commit/5d9a130ffb4ab3b1fe473dce5e4e0550c3f5c07a))


## v0.7.17 (2025-12-02)

### Bug Fixes

- Questdbstorage fixed for multithreading environment
  ([`80a8d08`](https://github.com/xLydianSoftware/Qubx/commit/80a8d089e3d360368d8894d761d30b1c5ddbf1af))


## v0.7.16 (2025-12-02)


## v0.7.15 (2025-12-01)


## v0.7.14 (2025-12-01)

### Bug Fixes

- Backtestsresultsmanager small fix
  ([`f1724fe`](https://github.com/xLydianSoftware/Qubx/commit/f1724febfff25650ff060d7223f7f969e944bcc0))

- Get_base_currency fixed
  ([`46062c7`](https://github.com/xLydianSoftware/Qubx/commit/46062c74fb2474359a2277e2ef9d0317f33e6698))

### Chores

- Storage open questions
  ([`e4850ef`](https://github.com/xLydianSoftware/Qubx/commit/e4850ef55187221045bf3e8e113d44b975092195))


## v0.7.13 (2025-11-12)

### Bug Fixes

- Fixed cusum indicator on interbar evemts updates
  ([`3fdab15`](https://github.com/xLydianSoftware/Qubx/commit/3fdab15340e2995b1edc67f11d243dbe4d6bee6c))

### Chores

- Decomposed tests
  ([`1aa3091`](https://github.com/xLydianSoftware/Qubx/commit/1aa30911a5e47aed178edcc91f3c168a9a0c37f1))


## v0.7.12 (2025-11-11)

### Bug Fixes

- Streaming indicators fixes
  ([`1b93a5d`](https://github.com/xLydianSoftware/Qubx/commit/1b93a5da52c94e7e0a9501df109d93300021c441))


## v0.7.11 (2025-11-05)


## v0.7.10 (2025-11-05)


## v0.7.9 (2025-11-04)

### Bug Fixes

- Add signatures
  ([`4e53dce`](https://github.com/xLydianSoftware/Qubx/commit/4e53dcec141a1d112b7b8ac60b6d3e4f21eaed8d))

- Supertrend fixed for smaller timeframe updates
  ([`d334053`](https://github.com/xLydianSoftware/Qubx/commit/d334053a6eeb1641c2526bd6173123ea0f428cfe))

### Features

- Supertrend streaming version
  ([`1406815`](https://github.com/xLydianSoftware/Qubx/commit/1406815e9017eddd87ec7f7e163f696bfe0a9beb))


## v0.7.8 (2025-11-04)


## v0.7.7 (2025-11-03)


## v0.7.6 (2025-11-02)

### Bug Fixes

- Fix OHLCV.from_dataframe()
  ([`c4fdfe0`](https://github.com/xLydianSoftware/Qubx/commit/c4fdfe01e9b7cf908913e0dc0061f96e0871f20f))

- Refactored cusum filter indicator
  ([`d7a4c30`](https://github.com/xLydianSoftware/Qubx/commit/d7a4c300ff8ca0cd3539bbc76d1cb16c5e5ecbfe))


## v0.7.5 (2025-10-31)

### Bug Fixes

- Fix pxd file
  ([`5aeb145`](https://github.com/xLydianSoftware/Qubx/commit/5aeb14566ff4f25023442bfd6d233517cf4f861a))

- Improve cusum speed
  ([`eb19f5e`](https://github.com/xLydianSoftware/Qubx/commit/eb19f5e5cdc080c12ca54a93e4736f67c684667e))

- Resample returns cached resamplers now to avoid recaclulations
  ([`2a4b459`](https://github.com/xLydianSoftware/Qubx/commit/2a4b4595eafbdb5d5dccbfc66f56dca20f7c5f32))

### Chores

- Description in pyi
  ([`b7c0d50`](https://github.com/xLydianSoftware/Qubx/commit/b7c0d5045dbe3d558087941be08254f8b5063637))

### Features

- Add macd indicator
  ([`72bb722`](https://github.com/xLydianSoftware/Qubx/commit/72bb722b13f325a5d1937eedeaf9ca1bb18912d5))

- Streaming version of cusum indicator
  ([`6bbc0c7`](https://github.com/xLydianSoftware/Qubx/commit/6bbc0c748f6a95dc6957466f2741f5822781252b))


## v0.7.4 (2025-10-30)

### Bug Fixes

- Fixed OHLC resample logic for correct indicators updates
  ([`c5792ea`](https://github.com/xLydianSoftware/Qubx/commit/c5792eace8ba3448706780475b2553d9926e4229))

### Chores

- Add streaming data tests for stdema
  ([`f810497`](https://github.com/xLydianSoftware/Qubx/commit/f8104978ceaa8957a4f72d776dfa5a26d9d4da98))

### Features

- Refactor stdema streaming indicator
  ([`2269a7d`](https://github.com/xLydianSoftware/Qubx/commit/2269a7df5e2fcf8082c045cce6a7fbff73fceb2b))

### Refactoring

- Resample method
  ([`ad711fe`](https://github.com/xLydianSoftware/Qubx/commit/ad711fe85d17a79752e3149fc758636b1dd62264))


## v0.7.3 (2025-10-28)


## v0.7.2 (2025-10-28)


## v0.6.95 (2025-10-24)


## v0.6.94 (2025-10-21)


## v0.6.93 (2025-10-15)


## v0.6.92 (2025-10-15)

### Bug Fixes

- Add expanduser on path in BacktestsResultsManager
  ([`4ec7f05`](https://github.com/xLydianSoftware/Qubx/commit/4ec7f0533e4a2c8576b6e8eccff5ecf8d7907e01))

- Aggregation for aggregated_liquidations uses right boundary timestamping now
  ([`bf6283f`](https://github.com/xLydianSoftware/Qubx/commit/bf6283fa86d667442e302ac90f298b5644799ccf))

- Fix .pd() in GenericSeries and timeframe detection in TypedGenericSeries transformer
  ([`6d8e472`](https://github.com/xLydianSoftware/Qubx/commit/6d8e4724caadb94c59ce35bab3445ab1bb9b3307))

- Timeframe detection issue fixed
  ([`5bd61ab`](https://github.com/xLydianSoftware/Qubx/commit/5bd61ab495d23281af720c32075be16586207e59))

### Chores

- Add definitions to .pxd
  ([`8296280`](https://github.com/xLydianSoftware/Qubx/commit/8296280de5de67e5a22444608e388ff1b11fd375))


## v0.6.91 (2025-10-09)

### Bug Fixes

- Add AggregatedLiquidations to Timestamped type alias
  ([`1232420`](https://github.com/xLydianSoftware/Qubx/commit/1232420f96c68efde330f2969d3b781b63181f1b))

- Add get_time_range implementation
  ([`1cef379`](https://github.com/xLydianSoftware/Qubx/commit/1cef37904f623a0d68cc7a749741faf411d61e21))

- Fi possible sorting bag in get_time_range in QuestDBStorage
  ([`7858e0d`](https://github.com/xLydianSoftware/Qubx/commit/7858e0d9062294fe44752e84d228243580895a24))

- Fix _find_time_idx in CSV reader
  ([`b74af66`](https://github.com/xLydianSoftware/Qubx/commit/b74af66e189250e1b3f646bc210a35d3f0b63bd6))

- Fix small issues
  ([`839555d`](https://github.com/xLydianSoftware/Qubx/commit/839555d28a946ddc875f69d6ecc5bfc078cb914c))

- Fix test
  ([`d5483d1`](https://github.com/xLydianSoftware/Qubx/commit/d5483d1fa5c5a462d06de5a3e0f02160f41634e4))

- Fix TradingSessionResult::from_file datetime formats reading
  ([`13f71a2`](https://github.com/xLydianSoftware/Qubx/commit/13f71a25f72a3c3aa11a521fdc9ef9d593e9e441))

- Fix wrong clause
  ([`1b7db81`](https://github.com/xLydianSoftware/Qubx/commit/1b7db81890cbe65902c132a8fbc56a024b129fdf))

- Fixed splitting utility
  ([`98751f9`](https://github.com/xLydianSoftware/Qubx/commit/98751f95c6e0f41f3228f40b4ef2a033c4bd5301))

- Implementation for new orderbook data format conversion
  ([`0e7d12e`](https://github.com/xLydianSoftware/Qubx/commit/0e7d12e7d434cfbf8f25fb12a2ca321d0515935f))

- Liquidations
  ([`69a540c`](https://github.com/xLydianSoftware/Qubx/commit/69a540cce904afbce00a8654572d7ef282649334))

- Lowercase fix
  ([`ca36074`](https://github.com/xLydianSoftware/Qubx/commit/ca36074e69fe19645b1b7a9a340a8f4a933b449e))

- Removed trailing commas in sql
  ([`e11b75c`](https://github.com/xLydianSoftware/Qubx/commit/e11b75c8f64c34e48d2e35640637d42179911602))

- Types control fixed
  ([`03c1e3c`](https://github.com/xLydianSoftware/Qubx/commit/03c1e3c58d6ab08056d5ce5ae751241a89a4ddbd))

### Chores

- Add docstrings to data contaiers
  ([`d05cc39`](https://github.com/xLydianSoftware/Qubx/commit/d05cc3944a3a06bf3587fd6593975681d69efaae))

- Removes old code
  ([`ccf0c0d`](https://github.com/xLydianSoftware/Qubx/commit/ccf0c0d6b9fdfe36887901ad9186e46ba30791c2))

- Small fix
  ([`e3a7235`](https://github.com/xLydianSoftware/Qubx/commit/e3a72357bc3254a7f763566a6cd5dc883bc7ba02))

### Features

- Add multi symbol reading support. Also handles chanksize reading
  ([`cf06e0f`](https://github.com/xLydianSoftware/Qubx/commit/cf06e0f4295d76f360e13ded58d568b64af88a5f))

- Questdb storage
  ([`112fc2e`](https://github.com/xLydianSoftware/Qubx/commit/112fc2ecb89a499f2f7148279b9efde72df29bbc))

### Refactoring

- Refactor names
  ([`99571fa`](https://github.com/xLydianSoftware/Qubx/commit/99571fa1056f401448ded6c1930959f685251524))

- Refactor storages, add storage registry and add transformable helpers
  ([`94f5add`](https://github.com/xLydianSoftware/Qubx/commit/94f5add5a10ff727f7054ad4660d7d02af591beb))


## v0.6.90 (2025-10-05)


## v0.6.89 (2025-10-04)


## v0.6.88 (2025-10-04)


## v0.6.87 (2025-10-03)


## v0.6.86 (2025-10-03)


## v0.6.85 (2025-09-29)


## v0.6.84 (2025-09-24)

### Features

- Add markdown export
  ([`1f80a50`](https://github.com/xLydianSoftware/Qubx/commit/1f80a5090ff158a9f40a7dedc2bea7456aa42f2a))


## v0.6.78 (2025-09-10)


## v0.6.77 (2025-09-09)


## v0.6.76 (2025-09-08)


## v0.6.75 (2025-09-08)


## v0.6.74 (2025-09-02)

### Chores

- Move training stops on indicator to trackers folder
  ([`5abb356`](https://github.com/xLydianSoftware/Qubx/commit/5abb356e05ddad01467245959d335e5e93f6a153))

### Features

- Add dots marker support to matplotlib LookingGlass
  ([`9c34d26`](https://github.com/xLydianSoftware/Qubx/commit/9c34d266826e86c77e60c65240984222e01c6382))

- Add swing indicator based stop loss tracker
  ([`fc49c19`](https://github.com/xLydianSoftware/Qubx/commit/fc49c1933af4d519daab4dfe6ad40559e303206c))


## v0.6.73 (2025-08-14)


## v0.6.72 (2025-08-06)

### Bug Fixes

- Improvedentrytracker(stoptakepositiontracker)
  ([`e237060`](https://github.com/xLydianSoftware/Qubx/commit/e23706052b506673c7d28ba445849710c62406a4))


## v0.6.71 (2025-08-04)


## v0.6.70 (2025-08-04)


## v0.6.69 (2025-08-04)

### Bug Fixes

- Show_latency_report in runner force INFO log level
  ([`5e38ea7`](https://github.com/xLydianSoftware/Qubx/commit/5e38ea74665b4f8d2ca1cebb1ade553f20bb0ab1))

### Chores

- Add wvf indicator
  ([`f2abf87`](https://github.com/xLydianSoftware/Qubx/commit/f2abf8759a4aa716d60f8d824f3d31b4b0c99a6f))

- Add wvf indicator
  ([`192f94f`](https://github.com/xLydianSoftware/Qubx/commit/192f94ff9f063d16f61bca697b114ac75029aa04))

### Features

- Add tolerance-based deduplication to CompositeReader
  ([`5a8d837`](https://github.com/xLydianSoftware/Qubx/commit/5a8d8375aae6e3675b29d2fd3cafba28408f4bea))

- Implement max_history enforcement in CCXT reader get_funding_payment method - Add
  concat_with_tolerance merge strategy with 1min default tolerance - Add per-symbol timestamp
  clustering algorithm for intelligent deduplication - Set keep='last' as default behavior for
  near-duplicate timestamps - Fix circular import issue by moving NoDataContinue import to local
  scope - Add comprehensive unit tests covering MultiIndex/single-index scenarios - Update existing
  tests to reflect new keep='last' default behavior

Key features: - Sophisticated timestamp grouping within tolerance windows - Per-symbol processing
  prevents cross-symbol interference - Support for both DataFrame and Series data structures -
  Configurable tolerance (1min default) and keep behavior - Backward compatibility with existing
  merge strategies

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>


## v0.6.68 (2025-07-30)

### Bug Fixes

- Fix update_by_bars Series method for take in account volumes / trade counts
  ([`f26fc37`](https://github.com/xLydianSoftware/Qubx/commit/f26fc37c837521885dabb55f180c15fa8d0d2a27))

- Fix volumes restoring in simulator
  ([`d5aeea4`](https://github.com/xLydianSoftware/Qubx/commit/d5aeea4bec33ca3e9139737e99f44ca88ed31465))

- Hover width in mpl backend
  ([`91f9d29`](https://github.com/xLydianSoftware/Qubx/commit/91f9d297c90bdb671bd03bf85744cfe48adeac7c))

- Includes pivot's times
  ([`075be44`](https://github.com/xLydianSoftware/Qubx/commit/075be446ca54725486af49a37977774b00ef269d))

- Makes LookingGlass mpl to be more in line with plotly
  ([`801c104`](https://github.com/xLydianSoftware/Qubx/commit/801c104ba596c94c5075940a20ba0f94ff98742b))

- Standardize Bar constructor to use keyword arguments
  ([`086865c`](https://github.com/xLydianSoftware/Qubx/commit/086865c62e93f6db09d636e67132d7055fbb60d0))

- Fix Tardis data provider to use keyword arguments for volume/bought_volume parameters - Fix core
  helpers Bar constructor to use proper keyword arguments - Fix series.pyx Bar constructor usage to
  use correct parameter order and keywords - Fix data readers to use keyword arguments for volume
  parameter - Update test files to use consistent keyword argument pattern - Ensures all Bar
  constructors use explicit keyword arguments instead of positional arguments

This prevents data corruption issues from incorrect parameter ordering and makes the code more
  maintainable.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>

- Trick to return empty plot in mpl backend
  ([`50fd0e9`](https://github.com/xLydianSoftware/Qubx/commit/50fd0e9a43cb104a0bdfd8cbede460687d7426df))

### Chores

- Add chart_signals to TradingSessionResult
  ([`8fb1662`](https://github.com/xLydianSoftware/Qubx/commit/8fb16620b92887256b84a8eb1e07dbaf7038c777))

- Adds pivot's indexes
  ([`e7eae09`](https://github.com/xLydianSoftware/Qubx/commit/e7eae091b5a43bf56c90dade93fa3490b8659f1e))

- Cleanup some pyx code
  ([`ffee456`](https://github.com/xLydianSoftware/Qubx/commit/ffee4566608010d865f4af6ed5b8c62c09acd64f))

- Fix handle_start_stop in chart_signals
  ([`c551b9f`](https://github.com/xLydianSoftware/Qubx/commit/c551b9f035f49a6571f32229ca72b575863b90bf))

### Features

- Add vector version of pivots_highs_lows indicator
  ([`53bdc36`](https://github.com/xLydianSoftware/Qubx/commit/53bdc36618df7a6b6a08dc4f00ae32117a3a0ef9))


## v0.6.67 (2025-07-21)


## v0.6.66 (2025-07-21)


## v0.6.65 (2025-07-16)


## v0.6.64 (2025-07-14)

### Bug Fixes

- Ensure accurate leverage calculations by handling capital as a float or sum of values
  ([`2ac656a`](https://github.com/xLydianSoftware/Qubx/commit/2ac656a39213f94a8f176805506d40a6bf708b27))

- F-string double quotes in get_fundamental_data
  ([`ff74f14`](https://github.com/xLydianSoftware/Qubx/commit/ff74f14c5a1e1bc3462e5fac7c7ad79a9ca03564))

- Fix aux data extension in InMemoryCachedReader
  ([`4942804`](https://github.com/xLydianSoftware/Qubx/commit/4942804f77f338ad9543999ed4f22369489b57d4))

- Fix travel_in_time method in InMemoryCachedReader
  ([`2f2b7ec`](https://github.com/xLydianSoftware/Qubx/commit/2f2b7ec6ef03ab530e638947fd7b8503615d5913))

- Resolve peek_historical_data issue in SimulatedDataProvider for filtered funding payment
  subscriptions
  ([`b77a80f`](https://github.com/xLydianSoftware/Qubx/commit/b77a80f6a5cc212b82a5972ccffa96a3d5043a10))

### Features

- Add get_funding_payment method to QuestDBConnector for retrieving funding payments with optional
  filtering
  ([`8e77412`](https://github.com/xLydianSoftware/Qubx/commit/8e77412584934a709f93a5d4212ec3704c8f6e9b))

### Refactoring

- Streamline portfolio data dumping and enhance funding handling for SWAP instruments
  ([`c8b2df6`](https://github.com/xLydianSoftware/Qubx/commit/c8b2df68c239f5b3bb113a6a19bffa06cbd623d9))


## v0.6.63 (2025-07-02)

### Features

- Add OhlcDict.__getitem__ for list/tuple keys
  ([`f588f00`](https://github.com/xLydianSoftware/Qubx/commit/f588f005551014ec0126c92888c8758fc67101fc))


## v0.6.62 (2025-07-01)


## v0.6.61 (2025-06-30)


## v0.6.60 (2025-06-30)

### Bug Fixes

- Add targets to TradingSessionResults class
  ([`d66f393`](https://github.com/xLydianSoftware/Qubx/commit/d66f393e8f26b60a056831b769bd9123f9b3fce6))

- Adds log mesage
  ([`5cf7791`](https://github.com/xLydianSoftware/Qubx/commit/5cf7791d2fddd0c3d6f50a349e82c4aa8005ab1f))

- Fix atr tests
  ([`002824b`](https://github.com/xLydianSoftware/Qubx/commit/002824b7a1daa992fe1c3a1b43c2c09e9c32a51b))

- Fix initialization stage processing - skip initializing signal if active target is present
  ([`8a6ce93`](https://github.com/xLydianSoftware/Qubx/commit/8a6ce9356659a9784fb8a2491d841ad15c52a7eb))

- Fix jupyter runner
  ([`d325eef`](https://github.com/xLydianSoftware/Qubx/commit/d325eef2306f68aa561945f9e89064c10d45531f))

- Fix target position processing
  ([`fbe9b08`](https://github.com/xLydianSoftware/Qubx/commit/fbe9b089f1aff142a912bb73dbfd8d0c4a1e601f))

- Fix test runner
  ([`c0e95c0`](https://github.com/xLydianSoftware/Qubx/commit/c0e95c07cfcac3b16e852cf5761c390a61a0ade7))

- Fix test runner
  ([`73b0f47`](https://github.com/xLydianSoftware/Qubx/commit/73b0f47f129638a4809d7463d02af479519f9416))

- Fix test runner: weird behavior of pytest fixtures !
  ([`ef8cb73`](https://github.com/xLydianSoftware/Qubx/commit/ef8cb73ae9a3c85ec89dae945161b13a0e615943))

- Fix test typo
  ([`392a507`](https://github.com/xLydianSoftware/Qubx/commit/392a5078856568120a96d3348b1556b199ac88e2))

- Fix tests in progress
  ([`8e1303a`](https://github.com/xLydianSoftware/Qubx/commit/8e1303ae99f61da1cae4be1b8aea41f8cb1b095b))

- Fixed tracker routing
  ([`893adbc`](https://github.com/xLydianSoftware/Qubx/commit/893adbcdb4fa7e4b5a4a2866bfcc424296b76c55))

- Processing class didn't intialize collections
  ([`e05b6f2`](https://github.com/xLydianSoftware/Qubx/commit/e05b6f2c4ce2ba0923536007a342fb444cde4218))

- Remove unused tracker map
  ([`5f16d6c`](https://github.com/xLydianSoftware/Qubx/commit/5f16d6c86635f32b2776ff71f212ef25f2b99532))

### Chores

- Add todo in trade method
  ([`768c134`](https://github.com/xLydianSoftware/Qubx/commit/768c13477ec3af559dfc076cdb8ac685a5a29a7c))

- Restore test
  ([`8ea36e5`](https://github.com/xLydianSoftware/Qubx/commit/8ea36e5db481a0ead6dc2b2c8a7d746fb4eaa65a))

### Features

- Active position implementation. SignalRiskPositionTracker is added
  ([`bdc0ab2`](https://github.com/xLydianSoftware/Qubx/commit/bdc0ab2523c8b237d892747ac5ab6cd2f18dfa86))

### Refactoring

- Add tests for active position
  ([`f643665`](https://github.com/xLydianSoftware/Qubx/commit/f6436658e5bacf5390fad19a75358b77954fb282))

- Processing
  ([`be2e9fa`](https://github.com/xLydianSoftware/Qubx/commit/be2e9fa6a74d2d82630b7d2050cda911137fc373))

- Refactor test scenario
  ([`2450023`](https://github.com/xLydianSoftware/Qubx/commit/2450023c31eb542bd1f66548e6a01a74e1ef1465))

- Remove unnecessary checks
  ([`8fac0e0`](https://github.com/xLydianSoftware/Qubx/commit/8fac0e09ccd61cdd04b8b00bf8ab307a48665b26))


## v0.6.59 (2025-06-23)


## v0.6.58 (2025-06-20)


## v0.6.57 (2025-06-20)

### Bug Fixes

- Fix finalization for only subscribed instruments in live mode to prevent altering OHLC data for
  other instruments
  ([`9c32f5a`](https://github.com/xLydianSoftware/Qubx/commit/9c32f5a8998f0c79dfaf490b9a1120cffb74bf6e))


## v0.6.56 (2025-06-17)

### Bug Fixes

- Finalize OHLC data for all instruments in the universe on trigger event
  ([`7915236`](https://github.com/xLydianSoftware/Qubx/commit/79152366f59f4384a4ccb6a4363125b39293ab24))


## v0.6.55 (2025-06-17)

### Bug Fixes

- Test_queue_monitoring_with_channel increased sleep time
  ([`ca2659c`](https://github.com/xLydianSoftware/Qubx/commit/ca2659c85ecce61ca3988a189e2e3a1acf74f966))

### Chores

- Removed some old code
  ([`e361582`](https://github.com/xLydianSoftware/Qubx/commit/e361582ad28aec4a71291cbca36e73d831913c54))

### Refactoring

- Added as_of parameter to find_instruments
  ([`8270535`](https://github.com/xLydianSoftware/Qubx/commit/827053570f826a6c5f80c147394f476b144d304e))

- Added reload interval to instruments mongo lookup
  ([`e027d78`](https://github.com/xLydianSoftware/Qubx/commit/e027d78f2e912f4aced972928087e4992af4f47b))

- Refactors instruments lookup and many small changes
  ([`20e3eff`](https://github.com/xLydianSoftware/Qubx/commit/20e3effc74ce7502fa5d0bbc2f96baa47f5a6c8a))


## v0.6.54 (2025-06-08)

### Bug Fixes

- Raise an error if no variable parameters were found for a simulation
  ([`c8a011f`](https://github.com/xLydianSoftware/Qubx/commit/c8a011f1f97805a09d1bc759075fd0d6e8a8e0be))

### Features

- Add variation_plot method to BacktestsResultsManager
  ([`d0230e5`](https://github.com/xLydianSoftware/Qubx/commit/d0230e55f037b52bc87cb2dea8d2c55e0efcafa4))


## v0.6.53 (2025-06-06)


## v0.6.52 (2025-06-06)


## v0.6.51 (2025-06-06)


## v0.6.50 (2025-06-06)

### Bug Fixes

- Fix ohlc data retrieval bug when receiving only one record
  ([`52c082a`](https://github.com/xLydianSoftware/Qubx/commit/52c082abd09c329c6ad877aac0dae329df3b776d))

### Chores

- Loggers code cleanup
  ([`d3bb431`](https://github.com/xLydianSoftware/Qubx/commit/d3bb43144f67cd05b0d73296620ae1cf3a6529c5))


## v0.6.49 (2025-06-06)


## v0.6.48 (2025-06-05)

### Bug Fixes

- Imports fix
  ([`afcc83f`](https://github.com/xLydianSoftware/Qubx/commit/afcc83f79f01cc98c258c91bbb9cd7b38926139c))

- Ome add support for deferred execution reports for stop market orders
  ([`1028832`](https://github.com/xLydianSoftware/Qubx/commit/1028832cb215ba7936a8887c06f5eecda59b604b))

- Stop order price validation
  ([`8b47c17`](https://github.com/xLydianSoftware/Qubx/commit/8b47c1761a300b9d4bfca0b1d13c27c86fae609a))


## v0.6.47 (2025-05-29)


## v0.6.46 (2025-05-29)

### Bug Fixes

- Add subscription's warmup in on_init stage
  ([`76944d5`](https://github.com/xLydianSoftware/Qubx/commit/76944d5db2975b30e5d8d5b94c609663917876ab))


## v0.6.45 (2025-05-29)


## v0.6.44 (2025-05-28)


## v0.6.43 (2025-05-27)

### Bug Fixes

- Choppiness indicator : parameter with_raw_indicator
  ([`60ea92a`](https://github.com/xLydianSoftware/Qubx/commit/60ea92a57aff5b693a99dfb9c48066c6a523cff0))

### Features

- Add rolling_zscore to ta.py
  ([`05514d7`](https://github.com/xLydianSoftware/Qubx/commit/05514d7ea7e6b7c312bc4078d71ffebb8d4851b3))


## v0.6.42 (2025-05-15)


## v0.6.41 (2025-05-07)

### Bug Fixes

- Moved instrument updates info to market data processor and fix on_fit check
  ([`f557a6e`](https://github.com/xLydianSoftware/Qubx/commit/f557a6eee61e7bc9e0ef63609d9e0b5a868f6b86))

### Chores

- Add extend_trading_results() method
  ([`6398f03`](https://github.com/xLydianSoftware/Qubx/commit/6398f0395676a9f4d7f771d1fc2fbdd4f98086a3))

- Add import of extend_trading_results
  ([`54edb3a`](https://github.com/xLydianSoftware/Qubx/commit/54edb3a72c0bad544cc132811ff6303d94b33ab5))


## v0.6.40 (2025-04-28)

### Bug Fixes

- Fix run_paper.sh script generation
  ([`de0e810`](https://github.com/xLydianSoftware/Qubx/commit/de0e810413801a71d849b50376905779e9b1b5a9))


## v0.6.39 (2025-04-28)

### Features

- Add run_paper.sh script generation
  ([`5179af9`](https://github.com/xLydianSoftware/Qubx/commit/5179af9fca15c5e6d413cc4b015ba582574b7e28))


## v0.6.38 (2025-04-25)

### Bug Fixes

- Fix simulations data readers configs in line with live runner
  ([`9294f06`](https://github.com/xLydianSoftware/Qubx/commit/9294f06ec577fdde62448dc2030549b9ab599fc6))


## v0.6.37 (2025-04-24)

### Bug Fixes

- Backtest results loading and saving fix
  ([`881a6a2`](https://github.com/xLydianSoftware/Qubx/commit/881a6a22ade3a339d4ff3057ce2ae1bb08a313f6))

- Error handling for order creation and cancellation
  ([`e402e07`](https://github.com/xLydianSoftware/Qubx/commit/e402e07cabbffb4d855c074b31b07b69e9c275a6))

- Feature manager prevents multiple features with the same signatures
  ([`f5c3c89`](https://github.com/xLydianSoftware/Qubx/commit/f5c3c8942e785b3ff7d1ff18e01019f313b5ce2a))

### Chores

- Fix type hints
  ([`71a1081`](https://github.com/xLydianSoftware/Qubx/commit/71a1081b4d889447ec6e45a5154c111a03da01d4))

### Features

- Add is_warmup_in_progress and is_paper_trading properties to IStrategyContext
  ([`07fb48a`](https://github.com/xLydianSoftware/Qubx/commit/07fb48afad689060a96b8e3c120acf1ce5d19b53))

- Notify error if error level is medium or higher
  ([`243385e`](https://github.com/xLydianSoftware/Qubx/commit/243385e2c73eb46b62dff8b536e63e7e3eb0d1cc))

- Start and stop timestamps are processed properly in HFT data reader
  ([`9558043`](https://github.com/xLydianSoftware/Qubx/commit/955804322029a22cb4822ad6ff7fd3aa30d0cc70))


## v0.6.36 (2025-04-18)


## v0.6.35 (2025-04-17)


## v0.6.34 (2025-04-17)


## v0.6.33 (2025-04-17)


## v0.6.32 (2025-04-16)


## v0.6.31 (2025-04-16)

### Bug Fixes

- Ccxt binance trade with zero price
  ([`b2749b8`](https://github.com/xLydianSoftware/Qubx/commit/b2749b83b44bb49c46912b9b2f3f0fca810f66f9))


## v0.6.30 (2025-04-16)


## v0.6.29 (2025-04-11)

### Bug Fixes

- Check if data is newer than previous update
  ([`8699f1e`](https://github.com/xLydianSoftware/Qubx/commit/8699f1eed76790e151fa03895965af597f078671))

### Chores

- Bump version
  ([`c78ae67`](https://github.com/xLydianSoftware/Qubx/commit/c78ae670e6dc0353c9dc6207bd232a15eff45db3))


## v0.6.27 (2025-04-10)


## v0.6.26 (2025-04-07)


## v0.6.25 (2025-04-07)


## v0.6.23 (2025-03-29)

### Bug Fixes

- Typo fix
  ([`797af9d`](https://github.com/xLydianSoftware/Qubx/commit/797af9da400f1680c1d24cb2c51157abde0a413c))


## v0.6.21 (2025-03-28)


## v0.6.20 (2025-03-28)

### Bug Fixes

- Update version
  ([`effe949`](https://github.com/xLydianSoftware/Qubx/commit/effe9492a5d3874beee7e18533d4b5b2f50c84b7))

### Features

- Redesign of simulated exchange
  ([`ef640a5`](https://github.com/xLydianSoftware/Qubx/commit/ef640a582413bc8d7cc13b6aa7510cc3854a4c54))

### Refactoring

- Introduce interface for simulated exchange
  ([`7f3b276`](https://github.com/xLydianSoftware/Qubx/commit/7f3b2760ffc9332daac51003559dab122fe5926b))

- Rename OmeReport to SimulatedExecutionReport
  ([`422a18a`](https://github.com/xLydianSoftware/Qubx/commit/422a18aab6f6a76aa9051917bbebf9107e6461bd))


## v0.6.19 (2025-03-27)

### Bug Fixes

- Do not raise exception when canceling order that is not found in OME
  ([`2d1bd93`](https://github.com/xLydianSoftware/Qubx/commit/2d1bd93a24daf5553bf9d6fc5c21f7aed75199b5))


## v0.6.18 (2025-03-26)


## v0.6.17 (2025-03-25)


## v0.6.16 (2025-03-24)


## v0.6.15 (2025-03-24)

### Bug Fixes

- Account.py
  ([`cf29ce5`](https://github.com/xLydianSoftware/Qubx/commit/cf29ce58cf0e8bfec529f93d4482933affe4ed80))

- Small fix
  ([`4644649`](https://github.com/xLydianSoftware/Qubx/commit/46446495c8e3bd4d7db3d7680ff7bdbe0de6c464))

### Features

- Add traded range from time tests
  ([`33ea0e0`](https://github.com/xLydianSoftware/Qubx/commit/33ea0e072cf7defef2462ce78a7abdaab315a801))

### Refactoring

- Tests refactoring
  ([`e909a73`](https://github.com/xLydianSoftware/Qubx/commit/e909a73760892e0090d4ace0bb49abd13890526a))


## v0.6.14 (2025-03-20)


## v0.6.13 (2025-03-20)

### Bug Fixes

- Part fills processing correctly
  ([`8a2a916`](https://github.com/xLydianSoftware/Qubx/commit/8a2a91605ec3880cfd16a541fd0c04e046b529ee))


## v0.6.12 (2025-03-19)


## v0.6.11 (2025-03-19)


## v0.6.10 (2025-03-19)


## v0.6.9 (2025-03-18)


## v0.6.8 (2025-03-18)


## v0.6.7 (2025-03-18)


## v0.6.6 (2025-03-18)

### Bug Fixes

- Small fix in broker (wrong signature)
  ([`1101c89`](https://github.com/xLydianSoftware/Qubx/commit/1101c89c2c7b7c27a20dd8ecd4496c9d08df69bb))

- Small fixes in broker and riskctrl
  ([`99ef920`](https://github.com/xLydianSoftware/Qubx/commit/99ef9202fedd3ceb63a7bb28b0c4876bb1057326))


## v0.6.5 (2025-03-13)

### Bug Fixes

- - deploy / release issues resolved
  ([`eea5696`](https://github.com/xLydianSoftware/Qubx/commit/eea5696db8df4e0bf7bb55938f8f6c75cbfea7e1))

- Add quote method to ActiveInstrument
  ([`44064ec`](https://github.com/xLydianSoftware/Qubx/commit/44064ec3e05bd8eba4cb2d2712391c36c0a92e26))

- Fix on_universe_change interceptor in jupyter runner and some other issues
  ([`9c80dfb`](https://github.com/xLydianSoftware/Qubx/commit/9c80dfbc548259a3984730b1d34588b20be1e565))

- Fix tests for release and deploy commands
  ([`4a995cc`](https://github.com/xLydianSoftware/Qubx/commit/4a995ccb2671e204e9f7825c1d497d8774762063))

- Indicator small fix
  ([`c0bcf59`](https://github.com/xLydianSoftware/Qubx/commit/c0bcf593c102db77dc7a0abf4890924a4ae42c45))

- Print new universe
  ([`ae702e7`](https://github.com/xLydianSoftware/Qubx/commit/ae702e7b455d6ee005c68a908afda9dbf5bdc5e7))

### Features

- Fix release and deploy commands for support composite strategies and pyx files
  ([`4173707`](https://github.com/xLydianSoftware/Qubx/commit/41737073a0ec3cf74d830c64926a76bfb1344c92))


## v0.6.4 (2025-03-07)

### Bug Fixes

- Broker and client executed at correct stop levels now
  ([`fe1e018`](https://github.com/xLydianSoftware/Qubx/commit/fe1e018d5f039ae731d8e0689669d65352b30772))

- Fix signals log symbol column name
  ([`07ce906`](https://github.com/xLydianSoftware/Qubx/commit/07ce906ee8a1cdec7bcc96270a3f08115368b39e))

- Ome with exact fill at signal price for market orders
  ([`db624f8`](https://github.com/xLydianSoftware/Qubx/commit/db624f85af81f7ac54473168e4a0018fc3d3e3d2))


## v0.6.3 (2025-03-07)

### Bug Fixes

- Drop unnecessary quote -> mid_price -> restored quote conversion
  ([`0af48d9`](https://github.com/xLydianSoftware/Qubx/commit/0af48d94c060fca75cf452e7b641aac8e9441547))

- Fix test
  ([`d57d6d1`](https://github.com/xLydianSoftware/Qubx/commit/d57d6d1042ccd590602e19a290526be52daa74ca))

- Fixes utils imports
  ([`1282af0`](https://github.com/xLydianSoftware/Qubx/commit/1282af01222b6e5b269443b25069b07721673a26))

- Hft data processing in OME
  ([`dcec98f`](https://github.com/xLydianSoftware/Qubx/commit/dcec98f36dd3a25de05f87df6d5108506753096a))

- Restored import of loader
  ([`923ec27`](https://github.com/xLydianSoftware/Qubx/commit/923ec2735941b0201a269cea002f83d8e5e5463b))

- Small fix
  ([`8fe18f6`](https://github.com/xLydianSoftware/Qubx/commit/8fe18f61b4699c32aa37ca9ea875bb632ab03d5c))

- Small fix in notebook
  ([`b69bf91`](https://github.com/xLydianSoftware/Qubx/commit/b69bf91dd6ed46d994fa7db514fdd18c7ffcf0a1))

- Small fixe
  ([`c42ca1f`](https://github.com/xLydianSoftware/Qubx/commit/c42ca1f27c8be6d318b13b607d1448985c21aee7))

- Support composite strategies in runner
  ([`0d6e192`](https://github.com/xLydianSoftware/Qubx/commit/0d6e192d95c83e2002b19bebabbf86998e69ad68))

### Features

- Add special case for execution price at signal price for market orders
  ([`6abf91c`](https://github.com/xLydianSoftware/Qubx/commit/6abf91c958490d854fdba7bc79d35e35f10a2c99))

- Adds test for real quotes
  ([`793f05e`](https://github.com/xLydianSoftware/Qubx/commit/793f05e70f0ac80a435db0499bde798f4d060702))

- Handy jupyter tools
  ([`dab691a`](https://github.com/xLydianSoftware/Qubx/commit/dab691a8453551c9fc00715a2b3dbbdf3d39c63d))

- Jupyter tools
  ([`1ee0dc2`](https://github.com/xLydianSoftware/Qubx/commit/1ee0dc20242767d4c6b261ed1078924ed992d544))

- Version update to 0.6.2
  ([`d6805ce`](https://github.com/xLydianSoftware/Qubx/commit/d6805ce4d05074eed6e49b0d2622a4d08c16ed16))


## v0.6.1 (2025-02-27)


## v0.6.0 (2025-02-26)


## v0.5.10 (2025-02-26)


## v0.5.8 (2025-02-24)


## v0.5.7 (2025-02-24)

### Bug Fixes

- Add candles reading interface to csv reader, so csv reader can be used in loader
  ([`4f5dfef`](https://github.com/xLydianSoftware/Qubx/commit/4f5dfef73c37dc40fa9cbf29bd4988d231c97e51))

- Add choppy identification method
  ([`a750e10`](https://github.com/xLydianSoftware/Qubx/commit/a750e100ef42cdabe17ae6642eda5794dc42d1fb))

- Add flag to recalculate indicators on closed bar
  ([`afabc64`](https://github.com/xLydianSoftware/Qubx/commit/afabc64506989d62058e8b984a234ddd21e8d6a9))

- Add processing of sort_by=None parameter to backtest management
  ([`25eaf3a`](https://github.com/xLydianSoftware/Qubx/commit/25eaf3a4c9d8c98f464c1d4290ce98f57cb386cf))

- Add support for signals in signals viewer
  ([`1a8ad45`](https://github.com/xLydianSoftware/Qubx/commit/1a8ad45997625797351e944babe5ed4b71cefb1a))

- Add tearsheet method to TradingSessionResult
  ([`6f9aefd`](https://github.com/xLydianSoftware/Qubx/commit/6f9aefd4fdffed7088693c979020900924ebcde0))

- Add volatility calculation and fix choppiness index calculation
  ([`335d62a`](https://github.com/xLydianSoftware/Qubx/commit/335d62a88f30c80e7078dc44722c995b20901fec))

- Additional test is added
  ([`fcc21e7`](https://github.com/xLydianSoftware/Qubx/commit/fcc21e7ef6d2c04422acafb5273f0171614d8f98))

- Adds abitlity to reserve desired amount of assets from trading
  ([`43d171e`](https://github.com/xLydianSoftware/Qubx/commit/43d171e733cc2f84d9819959f8670ffbeeb21b4b))

- Adds all spot symbols for Binance and Bitfinex
  ([`3b18464`](https://github.com/xLydianSoftware/Qubx/commit/3b1846466b51563b2926d428b5c5df65fa330b71))

- Adds check for preferred data type when probing data from reader
  ([`50a53f6`](https://github.com/xLydianSoftware/Qubx/commit/50a53f619280f3dce4aeb348a90c605717a81bd6))

- Adds client order id
  ([`01f0a76`](https://github.com/xLydianSoftware/Qubx/commit/01f0a769d87810f12e8c2a9365fd3e44c5dd4f50))

- Adds comments
  ([`8980808`](https://github.com/xLydianSoftware/Qubx/commit/89808081f95386803c4f060f31c7e83a636b743f))

- Adds control on attemptin to update series by past data
  ([`ce4b4f1`](https://github.com/xLydianSoftware/Qubx/commit/ce4b4f144e493e96696ee5399ac4264b618b213a))

- Adds debug level in simulate method
  ([`dabf472`](https://github.com/xLydianSoftware/Qubx/commit/dabf47264905e141819eab4d10361bdd0606d023))

- Adds decorator on jma method
  ([`1e3ddde`](https://github.com/xLydianSoftware/Qubx/commit/1e3ddde444072aa6c771b656e2ad9fb8f54dd82d))

- Adds docs
  ([`6b9a5bd`](https://github.com/xLydianSoftware/Qubx/commit/6b9a5bd8f98683911dfe9d46fdb2db9c9014f0b1))

- Adds exchange into paper broker
  ([`06a5172`](https://github.com/xLydianSoftware/Qubx/commit/06a51729bcfd68a870cd95544d082e1fa6ca03b5))

- Adds get_subscriptions_for_instrument method and it's test
  ([`30cf298`](https://github.com/xLydianSoftware/Qubx/commit/30cf2982fd46b59fb26cc2697801055761119119))

- Adds in memory data reader
  ([`bbaad09`](https://github.com/xLydianSoftware/Qubx/commit/bbaad09757b9031c7782a3ccdf948b873d6dd209))

- Adds notification on order's canceling and execution
  ([`20d7b2d`](https://github.com/xLydianSoftware/Qubx/commit/20d7b2ddd8e2ef38deaa7c23bbe255cc6830e5c9))

- Adds pyarrow dep in poetry
  ([`6f64cb5`](https://github.com/xLydianSoftware/Qubx/commit/6f64cb5ae6768d6d777720fc8ffbbb34212ed72a))

- Adds readme
  ([`dae8a4c`](https://github.com/xLydianSoftware/Qubx/commit/dae8a4cb4f9ddf0024dbe4e681b3218190752962))

- Adds rounding for limit order price
  ([`19dd4b0`](https://github.com/xLydianSoftware/Qubx/commit/19dd4b0e4ec0c5ba8f349e67af8b0a3719b51321))

- Adds small refactoring for Kama
  ([`2601b4c`](https://github.com/xLydianSoftware/Qubx/commit/2601b4cb18570c8450d5bfac5bb18bdfd32ede56))

- Adds some fields / methods interfaces into Bar class
  ([`818a807`](https://github.com/xLydianSoftware/Qubx/commit/818a807759c24a572d7dc9557d7166c34c8cfe57))

- Adds test for swings
  ([`92e8af9`](https://github.com/xLydianSoftware/Qubx/commit/92e8af9041f406059eacc63fc6274b3d6ad93441))

- Adds test on case for > 2 streams in slicer
  ([`9be06d0`](https://github.com/xLydianSoftware/Qubx/commit/9be06d018448033603b6de3a51d413d9321e86ca))

- Adds tests
  ([`e668a3c`](https://github.com/xLydianSoftware/Qubx/commit/e668a3ca6a231c6c517ae50ddd04f6ca85cf6ab8))

- Adds tests for ticks simulated data
  ([`64dd1d7`](https://github.com/xLydianSoftware/Qubx/commit/64dd1d747fdd8142abdbf2c5b0c1c4d36554b5e0))

- Allow scale position size by signal's value (flag in FixedRiskSizer)
  ([`0d93766`](https://github.com/xLydianSoftware/Qubx/commit/0d9376698bd635a50fb125755fe1734bf6f92199))

- Backtest managment
  ([`7d8b14d`](https://github.com/xLydianSoftware/Qubx/commit/7d8b14d3a12601adb67af1ab2b30a7586e075ca4))

- Balance calculations and part fills processing
  ([`4a8d846`](https://github.com/xLydianSoftware/Qubx/commit/4a8d8462d285f17d69137cab16def9f5b16ed8a4))

- Balance logging save fix
  ([`d9f1cea`](https://github.com/xLydianSoftware/Qubx/commit/d9f1ceaca1230be18720182799f6c6f665a22ea6))

- Bar trigger to be processed only once
  ([`4cb073a`](https://github.com/xLydianSoftware/Qubx/commit/4cb073a4550d3f9e18dc1ab25950c6bcaed7d52e))

- Bookmarks file
  ([`8880fe6`](https://github.com/xLydianSoftware/Qubx/commit/8880fe6ea1be712615e00478f7c03a0ba6a7dfd1))

- Changes info to debug messages in PortfolioRebalancerTracker
  ([`63a4aeb`](https://github.com/xLydianSoftware/Qubx/commit/63a4aeb2a544a2f1c515222d0ad752efc6974dd9))

- Choppyness index indicator fixed type
  ([`dc05d7e`](https://github.com/xLydianSoftware/Qubx/commit/dc05d7e4aa481d09f80495a76afcff6d7c6e62a4))

- Client side risk controller fix
  ([`aa0e1b5`](https://github.com/xLydianSoftware/Qubx/commit/aa0e1b53f39bae08aff66279186f4830c94bd46b))

- Commented some init code
  ([`4330cb7`](https://github.com/xLydianSoftware/Qubx/commit/4330cb7b413e72003773ea09d69718b107cc2775))

- Compare indicator fixed
  ([`13edccc`](https://github.com/xLydianSoftware/Qubx/commit/13edccc50e05418b4fa0dcd2eeaaa59d3ad6281d))

- Correct sentinel in simulator
  ([`0d3e24f`](https://github.com/xLydianSoftware/Qubx/commit/0d3e24f8f2a137a6782dda904a4e0f8078079c04))

- Crone simulation is fixed
  ([`f4a9114`](https://github.com/xLydianSoftware/Qubx/commit/f4a91147ae6dbf2decf28d95ffe52278c5312537))

- Csv candles reading error
  ([`754df62`](https://github.com/xLydianSoftware/Qubx/commit/754df62e486eaceb741bd3f87be9c41299f19b72))

- Data sniffer now knows how to use get_ranges from the reader
  ([`12d6c37`](https://github.com/xLydianSoftware/Qubx/commit/12d6c37e2407258cf589c9bc1e43ad11fbf9d3e2))

- Extracting strategy parameters before simualtion - strategy may alter intial parameters. Also it
  scans startegy mixings to get all parameters.
  ([`9594acb`](https://github.com/xLydianSoftware/Qubx/commit/9594acbdacc2024a18b232cadeac5ce894f48b07))

- F
  ([`e7cc972`](https://github.com/xLydianSoftware/Qubx/commit/e7cc972c7f09bd8646f006ca036ecfab13ce64a2))

- Fix color of sell orders
  ([`ed36341`](https://github.com/xLydianSoftware/Qubx/commit/ed36341ad073d99043d770a9a7257c6627d5c2c2))

- Fix description in list command
  ([`5fa555c`](https://github.com/xLydianSoftware/Qubx/commit/5fa555cf4ee9235612b83014b0bad3345aa508fc))

- Fix fixed risk sizer entry price detection
  ([`70411db`](https://github.com/xLydianSoftware/Qubx/commit/70411dbc06e3b671847ab5bee439d32629475d30))

- Fix pretty print for list method
  ([`e43b7cf`](https://github.com/xLydianSoftware/Qubx/commit/e43b7cff6e725363a72311d922abe5c61744c373))

- Fix queue tests
  ([`dc0e14a`](https://github.com/xLydianSoftware/Qubx/commit/dc0e14af35f6b6f6ed636b3115444e0c675a3f1f))

- Fix remove_instruments logic
  ([`52aba3c`](https://github.com/xLydianSoftware/Qubx/commit/52aba3ca9a433e3a839820a1bfa1ea1af9347e58))

- Fix risk manager orders cancelling processing
  ([`e98044f`](https://github.com/xLydianSoftware/Qubx/commit/e98044f56fcc51a8fd9826b7323f8c1055b58bd1))

- Fix typo
  ([`6c62945`](https://github.com/xLydianSoftware/Qubx/commit/6c6294581d0867bfd41075df2a44beb36b22e37b))

- Fixed portfolio weighter
  ([`ea8dbff`](https://github.com/xLydianSoftware/Qubx/commit/ea8dbff5bfe6cc900a8bd1fff82cab084f3e06bf))

- Fixed risk sizer
  ([`4212e5e`](https://github.com/xLydianSoftware/Qubx/commit/4212e5e183f8093784f41955c410cbf591be464e))

- Fixed states in stoptake basic class
  ([`2573a78`](https://github.com/xLydianSoftware/Qubx/commit/2573a78fe15698fd1a9425dc75f3ea9779e01ffd))

- Fixes accuracy in position class
  ([`66ea699`](https://github.com/xLydianSoftware/Qubx/commit/66ea69921d29d486fa86c967fc554730fd81ea42))

- Fixes broker mode in risk controller
  ([`35eae84`](https://github.com/xLydianSoftware/Qubx/commit/35eae8498089b5503e557d6ee812166ed6f2df0d))

- Fixes BrokerSideRiskController stop order issue
  ([`0d66b54`](https://github.com/xLydianSoftware/Qubx/commit/0d66b5415a6d3b55c5ac2dac5d126c55beed96f2))

- Fixes chunked reading in InMemory.... readers
  ([`93e3d7f`](https://github.com/xLydianSoftware/Qubx/commit/93e3d7fccb570096a5bd80c413f648efc06c773c))

- Fixes closing positions by zero signal
  ([`46ef696`](https://github.com/xLydianSoftware/Qubx/commit/46ef696a4c7fc1ea8aa82a103747b52ef42d2732))

- Fixes composite trackers issue
  ([`adbd2f1`](https://github.com/xLydianSoftware/Qubx/commit/adbd2f13f631a9380711a766f27edbdd911a5fd0))

- Fixes datareader test
  ([`a99083c`](https://github.com/xLydianSoftware/Qubx/commit/a99083c7f44783f3c7e42f4fa34bff5000bc10a9))

- Fixes datareader, adds correct implementation of position average price
  ([`0ebc73e`](https://github.com/xLydianSoftware/Qubx/commit/0ebc73e82ddbd8551b39348fafd6a46a79fe04de))

- Fixes default scheme for fees data
  ([`0149019`](https://github.com/xLydianSoftware/Qubx/commit/0149019e5f48a9a5001bed97be96692cfbe2ca47))

- Fixes equal intervals splitting function
  ([`cdbcf37`](https://github.com/xLydianSoftware/Qubx/commit/cdbcf3723d73b1ef4ef90c79a2620fe973b19ab4))

- Fixes exchange name for paper trading
  ([`731db68`](https://github.com/xLydianSoftware/Qubx/commit/731db68073654686c49efe6cf31015590a59770e))

- Fixes fixed risk calculations
  ([`faf1054`](https://github.com/xLydianSoftware/Qubx/commit/faf1054dfc33b93c9ef1a2c06c2ff485d494a85d))

- Fixes issue with swings and psar series renamed
  ([`1ec95b8`](https://github.com/xLydianSoftware/Qubx/commit/1ec95b8da4bea098449078a6446685006f791322))

- Fixes justfile
  ([`b7a35bc`](https://github.com/xLydianSoftware/Qubx/commit/b7a35bc01ba3f6e358c4d03f49e3c1eff1c4a224))

- Fixes Kama indcator and adds unit tests
  ([`5e05c53`](https://github.com/xLydianSoftware/Qubx/commit/5e05c53a1dc775cda3c4cedb2ddb90fb39ae9731))

- Fixes MarketDataProvider::ohlc logic
  ([`38b4bfa`](https://github.com/xLydianSoftware/Qubx/commit/38b4bfa180a5f02586d36be05160facc58de9cc5))

- Fixes math imports
  ([`2304589`](https://github.com/xLydianSoftware/Qubx/commit/230458910008b1228f6ea4aef0097df221f81526))

- Fixes non strategy config repr in TradingSessionResult
  ([`75c6fb8`](https://github.com/xLydianSoftware/Qubx/commit/75c6fb869728668f6bc2273ebf0f965bf28301d0))

- Fixes OhlcDict processing empty dataframes and removed annoying error message
  ([`0b651e8`](https://github.com/xLydianSoftware/Qubx/commit/0b651e84364f07af16d1e618a98e6b9b56a0de6e))

- Fixes pewma on streaming data
  ([`c809f47`](https://github.com/xLydianSoftware/Qubx/commit/c809f478c59babf50618cdd5d586785aa3c0357b))

- Fixes pewma_outliers_detector indicator in Qubx
  ([`034877b`](https://github.com/xLydianSoftware/Qubx/commit/034877b84ae963a027aa77892e913077bce01558))

- Fixes PnL calc in positions
  ([`6eb5676`](https://github.com/xLydianSoftware/Qubx/commit/6eb567609887c8d3c77be77332334193580c6b3e))

- Fixes progress bar and add dark ipywidgets background workaround trick
  ([`7712fbd`](https://github.com/xLydianSoftware/Qubx/commit/7712fbdf97f2571e350b4095656f34518faccc8a))

- Fixes riskcontrol case when limit take is triggered instantly
  ([`d072da5`](https://github.com/xLydianSoftware/Qubx/commit/d072da5438514d998a64a1075bd8c70e2815ba77))

- Fixes scheduler test
  ([`e371049`](https://github.com/xLydianSoftware/Qubx/commit/e371049d9162aa8cd43b66e2b98cffc92a0ed12f))

- Fixes search by naming in Series or DataFrame generated signals
  ([`e9a97d7`](https://github.com/xLydianSoftware/Qubx/commit/e9a97d702ad37209adc570f9b4e9ce1219bb03ad))

- Fixes short execution marker color
  ([`abc7847`](https://github.com/xLydianSoftware/Qubx/commit/abc78470d7cdfec2c6e0c80c225c85621d3b5f6a))

- Fixes simulator tests and removed quotes
  ([`7b6f990`](https://github.com/xLydianSoftware/Qubx/commit/7b6f990f446d5c896e840297c200767d1adf063a))

- Fixes sizer test
  ([`b8e015d`](https://github.com/xLydianSoftware/Qubx/commit/b8e015dd5ffecaa96226ca67cbde22e801461a64))

- Fixes some tests
  ([`ad6d1c8`](https://github.com/xLydianSoftware/Qubx/commit/ad6d1c8ceaa3b518fbad8c8b3b67457ae7b224e8))

- Fixes stupid typo
  ([`19d5082`](https://github.com/xLydianSoftware/Qubx/commit/19d5082898ada79e67e570a2503d02c0ac3eaa39))

- Fixes swings up/down trends names
  ([`7a104b0`](https://github.com/xLydianSoftware/Qubx/commit/7a104b0273499b54cbfa45fb173ea61907c6050f))

- Fixes tables naming for QuestDB connector
  ([`f850ae3`](https://github.com/xLydianSoftware/Qubx/commit/f850ae39f10d521d40196b8ef3c83673169baa78))

- Fixes test
  ([`c668e0d`](https://github.com/xLydianSoftware/Qubx/commit/c668e0d356e70f8de008be872fc9272e68aef790))

- Fixes test
  ([`ec192c6`](https://github.com/xLydianSoftware/Qubx/commit/ec192c6dc44b2dd718d79481bf0fecbb544bccf5))

- Fixes test
  ([`a1ac972`](https://github.com/xLydianSoftware/Qubx/commit/a1ac97242b80ced17980ce2bc4aa5c906c5eca81))

- Fixes test
  ([`7f890f9`](https://github.com/xLydianSoftware/Qubx/commit/7f890f996a3be25e1747d4f2a90adbcdc63e6e74))

- Fixes test
  ([`ea8a0ab`](https://github.com/xLydianSoftware/Qubx/commit/ea8a0ab742bc6c6a683389cc5b1984074f3fc583))

- Fixes test
  ([`df998ec`](https://github.com/xLydianSoftware/Qubx/commit/df998ec68f5c6077443eb028f96ca571dcceb385))

- Fixes test for loader
  ([`01ec273`](https://github.com/xLydianSoftware/Qubx/commit/01ec27347ba16b03b097dbfd10d65cd782c2d939))

- Fixes tests
  ([`1496cc5`](https://github.com/xLydianSoftware/Qubx/commit/1496cc5c7eb4d6251817493f20e2efdb143fb50d))

- Generated signals series might have non str names
  ([`96c7a8e`](https://github.com/xLydianSoftware/Qubx/commit/96c7a8e3ba7d9b727edfe04f02b3596a85b0bddc))

- Handling timeseries in TimeGuardedWrapper
  ([`a7b5286`](https://github.com/xLydianSoftware/Qubx/commit/a7b528667c9fc1c8ff1d81b78fd0dea0dd177b25))

- Handling when historical ohlcv is empty
  ([`b4e632b`](https://github.com/xLydianSoftware/Qubx/commit/b4e632b92c85b520335d1fd37c130ccbc10f3fc2))

- Hover
  ([`50ff613`](https://github.com/xLydianSoftware/Qubx/commit/50ff613ec2d670586838657ecfb618b6d6decd46))

- Hurst function docstring
  ([`3ff5d14`](https://github.com/xLydianSoftware/Qubx/commit/3ff5d147503186dd33b1f397318d62b2676e5b96))

- Increase version
  ([`01091a6`](https://github.com/xLydianSoftware/Qubx/commit/01091a6f352adc9ec1df53e63e9d2263f76fede9))

- Locator logic is fixed and added ability to process complex indicators
  ([`1b83b1e`](https://github.com/xLydianSoftware/Qubx/commit/1b83b1e97e77efbc985586b8f7db3e7790e0ab9d))

- Loggers fix (adds exchange in execution report)
  ([`e661229`](https://github.com/xLydianSoftware/Qubx/commit/e6612292c2a5928c3a517335af98ea5f82585911))

- Make dark plotly looks like mpl
  ([`94d75a7`](https://github.com/xLydianSoftware/Qubx/commit/94d75a7c2b80715286ad3bbb7a8e6968e7fea805))

- Makes tqdm progress smaller
  ([`8b1bbc3`](https://github.com/xLydianSoftware/Qubx/commit/8b1bbc3e72a77588bfbc266f17fcb3c31e679db2))

- Merged incoming
  ([`e50ffb4`](https://github.com/xLydianSoftware/Qubx/commit/e50ffb4d50c2e8cf9c93468e7ce7d28aca3104f7))

- More explainable warning
  ([`0bc0288`](https://github.com/xLydianSoftware/Qubx/commit/0bc0288b5de98d784c71d2141028f7514b0fa5be))

- Multiple small fixes and additional helpers
  ([`8c4acad`](https://github.com/xLydianSoftware/Qubx/commit/8c4acaddc9bc3fb2a1d11a7a9f3567399fd52594))

- Negative balance fix
  ([`059b71c`](https://github.com/xLydianSoftware/Qubx/commit/059b71c2bafce8d5737351c3e61ece32703d1012))

- New logo test
  ([`e8bee00`](https://github.com/xLydianSoftware/Qubx/commit/e8bee0032f70b5b11be4b606fed9e97b1611ca84))

- None title is not displayed in LG. chart_signals can use plugins
  ([`214e47d`](https://github.com/xLydianSoftware/Qubx/commit/214e47d2d9a8df83255bfd313a2a8a2fa6478428))

- Notebook
  ([`19649c6`](https://github.com/xLydianSoftware/Qubx/commit/19649c6e0cb318f6027162d063e531cff1a65c32))

- Now indicators are being updated in correct order
  ([`6cd6aa4`](https://github.com/xLydianSoftware/Qubx/commit/6cd6aa4b772b5acaa860bab41afd5d9e3ddef766))

- Ohlcdict key requirements led to simulation failure on instrument names started with number
  ([`258641a`](https://github.com/xLydianSoftware/Qubx/commit/258641a6f6f4361d127361f860b9a37009d45fd6))

- Position adjuster
  ([`ceff24c`](https://github.com/xLydianSoftware/Qubx/commit/ceff24ccfa0d321d3634a788add0c769901db111))

- Position test fixed
  ([`af6fb1e`](https://github.com/xLydianSoftware/Qubx/commit/af6fb1eed378232289efcbb4f49dfa65a09fbe53))

- Refactoring
  ([`5b2cf73`](https://github.com/xLydianSoftware/Qubx/commit/5b2cf73fc1ae0c305a75d80d2bb9f93e33a4f32d))

- Refactors
  ([`e6f9bf3`](https://github.com/xLydianSoftware/Qubx/commit/e6f9bf31f16a7346291d854cceb627f14aaa5760))

- Refactors code a bit
  ([`b56b584`](https://github.com/xLydianSoftware/Qubx/commit/b56b584bc39c1983ccf91564e4abe45679cb60e5))

- Refactors of update method with BatchEvent support
  ([`9f28647`](https://github.com/xLydianSoftware/Qubx/commit/9f28647979e263884e441ba121d66e00c41678e9))

- Refactors simulator and adds applying default warmups if it's not specified in strategy
  ([`52c5ee7`](https://github.com/xLydianSoftware/Qubx/commit/52c5ee77be62ef957293c2fb76172c918e70fd02))

- Refactors trackers/sizers methods arguments and fixes Portfolio balancer
  ([`2a6fed2`](https://github.com/xLydianSoftware/Qubx/commit/2a6fed2153dade637fd6e6488fd709367b1ccc62))

- Refactors typing
  ([`9fb0a3b`](https://github.com/xLydianSoftware/Qubx/commit/9fb0a3bf61d11b8b490f77e4785daf099e8fe704))

- Reformats file
  ([`e27d06e`](https://github.com/xLydianSoftware/Qubx/commit/e27d06ef0b6088dab94c6ec14ec13185603c0754))

- Removed bookmarks file
  ([`7f7be27`](https://github.com/xLydianSoftware/Qubx/commit/7f7be27e3f75a71be94e456fd536aaf5a36b15da))

- Removed file
  ([`3608d91`](https://github.com/xLydianSoftware/Qubx/commit/3608d91ecf54e73bc3c77b83ba04cff5d9b7be5c))

- Removed old load_data method
  ([`cb796cf`](https://github.com/xLydianSoftware/Qubx/commit/cb796cf8b024d218f7b46870cf993f8223f6d294))

- Removes _get_ohlc_data method from IExchangeServiceProvider
  ([`0f5d2f9`](https://github.com/xLydianSoftware/Qubx/commit/0f5d2f9e84019485fabb955d603a02e3fbdf36b3))

- Removes BatchEvent from processing
  ([`bac6186`](https://github.com/xLydianSoftware/Qubx/commit/bac6186a6cf08cc0c3dc44f51c60333fa0d4cfc5))

- Removes cell
  ([`df93fde`](https://github.com/xLydianSoftware/Qubx/commit/df93fdee2f27932924edaf44c021a6a677074432))

- Removes duplicated method for rounding average price
  ([`4ba3466`](https://github.com/xLydianSoftware/Qubx/commit/4ba3466537f051d86c8918b6c280eba34160600a))

- Removes leverage parameter from simulation method
  ([`feceb63`](https://github.com/xLydianSoftware/Qubx/commit/feceb63bf1aece710aeb71f8498d57a17b3d0d03))

- Removes logs
  ([`27197fe`](https://github.com/xLydianSoftware/Qubx/commit/27197fe170ed85b9bebd0ff4e7bc4174b264d785))

- Removes unnecessary files
  ([`a773343`](https://github.com/xLydianSoftware/Qubx/commit/a773343d747e0e57f206529ffa0dac8f6be02a7f))

- Removes unnecessary notebook
  ([`10be238`](https://github.com/xLydianSoftware/Qubx/commit/10be238389ef42c4bfafa0231ffc9ce11e10b24a))

- Removes unnecessary rebuilding
  ([`e715562`](https://github.com/xLydianSoftware/Qubx/commit/e715562fe1e2d06c199fef05998805bb4187f08e))

- Removes unnecessary Self import
  ([`08e1d06`](https://github.com/xLydianSoftware/Qubx/commit/08e1d0616433e7dd64a1fd835efa3eb8629d5cb1))

- Renamed methods and adds additional checks
  ([`0a9e968`](https://github.com/xLydianSoftware/Qubx/commit/0a9e9683952106a640df2a955732fa39a4b283ed))

- Requirements and version
  ([`6b82273`](https://github.com/xLydianSoftware/Qubx/commit/6b822731aa10672c100e0381744276fddb049435))

- Reserves and position restoring
  ([`a2e2690`](https://github.com/xLydianSoftware/Qubx/commit/a2e2690d408e90800f55257db7ff0cce6083c8b5))

- Set Qubx default log level to WARNING
  ([`b8edf23`](https://github.com/xLydianSoftware/Qubx/commit/b8edf235a6146d7bf5ca5bb84ecd06adcdb6cdc8))

- Show_portfolio flag
  ([`a54513a`](https://github.com/xLydianSoftware/Qubx/commit/a54513a17ebba898b22dc37eec9c64e085840e36))

- Simplest Gathering can handle entry at specified price by limit or stop orders
  ([`7d5efe2`](https://github.com/xLydianSoftware/Qubx/commit/7d5efe2868d56bb64602e016de5f648bbcb57203))

- Simulator stops backtesting if strategy fails more than N times a row
  ([`ba2c9fa`](https://github.com/xLydianSoftware/Qubx/commit/ba2c9fa8a2aaf6cc5f23de5dfc2da81fc20fbfda))

- Simulator with aux_data
  ([`a4b3445`](https://github.com/xLydianSoftware/Qubx/commit/a4b34456939070ca3551ba478902808a0b83621b))

- Small cosmetic changes
  ([`eb36897`](https://github.com/xLydianSoftware/Qubx/commit/eb36897cebabcd646f1203f44818c15de1303414))

- Small fix
  ([`9ec7cbb`](https://github.com/xLydianSoftware/Qubx/commit/9ec7cbb1c670b9a8af2c66b82dbcd7799a25e3fb))

- Small fix
  ([`658e96e`](https://github.com/xLydianSoftware/Qubx/commit/658e96eec3c7b2eab17eb34b8aa0d5f0c3ab6687))

- Small fix in handler
  ([`06b15fe`](https://github.com/xLydianSoftware/Qubx/commit/06b15fe77ae753ce2c9b6f06828f26bbaa491942))

- Small fixe for py < 3.12
  ([`52f8792`](https://github.com/xLydianSoftware/Qubx/commit/52f8792935406d3dd30a8aad57150c87e5f6edc1))

- Small fixes
  ([`edccfce`](https://github.com/xLydianSoftware/Qubx/commit/edccfce81ecbac0083eacd6ad6afb03db4aa8448))

- Small fixes
  ([`caa19de`](https://github.com/xLydianSoftware/Qubx/commit/caa19dea7a422d38a91f72c3b68ceb07b9611ccf))

- Small fixes in tests
  ([`bcd67cf`](https://github.com/xLydianSoftware/Qubx/commit/bcd67cf327cb0c8b0ec53053a5076b51591da90d))

- Small names fixes
  ([`f8351b6`](https://github.com/xLydianSoftware/Qubx/commit/f8351b620a770898f0edcde905a5381e5bc57fde))

- Small one
  ([`7830b44`](https://github.com/xLydianSoftware/Qubx/commit/7830b441550a4014b23b4122f1582ae6dd9a33a7))

- Small one
  ([`cb33fdf`](https://github.com/xLydianSoftware/Qubx/commit/cb33fdf21d2a490c7d6f284cadcfe1d0d05f7117))

- Small refactoring
  ([`8e01c35`](https://github.com/xLydianSoftware/Qubx/commit/8e01c35f177a6686dbaac17581d985a7c8d8a4d0))

- Small tests
  ([`a8d9dce`](https://github.com/xLydianSoftware/Qubx/commit/a8d9dce4343f49416edd1112838944786fc849b8))

- Small typo in definitions
  ([`ecba69c`](https://github.com/xLydianSoftware/Qubx/commit/ecba69c58d4f9f0b3397cdc7d68e3f000483b3b0))

- Some experiments
  ([`9527824`](https://github.com/xLydianSoftware/Qubx/commit/9527824a40f02bf735eadc333f1a6456d6a01710))

- Some fixes of signals
  ([`8b11385`](https://github.com/xLydianSoftware/Qubx/commit/8b11385e757354f8c324131c9452a4dad447cbae))

- Strategy simulation doesn't stop after max number of failures in a row
  ([`b58e7c4`](https://github.com/xLydianSoftware/Qubx/commit/b58e7c4102476c41a1f89509645f359f8b5e0a0f))

- Tearsheet's title overflow
  ([`e497713`](https://github.com/xLydianSoftware/Qubx/commit/e497713e0f33fa9a3d2c3de94d683705dec96790))

- Temp fix of ccxt_integration_test
  ([`762c79b`](https://github.com/xLydianSoftware/Qubx/commit/762c79b55b2cdd4f47d99de9f999a06327e05782))

- Temporary disabled import imp module
  ([`c45acfa`](https://github.com/xLydianSoftware/Qubx/commit/c45acfacd160f26cbaee48cc7f1cb53f0a6b8e9c))

- Temporary sync ccxt trading connector (WIP)
  ([`0b94fbd`](https://github.com/xLydianSoftware/Qubx/commit/0b94fbdf87bf23a30914d85498018ffe273be360))

- Testing taregt position
  ([`e3aa69d`](https://github.com/xLydianSoftware/Qubx/commit/e3aa69d24260bfb63a0fbe76a62f7d33ad3cd645))

- Tests
  ([`d2c8bc2`](https://github.com/xLydianSoftware/Qubx/commit/d2c8bc28263d7eb897d73eb124849f80cee3b025))

- Tests loop
  ([`398e4f5`](https://github.com/xLydianSoftware/Qubx/commit/398e4f5972bcaf9d7cdd467f2c59b75c6623395a))

- Trackers test fixed
  ([`ef737f4`](https://github.com/xLydianSoftware/Qubx/commit/ef737f48637dd75a11f8780c5b7dd871eff2b854))

- Tries to fix pickling cython classes issue
  ([`7850fbd`](https://github.com/xLydianSoftware/Qubx/commit/7850fbd20b01d80676746bb83cb1c3248d00adb9))

- Typo
  ([`21b50e9`](https://github.com/xLydianSoftware/Qubx/commit/21b50e9b59341b4ac6fdfd0892c5529a173d1691))

- Typo
  ([`c92c2ce`](https://github.com/xLydianSoftware/Qubx/commit/c92c2ce61d60e3d429cc69cee7d8d024a27a307f))

- Typo fixed
  ([`74eedf7`](https://github.com/xLydianSoftware/Qubx/commit/74eedf71f857a4f619767cc23cb3db6f72c522b8))

- Typo in cached reader
  ([`2a8db55`](https://github.com/xLydianSoftware/Qubx/commit/2a8db553be2740ea41f888fe8ad0ec13716f4b7f))

- Typo in position update
  ([`753acf8`](https://github.com/xLydianSoftware/Qubx/commit/753acf890a7fdf6c34922914c5449303466084b1))

- Typo in signal processing
  ([`e725fd8`](https://github.com/xLydianSoftware/Qubx/commit/e725fd872eac016e68e77637eb7f6f66ddcafc59))

- Update
  ([`1a37ed6`](https://github.com/xLydianSoftware/Qubx/commit/1a37ed608e627192f6d13ee3562a11da10458d3b))

- Upper case for debug level
  ([`c0f9d09`](https://github.com/xLydianSoftware/Qubx/commit/c0f9d09cee005069582c953f6662c9351db458fc))

- Use 1min resample if not specified for candle builder
  ([`4b374b9`](https://github.com/xLydianSoftware/Qubx/commit/4b374b9d9c329f48d6859baf0cf25e06505a5ca6))

- Use apply_async for loogers
  ([`4f2cc00`](https://github.com/xLydianSoftware/Qubx/commit/4f2cc0008198fa8b0c0e8af8ae702c1a58f87617))

- Variate now accepts function
  ([`ba64a6b`](https://github.com/xLydianSoftware/Qubx/commit/ba64a6bcea00ac6e076f8f18996020bd231dc697))

- Version increase
  ([`6162d7c`](https://github.com/xLydianSoftware/Qubx/commit/6162d7cd267f028320ffa6f2b97213af3f465a1d))

- Version increasign
  ([`6724466`](https://github.com/xLydianSoftware/Qubx/commit/67244669e95be16a43879188727d92bf87cbd004))

- Version update
  ([`6369159`](https://github.com/xLydianSoftware/Qubx/commit/636915973425171cc56634c271595ed4ddad8c47))

- Version update
  ([`8fbc2c0`](https://github.com/xLydianSoftware/Qubx/commit/8fbc2c0b8dac6a74d84e4ba15695f2c3f4b32c93))

- When subscribing to new symbols it's also need to submit last quote to OME in simulator
  ([`466b44e`](https://github.com/xLydianSoftware/Qubx/commit/466b44e83a47435e52dae464421782fece1c145c))

### Chores

- Add equity method, variate fixes
  ([`674e9e6`](https://github.com/xLydianSoftware/Qubx/commit/674e9e61f13a268dd91efcb4c904b0385bdd6e7d))

- Add historical quote processing
  ([`9d83d19`](https://github.com/xLydianSoftware/Qubx/commit/9d83d19ffe15236af9e8ebc89e243ba0359491a9))

- Add qubx console utility
  ([`6b83c34`](https://github.com/xLydianSoftware/Qubx/commit/6b83c341b0ebf40d0e0710941b372780a1e2220d))

- Add type hints for kama, atr, etc
  ([`cebf79c`](https://github.com/xLydianSoftware/Qubx/commit/cebf79c33ec02afb2132f04e87b024676588b681))

- Adds abitlity to generate mpl tearsheet
  ([`763259e`](https://github.com/xLydianSoftware/Qubx/commit/763259eef1b8ca88a79cac6ab34dc8d953440e2c))

- Adds additional test
  ([`54b843a`](https://github.com/xLydianSoftware/Qubx/commit/54b843a9a15c834c66a779a7bd148b478edf8634))

- Adds balance logger
  ([`dc20076`](https://github.com/xLydianSoftware/Qubx/commit/dc200764916484f5e8f704ed7eb877c8ceb377e4))

- Adds definitions for indicators
  ([`7501823`](https://github.com/xLydianSoftware/Qubx/commit/750182383360ba13b39944388684942f520ccb1c))

- Adds deps and useful utility in pandas
  ([`f10f85d`](https://github.com/xLydianSoftware/Qubx/commit/f10f85d9ed1c3ab52dd66ca83133058f8abfe0ae))

- Adds example of account config
  ([`9f2032a`](https://github.com/xLydianSoftware/Qubx/commit/9f2032ac6613df7ce427b6d3b6df7a15aab532fa))

- Adds hyperliquid symbols loading
  ([`ee796f4`](https://github.com/xLydianSoftware/Qubx/commit/ee796f4b3c913449f9357c3ef3763fc7ced9d256))

- Adds impl for get_historical_ohlc method
  ([`c3bac4e`](https://github.com/xLydianSoftware/Qubx/commit/c3bac4eee2c146effac4d0f27781913ee2ee80b7))

- Adds implementation for QuestDBReader::get_names
  ([`ba784ca`](https://github.com/xLydianSoftware/Qubx/commit/ba784cab85ebcc78217f8112bfd96689a9c52b91))

- Adds jupyter as option to run strategy
  ([`c81f4bb`](https://github.com/xLydianSoftware/Qubx/commit/c81f4bb6ed1fc5236c878579aabdfa7e7cd610e6))

- Adds lag indicator, fixes indicator calculation on already formed series
  ([`a188b77`](https://github.com/xLydianSoftware/Qubx/commit/a188b7714c2e8230f53fad00a6c0f3f64ed34df6))

- Adds market value and it's sum to PnL report
  ([`67c344f`](https://github.com/xLydianSoftware/Qubx/commit/67c344f58cf5939879a6c12b8a4718c1e87af820))

- Adds metrics and tearcheets for Qubx
  ([`fa3bcf8`](https://github.com/xLydianSoftware/Qubx/commit/fa3bcf8fc7fa0e81cf46b312cced0b97c6f2ee78))

- Adds more detailed logs on errors
  ([`752e82d`](https://github.com/xLydianSoftware/Qubx/commit/752e82de9e26da21450ec301766a537a79c62f68))

- Adds more details in exception
  ([`d3bcc03`](https://github.com/xLydianSoftware/Qubx/commit/d3bcc03fcd9a27a2c4e4e3f1c8dc866dc2e8ae9d))

- Adds more logs on latency
  ([`05286e9`](https://github.com/xLydianSoftware/Qubx/commit/05286e930c9f1c963df26a2c6e9b7fad5fb630e6))

- Adds new runner
  ([`cb7261b`](https://github.com/xLydianSoftware/Qubx/commit/cb7261b7e7bdb7018c31fdfaa7c67dc2cbff600b))

- Adds ohlc_plot method
  ([`53461f0`](https://github.com/xLydianSoftware/Qubx/commit/53461f014c676ba9bd749bbd4a879a7d8e6632e9))

- Adds order processing
  ([`a4dc38a`](https://github.com/xLydianSoftware/Qubx/commit/a4dc38aea255e82664321f08eedea8b1e8096557))

- Adds pandas ta tests from Qube
  ([`5210a78`](https://github.com/xLydianSoftware/Qubx/commit/5210a7822becef16877de09f0209412b1c44dd71))

- Adds portfolio performance metrics
  ([`c1d941a`](https://github.com/xLydianSoftware/Qubx/commit/c1d941ae00299a599c99357086576ca09d424dcf))

- Adds pyi definitions for series methods and increases version
  ([`2a916df`](https://github.com/xLydianSoftware/Qubx/commit/2a916df75487737ee031918bd421a272ce4f3758))

- Adds README memo
  ([`a4c2a77`](https://github.com/xLydianSoftware/Qubx/commit/a4c2a775c587bbff097665e2fe93048e8105725b))

- Adds run_id to log writer
  ([`e395bf2`](https://github.com/xLydianSoftware/Qubx/commit/e395bf2ea31e132365c1d44ab58721deafdb09f4))

- Adds some helpers: this_project_root() method
  ([`e8da14c`](https://github.com/xLydianSoftware/Qubx/commit/e8da14cc5b99b28a4e18d282675cc0613daf32f0))

- Adds suffix for QuestDB reader
  ([`18fb5c5`](https://github.com/xLydianSoftware/Qubx/commit/18fb5c5346c76611095a4a0ad9ebcab032cfced2))

- Adds test for position average price
  ([`69b3ea1`](https://github.com/xLydianSoftware/Qubx/commit/69b3ea15a214225b686970ff56166bac0645a857))

- Adds test for tracking and gathering
  ([`822f47f`](https://github.com/xLydianSoftware/Qubx/commit/822f47f77ceb8bf5fa68fe130b93f4b9e24e3af1))

- Adds test on ohlc history in context
  ([`8b3e8fb`](https://github.com/xLydianSoftware/Qubx/commit/8b3e8fb92e982d62149e3f44e0b414e5a505b7b0))

- Adds tests for IterableSimulatorData
  ([`ba2c912`](https://github.com/xLydianSoftware/Qubx/commit/ba2c91257cfd38dbfc3798a57937543cc3ecdd7f))

- Adds tests for OHLCV listening
  ([`981f182`](https://github.com/xLydianSoftware/Qubx/commit/981f1820e5fcee5f77ed4a26bbce4ca68fc46a7f))

- Adds tests for trackers
  ([`32eaf5a`](https://github.com/xLydianSoftware/Qubx/commit/32eaf5a8c07aecf25b789aa3ddc9ac8682a805ad))

- Adds tests on locator
  ([`39f126b`](https://github.com/xLydianSoftware/Qubx/commit/39f126b5864fdff7072fa6c1e999e0a365353c5a))

- Adds utils for cython re-compilation for
  ([`fb74016`](https://github.com/xLydianSoftware/Qubx/commit/fb74016d3f9727d353ff105366d32674c7772008))

- Adjusting open_close_time_indent
  ([`96764f3`](https://github.com/xLydianSoftware/Qubx/commit/96764f3e3fe5f66be72cb84da4d850ff4a50e5d1))

- Before refactoring
  ([`54ac055`](https://github.com/xLydianSoftware/Qubx/commit/54ac05524840df11744ca14c97e4bb9db91cb7e9))

- Before renaming Subtype -> DataType
  ([`7bd15a2`](https://github.com/xLydianSoftware/Qubx/commit/7bd15a202e527d24cf565b3d7058ceaff1669b46))

- Big refactoring needed for simulator
  ([`eba6719`](https://github.com/xLydianSoftware/Qubx/commit/eba671984967662d5332eb7bd1ea29ebea993d86))

- Calls StrategyContext.stop() after simulation is finished. Adds latency report printing.
  ([`2c5a9d6`](https://github.com/xLydianSoftware/Qubx/commit/2c5a9d6e3aacbca36336ad89a22c3e8292bc7a06))

- Changed debug logging format
  ([`25e7173`](https://github.com/xLydianSoftware/Qubx/commit/25e7173f93e75dd470922fffd76e37f7e2a2b5fb))

- Changes version and update ignoring
  ([`8d319bd`](https://github.com/xLydianSoftware/Qubx/commit/8d319bd313f793cdf39a6e4dad036d37cfb9fa27))

- Cleanup runner config structure
  ([`b7cb94b`](https://github.com/xLydianSoftware/Qubx/commit/b7cb94b1453edad2b7eea47c3f7474a2687b2070))

- Compare indicator
  ([`913adaf`](https://github.com/xLydianSoftware/Qubx/commit/913adaf3b1c3cfef815b6c739838d3fefa6aace2))

- First QuestDB implementation
  ([`a8850e5`](https://github.com/xLydianSoftware/Qubx/commit/a8850e5bf8f28772cc60cf997ffb0c804f8b8c42))

- First working version of simulated broker
  ([`5cd44b6`](https://github.com/xLydianSoftware/Qubx/commit/5cd44b680a445b1dd6ded9ec823f577943b60cf7))

- Fixes names presentation for indicators
  ([`6c44067`](https://github.com/xLydianSoftware/Qubx/commit/6c440679f4c57b696a8b04212ddbb5ae8da0c2df))

- Fixes poetry building system
  ([`01c8044`](https://github.com/xLydianSoftware/Qubx/commit/01c8044e30dc6aaf8bdffb8dfc20d52f802e5e72))

- Fixes printing exceptions
  ([`e57c9da`](https://github.com/xLydianSoftware/Qubx/commit/e57c9da6b79510969b4d4b47e917af3856e29106))

- Fixes tests
  ([`05f58dc`](https://github.com/xLydianSoftware/Qubx/commit/05f58dcb629dc95a60f1e4220816cc78b437be83))

- Idea on OrderBook custom connector
  ([`b6ce203`](https://github.com/xLydianSoftware/Qubx/commit/b6ce20329280781b59ef5f92369d60d4bf132f59))

- Implement universe updates
  ([`29b5c34`](https://github.com/xLydianSoftware/Qubx/commit/29b5c3451b6bc2d4b424d03163f327711d1c6a40))

- Implements on_fit processing when all data is ready
  ([`35042f0`](https://github.com/xLydianSoftware/Qubx/commit/35042f0e72b331442b7b3478bedc465f4a81a6a7))

- Increment version for MultiQuestDB connector
  ([`ad1f69e`](https://github.com/xLydianSoftware/Qubx/commit/ad1f69e9c8387c3282a7a826c10eabec315026b6))

- Logger prints simulated timestamps when backtesting running
  ([`26be10d`](https://github.com/xLydianSoftware/Qubx/commit/26be10d8a41d17262ee26d644abbf4868dbd3118))

- Merging main
  ([`cd8ee5a`](https://github.com/xLydianSoftware/Qubx/commit/cd8ee5aed0224b6f63026a0cfbd102f53ff6f11b))

- Metric fixes, add indicator typing, swings middles
  ([`55181e9`](https://github.com/xLydianSoftware/Qubx/commit/55181e9675acc6a11d9219298347626b879a89ff))

- More clear way to represent indicators as function
  ([`5f80b58`](https://github.com/xLydianSoftware/Qubx/commit/5f80b5896933bc370c9a850b9704427a253fac57))

- More convenient parameters log
  ([`a9e3d3f`](https://github.com/xLydianSoftware/Qubx/commit/a9e3d3fa71cdd6f16889a04a43b670e8d0c6eea2))

- Move pnl to bottom in chart signals
  ([`9c5d987`](https://github.com/xLydianSoftware/Qubx/commit/9c5d9870bad37a39077e1f78a0da0c7f3d2a9909))

- Moved some pandas utils from Qube1
  ([`b43c615`](https://github.com/xLydianSoftware/Qubx/commit/b43c615a937a441f0628db41bd9c54a3f79f2102))

- Moves indicators into separate module. Adds
  ([`bc16ad0`](https://github.com/xLydianSoftware/Qubx/commit/bc16ad0850a3ba775867b09ab6b5a2eb7e71d134))

- New backtester initial version
  ([`6602929`](https://github.com/xLydianSoftware/Qubx/commit/66029293961a50b05bfb18f82cf2a2628ba4bdc3))

- New impl for readers and transformers
  ([`fdf0935`](https://github.com/xLydianSoftware/Qubx/commit/fdf0935f1b843997608a46d04685702c1afe6c9f))

- New logo, adds latency measurements
  ([`d5edbef`](https://github.com/xLydianSoftware/Qubx/commit/d5edbef52ba3ddd752f45ea5b4d65bf2886f5235))

- Ome functionality is done
  ([`57cc843`](https://github.com/xLydianSoftware/Qubx/commit/57cc84386283ecb531e79a72bb0ab1da746c0c38))

- Provide exchange name in StrategyContext
  ([`fe3e9e3`](https://github.com/xLydianSoftware/Qubx/commit/fe3e9e3e0d52d5b22a9a64941addd96ed4309c5b))

- Refactoring in progress
  ([`60d5c9a`](https://github.com/xLydianSoftware/Qubx/commit/60d5c9a96457f650c6d2f56e6668dc2f44d4893a))

- Refactors
  ([`85c3fa4`](https://github.com/xLydianSoftware/Qubx/commit/85c3fa4b3ddce9eca8b21d3454fec84f0c81b6f6))

- Refactors experiments notebooks
  ([`7f52fd5`](https://github.com/xLydianSoftware/Qubx/commit/7f52fd50ab4fd525beacc4d1d7bd3bea9fcc87de))

- Refactors indicator
  ([`4e8fed2`](https://github.com/xLydianSoftware/Qubx/commit/4e8fed2f70dfbca6b0c65011f9d78dea4843f39e))

- Refactors IndicatorOHLC
  ([`ec4c0d9`](https://github.com/xLydianSoftware/Qubx/commit/ec4c0d9dac7c00da69f7ecc34100b26ccee85064))

- Refactors portfolio logging subsystem
  ([`22e30d2`](https://github.com/xLydianSoftware/Qubx/commit/22e30d23ee4caeda95ff5b38fc01a2762c5e9c61))

- Refactors risk manager logic
  ([`4ccd72c`](https://github.com/xLydianSoftware/Qubx/commit/4ccd72cf66533a6748982bc681440c84f8a2936e))

- Remove installation cython compiling hook
  ([`c7793d8`](https://github.com/xLydianSoftware/Qubx/commit/c7793d853f54016d1c140b420cc48eba3d60be60))

- Removed is_simulation checking on subscription processor
  ([`b179b15`](https://github.com/xLydianSoftware/Qubx/commit/b179b15505722c1ddc0e6cab74b156e401b8f353))

- Removes some unnecessary files
  ([`6a6f3f7`](https://github.com/xLydianSoftware/Qubx/commit/6a6f3f7210de5b8fa0db5a2b76025f3d04e098ec))

- Renamed repo and cleaned up code
  ([`d5f3b2a`](https://github.com/xLydianSoftware/Qubx/commit/d5f3b2a95e1e4f9ad3780c50fc52b185448748e0))

- Renamed to Qubx
  ([`f2acf50`](https://github.com/xLydianSoftware/Qubx/commit/f2acf50bbce38082989a326b9c48a8e19121cb48))

- Renames method in Indicator class
  ([`bebee0c`](https://github.com/xLydianSoftware/Qubx/commit/bebee0ce03689d66eb1f4a0770693d9ab6f0d33f))

- Renames options argument in trade method
  ([`152350c`](https://github.com/xLydianSoftware/Qubx/commit/152350cafa78f1702c5686383e114d860f679968))

- Simplified configuration for trigger and
  ([`c80b006`](https://github.com/xLydianSoftware/Qubx/commit/c80b006e61eb92cf66911eef6cdfe838d971e542))

- Small additions
  ([`35cf392`](https://github.com/xLydianSoftware/Qubx/commit/35cf39297ef8845d0278eb641e96af9d6bbfca98))

- Small changes
  ([`4e2de17`](https://github.com/xLydianSoftware/Qubx/commit/4e2de17f1eb031dd0d2f6c75c03fed06982c31d7))

- Small refactoring
  ([`ee5964d`](https://github.com/xLydianSoftware/Qubx/commit/ee5964d3e6ec531fc560d14de35dccd877d5e965))

- Small refactoring
  ([`9e45868`](https://github.com/xLydianSoftware/Qubx/commit/9e45868f51b0016340e8df7c73e89accef938118))

- Small refactoring
  ([`c9b1464`](https://github.com/xLydianSoftware/Qubx/commit/c9b14640f0a8e40b84c383df0e4ac95c4ed38a0b))

- Small refactoring
  ([`a34e5ed`](https://github.com/xLydianSoftware/Qubx/commit/a34e5edb9a79b74ca8d52d2b78abc15e2f854f14))

- Small renamings
  ([`a50ba98`](https://github.com/xLydianSoftware/Qubx/commit/a50ba98c063feee633f97db07b815503f726582a))

- Some debug adjustment
  ([`c8133f8`](https://github.com/xLydianSoftware/Qubx/commit/c8133f8be367352165f99ae4b767d1286b3ccdd7))

- Some generalization of QuestDB reader class
  ([`51fa13b`](https://github.com/xLydianSoftware/Qubx/commit/51fa13bc4d99bf54bc646c82e9e68cea2424d484))

- Store working version before refactoring
  ([`f51fd25`](https://github.com/xLydianSoftware/Qubx/commit/f51fd2538d6c453ae7ad56319a2a7225120ce5d4))

- Strats design
  ([`e4d2a38`](https://github.com/xLydianSoftware/Qubx/commit/e4d2a3860faa78b09a2eabbb4255bf42b59cd89d))

- Test for account processor
  ([`14ea724`](https://github.com/xLydianSoftware/Qubx/commit/14ea724cfa58799b3f515279c97be70f2ca2c6e4))

- Test for BFS
  ([`50091ea`](https://github.com/xLydianSoftware/Qubx/commit/50091ead6614826f71fdc7b339e1bb078472648c))

- Tests for stop orders
  ([`1c20118`](https://github.com/xLydianSoftware/Qubx/commit/1c2011893cbd258694793b7096485d1bfce3c07e))

- Update version
  ([`bfc37eb`](https://github.com/xLydianSoftware/Qubx/commit/bfc37eba2b56a7d1f1d98833ed1f432335db4005))

- Validation of simulator
  ([`cbbc8b0`](https://github.com/xLydianSoftware/Qubx/commit/cbbc8b093b63a47740628489c6d357b8395eb14a))

- Version 0.3.0
  ([`2e9a686`](https://github.com/xLydianSoftware/Qubx/commit/2e9a68631cf6f196bf39ae7c17478623a2d7a121))

- Version increase
  ([`8d3c066`](https://github.com/xLydianSoftware/Qubx/commit/8d3c066c35bab138ec12a1a1138b046101bc107e))

- Version increase
  ([`a4bdc1b`](https://github.com/xLydianSoftware/Qubx/commit/a4bdc1b38cd47330c83072e10e3b87f88489e441))

- Version increased
  ([`d275c36`](https://github.com/xLydianSoftware/Qubx/commit/d275c36af5439876d317a1d6c00916afde9186c5))

- Version increased
  ([`7c7bde5`](https://github.com/xLydianSoftware/Qubx/commit/7c7bde51da523a1dff4a870c4329afa8e73d69bc))

- Version increasing
  ([`e4db070`](https://github.com/xLydianSoftware/Qubx/commit/e4db0706e88d17620579d9cca1040681058149d9))

- Version increment
  ([`e443eaf`](https://github.com/xLydianSoftware/Qubx/commit/e443eaf2eb03463113791ce79a019c9389c2b1ad))

- Version increment
  ([`c21eb1d`](https://github.com/xLydianSoftware/Qubx/commit/c21eb1dfb931163d003341eacf5e1328252f55d9))

- Version increment
  ([`af9a161`](https://github.com/xLydianSoftware/Qubx/commit/af9a1612fcce1d009d9e53fd5dde4e36e1681da6))

- Version increment
  ([`4ab5647`](https://github.com/xLydianSoftware/Qubx/commit/4ab5647860586c8eb1a257ce624dbbbbadcb2211))

- Version increment
  ([`4209eb7`](https://github.com/xLydianSoftware/Qubx/commit/4209eb718ba31f41bad756e14ddc69bcf154e413))

- Version update
  ([`161fb24`](https://github.com/xLydianSoftware/Qubx/commit/161fb24ba54a71e791f6d98e94d60f2d5d1e98e1))

- Wip for orders processing
  ([`38990be`](https://github.com/xLydianSoftware/Qubx/commit/38990be6796ad46b796ed9e3f1951096c8479aba))

### Documentation

- Add docs to interfaces
  ([`ca3056d`](https://github.com/xLydianSoftware/Qubx/commit/ca3056d131e924d4897cf196a1bd0040697ec233))

### Features

- - high level prototypes for strategy and backtester
  ([`1f32017`](https://github.com/xLydianSoftware/Qubx/commit/1f32017714389299f2de36676ab0f482e9acf279))

- Add description to strategy config
  ([`836f956`](https://github.com/xLydianSoftware/Qubx/commit/836f95621350165ae8e4b276b40678767b568f90))

- Add load_config method to backtester management
  ([`3302bcd`](https://github.com/xLydianSoftware/Qubx/commit/3302bcd40801045cf04349fdbf1ebcd6c27f794a))

- Add performance export for TradingSessionResults.to_file()
  ([`40c2e50`](https://github.com/xLydianSoftware/Qubx/commit/40c2e50c4c6db0bf81ec7276998073ca939bfb10))

- Add simulation config file to results
  ([`34e98fe`](https://github.com/xLydianSoftware/Qubx/commit/34e98fe4b925f67ae7e3c933280deff6d76d871a))

- Add support for conditions in variate
  ([`798b6c8`](https://github.com/xLydianSoftware/Qubx/commit/798b6c8d1674df2dcdaeb1d97400eea6062f8b33))

- Add variations support to backtests results manager
  ([`8ef4f5f`](https://github.com/xLydianSoftware/Qubx/commit/8ef4f5f3f0c4431819935b8a8408e0fa65c53910))

- Add variations to the simulation results
  ([`1a6ba6f`](https://github.com/xLydianSoftware/Qubx/commit/1a6ba6f3dfaef2712dea914a612cea1def79b0f5))

- Adds a test
  ([`ce84fe8`](https://github.com/xLydianSoftware/Qubx/commit/ce84fe82d8628ec13d9f20abc113717b6282aeaf))

- Adds ability to convert Qubx portfolio to Qube presentation
  ([`c48da95`](https://github.com/xLydianSoftware/Qubx/commit/c48da95205cbc2d6799aeaf1cb6fd37a208f2fdf))

- Adds actual timestamps of spotted pivots in swings indicator
  ([`a733e95`](https://github.com/xLydianSoftware/Qubx/commit/a733e9551b96d6c29738409715db7e7f6bd59dc7))

- Adds advanced trackers with entry improvements
  ([`7371a17`](https://github.com/xLydianSoftware/Qubx/commit/7371a17a05c6f1b3389184775e3da74c827ef65c))

- Adds binance data loader
  ([`8a5c6c6`](https://github.com/xLydianSoftware/Qubx/commit/8a5c6c645f06c7d9f8a79236450548e0cead8407))

- Adds example doc in loader
  ([`3fbf686`](https://github.com/xLydianSoftware/Qubx/commit/3fbf686f31d16a17709d6bdc121ca642a48d1130))

- Adds exchange method to context
  ([`2a2cf63`](https://github.com/xLydianSoftware/Qubx/commit/2a2cf63c97c558e07fc4b7f7288c4999ca7f7d56))

- Adds exchange() method to IMarketManager interface
  ([`89af1af`](https://github.com/xLydianSoftware/Qubx/commit/89af1af2196df1a23787cd6abbbe60aaa5de3081))

- Adds fees lookup with configuration etc
  ([`7b26066`](https://github.com/xLydianSoftware/Qubx/commit/7b26066f3ae90ed39c3a25b8dfeba35d9176fcda))

- Adds get_historical_ohlc to simulated broker
  ([`8d6c9ed`](https://github.com/xLydianSoftware/Qubx/commit/8d6c9ed3188b5dc8cb9c564e10f24c9bcdcf4477))

- Adds Instrument class and lookup for
  ([`dce112f`](https://github.com/xLydianSoftware/Qubx/commit/dce112fd026a0f3d84a7d6b9fcbfe67eb98f8d27))

- Adds Kama indicator
  ([`3493fc6`](https://github.com/xLydianSoftware/Qubx/commit/3493fc6218e0e4ff239cdb550b3f8c0f36b3beb8))

- Adds Kraken symbols info getting for lookup
  ([`29bb872`](https://github.com/xLydianSoftware/Qubx/commit/29bb872085a934aca60777e928bec9e387fe50f1))

- Adds list of coins in the update_binance_data_storage
  ([`9bb5b0f`](https://github.com/xLydianSoftware/Qubx/commit/9bb5b0f9d161322c980975ef2288d6564fef16d1))

- Adds loc method to TimeSeries class
  ([`54c0b59`](https://github.com/xLydianSoftware/Qubx/commit/54c0b598845c6e9148b7c5dfa12bde352f14a51b))

- Adds locator so it's possible to do slicing and searching in TimeSeries
  ([`c4c1de0`](https://github.com/xLydianSoftware/Qubx/commit/c4c1de02314cd4ce78c71f5609fcbc706fa1cd21))

- Adds lowest and highest indicators
  ([`1a714f8`](https://github.com/xLydianSoftware/Qubx/commit/1a714f8e40c97e1e77ada2569be830a8e620772b))

- Adds method for finding aux instrument
  ([`5bead17`](https://github.com/xLydianSoftware/Qubx/commit/5bead17ab23ef61fb9b2d09cdb37bb8744f9131a))

- Adds ohlc data subscriber
  ([`55ce636`](https://github.com/xLydianSoftware/Qubx/commit/55ce636efaf0a0aa981f0dbdc358559a4aa68b58))

- Adds ohlc data subscriber
  ([`d6f8749`](https://github.com/xLydianSoftware/Qubx/commit/d6f87498db7d330158940bf2dc5dbdcd15d4563f))

- Adds OhlcDict wrapper to loader's dict output
  ([`f18b041`](https://github.com/xLydianSoftware/Qubx/commit/f18b0419cabcdd7f7120297db9f250bcf2241c90))

- Adds OHLCV.from_dataframe(pd.DataFrame) static method. Resets watchdog before simulation.
  ([`9771ef6`](https://github.com/xLydianSoftware/Qubx/commit/9771ef67382be694066ed3d9e6bdf343089c6447))

- Adds packed binance symbols meta-data
  ([`18ff1ad`](https://github.com/xLydianSoftware/Qubx/commit/18ff1adb56816f17fac8aa01e661c76d1eb9aa4a))

- Adds pandas csv reader
  ([`2a3e2ab`](https://github.com/xLydianSoftware/Qubx/commit/2a3e2ab91da7182b20f4cf905f652c3a355d1be8))

- Adds positions updates by current prices
  ([`c5a31f6`](https://github.com/xLydianSoftware/Qubx/commit/c5a31f61dfb34ccd45b20b9604c10e3016c279f0))

- Adds RestoredBarsFromOHLC transformer
  ([`a04793b`](https://github.com/xLydianSoftware/Qubx/commit/a04793bc452923bc188b991363d9bb6d31819a60))

- Adds signals logging
  ([`3ac71d8`](https://github.com/xLydianSoftware/Qubx/commit/3ac71d8b651a9ef8f14fe2592994e4416d96f60c))

- Adds test for pewma indicator
  ([`7f8da86`](https://github.com/xLydianSoftware/Qubx/commit/7f8da86ca240ae0634d9b4b3d6bdfd9ac39e0b21))

- Adds tests for psar OHLC based indicator
  ([`86d729f`](https://github.com/xLydianSoftware/Qubx/commit/86d729f8b32f1af1a05ccd9182ff295114751525))

- Adds tests on simulation data recognition
  ([`8165692`](https://github.com/xLydianSoftware/Qubx/commit/816569252e32317d746afc5c8cca8b6b496a8dc8))

- Adds TimestampedDict type for representing arbitrary data, also
  ([`cf450d4`](https://github.com/xLydianSoftware/Qubx/commit/cf450d4a7fd9452b213a6f91a050fe919e1ca62f))

- Atr based risk manager - first version
  ([`c157baf`](https://github.com/xLydianSoftware/Qubx/commit/c157baf1a7cf0fa5ffd617307dd8ba291df76a4f))

- Atr indicator is added
  ([`8fcfefe`](https://github.com/xLydianSoftware/Qubx/commit/8fcfefe20ac26542a25707635aa2b6ba8bc2c8a5))

- Bar triggers - very first impl tested
  ([`ef58f00`](https://github.com/xLydianSoftware/Qubx/commit/ef58f005c401fe096d86b13644a2101554c4241f))

- Constant capital rtisk sizer
  ([`3f16512`](https://github.com/xLydianSoftware/Qubx/commit/3f16512b420f2637f80b8b66f62196fe5ca575f7))

- Customize dataframes look in notebooks
  ([`59f7eab`](https://github.com/xLydianSoftware/Qubx/commit/59f7eab2af34f1aaa47641a2b89bfd5619282dc2))

- Dynamic mixin of strategies
  ([`41b4f0e`](https://github.com/xLydianSoftware/Qubx/commit/41b4f0e4c85a0b0e05345c30037a4948443f769e))

- Emulating quotes / trades from ohlc data
  ([`fbfe6cc`](https://github.com/xLydianSoftware/Qubx/commit/fbfe6cc53ca4ba76ad97b5ca246a2a198ce97674))

- Finishes Position and Transaction costs classes
  ([`3604d56`](https://github.com/xLydianSoftware/Qubx/commit/3604d56ceef05d25d153e17c22634e0368a25e9a))

- First version of data readers with pyarrow as engine
  ([`ffb1012`](https://github.com/xLydianSoftware/Qubx/commit/ffb101224fa9fede3c244b14f5e3937d57491a45))

- First version of StrategyCtx
  ([`a366c0f`](https://github.com/xLydianSoftware/Qubx/commit/a366c0fcc55be1f7d7ddfbace5614624b1b4330e))

- Fix list method to backtester management
  ([`add380f`](https://github.com/xLydianSoftware/Qubx/commit/add380f496d63984959e9627287c68d97ee02369))

- Flag that indicates whether calculate indicators on closed bars
  ([`440ec95`](https://github.com/xLydianSoftware/Qubx/commit/440ec9585a474c8ecf973cc43c1267dab6a4b99b))

- Loading positions and active orders from exchange during startegy starting
  ([`761dece`](https://github.com/xLydianSoftware/Qubx/commit/761decefd48c78ac4d0094125b56a47ec7f12d75))

- New tracker implementation
  ([`3073b7f`](https://github.com/xLydianSoftware/Qubx/commit/3073b7f20b0823220ebc061ad57f7885a93900b7))

- Ohlc based indicators and PSAR indicator
  ([`9667f65`](https://github.com/xLydianSoftware/Qubx/commit/9667f6588016c965aef4ebcc4f3c49ea5b5e2198))

- Pewma indicator on streaming data is added
  ([`95f6c0b`](https://github.com/xLydianSoftware/Qubx/commit/95f6c0b127d5c7ec84c73c61d9b3f1361f0eb550))

- Pewma_outliers_detector is added
  ([`299e185`](https://github.com/xLydianSoftware/Qubx/commit/299e185cabc87e9b242783baf904736a56e26622))

- Portfolio, executions and positions loggers
  ([`b24c78d`](https://github.com/xLydianSoftware/Qubx/commit/b24c78db216135c20e081e40fbe1d97247b4421c))

- Refactors recognize_simulation_setups methos and adds test
  ([`9e6c53f`](https://github.com/xLydianSoftware/Qubx/commit/9e6c53f09b9e3c8094d7d477af6cee4fc7b7d3b2))

- Removes "hist_" prefix conversion
  ([`dea96aa`](https://github.com/xLydianSoftware/Qubx/commit/dea96aa53cde9c57b5a32c3945dbac34e6bfeb9d))

- Simulator uses default schedule if detected
  ([`a99ee6b`](https://github.com/xLydianSoftware/Qubx/commit/a99ee6becabfe010477ff576d94115b870822c5e))

- Swings detector in cython
  ([`52154d6`](https://github.com/xLydianSoftware/Qubx/commit/52154d65bae0c3e32089375a24ea1205addcecb6))

- Tests for on_formed_bar indicators calculations
  ([`5a6b018`](https://github.com/xLydianSoftware/Qubx/commit/5a6b0189f659295e7eab8091157f741996aa2024))

- Transfers some useful utils from Qube1
  ([`7c12713`](https://github.com/xLydianSoftware/Qubx/commit/7c127134d7dd0fb65d1b2aceb64d1da7a0c0a3d0))

- Wip - set_unverse with position close policies
  ([`a11d1c5`](https://github.com/xLydianSoftware/Qubx/commit/a11d1c5e33393e321ce0e5ddf7528a6612a7f483))

- Wip - set_unverse with position wait_for_change policy
  ([`a19d11a`](https://github.com/xLydianSoftware/Qubx/commit/a19d11aed93837c359e6c47a7f6666a9ed43b886))

### Refactoring

- Cleanup quotes / bars emulating code
  ([`399f935`](https://github.com/xLydianSoftware/Qubx/commit/399f935477a2b4f1d0474ebd2e4aadb0adb4ba53))

- Moves plotting to utils module
  ([`3267357`](https://github.com/xLydianSoftware/Qubx/commit/3267357d87166443cf0346d2a18d622bdcffe440))

- Redone recognizer
  ([`b23c1cb`](https://github.com/xLydianSoftware/Qubx/commit/b23c1cb7647af7c31a428a62d6eedee08b253a1f))

- Refactoring IteratedDataStreamsSlicer class
  ([`eef8faa`](https://github.com/xLydianSoftware/Qubx/commit/eef8faa288f8c6485827fbaacfcd6fdc6366de54))

- Refactors how simulator processes scheduled events
  ([`667c179`](https://github.com/xLydianSoftware/Qubx/commit/667c1795c719b506afd32e13ec1c4599f230da48))

- Renames get_instrument -> query_instrument
  ([`f6bd440`](https://github.com/xLydianSoftware/Qubx/commit/f6bd440dad729e80b36be265b8a60a7a007b10a9))

- Simulator utils
  ([`444db28`](https://github.com/xLydianSoftware/Qubx/commit/444db28766dc03c945887eadf44d222a1d73ec25))

- Timeguard class as decorator on DataReader
  ([`626ee3a`](https://github.com/xLydianSoftware/Qubx/commit/626ee3a8c874e605e2a762fbc9f078eb84544ef8))

### Testing

- Adds some memory tests
  ([`250ca4d`](https://github.com/xLydianSoftware/Qubx/commit/250ca4da4bdc141cb5521982465f16b54d0b7d54))
