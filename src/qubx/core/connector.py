from typing import Protocol, runtime_checkable

from qubx.core.basics import CtrlChannel, Instrument, Order, OrderRequest, Timestamped, WalletMove, dt_64
from qubx.core.events import ChannelMessage


class ChannelEmitter:
    """Gives a connector its ``send`` helper for emitting events on the channel.

    ``send`` is NOT part of the IConnector contract — the framework never calls it; the
    connector emits its own lifecycle / market-data events. Concrete connectors set
    ``channel`` in their constructor and inherit ``send`` here instead of each redefining
    it.
    """

    channel: CtrlChannel

    def send(self, event: ChannelMessage) -> None:
        self.channel.send(event)

    # Default for every connector whose venue has no cash market to convert on (simulation,
    # and the venues that hold a single currency). Lives here rather than as a bare protocol
    # member so an out-of-tree connector keeps satisfying IConnector without implementing it.
    def convert_currency(
        self,
        from_currency: str,
        to_currency: str,
        amount: float,
        *,
        limit_price: float | None = None,
        max_slippage_bps: float = 10.0,
    ) -> str:
        raise NotImplementedError(f"{type(self).__name__} does not support currency conversion")

    # Default for every venue with a single wallet per account (simulation included).
    def wallet_moves(self) -> list[WalletMove]:
        return []

    def move_funds(self, currency: str | None, src: str, dst: str, amount: float | None = None) -> str:
        raise NotImplementedError(f"{type(self).__name__} does not support moving funds between wallets")

    def debt_repayments(self) -> list[str]:
        return []

    def repay_debt(self, currency: str, amount: float | None = None) -> str:
        raise NotImplementedError(f"{type(self).__name__} does not support repaying debt")


@runtime_checkable
class IMarketDataSink(Protocol):
    """Connector that consumes market data locally — a simulated venue feeding its OME.

    Not part of the IConnector contract: live connectors execute at the venue and never
    see market data. The processing manager narrows to this protocol before feeding a
    paper-trading connector.
    """

    def process_market_data(self, instrument: Instrument, data: Timestamped) -> None: ...


@runtime_checkable
class IConnector(Protocol):
    exchange_name: str

    def submit_order(self, request: OrderRequest) -> None: ...

    # cancel/update/request_order_status take the whole ``Order`` the AccountManager already
    # holds (it is the single source of order state). The connector reads everything the venue
    # call needs straight off it — both ids (it prefers ``venue_order_id`` when present, the
    # venue's own id, and falls back to ``client_order_id``, the only id known before the ack),
    # the ``instrument`` (HL asset index, ccxt symbol), and ``side``/``type``/``time_in_force``
    # that an edit-as-cancel-and-replace needs. This keeps every connector a stateless adapter:
    # no per-order cache. The ``Order`` is READ-ONLY — the connector must extract what it needs
    # SYNCHRONOUSLY (before scheduling any async venue call) and never mutate it, so the async
    # path never races the AM mutating the live object. Resulting events carry both ids, so the
    # AM routes by either.
    def cancel_order(self, order: Order) -> None: ...

    # ``quantity`` is the order's new TOTAL size including everything already filled;
    # None means unchanged. Connectors translate this to their venue's amend dialect on the
    # wire (e.g. HL's "replacement" dialect wants total - filled), but MUST echo the
    # requested total (or None) back in OrderUpdatedEvent.new_quantity — never the wire figure.
    def update_order(self, order: Order, *, price: float | None = None, quantity: float | None = None) -> None: ...

    def request_order_status(self, order: Order) -> None: ...

    # - include_orders=True: open orders (regular + algo/trigger) + positions + balances
    #   (startup discovery / periodic sweep); False: positions + balances only —
    #   open_orders=None, reconcile skips order diffing
    def request_snapshot(self, include_orders: bool = True) -> None: ...

    # Fetch trades for ``instrument`` since ``since`` (venue clock) and emit one DealEvent per
    # trade — recovers executions missed behind a position size diff (the Reconciler's
    # ConfirmPositionBySnapshot → RequestHistDeals).
    def request_hist_deals(self, instrument: Instrument, since: dt_64) -> None: ...

    def is_ws_ready(self) -> bool: ...
    def reconnect(self) -> bool: ...  # synchronous WS reconnect; returns success
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def make_client_id(self, suggested: str) -> str: ...

    @property
    def is_simulated_trading(self) -> bool: ...

    # Per-instrument venue settings.
    def get_instrument_leverage(self, instrument: Instrument) -> float | None: ...
    def get_max_instrument_leverage(self, instrument: Instrument) -> float | None: ...
    def get_max_instrument_notional(self, instrument: Instrument) -> float: ...
    def get_margin_mode(self, instrument: Instrument) -> str | None: ...
    def get_adl_level(self, instrument: Instrument) -> int | None: ...

    # Requests the CONFIGURED leverage, not the venue's maximum — that is a venue-reported
    # cap, read with get_instrument_leverage. Does not block and returns nothing: the
    # caller is the ProcessorThread, so the venue call goes off-thread and a refusal comes
    # back as a VenueOperationError on on_error. The connector skips the call when its cached configured
    # value already matches, and clamps the request to the cached venue maximum.
    def set_instrument_leverage(self, instrument: Instrument, leverage: float) -> None: ...

    def set_margin_mode(self, instrument: Instrument, mode: str) -> bool: ...

    # Cash, not exposure: swaps one currency for another on the venue's own market for the
    # pair and is never registered with the AccountManager — the only trace it leaves is the
    # balances of the next snapshot. Returns the conversion's id immediately (the venue round
    # trip runs off-thread, so the ProcessorThread keeps draining) and emits EXACTLY ONE
    # CurrencyConversionEvent per accepted call, failures included — only argument mistakes
    # raise. ONE IOC attempt: nothing rests on the book, and a short fill comes back as a
    # PARTIAL record, since only the caller knows how much it still needs once balances have
    # moved. ``amount`` is denominated in ``from_currency``; ``limit_price`` (in the market's
    # quote terms) bounds the fill absolutely, ``max_slippage_bps`` only relative to the book.
    # Venues without a cash market raise NotImplementedError.
    def convert_currency(
        self,
        from_currency: str,
        to_currency: str,
        amount: float,
        *,
        limit_price: float | None = None,
        max_slippage_bps: float = 10.0,
    ) -> str: ...

    # The wallet-to-wallet moves this connector's venue supports; empty on single-wallet venues.
    def wallet_moves(self) -> list[WalletMove]: ...

    # Cash moved between wallets of ONE account (keys from the Balance.wallets vocabulary),
    # never registered with the AccountManager — the only trace is the balances of the next
    # snapshot. Returns the move's id immediately (the venue round trip runs off-thread) and
    # emits EXACTLY ONE FundsMovedEvent per accepted call, failures included. A move not in
    # wallet_moves() or bad arguments (``amount`` missing where required, given where the venue
    # takes none) raise ValueError synchronously; ``currency=None`` means every eligible
    # currency. Venues without wallets raise NotImplementedError.
    def move_funds(self, currency: str | None, src: str, dst: str, amount: float | None = None) -> str: ...

    # The Balance.liabilities kinds repay_debt pays down; empty where the venue can't repay.
    def debt_repayments(self) -> list[str]: ...

    # Pays down debt in ``currency`` from the account's own cash, never registered with the
    # AccountManager — the only trace is the balances of the next snapshot. Returns the
    # repayment's id immediately and emits EXACTLY ONE DebtRepaidEvent per accepted call,
    # failures included. ``amount=None`` repays everything owed in the declared kinds. Bad
    # arguments raise ValueError synchronously; venues that can't repay raise NotImplementedError.
    def repay_debt(self, currency: str, amount: float | None = None) -> str: ...
