import numpy as np
import pandas as pd
from pytest import approx

from qubx.connectors.ccxt.utils import (
    ccxt_convert_deal_info,
    ccxt_convert_order_info,
    ccxt_extract_deals_from_exec,
    ccxt_restore_position_from_deals,
)
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import ZERO_COSTS, CtrlChannel, Deal, Instrument, ITimeProvider, Position, dt_64
from qubx.core.lookups import lookup
from tests.qubx.connectors.ccxt.data.ccxt_responses import (
    C1,
    C2,
    C3,
    C4,
    HIST,
    C5new,
    C6ex,
    C7cancel,
    buy_RAREUSDT1,
    buy_RAREUSDT2,
    buy_RAREUSDT3,
    execs_ACA,
    execs_SUPERUSDT1,
    execs_SUPERUSDT2,
    execs_SUPERUSDT3,
    sell_RAREUSDT1,
    sell_RAREUSDT2,
)
from tests.qubx.core.utils_test import DummyTimeProvider

N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)


class TestStrats:
    def test_ccxt_exec_report_conversion(self):
        instrument = lookup.find_symbol("BINANCE", "ACAUSDT")
        assert instrument is not None
        # - execution reports
        for o in [
            ccxt_convert_order_info(instrument, C1),
            ccxt_convert_order_info(instrument, C2),
            ccxt_convert_order_info(instrument, C3),
            ccxt_convert_order_info(instrument, C4),
            ccxt_convert_order_info(instrument, C5new),
            ccxt_convert_order_info(instrument, C6ex),
            ccxt_convert_order_info(instrument, C7cancel),
        ]:
            print(o)
        print("-" * 50)

        print(ccxt_convert_order_info(instrument, C5new))
        print(ccxt_convert_order_info(instrument, C6ex))
        print(ccxt_convert_order_info(instrument, C7cancel))

        print("#" * 50)

        # - historical records
        for h in HIST:
            i = lookup.find_symbol("BINANCE.UM", h["info"]["symbol"])
            if i is not None:
                o = ccxt_convert_order_info(i, h)
                print(o)

    def test_ccxt_hist_trades_conversion(self):
        raw = {
            "info": {
                "symbol": "RAYUSDT",
                "id": "56324015",
                "orderId": "536752004",
                "orderListId": "-1",
                "price": "2.11290000",
                "qty": "2.40000000",
                "quoteQty": "5.07096000",
                "commission": "0.00000648",
                "commissionAsset": "BNB",
                "time": "1712497717270",
                "isBuyer": True,
                "isMaker": False,
                "isBestMatch": True,
            },
            "timestamp": 1712497717270,
            "datetime": "2024-04-07T13:48:37.270Z",
            "symbol": "RAY/USDT",
            "id": "56324015",
            "order": "536752004",
            "type": None,
            "side": "buy",
            "takerOrMaker": "taker",
            "price": 2.1129,
            "amount": 2.4,
            "cost": 5.07096,
            "fee": {"cost": 6.48e-06, "currency": "BNB"},
            "fees": [{"cost": 6.48e-06, "currency": "BNB"}],
        }
        print(ccxt_convert_deal_info(raw))

    def test_position_restoring_from_deals(self):
        deals = [
            Deal(
                "0",
                1,
                time=pd.Timestamp("2024-04-07 13:04:36.975000"),
                amount=0.5,
                price=180.84,
                aggressive=True,
                fee_amount=0.00011542,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "1",
                1,
                time=pd.Timestamp("2024-04-07 13:09:22.644000"),
                amount=-0.5,
                price=181.12,
                aggressive=True,
                fee_amount=0.00011562,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "2",
                1,
                time=pd.Timestamp("2024-04-07 13:48:37.611000"),
                amount=0.11,
                price=181.67,
                aggressive=True,
                fee_amount=2.544e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "3",
                1,
                time=pd.Timestamp("2024-04-07 13:48:37.611000"),
                amount=0.11,
                price=181.68,
                aggressive=True,
                fee_amount=2.544e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "4",
                1,
                time=pd.Timestamp("2024-04-07 13:48:37.611000"),
                amount=0.11,
                price=181.69,
                aggressive=True,
                fee_amount=2.544e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "5",
                1,
                time=pd.Timestamp("2024-04-07 13:48:37.611000"),
                amount=0.22,
                price=181.69,
                aggressive=True,
                fee_amount=5.09e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "6",
                1,
                time=pd.Timestamp("2024-04-07 14:12:34.624000"),
                amount=-0.55,
                price=181.29,
                aggressive=True,
                fee_amount=0.00012728,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "7",
                1,
                time=pd.Timestamp("2024-04-07 14:16:46.048000"),
                amount=0.7,
                price=181.32,
                aggressive=True,
                fee_amount=0.00016175,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "8",
                1,
                time=pd.Timestamp("2024-04-07 14:17:47.396000"),
                amount=-0.7,
                price=181.36,
                aggressive=True,
                fee_amount=0.00016176,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "9",
                1,
                time=pd.Timestamp("2024-04-07 14:18:25.864000"),
                amount=0.13,
                price=181.36,
                aggressive=True,
                fee_amount=3.005e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "a",
                1,
                time=pd.Timestamp("2024-04-07 14:18:25.864000"),
                amount=0.11,
                price=181.36,
                aggressive=True,
                fee_amount=2.543e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "b",
                1,
                time=pd.Timestamp("2024-04-07 14:18:25.864000"),
                amount=0.76,
                price=181.36,
                aggressive=True,
                fee_amount=0.00076,
                fee_currency="SOL",
            ),  # type: ignore
        ]

        instr1: Instrument = lookup.find_symbol("BINANCE", "SOLUSDT")  # type: ignore
        pos1 = Position(instr1)  # type: ignore
        vol1 = np.sum([d.amount for d in deals]) - instr1.round_size_up(
            deals[-1].fee_amount if deals[-1].fee_amount else 0
        )

        pos1 = ccxt_restore_position_from_deals(pos1, vol1, deals)
        assert N(pos1.quantity, instr1.lot_size) == vol1

        deals = [
            Deal(
                "0",
                2,
                time=pd.Timestamp("2024-04-07 12:40:41.717000"),
                amount=0.154,
                price=587.1,
                aggressive=True,
                fee_amount=0.0001155,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "1",
                2,
                time=pd.Timestamp("2024-04-07 12:41:59.307000"),
                amount=-0.154,
                price=586.6,
                aggressive=True,
                fee_amount=0.00011472,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "2",
                2,
                time=pd.Timestamp("2024-04-07 12:44:45.991000"),
                amount=-0.199,
                price=588.5,
                aggressive=True,
                fee_amount=0.00014922,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "3",
                2,
                time=pd.Timestamp("2024-04-08 12:45:49.738000"),
                amount=0.025,
                price=594.1,
                aggressive=True,
                fee_amount=1.875e-05,
                fee_currency="BNB",
            ),  # type: ignore
            Deal(
                "4",
                2,
                time=pd.Timestamp("2024-04-08 12:48:37.543000"),
                amount=0.011,
                price=594.0,
                aggressive=True,
                fee_amount=8.25e-06,
                fee_currency="BNB",
            ),  # type: ignore
        ]

        instr2 = lookup.find_symbol("BINANCE", "BNBUSDT")
        assert instr2 is not None
        pos2 = Position(instr2)
        vol2 = np.sum([d.amount for d in deals]) - instr2.round_size_up(np.sum([d.fee_amount for d in deals]))  # type: ignore

        pos2 = ccxt_restore_position_from_deals(pos2, vol2, deals)
        assert N(pos2.quantity, instr2.lot_size) == vol2

    def test_account_processor_from_ccxt_reports(self):
        acc = BasicAccountProcessor(
            account_id="TestAcc1",
            time_provider=DummyTimeProvider(),
            base_currency="USDT",
            exchange="BINANCE",
            initial_capital=100,
        )
        acc.attach_positions(
            Position(lookup.find_symbol("BINANCE", "RAREUSDT")),  # type: ignore
            Position(lookup.find_symbol("BINANCE", "SUPERUSDT")),  # type: ignore
            Position(lookup.find_symbol("BINANCE", "ACAUSDT")),  # type: ignore
        )

        for exs in [
            *execs_ACA,
            buy_RAREUSDT1,
            buy_RAREUSDT2,
            buy_RAREUSDT3,
            sell_RAREUSDT1,
            sell_RAREUSDT2,
            execs_SUPERUSDT1,
            execs_SUPERUSDT2,
            execs_SUPERUSDT3,
        ]:
            for report in exs:
                symbol = report["info"]["s"]
                order = ccxt_convert_order_info(symbol, report)
                deals = ccxt_extract_deals_from_exec(report)
                acc.process_deals(symbol, deals)
                acc.process_order(order)

        print("- " * 50)
        print(pd.DataFrame.from_dict(acc.position_report()).T)
        print("- " * 50)
        print(f"Capital: {acc.get_capital()}")
        print(f"Margin Capital: {acc.get_total_capital()}")
        print(f"Net leverage: {acc.get_net_leverage()}")
        print(f"Gross leverage: {acc.get_gross_leverage()}")
