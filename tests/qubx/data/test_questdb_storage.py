import numpy as np
import pandas as pd

from qubx.core.basics import DataType, TimestampedDict
from qubx.core.series import Quote, Trade
from qubx.data.storage import RawData, RawMultiData
from qubx.data.storages.questdb import xLTableInfo
from qubx.data.transformers import OHLCVSeries, PandasFrame, TickSeries, TypedRecords


class TestQuestDbStorages:
    def test_xldecoding(self):
        x1 = xLTableInfo.decode_table_name("gateio.swap.candles_1m")
        assert x1 is not None
        assert x1.exchange == "GATEIO"
        assert x1.market_type == "SWAP"
        assert x1.dtype == DataType.OHLC
        assert x1.data_timeframe == "1m"

        x2 = xLTableInfo.decode_table_name("any_custom_table")
        assert x2 is None

        x3 = xLTableInfo.decode_table_name("binance.umswap.funding_payment")
        assert x3 is not None
        assert x3.exchange == "BINANCE.UM"
        assert x3.market_type == "SWAP"
        assert x3.dtype == DataType.FUNDING_PAYMENT
        assert x3.data_timeframe is None

        x4 = xLTableInfo.decode_table_name("test1.universe.factors")
        assert x4 is not None
        assert x4.exchange == "TEST1"
        assert x4.market_type == "UNIVERSE"
        assert x4.dtype == DataType.RECORD
        assert x4.alias_for_record_type == "factors"
