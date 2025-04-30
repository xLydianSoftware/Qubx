""" "
Here stuff we want to have in every Jupyter notebook after calling %qubx magic
"""

import qubx
from qubx import runtime_env
from qubx.utils.misc import add_project_to_system_path, logo


def np_fmt_short():
    # default np output is 75 columns so extend it a bit and suppress scientific fmt for small floats
    np.set_printoptions(linewidth=240, suppress=True)


def np_fmt_reset():
    # reset default np printing options
    np.set_printoptions(
        edgeitems=3,
        infstr="inf",
        linewidth=75,
        nanstr="nan",
        precision=8,
        suppress=False,
        threshold=1000,
        formatter=None,
    )


if runtime_env() in ["notebook", "shell"]:
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # -- all imports below will appear in notebook after calling %%qubx magic ---
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - Common stuff - - - -
    from datetime import time, timedelta  # noqa: F401

    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401

    # - - - - Charting stuff - - - -
    from matplotlib import pyplot as plt  # noqa: F401
    from tqdm.auto import tqdm  # noqa: F401

    # - - - - TA stuff and indicators - - - -
    import qubx.pandaz.ta as pta  # noqa: F401
    import qubx.ta.indicators as ta  # noqa: F401
    from qubx.backtester.optimization import variate  # noqa: F401

    # - - - - Simulator stuff - - - -
    from qubx.backtester.simulator import simulate  # noqa: F401

    # - - - - Portfolio analysis - - - -
    from qubx.core.metrics import (  # noqa: F401
        chart_signals,
        drop_symbols,
        extend_trading_results,
        get_symbol_pnls,
        pick_symbols,
        pnl,
        portfolio_metrics,
        tearsheet,
    )
    from qubx.data.helpers import loader  # noqa: F401

    # - - - - Data reading - - - -
    from qubx.data.readers import (  # noqa: F401
        AsOhlcvSeries,
        AsPandasFrame,
        AsQuotes,
        AsTimestampedRecords,
        CsvStorageDataReader,
        MultiQdbConnector,
        QuestDBConnector,
        RestoreTicksFromOHLC,
    )

    # - - - - Utils - - - -
    from qubx.pandaz.utils import (  # noqa: F401
        continuous_periods,
        drop_duplicated_indexes,
        generate_equal_date_ranges,
        ohlc_resample,
        retain_columns_and_join,
        rolling_forward_test_split,
        scols,
        srows,
    )
    from qubx.utils.charting.lookinglass import LookingGlass  # noqa: F401
    from qubx.utils.charting.mpl_helpers import fig, ohlc_plot, plot_trends, sbp, subplot  # noqa: F401
    from qubx.utils.misc import this_project_root  # noqa: F401

    # - setup short numpy output format
    np_fmt_short()

    # - add project home to system path
    add_project_to_system_path()

    # show logo first time
    if not hasattr(qubx.QubxMagics, "__already_initialized__"):
        setattr(qubx.QubxMagics, "__already_initialized__", True)
        logo()
