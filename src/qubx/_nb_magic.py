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
    import numpy as np  # type: ignore # noqa: F401
    import pandas as pd  # type: ignore # noqa: F401

    # - - - - Charting stuff - - - -
    from matplotlib import pyplot as plt  # type: ignore # noqa: F401
    from tqdm.auto import tqdm  # type: ignore # noqa: F401

    # - - - - TA stuff and indicators - - - -
    import qubx.pandaz.ta as pta  # type: ignore # noqa: F401
    import qubx.ta.indicators as ta  # type: ignore # noqa: F401
    from qubx.backtester.optimization import variate  # type: ignore # noqa: F401

    # - - - - Simulator stuff - - - -
    from qubx.backtester.simulator import simulate  # type: ignore # noqa: F401

    # - - - - Portfolio analysis - - - -
    from qubx.core.metrics import (  # type: ignore # noqa: F401
        calculate_leverage_per_symbol,  # type: ignore
        calculate_pnl_per_symbol,  # type: ignore
        chart_signals,  # type: ignore
        combine_sessions,  # type: ignore
        drop_symbols,  # type: ignore
        extend_trading_results,  # type: ignore
        find_session,  # type: ignore
        find_sessions,  # type: ignore
        get_symbol_pnls,  # type: ignore
        pick_symbols,  # type: ignore
        pnl,  # type: ignore
        portfolio_metrics,  # type: ignore
        tearsheet,  # type: ignore
    )

    # - - - - Data storages - - - -
    from qubx.data import (  # noqa: F401
        CachedStorage,  # type: ignore
        CsvStorage,  # type: ignore
        QuestDBStorage,  # type: ignore
    )
    from qubx.data.registry import StorageRegistry  # noqa: F401  # type: ignore

    # - - - - Utils - - - -
    from qubx.pandaz.utils import (  # noqa: F401
        continuous_periods,  # type: ignore
        drop_duplicated_indexes,  # type: ignore
        generate_equal_date_ranges,  # type: ignore
        ohlc_resample,  # type: ignore
        retain_columns_and_join,  # type: ignore
        rolling_forward_test_split,  # type: ignore
        scols,  # type: ignore
        srows,  # type: ignore
    )
    from qubx.utils.charting.lookinglass import LookingGlass  # type: ignore # noqa: F401
    from qubx.utils.charting.mpl_helpers import fig, ohlc_plot, plot_trends, sbp, subplot  # type: ignore # noqa: F401
    from qubx.utils.misc import this_project_root  # type: ignore # noqa: F401

    # - setup short numpy output format
    np_fmt_short()

    # - add project home to system path
    add_project_to_system_path()

    # show logo first time
    if not hasattr(qubx.QubxMagics, "__already_initialized__"):
        setattr(qubx.QubxMagics, "__already_initialized__", True)
        logo()
