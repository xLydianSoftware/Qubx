import re
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from qubx.core.metrics import TradingSessionResult
from qubx.utils.misc import blue, cyan, green, magenta, red, yellow


class BacktestsResultsManager:
    """
    Manager class for handling backtesting results.

    This class provides functionality to load, list and manage backtesting results stored in zip files.
    Each result contains trading session information and metrics that can be loaded and analyzed.

    Parameters
    ----------
    path : str
        Path to directory containing backtesting result zip files

    Methods
    -------
    - reload()
        Reloads all backtesting results from the specified path
    - list(regex="", with_metrics=False)
        Lists all backtesting results, optionally filtered by regex and including metrics
    - load(name)
        Loads a specific backtesting result by name
    - load_config(name)
        Loads the configuration YAML file for a specific backtest result
    - delete(name)
        Deletes one or more backtest results
    """

    def __init__(self, path: str):
        self.path = path
        self.reload()

    def reload(self) -> "BacktestsResultsManager":
        self.results = {}
        self.variations = {}

        _vars = defaultdict(list)
        names = defaultdict(lambda: 0)
        for p in Path(self.path).glob("**/*.zip"):
            with zipfile.ZipFile(p, "r") as zip_ref:
                try:
                    info = yaml.safe_load(zip_ref.read("info.yml"))
                    info["path"] = str(p)
                    n = info.get("name", "")
                    var_set_name = info.get("variation_name", "")

                    # - put variations aside
                    if var_set_name:
                        _vars[var_set_name].append(info)
                        continue

                    _new_name = n if names[n] == 0 else f"{n}.{names[n]}"
                    names[n] += 1
                    info["name"] = _new_name
                    self.results[_new_name] = info
                except Exception:
                    pass

        # - reindex
        _idx = 1
        for n in sorted(self.results.keys()):
            self.results[n]["idx"] = _idx
            _idx += 1

        # - reindex variations at the end
        for n in sorted(_vars.keys()):
            self.variations[_idx] = {
                "name": n,
                "idx": _idx,
                "variations": _vars[n],
                "created": pd.Timestamp(_vars[n][0].get("creation_time", "")).round("1s"),
                "author": _vars[n][0].get("author", ""),
                "description": _vars[n][0].get("description", ""),
            }
            _idx += 1

        return self

    def __getitem__(
        self, name: str | int | list[int] | list[str] | slice
    ) -> TradingSessionResult | list[TradingSessionResult]:
        return self.load(name)

    def load(
        self, name_or_idx: str | int | list[int] | list[str] | slice
    ) -> TradingSessionResult | list[TradingSessionResult]:
        match name_or_idx:
            case list():
                return [self.load(i) for i in name_or_idx]  # type: ignore
            case str():
                return [self.load(i) for i in self._find_indices(name_or_idx)]  # type: ignore
            case slice():
                return [
                    self.load(i)
                    for i in range(name_or_idx.start, name_or_idx.stop, name_or_idx.step if name_or_idx.step else 1)
                ]  # type: ignore
            case int():
                if name_or_idx > len(self.results) and name_or_idx in self.variations:
                    return [
                        TradingSessionResult.from_file(v.get("path", ""))
                        for v in self.variations[name_or_idx].get("variations", [])
                    ]

                # - load by index
                for info in self.results.values():
                    if info.get("idx", -1) == name_or_idx:
                        return TradingSessionResult.from_file(info["path"])

        raise ValueError(f"No result found for '{name_or_idx}' !")

    def load_config(self, name: str | int) -> str:
        """Load the configuration YAML file for a specific backtest result.

        Args:
            name (str | int): The name or index of the backtest result. If str, matches against the backtest name.
                            If int, matches against the backtest index.

        Returns:
            str: The contents of the configuration YAML file as a string.

        Raises:
            ValueError: If no backtest result is found matching the provided name/index.
        """
        p = None
        for info in self.results.values():
            match name:
                case int():
                    if info.get("idx", -1) == name:
                        n = info.get("name", "")
                        p = info.get("path", {})
                        break
                case str():
                    if info.get("name", "") == name:
                        n = info.get("name", "")
                        p = info.get("path", {})
                        break
        if p is None:
            raise ValueError(f"No result found for {name}")

        # - name may have .1, .2, etc. so we need to remove it
        n = n.split(".")[0] if "." in n else n
        with zipfile.ZipFile(p, "r") as zip_ref:
            return zip_ref.read(f"{n}.yaml").decode("utf-8")

    def delete(self, name: str | int | list[int] | list[str] | slice):
        """Delete one or more backtest results.

        Args:
            name: Identifier(s) for the backtest result(s) to delete. Can be:
                - str: Name of backtest or regex pattern to match multiple backtests
                - int: Index of specific backtest
                - list[int]: List of backtest indices
                - list[str]: List of backtest names
                - slice: Range of backtest indices to delete

        Prints:
            Message confirming which backtest(s) were deleted, or error if none found.
            Deleted backtest names are shown in red text.

        Note:
            - For string names, supports regex pattern matching against backtest names and strategy class names
            - Deletes the underlying results files and reloads the results index
            - Operation is irreversible
        """

        def _del_idx(idx):
            for info in self.results.values():
                if info.get("idx", -1) == idx:
                    Path(info["path"]).unlink()
                    return info.get("name", idx)
            return None

        match name:
            case str():
                nms = [_del_idx(i) for i in self._find_indices(name)]
                self.reload()
                print(f" -> Deleted {red(', '.join(nms))} ...")
                return

            case list():
                nms = [_del_idx(i) for i in name]
                self.reload()
                print(f" -> Deleted {red(', '.join(nms))} ...")
                return

            case slice():
                nms = [_del_idx(i) for i in range(name.start, name.stop, name.step if name.step else 1)]
                self.reload()
                print(f" -> Deleted {red(', '.join(nms))} ...")
                return

        for info in self.results.values():
            match name:
                case int():
                    if info.get("idx", -1) == name:
                        Path(info["path"]).unlink()
                        print(f" -> Deleted {red(info.get('name', name))} ...")
                        self.reload()
                        return
                case str():
                    if info.get("name", "") == name:
                        Path(info["path"]).unlink()
                        print(f" -> Deleted {red(info.get('name', name))} ...")
                        self.reload()
                        return
        print(f" -> No results found for {red(name)} !")

    def _find_indices(self, regex: str):
        for n in sorted(self.results.keys()):
            info = self.results[n]
            s_cls = info.get("strategy_class", "").split(".")[-1]

            try:
                if not re.match(regex, n, re.IGNORECASE):
                    # if not re.match(regex, s_cls, re.IGNORECASE):
                    continue
            except Exception:
                if regex.lower() != n.lower() and regex.lower() != s_cls.lower():
                    continue

            yield info.get("idx", -1)

    def list(
        self,
        regex: str = "",
        with_metrics=True,
        params=False,
        as_table=False,
        pretty_print=False,
        sort_by: str | None = "sharpe",
        ascending=False,
        show_variations=True,
    ):
        """List backtesting results with optional filtering and formatting.

        Args:
            - regex (str, optional): Regular expression pattern to filter results by strategy name or class. Defaults to "".
            - with_metrics (bool, optional): Whether to include performance metrics in output. Defaults to True.
            - params (bool, optional): Whether to display strategy parameters. Defaults to False.
            - as_table (bool, optional): Return results as a pandas DataFrame instead of printing. Defaults to False.

        Returns:
            - Optional[pd.DataFrame]: If as_table=True, returns a DataFrame containing the results sorted by creation time.
            - Otherwise prints formatted results to console.
        """
        _t_rep = []
        for n in sorted(self.results.keys()):
            info = self.results[n]
            s_cls = info.get("strategy_class", "").split(".")[-1]

            if regex:
                if not re.match(regex, n, re.IGNORECASE):
                    # if not re.match(regex, s_cls, re.IGNORECASE):
                    continue

            name = info.get("name", "")
            smbs = ", ".join(info.get("symbols", list()))
            start = pd.Timestamp(info.get("start", "")).round("1s")
            stop = pd.Timestamp(info.get("stop", "")).round("1s")
            dscr = info.get("description", "")
            created = pd.Timestamp(info.get("creation_time", "")).round("1s")
            metrics = info.get("performance", {})
            author = info.get("author", "")
            _s = f"{yellow(str(info.get('idx')))} - {red(name)} ::: {magenta(created)} by {cyan(author)}"

            _one_line_dscr = ""
            if dscr:
                dscr = dscr.split("\n")
                for _d in dscr:
                    _s += f"\n\t{magenta('# ' + _d)}"
                    _one_line_dscr += "\u25cf " + _d + "\n"

            _s += f"\n\tstrategy: {green(s_cls)}"
            _s += f"\n\tinterval: {blue(start)} - {blue(stop)}"
            _s += f"\n\tcapital: {blue(info.get('capital', ''))} {info.get('base_currency', '')} ({info.get('commissions', '')})"
            _s += f"\n\tinstruments: {blue(smbs)}"
            if params:
                formats = ["{" + f":<{i}" + "}" for i in [50]]
                _p = pd.DataFrame.from_dict(info.get("parameters", {}), orient="index")
                for i in _p.to_string(
                    max_colwidth=30,
                    header=False,
                    formatters=[(lambda x: cyan(fmt.format(str(x)))) for fmt in formats],
                    justify="left",
                ).split("\n"):
                    _s += f"\n\t  |  {yellow(i)}"

            if not as_table:
                print(_s)

            if with_metrics:
                _m_repr = (
                    pd.DataFrame.from_dict(metrics, orient="index")
                    .T[["gain", "cagr", "sharpe", "qr", "max_dd_pct", "mdd_usd", "fees", "execs"]]
                    .astype(float)
                )
                _m_repr = _m_repr.round(3).to_string(index=False)
                _h, _v = _m_repr.split("\n")
                if not as_table:
                    print("\t " + red(_h))
                    print("\t " + cyan(_v))

            if not as_table:
                print()
            else:
                metrics = {
                    m: round(v, 3)
                    for m, v in metrics.items()
                    if m in ["gain", "cagr", "sharpe", "qr", "max_dd_pct", "mdd_usd", "fees", "execs"]
                }
                _t_rep.append(
                    {"Index": info.get("idx", ""), "Strategy": name}
                    | metrics
                    | {
                        "start": start,
                        "stop": stop,
                        "Created": created,
                        "Author": author,
                        "Description": _one_line_dscr,
                    },
                )

        # - variations (only if not as_table for the time being)
        if not as_table and show_variations:
            for _i, vi in self.variations.items():
                n = vi.get("name", "")
                if regex:
                    if not re.match(regex, n, re.IGNORECASE):
                        continue

                _s = f"{yellow(str(_i))} - {red(str(n))} set of {len(vi.get('variations'))} variations ::: {magenta(vi.get('created'))} by {cyan(vi.get('author'))}"

                dscr = vi.get("description", "").split("\n")
                for _d in dscr:
                    _s += f"\n\t{magenta('# ' + _d)}"

                _mtrx = {}
                for v in vi.get("variations", []):
                    _nm = v.get("name", "")
                    _nm = _nm.split("_")[-1].strip("()")
                    _mtrx[_nm] = v.get("performance", {})

                _m_repr = pd.DataFrame.from_dict(_mtrx, orient="index")[
                    ["gain", "cagr", "sharpe", "qr", "max_dd_pct", "mdd_usd", "fees", "execs"]
                ].astype(float)
                _m_repr = _m_repr.round(3)
                _m_repr = _m_repr.sort_values(by=sort_by, ascending=ascending) if sort_by else _m_repr
                _m_repr = _m_repr.to_string(index=True)

                print(_s)
                for _i, _l in enumerate(_m_repr.split("\n")):
                    if _i == 0:
                        print("\t " + red(_l))
                    else:
                        print("\t " + blue(_l))

        if as_table:
            _df = pd.DataFrame.from_records(_t_rep, index="Index")
            _df = _df.sort_values(by=sort_by, ascending=ascending) if sort_by else _df
            if pretty_print:
                from IPython.display import HTML

                return HTML(
                    _df.to_html()
                    .replace("\\n", "<br><hr style='border-color: #005000; '/>")
                    .replace("<td>", '<td align="left" valign="top">')
                )
            return _df
