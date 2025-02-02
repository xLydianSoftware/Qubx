import re
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from qubx.core.metrics import TradingSessionResult, _pfl_metrics_prepare
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
    reload()
        Reloads all backtesting results from the specified path
    list(regex="", with_metrics=False)
        Lists all backtesting results, optionally filtered by regex and including metrics
    load(name)
        Loads a specific backtesting result by name
    """

    def __init__(self, path: str):
        self.path = path
        self.reload()

    def reload(self) -> "BacktestsResultsManager":
        self.results = {}
        names = defaultdict(lambda: 0)
        for p in Path(self.path).glob("**/*.zip"):
            with zipfile.ZipFile(p, "r") as zip_ref:
                try:
                    info = yaml.safe_load(zip_ref.read("info.yml"))
                    info["path"] = str(p)
                    n = info.get("name", "")
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

        return self

    def __getitem__(
        self, name: str | int | list[int] | list[str] | slice
    ) -> TradingSessionResult | list[TradingSessionResult]:
        return self.load(name)

    def load(
        self, name: str | int | list[int] | list[str] | slice
    ) -> TradingSessionResult | list[TradingSessionResult]:
        match name:
            case list():
                return [self.load(i) for i in name]
            case slice():
                return [self.load(i) for i in range(name.start, name.stop, name.step if name.step else 1)]

        for info in self.results.values():
            match name:
                case int():
                    if info.get("idx", -1) == name:
                        return TradingSessionResult.from_file(info["path"])
                case str():
                    if info.get("name", "") == name:
                        return TradingSessionResult.from_file(info["path"])
        raise ValueError(f"No result found for {name}")

    def delete(self, name: str | int | list[int] | list[str] | slice):
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
                    if not re.match(regex, s_cls, re.IGNORECASE):
                        continue
            except Exception:
                if regex.lower() != n.lower() and regex.lower() != s_cls.lower():
                    continue

            yield info.get("idx", -1)

    def list(self, regex: str = "", with_metrics=False, params=False):
        for n in sorted(self.results.keys()):
            info = self.results[n]
            s_cls = info.get("strategy_class", "").split(".")[-1]

            if regex:
                if not re.match(regex, n, re.IGNORECASE):
                    if not re.match(regex, s_cls, re.IGNORECASE):
                        continue

            name = info.get("name", "")
            smbs = ", ".join(info.get("symbols", list()))
            start = pd.Timestamp(info.get("start", "")).round("1s")
            stop = pd.Timestamp(info.get("stop", "")).round("1s")
            dscr = info.get("description", "")
            _s = f"{yellow(str(info.get('idx')))} - {red(name)} ::: {magenta(pd.Timestamp(info.get('creation_time', '')).round('1s'))} by {cyan(info.get('author', ''))}"

            if dscr:
                dscr = dscr.split("\n")
                for _d in dscr:
                    _s += f"\n\t{magenta('# ' + _d)}"

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
            print(_s)

            if with_metrics:
                r = TradingSessionResult.from_file(info["path"])
                metric = _pfl_metrics_prepare(r, True, 365)
                _m_repr = str(metric[0][["Gain", "Cagr", "Sharpe", "Max dd pct", "Qr", "Fees"]].round(3)).split("\n")[
                    :-1
                ]
                for i in _m_repr:
                    print("\t " + cyan(i))
            print()
