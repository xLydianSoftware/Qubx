import asyncio
import concurrent.futures
import getpass
import hashlib
import os
import re
import string
import sys
import time
from collections import OrderedDict, defaultdict, deque, namedtuple
from collections.abc import Callable
from functools import wraps
from os.path import abspath, exists, expanduser, relpath
from pathlib import Path
from threading import Lock
from typing import Any, Awaitable, Union

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def version() -> str:
    # - check current version
    version = "Dev"
    try:
        import importlib_metadata

        version = importlib_metadata.version("qubx")
    except:  # noqa: E722
        pass

    return version


def install_pyx_recompiler_for_dev():
    from ._pyxreloader import pyx_install_loader

    # if version().lower() == 'dev':
    print(f" >  [{green('dev')}] {red('installing cython rebuilding hook')}")
    pyx_install_loader(["qubx.core", "qubx.ta", "qubx.data", "qubx.strategies"])


_QUBX_FLDR = None


def get_local_qubx_folder() -> str:
    global _QUBX_FLDR

    if _QUBX_FLDR is None:
        _QUBX_FLDR = makedirs(os.getenv("QUBXSTORAGE", os.path.expanduser("~/.qubx")))

    return _QUBX_FLDR


def get_current_user() -> str:
    """
    Get current user's username.
    """
    return getpass.getuser()


def this_project_root(path: str = ".") -> Path | None:
    """
    Tries to find current research project root.
    This is convenient when need to get relative paths in notebook in research project.
    """
    _toml = Path("pyproject.toml")
    _x = Path(abspath(expanduser(path)))
    _terminator = str(_x.root)
    while str(_x) != _terminator:
        if (_x / _toml).exists():
            return _x
        _x = _x.parent
    return None


def add_project_to_system_path(project_folder: str = "~/projects"):
    """
    Add path to projects folder to system python path to be able importing any modules from project
    from test.Models.handy_utils import some_module
    """
    # we want to track folders with these files as separate paths
    toml = Path("pyproject.toml")
    src = Path("src")

    try:
        prj = Path(expanduser(project_folder)).resolve()
    except ValueError as e:
        # This error can occur on Windows if user folder and python file are on different drives
        print(f"Qubx> Error during get path to projects folder:\n{e}")
    else:
        insert_path_iff = lambda p: (sys.path.insert(0, p.as_posix()) if p.as_posix() not in sys.path else None)  # noqa: E731
        if prj.exists():
            # If the project folder itself is a Python package, add its parent to path
            if (prj / "__init__.py").exists():
                insert_path_iff(prj.parent)
            else:
                insert_path_iff(prj)

            for di in prj.iterdir():
                _src = di / src
                if (di / toml).exists():
                    # when we have src/
                    if _src.exists() and _src.is_dir():
                        insert_path_iff(_src)
                    else:
                        insert_path_iff(di)
                elif di.is_dir() and ((di / "__init__.py").exists() or any(f.suffix == ".py" for f in di.iterdir() if f.is_file())):
                    # Add parent directory so we can import the strategy package by name
                    insert_path_iff(di.parent)
        else:
            print(f"Qubx> Cant find {project_folder} folder for adding to python path !")


def class_import(name: str):
    """
    Import class by its name.

    For example:
    >>> class_import("qubx.core.data.DataProvider")
    <class 'qubx.core.data.DataProvider'>
    """
    components = name.split(".")
    clz = components[-1]
    mod = __import__(".".join(components[:-1]), fromlist=[clz])
    mod = getattr(mod, clz)
    return mod


def is_localhost(host):
    return host.lower() == "localhost" or host == "127.0.0.1"


def __wrap_with_color(code):
    def inner(text, bold=False):
        c = code
        if bold:
            c = "1;%s" % c
        return "\033[%sm%s\033[0m" % (c, text)

    return inner


red, green, yellow, blue, magenta, cyan, white = (
    __wrap_with_color("31"),
    __wrap_with_color("32"),
    __wrap_with_color("33"),
    __wrap_with_color("34"),
    __wrap_with_color("35"),
    __wrap_with_color("36"),
    __wrap_with_color("37"),
)


def logo():
    """
    Some fancy Qubx logo
    """
    print(
        f"""
⠀⠀⡰⡖⠒⠒⢒⢦⠀⠀   
⠀⢠⠃⠈⢆⣀⣎⣀⣱⡀  {red("QUBX")} | {cyan("Quantitative Backtesting Environment")} 
⠀⢳⠒⠒⡞⠚⡄⠀⡰⠁         (c) 2025, ver. {magenta(version().rstrip())}
⠀⠀⠱⣜⣀⣀⣈⣦⠃⠀⠀⠀ 
        """
    )


class Struct:
    """
    Dynamic structure (similar to matlab's struct it allows to add new properties dynamically)

    >>> a = Struct(x=1, y=2)
    >>> a.z = 'Hello'
    >>> print(a)

    Struct(x=1, y=2, z='Hello')

    >>> Struct(a=234, b=Struct(c=222)).to_dict()

    {'a': 234, 'b': {'c': 222}}

    >>> Struct({'a': 555}, a=123, b=Struct(c=222)).to_dict()

    {'a': 123, 'b': {'c': 222}}
    """

    def __init__(self, *args, **kwargs):
        _odw = OrderedDict(**kwargs)
        if args:
            if isinstance(args[0], dict):
                _odw = OrderedDict(Struct.dict2struct(args[0]).to_dict()) | _odw
            elif isinstance(args[0], Struct):
                _odw = args[0].to_dict() | _odw
        self.__initialize(_odw.keys(), _odw.values())

    def __initialize(self, fields, values):
        self._fields = list(fields)
        self._meta = namedtuple("Struct", " ".join(fields))
        self._inst = self._meta(*values)

    def fields(self) -> list:
        return self._fields

    def __getitem__(self, idx: int):
        return getattr(self._inst, self._fields[idx])

    def __getattr__(self, k):
        return getattr(self._inst, k)

    def __or__(self, other: Union[dict, "Struct"]):
        if isinstance(other, dict):
            other = Struct.dict2struct(other)
        elif not isinstance(other, Struct):
            raise ValueError(f"Can't union with object of {type(other)} type ")
        for f in other.fields():
            self.__setattr__(f, other.__getattr__(f))
        return self

    def __dir__(self):
        return self._fields

    def __repr__(self):
        return self._inst.__repr__()

    def __setattr__(self, k, v):
        if k not in ["_inst", "_meta", "_fields"]:
            new_vals = {**self._inst._asdict(), **{k: v}}
            self.__initialize(new_vals.keys(), new_vals.values())
        else:
            super().__setattr__(k, v)

    def __getstate__(self):
        return self._inst._asdict()

    def __setstate__(self, state):
        self.__init__(**state)

    def __ms2d(self, m) -> dict:
        r = {}
        for f in m._fields:
            v = m.__getattr__(f)
            r[f] = self.__ms2d(v) if isinstance(v, Struct) else v
        return r

    def to_dict(self) -> dict:
        """
        Return this structure as dictionary
        """
        return self.__ms2d(self)

    def copy(self) -> "Struct":
        """
        Returns copy of this structure
        """
        return Struct(self.to_dict())

    @staticmethod
    def dict2struct(d: dict) -> "Struct":
        """
        Convert dictionary to structure
        >>> s = dict2struct({'f_1_0': 1, 'z': {'x': 1, 'y': 2}})
        >>> print(s.z.x)
        1
        """
        m = Struct()
        for k, v in d.items():
            # skip if key is not valid identifier
            if not k.isidentifier():
                print(f"Struct> {k} doesn't look like as identifier - skip it")
                continue
            if isinstance(v, dict):
                v = Struct.dict2struct(v)
            m.__setattr__(k, v)
        return m


def makedirs(path: str, *args) -> str:
    path = os.path.expanduser(os.path.join(*[path, *args]))
    if not exists(path):
        os.makedirs(path, exist_ok=True)
    return path


class Stopwatch:
    """
    Stopwatch timer for performance
    """

    starts: dict[str | None, int] = {}
    counts: dict[str | None, int] = defaultdict(lambda: 0)
    latencies: dict[str | None, int] = {}
    _current_scope: str | None = None

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Stopwatch, cls).__new__(cls)
        return cls.instance

    def start(self, scope: str | None):
        self.starts[scope] = time.perf_counter_ns()
        self.counts[scope] += 1

    def stop(self, scope: str | None = None) -> int | None:
        t = time.perf_counter_ns()
        s = self.starts.get(scope, None)
        lat = None
        if s:
            lat = t - s
            n = self.counts[scope]
            self.latencies[scope] = (self.latencies.get(scope, lat) * (n - 1) + lat) // n
            del self.starts[scope]
        return lat

    def latency_sec(self, scope: str | None) -> float:
        return self.latencies.get(scope, 0) / 1e9

    def watch(self, scope="global"):
        def _decorator(func):
            info = scope + "." + func.__name__

            def wrapper(*args, **kwargs):
                self.start(info)
                output = func(*args, **kwargs)
                self.stop(info)
                return output

            return wrapper

        return _decorator

    def reset(self):
        self.starts.clear()
        self.counts.clear()
        self.latencies.clear()

    def __str__(self) -> str:
        r = ""
        for l in self.latencies.keys():
            r += f"\n\t<w>{l}</w> took <r>{self.latency_sec(l):.7f}</r> secs"
        return r

    def __enter__(self):
        self.start(self._current_scope)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(self._current_scope)

    def __call__(self, scope: str | None = "global"):
        self._current_scope = scope
        return self

    @classmethod
    def latency_report(cls) -> pd.DataFrame | None:
        if not hasattr(cls, "instance"):
            return None
        sw = cls.instance
        scope_to_latency_sec = {scope: sw.latency_sec(scope) for scope in sw.latencies.keys()}
        scope_to_count = {l: sw.counts[l] for l in scope_to_latency_sec.keys()}
        scope_to_total_time = {scope: scope_to_count[scope] * lat for scope, lat in scope_to_latency_sec.items()}
        # create pandas datafrmae from dictionaries
        lats = pd.DataFrame(
            {
                "scope": list(scope_to_latency_sec.keys()),
                "latency": list(scope_to_latency_sec.values()),
                "count": list(scope_to_count.values()),
                "total_time": list(scope_to_total_time.values()),
            }
        )
        lats["latency"] = lats["latency"].apply(lambda x: f"{x:.4f}")
        lats["total_time (min)"] = lats["total_time"].apply(lambda x: f"{x / 60:.4f}")
        lats.drop(columns=["total_time"], inplace=True)
        return lats


def quotify(sx: str | list[str], quote="USDT"):
    """
    Make XXX<quote> from anything if that anything doesn't end with <quote>
    """
    if isinstance(sx, str):
        return (sx if sx.endswith(quote) else sx + quote).upper()
    elif isinstance(sx, (list, set, tuple)):
        return [quotify(s, quote) for s in sx]
    raise ValueError("Can't process input data !")


def dequotify(sx: str | list[str], quote="USDT"):
    """
    Turns XXX<quote> to XXX (reverse of quotify)
    """
    if isinstance(sx, str):
        quote = quote.upper()
        if (s := sx.upper()).endswith(quote):
            s = s.split(":")[1] if ":" in s else s  # remove exch: if presented
            return s.split(quote)[0]
    elif isinstance(sx, (list, set, tuple)):
        return [dequotify(s, quote) for s in sx]

    raise ValueError("Can't process input data !")


class ProgressParallel(joblib.Parallel):
    def __init__(self, *args, **kwargs):
        self.total = kwargs.pop("total", None)
        self.silent = kwargs.pop("silent", False)
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.silent:
            return joblib.Parallel.__call__(self, *args, **kwargs)
        with tqdm(total=self.total) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self.silent:
            return
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class AsyncThreadLoop:
    """
    Helper class to submit coroutines to asyncio loop from separate thread.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def submit(self, coro: Awaitable) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


def synchronized(func: Callable):
    """Decorator that ensures only one thread can execute the decorated function at a time."""
    lock = Lock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)

    return wrapper


class TimeLimitedDeque(deque):
    """
    A deque that removes elements older than a given time limit.
    Assumes that elements are inserted in increasing order of time.
    """

    def __init__(self, time_limit: str, time_key=lambda x: x[0], unit="ns", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_limit = pd.Timedelta(time_limit).to_timedelta64()
        self.unit = unit
        self.time_key = lambda x: self._to_datetime64(time_key(x))

    def append(self, item):
        super().append(item)
        self._remove_old_elements()

    def __getitem__(self, idx) -> list[Any]:
        if isinstance(idx, slice) and (isinstance(idx.start, str) or isinstance(idx.stop, str)):
            start_loc, end_loc = 0, len(self)
            if idx.start is not None:
                start = self._to_datetime64(idx.start)
                while start_loc < len(self) and self.time_key(self[start_loc]) < start:
                    start_loc += 1
            if idx.stop is not None:
                stop = self._to_datetime64(idx.stop)
                while end_loc > 0 and self.time_key(self[end_loc - 1]) > stop:
                    end_loc -= 1
            return list(self)[start_loc:end_loc]
        else:
            return super().__getitem__(idx)

    def appendleft(self, item):
        raise NotImplementedError("appendleft is not supported for TimeLimitedDeque")

    def extendleft(self, items):
        raise NotImplementedError("extendleft is not supported for TimeLimitedDeque")

    def _remove_old_elements(self):
        if not self:
            return
        current_time = self.time_key(self[-1])
        while self and (current_time - self.time_key(self[0])) > self.time_limit:
            self.popleft()

    def _to_datetime64(self, time):
        return np.datetime64(time, self.unit)


__VOWS = "aeiou"
__CONS = "".join(sorted(set(string.ascii_lowercase) - set(__VOWS)))


def generate_name(content: Any, n1, ns=0) -> str:
    """
    Generates short unique name for given content.

    >>> print(generate_name("Qubix Trading Platform, (c) 2025", 8))
    >>> 'Pojituke'
    """
    __NV, __NC = len(__VOWS), len(__CONS)
    hdg = hashlib.sha256(str(content).encode("utf-8")).hexdigest().upper()
    w = ""
    for i, x in enumerate(hdg[ns : n1 + ns]):
        if i % 2 == 0:
            w += __CONS[int(x, 16) % __NC]
        else:
            w += __VOWS[int(x, 16) % __NV]
    return w[0].upper() + w[1:]


def string_shortener(s: str) -> str:
    """
    Removes all vovels and squeeze repeating symbols
    >>> print(string_shortener("QubxAssetManager"))
    >>> 'QbxAstMngr'
    """
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiou]", "", s))


def load_qubx_resources_as_json(path: Path | str) -> list[dict]:
    """
    Reads a JSON file from resource module
    """
    import importlib.resources
    import json

    if isinstance(path, str):
        path = Path(path)

    if path.suffix != ".json":
        path = path.with_suffix(path.suffix + ".json")

    data = []
    try:
        res_path = importlib.resources.files("qubx.resources") / path  # type: ignore
        with res_path.open() as f:
            data = json.load(f)
    except Exception as e:
        raise e

    return data


def load_qubx_resources_as_text(path: Path | str) -> str:
    """
    Reads a text file from resource module
    """
    import importlib.resources

    if isinstance(path, str):
        path = Path(path)

    try:
        res_path = importlib.resources.files("qubx.resources") / path  # type: ignore
        with res_path.open() as f:
            data = f.read()
    except Exception as e:
        raise e

    return data
