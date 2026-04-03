import json as _json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import stackprinter
from loguru import logger

# - TODO: import some main methods from packages

_COLOR_TAG_RE = re.compile(r"</?[a-z_]+>")
_SKIP_EXTRA_KEYS = frozenset(("user", "end", "stack"))


def runtime_env():
    """
    Check what environment this script is being run under
    :return: environment name, possible values:
             - 'notebook' jupyter notebook
             - 'shell' any interactive shell (ipython, PyCharm's console etc)
             - 'python' standard python interpreter
             - 'unknown' can't recognize environment
    """
    try:
        from IPython.core.getipython import get_ipython

        shell = get_ipython().__class__.__name__

        if shell == "ZMQInteractiveShell":  # Jupyter notebook or qtconsole
            return "notebook"
        elif shell.endswith("TerminalInteractiveShell"):  # Terminal running IPython
            return "shell"
        else:
            return "unknown"  # Other type (?)
    except (NameError, ImportError):
        return "python"  # Probably standard Python interpreter


# ---------------------------------------------------------------------------
#  LogContext — single source of truth for all mutable logging state
# ---------------------------------------------------------------------------


@dataclass
class LogContext:
    """All mutable logging state in one place — explicit, greppable, mockable."""

    bot_id: str | None = None
    instance_id: str | None = None
    phase: str | None = None
    time_provider: Any = None


_log_context = LogContext()


def get_log_context() -> LogContext:
    """Return the global log context (useful for testing / direct access)."""
    return _log_context


# ---------------------------------------------------------------------------
#  Formatting helpers
# ---------------------------------------------------------------------------


def _resolve_provider_time(time_provider) -> str | None:
    """Convert time_provider.time() to a display string, or None on failure."""
    try:
        import pandas as pd

        dt = time_provider.time()
        if isinstance(dt, int):
            return pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return dt.astype("datetime64[us]").item().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except Exception:
        return None


def _resolve_timestamp(fallback="{time:YYYY-MM-DD HH:mm:ss.SSS}") -> str:
    """Return resolved time-provider string, or *fallback* (a loguru placeholder) on failure."""
    tp = _log_context.time_provider
    if tp is not None:
        now = _resolve_provider_time(tp)
        if now is not None:
            return now
    return fallback


def _format_phase_plain(record) -> str:
    """Return plain phase tag from record extras."""
    phase = record["extra"].get("phase")
    if phase == "warmup":
        return "[WARMUP] "
    elif phase == "live":
        return "[LIVE] "
    return ""


def formatter(record):
    """Console formatter — colored, with emoji level icons, no identity."""
    end = record["extra"].get("end", "\n")
    fmt = "<lvl>{message}</lvl>%s" % end
    if record["level"].name in {"WARNING", "SNAKY"}:
        fmt = "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - %s" % fmt

    ts = _resolve_timestamp()
    if ts.startswith("{"):
        ts = f"<green>{ts}</green>"
    else:
        ts = f"<lc>{ts}</lc>"

    phase = record["extra"].get("phase")
    phase_tag = ""
    if phase == "warmup":
        phase_tag = "<yellow>[WARMUP]</yellow> "
    elif phase == "live":
        phase_tag = "<green>[LIVE]</green> "

    prefix = f"{ts} [ <level>{record['level'].icon}</level> ] {phase_tag}<cyan>({{module}})</cyan> "

    if record["exception"] is not None:
        record["extra"]["stack"] = stackprinter.format(record["exception"], style="darkbg3")
        fmt += "\n{extra[stack]}\n"

    if record["level"].name in {"TEXT"}:
        prefix = ""

    return prefix + fmt


def file_formatter(record):
    """Plain-text formatter for file sinks — no colors, no emojis, no identity."""
    # Embed the stripped message as a literal (escape braces so loguru won't parse them)
    clean_msg = _COLOR_TAG_RE.sub("", record["message"]).replace("{", "{{").replace("}", "}}")
    end = record["extra"].get("end", "\n")
    msg = clean_msg + end

    if record["level"].name in {"WARNING", "SNAKY"}:
        msg = "{name}:{function}:{line} - " + msg

    ts = _resolve_timestamp()
    phase = _format_phase_plain(record)
    level_name = record["level"].name
    prefix = ts + " [ " + f"{level_name:<8}" + " ] " + phase + "({module}) "

    if record["exception"] is not None:
        record["extra"]["stack"] = stackprinter.format(record["exception"], style="plaintext")
        msg += "\n{extra[stack]}\n"

    if record["level"].name == "TEXT":
        prefix = ""

    return prefix + msg


# ---------------------------------------------------------------------------
#  JSON sink
# ---------------------------------------------------------------------------


class JsonSink:
    """Emits one JSON line per log record for Loki / Promtail ingestion."""

    def __init__(self, context: LogContext):
        self._ctx = context

    def write(self, message):
        record = message.record
        entry = {
            "timestamp": self._get_timestamp(record),
            "level": record["level"].name,
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": _COLOR_TAG_RE.sub("", record["message"]),
        }
        for key, val in record["extra"].items():
            if key not in _SKIP_EXTRA_KEYS:
                entry[key] = val
        if record["exception"] is not None:
            entry["exception"] = stackprinter.format(record["exception"], style="plaintext")
        sys.stdout.write(_json.dumps(entry, default=str) + "\n")
        sys.stdout.flush()

    def _get_timestamp(self, record) -> str:
        tp = self._ctx.time_provider
        if tp is not None:
            try:
                import pandas as pd

                dt = tp.time()
                if isinstance(dt, int):
                    return pd.Timestamp(dt).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                return dt.astype("datetime64[us]").item().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            except Exception:
                pass
        return record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
#  Patcher — injects bound fields into every log record
# ---------------------------------------------------------------------------


def _patcher(record):
    ctx = _log_context
    if ctx.bot_id:
        record["extra"]["bot_id"] = ctx.bot_id
    if ctx.instance_id:
        record["extra"]["instance_id"] = ctx.instance_id
    if ctx.phase:
        record["extra"]["phase"] = ctx.phase


# ---------------------------------------------------------------------------
#  QubxLogConfig — thin façade for setup / bind operations
# ---------------------------------------------------------------------------


class QubxLogConfig:
    @staticmethod
    def get_log_level():
        env_level = os.getenv("QUBX_LOG_LEVEL")
        if env_level:
            return env_level
        from qubx.config import settings

        return settings.log_level

    @staticmethod
    def set_log_level(level: str):
        os.environ["QUBX_LOG_LEVEL"] = level
        QubxLogConfig.setup_logger(level)

    @staticmethod
    def setup_logger(level: str | None = None, colorize: bool = True):
        logger.remove()
        level = level or QubxLogConfig.get_log_level()

        log_format = os.getenv("QUBX_LOG_FORMAT", "text").lower()
        if log_format == "json":
            sink = JsonSink(_log_context)
            logger.add(sink.write, level=level, enqueue=True, backtrace=True, diagnose=True)
        else:
            logger.add(
                sys.stdout,
                format=formatter,
                colorize=colorize,
                level=level,
                enqueue=True,
                backtrace=True,
                diagnose=True,
            )

        logger.configure(patcher=_patcher)

    @staticmethod
    def bind_platform_identity(bot_id: str | None = None, instance_id: str | None = None):
        """Bind platform identity fields to all log messages globally."""
        _log_context.bot_id = bot_id
        _log_context.instance_id = instance_id

    @staticmethod
    def bind_phase(phase: str | None):
        """Bind phase (warmup/live) to all log messages globally."""
        _log_context.phase = phase

    @staticmethod
    def bind_time_provider(time_provider=None):
        """Bind a time provider for timestamps in both text and JSON logs."""
        _log_context.time_provider = time_provider


QubxLogConfig.setup_logger()


# registering magic for jupyter notebook
if runtime_env() in ["notebook", "shell"]:
    from IPython.core.getipython import get_ipython
    from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class

    try:
        from qubx.utils.charting.mpl_helpers import set_mpl_theme
    except ImportError:
        set_mpl_theme = None  # matplotlib not available (e.g. deployed strategy)

    @magics_class
    class QubxMagics(Magics):
        # process data manager
        __manager = None

        @line_magic
        def qubx(self, line: str):
            self.qubx_setup("dark" + " " + line)

        @line_magic
        def qubxd(self, line: str):
            self.qubx_setup("dark" + " " + line)

        @line_magic
        def qubxl(self, line: str):
            self.qubx_setup("light" + " " + line)

        @line_magic
        def qubx_setup(self, line: str):
            """
            QUBX framework initialization
            """
            import os

            args = [x.strip() for x in line.split(" ")]

            # setup cython dev hooks - only if 'dev' is passed as argument
            if line and "dev" in args:
                from qubx.utils.misc import install_pyx_recompiler_for_dev

                install_pyx_recompiler_for_dev()

            tpl_path = os.path.join(os.path.dirname(__file__), "_nb_magic.py")
            with open(tpl_path, "r", encoding="utf8") as myfile:
                s = myfile.read()

            exec(s, self.shell.user_ns)

            # setup more funcy mpl theme instead of ugly default
            if line and set_mpl_theme is not None:
                if "dark" in line.lower():
                    set_mpl_theme("dark")
                    # - temporary workaround for vscode - dark theme not applying to ipywidgets in notebook
                    # - see https://github.com/microsoft/vscode-jupyter/issues/7161
                    if runtime_env() == "notebook":
                        _vscode_clr_trick = """from IPython.display import display, HTML; display(HTML("<style> .cell-output-ipywidget-background { background-color: transparent !important; } :root { --jp-widgets-color: var(--vscode-editor-foreground); --jp-widgets-font-size: var(--vscode-editor-font-size); } .widget-hprogress, .jupyter-widget-hprogress { height: 16px; align-self: center; kj} table.dataframe, .dataframe td, .dataframe tr { border: 1px solid #55554a85; border-collapse: collapse; color: #859548d9 !important; } .dataframe th { border: 1px solid #55554a85; border-collapse: collapse; background-color: #010101 !important; color: #177 !important; } </style>"))"""
                        exec(_vscode_clr_trick, self.shell.user_ns)

                elif "light" in line.lower():
                    set_mpl_theme("light")

        def _get_manager(self):
            if self.__manager is None:
                import multiprocessing as m

                self.__manager = m.Manager()
            return self.__manager

        @line_cell_magic
        def proc(self, line, cell=None):
            """
            Run cell in separate process

            >>> %%proc x, y as MyProc1
            >>> x.set('Hello')
            >>> y.set([1,2,3,4])

            """
            import multiprocessing as m
            import re
            import time

            # create ext args
            name = None
            if line:
                # check if custom process name was provided
                if " as " in line:
                    line, name = line.split("as")
                    if not name.isspace():
                        name = name.strip()
                    else:
                        print('>>> Process name must be specified afer "as" keyword !')
                        return

                ipy = get_ipython()
                for a in [x for x in re.split(r"[\s,;]", line.strip()) if x]:
                    ipy.push({a: self._get_manager().Value(None, None)})

            # code to run
            lines = "\n".join(["    %s" % x for x in cell.split("\n")])

            def fn():
                result = get_ipython().run_cell(lines)

                # send errors to parent
                if result.error_before_exec:
                    raise result.error_before_exec

                if result.error_in_exec:
                    raise result.error_in_exec

            t_start = str(time.time()).replace(".", "_")
            f_id = f"proc_{t_start}" if name is None else name
            if self._is_task_name_already_used(f_id):
                f_id = f"{f_id}_{t_start}"

            task = m.Process(target=fn, name=f_id)
            task.start()
            print(" -> Task %s is started" % f_id)

        def _is_task_name_already_used(self, name):
            import multiprocessing as m

            for p in m.active_children():
                if p.name == name:
                    return True
            return False

        @line_magic
        def list_proc(self, line):
            import multiprocessing as m

            for p in m.active_children():
                print(p.name)

        @line_magic
        def kill_proc(self, line):
            import multiprocessing as m

            for p in m.active_children():
                if line and p.name.startswith(line):
                    p.terminate()

    # - registering magic here
    get_ipython().register_magics(QubxMagics)  # type: ignore
