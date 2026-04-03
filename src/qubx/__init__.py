import json as _json
import os
import sys
from typing import Callable

import stackprinter
from loguru import logger

# - TODO: import some main methods from packages


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


def format_platform_identity(record) -> str:
    """Return a colored identity prefix from record extras (bot_id / instance_id)."""
    bot_id = record["extra"].get("bot_id")
    instance_id = record["extra"].get("instance_id")
    if bot_id or instance_id:
        parts = []
        if bot_id:
            parts.append(f"bot={bot_id}")
        if instance_id:
            parts.append(f"inst={instance_id}")
        return "<magenta>[%s]</magenta> " % " ".join(parts)
    return ""


def format_phase(record) -> str:
    """Return a colored phase tag from record extras."""
    phase = record["extra"].get("phase")
    if phase == "warmup":
        return "<yellow>[WARMUP]</yellow> "
    elif phase == "live":
        return "<green>[LIVE]</green> "
    return ""


def formatter(record):
    end = record["extra"].get("end", "\n")
    fmt = "<lvl>{message}</lvl>%s" % end
    if record["level"].name in {"WARNING", "SNAKY"}:
        fmt = "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - %s" % fmt

    identity = format_platform_identity(record)
    phase = format_phase(record)
    prefix = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [ <level>%s</level> ] %s%s<cyan>({module})</cyan> "
        % (record["level"].icon, identity, phase)
    )

    if record["exception"] is not None:
        # stackprinter.set_excepthook(style='darkbg2')
        record["extra"]["stack"] = stackprinter.format(record["exception"], style="darkbg3")
        fmt += "\n{extra[stack]}\n"

    if record["level"].name in {"TEXT"}:
        prefix = ""

    return prefix + fmt


class QubxLogConfig:
    @staticmethod
    def get_log_level():
        # Env var takes priority (for CLI --log-level override), then settings
        env_level = os.getenv("QUBX_LOG_LEVEL")
        if env_level:
            return env_level
        from qubx.config import settings

        return settings.log_level

    @staticmethod
    def set_log_level(level: str):
        os.environ["QUBX_LOG_LEVEL"] = level
        QubxLogConfig.setup_logger(level)

    _COLOR_TAG_RE = None

    @staticmethod
    def _strip_color_tags(text: str) -> str:
        """Remove loguru color markup tags like <yellow>...</yellow> from text."""
        import re

        if QubxLogConfig._COLOR_TAG_RE is None:
            QubxLogConfig._COLOR_TAG_RE = re.compile(r"</?[a-z_]+>")
        return QubxLogConfig._COLOR_TAG_RE.sub("", text)

    @staticmethod
    def _json_sink(message):
        """Emit one JSON line per log record for Loki/Promtail ingestion."""
        record = message.record
        entry = {
            "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record["level"].name,
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": QubxLogConfig._strip_color_tags(record["message"]),
        }
        # Merge platform identity and any other extras (skip internal loguru keys)
        for key, val in record["extra"].items():
            if key not in ("user", "end", "stack"):
                entry[key] = val
        if record["exception"] is not None:
            entry["exception"] = stackprinter.format(record["exception"], style="plaintext")
        sys.stdout.write(_json.dumps(entry, default=str) + "\n")
        sys.stdout.flush()

    @staticmethod
    def setup_logger(level: str | None = None, custom_formatter: Callable | None = None, colorize: bool = True):
        global logger

        config = {
            "handlers": [
                {"sink": sys.stdout, "format": "{time} - {message}"},
            ],
            "extra": {"user": "someone"},
        }
        logger.configure(**config)
        logger.remove(None)

        level = level or QubxLogConfig.get_log_level()

        # Check if JSON format is requested (for Loki/container deployments)
        log_format = os.getenv("QUBX_LOG_FORMAT", "text").lower()
        if log_format == "json":
            logger.add(
                QubxLogConfig._json_sink,
                level=level,
                enqueue=True,
                backtrace=True,
                diagnose=True,
            )
            # No colorize opt needed for JSON
            return

        # Default: human-readable text format
        logger.add(
            sys.stdout,
            format=custom_formatter or formatter,
            colorize=colorize,
            level=level,
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
        logger = logger.opt(colors=colorize)

    _bot_id: str | None = None
    _instance_id: str | None = None
    _phase: str | None = None

    @staticmethod
    def _update_patcher():
        """Reconfigure loguru with a single patcher that applies all bound fields."""
        def patcher(record):
            if QubxLogConfig._bot_id:
                record["extra"]["bot_id"] = QubxLogConfig._bot_id
            if QubxLogConfig._instance_id:
                record["extra"]["instance_id"] = QubxLogConfig._instance_id
            if QubxLogConfig._phase:
                record["extra"]["phase"] = QubxLogConfig._phase

        logger.configure(patcher=patcher)

    @staticmethod
    def bind_platform_identity(bot_id: str | None = None, instance_id: str | None = None):
        """Bind platform identity fields (bot_id, instance_id) to all log messages globally."""
        QubxLogConfig._bot_id = bot_id
        QubxLogConfig._instance_id = instance_id
        QubxLogConfig._update_patcher()

    @staticmethod
    def bind_phase(phase: str | None):
        """Bind phase (warmup/live) to all log messages globally."""
        QubxLogConfig._phase = phase
        QubxLogConfig._update_patcher()


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
