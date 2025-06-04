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


def formatter(record):
    end = record["extra"].get("end", "\n")
    fmt = "<lvl>{message}</lvl>%s" % end
    if record["level"].name in {"WARNING", "SNAKY"}:
        fmt = "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - %s" % fmt

    prefix = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [ <level>%s</level> ] <cyan>({module})</cyan> "
        % record["level"].icon
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
        return os.getenv("QUBX_LOG_LEVEL", "WARNING")

    @staticmethod
    def set_log_level(level: str):
        os.environ["QUBX_LOG_LEVEL"] = level
        QubxLogConfig.setup_logger(level)

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
        # Add stdout handler with enqueue=True for thread/process safety
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


QubxLogConfig.setup_logger()


# registering magic for jupyter notebook
if runtime_env() in ["notebook", "shell"]:
    from IPython.core.getipython import get_ipython
    from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class

    from qubx.utils.charting.mpl_helpers import set_mpl_theme

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
            if line:
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
                for a in [x for x in re.split(r"[\ ,;]", line.strip()) if x]:
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
