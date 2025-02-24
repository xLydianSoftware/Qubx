import datetime
import itertools
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import toml
from Cython.Build import build_ext, cythonize
from Cython.Compiler import Options
from Cython.Compiler.Version import version as cython_compiler_version
from setuptools import Distribution, Extension

RED, BLUE, GREEN, YLW, RES = "\033[31m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"


BUILD_MODE = os.getenv("BUILD_MODE", "release")
PROFILE_MODE = bool(os.getenv("PROFILE_MODE", ""))
ANNOTATION_MODE = bool(os.getenv("ANNOTATION_MODE", ""))
BUILD_DIR = "build/optimized"
COPY_TO_SOURCE = os.getenv("COPY_TO_SOURCE", "true") == "true"
PYO3_ONLY = os.getenv("PYO3_ONLY", "false") == "true"  # Skip Cython compilation if true

################################################################################
#  CYTHON BUILD
################################################################################
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html

Options.docstrings = True  # Include docstrings in modules
Options.fast_fail = True  # Abort the compilation on the first error occurred
Options.annotate = ANNOTATION_MODE  # Create annotated HTML files for each .pyx
if ANNOTATION_MODE:
    Options.annotate_coverage_xml = "coverage.xml"
Options.fast_fail = True  # Abort compilation on first error
Options.warning_errors = True  # Treat compiler warnings as errors
Options.extra_warnings = True

CYTHON_COMPILER_DIRECTIVES = {
    "language_level": "3",
    "cdivision": True,  # If division is as per C with no check for zero (35% speed up)
    "nonecheck": True,  # Insert extra check for field access on C extensions
    "embedsignature": True,  # If docstrings should be embedded into C signatures
    "profile": PROFILE_MODE,  # If we're debugging or profiling
    "linetrace": PROFILE_MODE,  # If we're debugging or profiling
    "warn.maybe_uninitialized": True,
}


def _build_extensions() -> list[Extension]:
    # Regarding the compiler warning: #warning "Using deprecated NumPy API,
    # disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
    # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api
    # From the Cython docs: "For the time being, it is just a warning that you can ignore."
    define_macros: list[tuple[str, str | None]] = [
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    ]
    if PROFILE_MODE or ANNOTATION_MODE:
        # Profiling requires special macro directives
        define_macros.append(("CYTHON_TRACE", "1"))

    extra_compile_args = []
    extra_link_args = []
    # extra_link_args = RUST_LIBS

    if platform.system() != "Windows":
        # Suppress warnings produced by Cython boilerplate
        extra_compile_args.append("-Wno-unreachable-code")
        if BUILD_MODE == "release":
            extra_compile_args.append("-O2")
            extra_compile_args.append("-pipe")

    if platform.system() == "Windows":
        extra_link_args += [
            "AdvAPI32.Lib",
            "bcrypt.lib",
            "Crypt32.lib",
            "Iphlpapi.lib",
            "Kernel32.lib",
            "ncrypt.lib",
            "Netapi32.lib",
            "ntdll.lib",
            "Ole32.lib",
            "OleAut32.lib",
            "Pdh.lib",
            "PowrProf.lib",
            "Psapi.lib",
            "schannel.lib",
            "secur32.lib",
            "Shell32.lib",
            "User32.Lib",
            "UserEnv.Lib",
            "WS2_32.Lib",
        ]

    print(f">> {GREEN} Creating C extension modules...{RES}")
    print(f"define_macros={define_macros}")
    print(f"extra_compile_args={extra_compile_args}")

    return [
        Extension(
            name=str(pyx.relative_to("src")).replace(os.path.sep, ".")[:-4],
            sources=[str(pyx)],
            include_dirs=[np.get_include()],  # , *RUST_INCLUDES],
            define_macros=define_macros,
            language="c",
            extra_link_args=extra_link_args,
            extra_compile_args=extra_compile_args,
        )
        for pyx in itertools.chain(Path("src/qubx").rglob("*.pyx"))
    ]


def _build_distribution(extensions: list[Extension]) -> Distribution:
    nthreads = os.cpu_count() or 1
    if platform.system() == "Windows":
        nthreads = min(nthreads, 60)
    print(f"nthreads={nthreads}")

    distribution = Distribution(
        {
            "name": "qubx",
            "ext_modules": cythonize(
                module_list=extensions,
                compiler_directives=CYTHON_COMPILER_DIRECTIVES,
                nthreads=nthreads,
                build_dir=BUILD_DIR,
                gdb_debug=PROFILE_MODE,
            ),
            "zip_safe": False,
        },
    )
    return distribution


def _copy_build_dir_to_project(cmd: build_ext) -> None:
    # Copy built extensions back to the project tree
    for output in cmd.get_outputs():
        relative_extension = Path("src") / Path(output).relative_to(cmd.build_lib)
        if not Path(output).exists():
            continue

        # Copy the file and set permissions
        shutil.copyfile(output, relative_extension)
        mode = relative_extension.stat().st_mode
        mode |= (mode & 0o444) >> 2
        relative_extension.chmod(mode)

    print(f" >> {GREEN}Copied all compiled dynamic library files into source{RES}")


def _strip_unneeded_symbols() -> None:
    try:
        print(f" >> {YLW}Stripping unneeded symbols from binaries...{RES}")
        for so in itertools.chain(Path("src/qubx").rglob("*.so")):
            if platform.system() == "Linux":
                strip_cmd = ["strip", "--strip-unneeded", so]
            elif platform.system() == "Darwin":
                strip_cmd = ["strip", "-x", so]
            else:
                raise RuntimeError(f"Cannot strip symbols for platform {platform.system()}")
            subprocess.run(
                strip_cmd,  # type: ignore [arg-type] # noqa
                check=True,
                capture_output=True,
            )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error when stripping symbols.\n{e}") from e


def build() -> None:
    """
    Construct the extensions and distribution.
    """
    # _build_rust_libs()
    # _copy_rust_dylibs_to_project()

    if not PYO3_ONLY:  # Only build Cython extensions if PYO3_ONLY is false
        # Create C Extensions to feed into cythonize()
        extensions = _build_extensions()
        distribution = _build_distribution(extensions)

        # Build and run the command
        print(f">> {GREEN} Compiling C extension modules...{RES}")
        cmd: build_ext = build_ext(distribution)
        # if PARALLEL_BUILD:
        # cmd.parallel = os.cpu_count()
        cmd.ensure_finalized()
        cmd.run()

        if COPY_TO_SOURCE:
            # Copy the build back into the source tree for development and wheel packaging
            _copy_build_dir_to_project(cmd)

    if BUILD_MODE == "release" and platform.system() in ("Linux", "Darwin"):
        # Only strip symbols for release builds
        _strip_unneeded_symbols()


if __name__ == "__main__":
    qubx_platform = toml.load("pyproject.toml")["tool"]["poetry"]["version"]
    print(BLUE)
    print("=====================================================================")
    print(f"Qubx Builder {qubx_platform}")
    print("=====================================================================\033[0m")
    print(f"System: {GREEN}{platform.system()} {platform.machine()}{RES}")
    # print(f"Clang:  {GREEN}{_get_clang_version()}{RES}")
    # print(f"Rust:   {GREEN}{_get_rustc_version()}{RES}")
    print(f"Python: {GREEN}{platform.python_version()}{RES}")
    print(f"Cython: {GREEN}{cython_compiler_version}{RES}")
    print(f"NumPy:  {GREEN}{np.__version__}{RES}\n")

    print(f"BUILD_MODE={BUILD_MODE}")
    print(f"BUILD_DIR={BUILD_DIR}")
    print(f"PROFILE_MODE={PROFILE_MODE}")
    print(f"ANNOTATION_MODE={ANNOTATION_MODE}")
    print(f"PYO3_ONLY={PYO3_ONLY}")
    print(f"COPY_TO_SOURCE={COPY_TO_SOURCE}")

    print(f">> {GREEN}Starting build...{RES}")
    ts_start = datetime.datetime.now(datetime.timezone.utc)
    build()
    print(f"Build time: {YLW}{datetime.datetime.now(datetime.timezone.utc) - ts_start}{RES}")
    print(GREEN + "Build completed" + RES)

# # See if Cython is installed
# try:
#     from Cython.Build import cythonize
# # Do nothing if Cython is not available
# except ImportError:
#     # Got to provide this function. Otherwise, poetry will fail
#     def build(setup_kwargs):
#         pass

# # Cython is installed. Compile
# else:
#     from setuptools import Extension
#     from setuptools.dist import Distribution
#     from distutils.command.build_ext import build_ext

#     # This function will be executed in setup.py:
#     def build(setup_kwargs):
#         # The file you want to compile
#         extensions = [
#             "src/qubx/core/series.pyx",
#             "src/qubx/core/utils.pyx",
#             "src/qubx/ta/indicators.pyx",
#         ]

#         # gcc arguments hack: enable optimizations
#         os.environ['CFLAGS'] = '-O3'

#         # Build
#         import numpy as np
#         setup_kwargs.update({
#             'ext_modules': cythonize(
#                 extensions,
#                 language_level=3,
#                 compiler_directives={'linetrace': True},
#                 include_path=[np.get_include()]
#             ),
#             'cmdclass': {'build_ext': build_ext}
#         })
