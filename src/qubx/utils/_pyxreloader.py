import glob
import importlib
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ExtensionFileLoader, SourceFileLoader
from importlib.util import spec_from_file_location
from typing import List

PYX_EXT = ".pyx"
PYXDEP_EXT = ".pyxdep"
PYXBLD_EXT = ".pyxbld"


def handle_dependencies(pyxfilename):
    testing = "_test_files" in globals()
    dependfile = os.path.splitext(pyxfilename)[0] + PYXDEP_EXT

    # by default let distutils decide whether to rebuild on its own
    # (it has a better idea of what the output file will be)

    # but we know more about dependencies so force a rebuild if
    # some of the dependencies are newer than the pyxfile.
    if os.path.exists(dependfile):
        with open(dependfile) as fid:
            depends = fid.readlines()
        depends = [depend.strip() for depend in depends]

        # gather dependencies in the "files" variable
        # the dependency file is itself a dependency
        files = [dependfile]
        for depend in depends:
            fullpath = os.path.join(os.path.dirname(dependfile), depend)
            files.extend(glob.glob(fullpath))

        # if any file that the pyxfile depends upon is newer than
        # the pyx file, 'touch' the pyx file so that distutils will
        # be tricked into rebuilding it.
        for file in files:
            from distutils.dep_util import newer

            if newer(file, pyxfilename):
                print("Rebuilding %s because of %s", pyxfilename, file)
                filetime = os.path.getmtime(file)
                os.utime(pyxfilename, (filetime, filetime))


def handle_special_build(modname, pyxfilename):
    try:
        import imp
    except:
        return None, None
    special_build = os.path.splitext(pyxfilename)[0] + PYXBLD_EXT
    ext = None
    setup_args = {}
    if os.path.exists(special_build):
        # globls = {}
        # locs = {}
        # execfile(special_build, globls, locs)
        # ext = locs["make_ext"](modname, pyxfilename)
        with open(special_build) as fid:
            mod = imp.load_source("XXXX", special_build, fid)
        make_ext = getattr(mod, "make_ext", None)
        if make_ext:
            ext = make_ext(modname, pyxfilename)
            assert ext and ext.sources, "make_ext in %s did not return Extension" % special_build
        make_setup_args = getattr(mod, "make_setup_args", None)
        if make_setup_args:
            setup_args = make_setup_args()
            assert isinstance(setup_args, dict), "make_setup_args in %s did not return a dict" % special_build
        assert set or setup_args, "neither make_ext nor make_setup_args %s" % special_build
        ext.sources = [os.path.join(os.path.dirname(special_build), source) for source in ext.sources]
    return ext, setup_args


def get_distutils_extension(modname, pyxfilename, language_level=None):
    extension_mod, setup_args = handle_special_build(modname, pyxfilename)
    if not extension_mod:
        if not isinstance(pyxfilename, str):
            # distutils is stupid in Py2 and requires exactly 'str'
            # => encode accidentally coerced unicode strings back to str
            pyxfilename = pyxfilename.encode(sys.getfilesystemencoding())
        from distutils.extension import Extension

        extension_mod = Extension(name=modname, sources=[pyxfilename])
        if language_level is not None:
            extension_mod.cython_directives = {"language_level": language_level}
    return extension_mod, setup_args


def build_module(
    name,
    pyxfilename,
    user_setup_args,
    pyxbuild_dir=None,
    inplace=False,
    language_level=None,
    build_in_temp=False,
    reload_support=True,
):
    assert os.path.exists(pyxfilename), "Path does not exist: %s" % pyxfilename
    handle_dependencies(pyxfilename)

    extension_mod, setup_args = get_distutils_extension(name, pyxfilename, language_level)
    build_in_temp = True
    sargs = user_setup_args.copy() if user_setup_args else dict()
    sargs.update(setup_args)
    build_in_temp = sargs.pop("build_in_temp", build_in_temp)

    from pyximport import pyxbuild

    olddir = os.getcwd()
    common = ""
    if pyxbuild_dir:
        # Windows concatenates the pyxbuild_dir to the pyxfilename when
        # compiling, and then complains that the filename is too long
        common = os.path.commonprefix([pyxbuild_dir, pyxfilename])
    if len(common) > 30:
        pyxfilename = os.path.relpath(pyxfilename)
        pyxbuild_dir = os.path.relpath(pyxbuild_dir)
        os.chdir(common)
    try:
        so_path = pyxbuild.pyx_to_dll(
            pyxfilename,
            extension_mod,
            force_rebuild=1,
            build_in_temp=build_in_temp,
            pyxbuild_dir=pyxbuild_dir,
            setup_args=sargs,
            inplace=inplace,
            reload_support=reload_support,
        )
    finally:
        os.chdir(olddir)
    so_path = os.path.join(common, so_path)
    assert os.path.exists(so_path), "Cannot find: %s" % so_path

    junkpath = os.path.join(
        os.path.dirname(so_path), name + "_*"
    )  # very dangerous with --inplace ? yes, indeed, trying to eat my files ;)
    junkstuff = glob.glob(junkpath)
    for path in junkstuff:
        if path != so_path:
            try:
                os.remove(path)
            except IOError:
                print("Couldn't remove %s", path)

    return so_path


def load_module(
    name, pyxfilename, pyxbuild_dir=None, is_package=False, build_inplace=False, language_level=None, so_path=None
):
    try:
        import imp
    except:
        return None
    try:
        if so_path is None:
            if is_package:
                module_name = name + ".__init__"
            else:
                module_name = name
            so_path = build_module(
                module_name, pyxfilename, pyxbuild_dir, inplace=build_inplace, language_level=language_level
            )
        mod = imp.load_dynamic(name, so_path)
        if is_package and not hasattr(mod, "__path__"):
            mod.__path__ = [os.path.dirname(so_path)]
        assert mod.__file__ == so_path, (mod.__file__, so_path)
    except Exception as failure_exc:
        print("Failed to load extension module: %r" % failure_exc)
        # if pyxargs.load_py_module_on_import_failure and pyxfilename.endswith('.py'):
        if False and pyxfilename.endswith(".py"):
            # try to fall back to normal import
            mod = imp.load_source(name, pyxfilename)
            assert mod.__file__ in (pyxfilename, pyxfilename + "c", pyxfilename + "o"), (mod.__file__, pyxfilename)
        else:
            tb = sys.exc_info()[2]
            import traceback

            exc = ImportError(
                "Building module %s failed: %s" % (name, traceback.format_exception_only(*sys.exc_info()[:2]))
            )
            if sys.version_info[0] >= 3:
                raise exc.with_traceback(tb)
            else:
                exec("raise exc, None, tb", {"exc": exc, "tb": tb})
    return mod


class PyxImportLoader(ExtensionFileLoader):
    def __init__(self, filename, setup_args, pyxbuild_dir, inplace, language_level, reload_support):
        module_name = os.path.splitext(os.path.basename(filename))[0]
        super().__init__(module_name, filename)
        self._pyxbuild_dir = pyxbuild_dir
        self._inplace = inplace
        self._language_level = language_level
        self._setup_args = setup_args
        self._reload_support = reload_support

    def create_module(self, spec):
        try:
            # print(f"CREATING MODULE: {spec.name} -> {spec.origin}")
            so_path = build_module(
                spec.name,
                pyxfilename=spec.origin,
                user_setup_args=self._setup_args,
                pyxbuild_dir=self._pyxbuild_dir,
                inplace=self._inplace,
                language_level=self._language_level,
                reload_support=self._reload_support,
            )
            self.path = so_path
            spec.origin = so_path
            return super().create_module(spec)
        except Exception as failure_exc:
            # print("LOADING on FAILURE MODULE")
            # if pyxargs.load_py_module_on_import_failure and spec.origin.endswith('.pyx'):
            if False and spec.origin.endswith(PYX_EXT):
                spec = importlib.util.spec_from_file_location(
                    spec.name, spec.origin, loader=SourceFileLoader(spec.name, spec.origin)
                )
                mod = importlib.util.module_from_spec(spec)
                assert mod.__file__ in (spec.origin, spec.origin + "c", spec.origin + "o"), (mod.__file__, spec.origin)
                return mod
            else:
                tb = sys.exc_info()[2]
                import traceback

                exc = ImportError(
                    "Building module %s failed: %s" % (spec.name, traceback.format_exception_only(*sys.exc_info()[:2]))
                )
                raise exc.with_traceback(tb)

    def exec_module(self, module):
        try:
            # print(f"EXEC MODULE: {module}")
            return super().exec_module(module)
        except Exception as failure_exc:
            import traceback

            print("Failed to load extension module: %r" % failure_exc)
            raise ImportError(
                "Executing module %s failed %s"
                % (module.__file__, traceback.format_exception_only(*sys.exc_info()[:2]))
            )


class CustomPyxImportMetaFinder(MetaPathFinder):
    def __init__(
        self,
        modules_to_check: List[str],
        extension=PYX_EXT,
        setup_args=None,
        pyxbuild_dir=None,
        inplace=False,
        language_level=None,
        reload_support=True,
    ):
        self.valid_modules = modules_to_check
        self.pyxbuild_dir = pyxbuild_dir
        self.inplace = inplace
        self.language_level = language_level
        self.extension = extension
        self.setup_args = setup_args if setup_args else dict()
        self.reload_support = reload_support

    def find_spec(self, fullname, path, target=None):
        def _is_valid(module):
            if not self.valid_modules:
                return True
            for m in self.valid_modules:
                if module.startswith(m):
                    return True
            return False

        if not path:
            path = [os.getcwd()]  # top level import --
        if "." in fullname:
            *parents, name = fullname.split(".")
        else:
            name = fullname
        for entry in path:
            if os.path.isdir(os.path.join(entry, name)):
                # this module has child modules
                filename = os.path.join(entry, name, "__init__" + self.extension)
                submodule_locations = [os.path.join(entry, name)]
            else:
                filename = os.path.join(entry, name + self.extension)
                submodule_locations = None
            if not os.path.exists(filename):
                continue

            if not _is_valid(fullname):
                continue

            return spec_from_file_location(
                fullname,
                filename,
                loader=PyxImportLoader(
                    filename, self.setup_args, self.pyxbuild_dir, self.inplace, self.language_level, self.reload_support
                ),
                submodule_search_locations=submodule_locations,
            )

        return None  # we don't know how to import this


__pyx_finder_installed = False


def pyx_install_loader(modules_to_check: List[str]):
    import numpy as np
    import pyximport

    global __pyx_finder_installed

    if not __pyx_finder_installed:
        build_dir = os.path.expanduser("~/.pyxbld")
        setup_args = {"include_dirs": np.get_include()}
        sys.meta_path.insert(
            0,
            CustomPyxImportMetaFinder(
                modules_to_check,
                PYX_EXT,
                setup_args=setup_args,
                pyxbuild_dir=build_dir,
                language_level=3,
                reload_support=True,
            ),
        )
        pyximport.install(setup_args=setup_args, build_dir=build_dir, reload_support=True, language_level=3)
