"""Tests for the ModuleResolver recursive import resolution."""

import os

import pytest

from qubx.cli.resolver import ModuleResolver


def _write(path: str, content: str = "") -> str:
    """Helper: write content to a file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def _make_src_layout(tmp_path, pkg_name="mypkg"):
    """Create a src-layout project skeleton and return (project_root, package_root)."""
    project_root = str(tmp_path)
    package_root = os.path.join(project_root, "src", pkg_name)
    os.makedirs(package_root, exist_ok=True)
    _write(os.path.join(package_root, "__init__.py"))
    return project_root, package_root


def _make_flat_layout(tmp_path, pkg_name="mypkg"):
    """Create a flat-layout project skeleton and return (project_root, package_root)."""
    project_root = str(tmp_path)
    package_root = os.path.join(project_root, pkg_name)
    os.makedirs(package_root, exist_ok=True)
    _write(os.path.join(package_root, "__init__.py"))
    return project_root, package_root


# ---------------------------------------------------------------------------
# 1. Linear chain: A → B → C, all 3 collected
# ---------------------------------------------------------------------------
class TestLinearChain:
    def test_all_three_collected(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        a = _write(os.path.join(pkg, "a.py"), "from mypkg.b import something\n")
        b = _write(os.path.join(pkg, "b.py"), "from mypkg.c import other\n")
        c = _write(os.path.join(pkg, "c.py"), "x = 1\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        files = resolver.collect_internal_files([a])

        assert a in files
        assert b in files
        assert c in files


# ---------------------------------------------------------------------------
# 2. Branching: A → B + C, B → D, all 4 collected
# ---------------------------------------------------------------------------
class TestBranching:
    def test_all_four_collected(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        a = _write(os.path.join(pkg, "a.py"), "from mypkg.b import x\nfrom mypkg.c import y\n")
        b = _write(os.path.join(pkg, "b.py"), "from mypkg.d import z\n")
        c = _write(os.path.join(pkg, "c.py"), "y = 1\n")
        d = _write(os.path.join(pkg, "d.py"), "z = 2\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        files = resolver.collect_internal_files([a])

        assert {a, b, c, d} == files


# ---------------------------------------------------------------------------
# 3. Circular imports: A ↔ B, no infinite loop
# ---------------------------------------------------------------------------
class TestCircularImports:
    def test_no_infinite_loop(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        a = _write(os.path.join(pkg, "a.py"), "from mypkg.b import foo\n")
        b = _write(os.path.join(pkg, "b.py"), "from mypkg.a import bar\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        files = resolver.collect_internal_files([a])

        assert a in files
        assert b in files


# ---------------------------------------------------------------------------
# 4. External boundary: imports numpy + internal, only internal followed
# ---------------------------------------------------------------------------
class TestExternalBoundary:
    def test_external_not_followed_but_collected(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        a = _write(os.path.join(pkg, "a.py"), "import numpy\nfrom mypkg.b import x\n")
        b = _write(os.path.join(pkg, "b.py"), "x = 1\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        internal = resolver.collect_internal_files([a])
        external = resolver.collect_external_imports(internal)

        assert a in internal
        assert b in internal
        assert "numpy" in external


# ---------------------------------------------------------------------------
# 5. Multiple entry points: two files, union of deps
# ---------------------------------------------------------------------------
class TestMultipleEntryPoints:
    def test_union_of_deps(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        a = _write(os.path.join(pkg, "a.py"), "from mypkg.shared import x\n")
        b = _write(os.path.join(pkg, "b.py"), "from mypkg.only_b import y\n")
        shared = _write(os.path.join(pkg, "shared.py"), "x = 1\n")
        only_b = _write(os.path.join(pkg, "only_b.py"), "y = 2\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        files = resolver.collect_internal_files([a, b])

        assert {a, b, shared, only_b} == files


# ---------------------------------------------------------------------------
# 6. Mixed entries: one internal + one external package, only internal followed
# ---------------------------------------------------------------------------
class TestMixedEntries:
    def test_only_internal_followed(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        a = _write(os.path.join(pkg, "a.py"), "from mypkg.b import x\n")
        b = _write(os.path.join(pkg, "b.py"), "x = 1\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        # Pass an existing internal file + a non-existent external path
        files = resolver.collect_internal_files([a, "/nonexistent/external.py"])

        assert a in files
        assert b in files
        assert len(files) == 2


# ---------------------------------------------------------------------------
# 7. Relative imports: from .sibling import X resolved and followed
# ---------------------------------------------------------------------------
class TestRelativeImports:
    def test_relative_imports_followed(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        sub = os.path.join(pkg, "sub")
        os.makedirs(sub, exist_ok=True)
        _write(os.path.join(sub, "__init__.py"))
        a = _write(os.path.join(sub, "a.py"), "from .sibling import x\n")
        sibling = _write(os.path.join(sub, "sibling.py"), "x = 1\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        files = resolver.collect_internal_files([a])

        assert a in files
        assert sibling in files


# ---------------------------------------------------------------------------
# 8. Package imports: directory import → __init__.py scanned, re-exports followed
# ---------------------------------------------------------------------------
class TestPackageImports:
    def test_init_scanned_reexports_followed(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        sub = os.path.join(pkg, "sub")
        os.makedirs(sub, exist_ok=True)
        init = _write(os.path.join(sub, "__init__.py"), "from mypkg.sub.impl import core_func\n")
        impl = _write(os.path.join(sub, "impl.py"), "def core_func(): pass\n")
        a = _write(os.path.join(pkg, "a.py"), "from mypkg.sub import core_func\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        files = resolver.collect_internal_files([a])

        assert a in files
        assert init in files
        assert impl in files


# ---------------------------------------------------------------------------
# 9. Transitive external: A → B → import matplotlib, matplotlib found
# ---------------------------------------------------------------------------
class TestTransitiveExternal:
    def test_transitive_external_detected(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        a = _write(os.path.join(pkg, "a.py"), "from mypkg.b import plot\n")
        b = _write(os.path.join(pkg, "b.py"), "import matplotlib\ndef plot(): pass\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        internal, external = resolver.resolve([a])

        assert a in internal
        assert b in internal
        assert "matplotlib" in external


# ---------------------------------------------------------------------------
# 10. Missing module: no crash, logged warning
# ---------------------------------------------------------------------------
class TestMissingModule:
    def test_missing_module_no_crash(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        a = _write(os.path.join(pkg, "a.py"), "from mypkg.nonexistent import x\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        files = resolver.collect_internal_files([a])

        # Should still include the entry file, just not crash
        assert a in files
        assert len(files) == 1


# ---------------------------------------------------------------------------
# 11. ensure_init_files: deep file gets parent __init__.py added
# ---------------------------------------------------------------------------
class TestEnsureInitFiles:
    def test_parent_inits_added(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        sub1 = os.path.join(pkg, "sub1")
        sub2 = os.path.join(sub1, "sub2")
        os.makedirs(sub2, exist_ok=True)
        pkg_init = _write(os.path.join(pkg, "__init__.py"))
        sub1_init = _write(os.path.join(sub1, "__init__.py"))
        sub2_init = _write(os.path.join(sub2, "__init__.py"))
        deep_file = _write(os.path.join(sub2, "deep.py"), "x = 1\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        augmented = resolver.ensure_init_files({deep_file})

        assert deep_file in augmented
        assert pkg_init in augmented
        assert sub1_init in augmented
        assert sub2_init in augmented


# ---------------------------------------------------------------------------
# 12. .pyx handling: resolved, included, regex import extraction
# ---------------------------------------------------------------------------
class TestPyxHandling:
    def test_pyx_resolved_and_included(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        # .pyx with Cython syntax that would fail AST parse
        pyx = _write(
            os.path.join(pkg, "fast.pyx"),
            "cimport numpy as np\nimport pandas\nfrom mypkg.helper import x\n",
        )
        a = _write(os.path.join(pkg, "a.py"), "from mypkg.fast import compute\n")
        helper = _write(os.path.join(pkg, "helper.py"), "x = 1\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")

        # resolve_module_to_file should find .pyx
        resolved = resolver.resolve_module_to_file(["mypkg", "fast"])
        assert resolved == pyx

        # collect_internal_files should include the .pyx file
        internal = resolver.collect_internal_files([a])
        assert pyx in internal

        # collect_external_imports should extract via regex
        external = resolver.collect_external_imports({pyx})
        assert "pandas" in external

    def test_pyx_regex_extraction(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        pyx = _write(
            os.path.join(pkg, "algo.pyx"),
            "import numpy\nfrom scipy.stats import norm\ncdef int x = 1\n",
        )

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        external = resolver.collect_external_imports({pyx})

        assert "numpy" in external
        assert "scipy" in external
        assert "mypkg" not in external  # internal should be excluded


# ---------------------------------------------------------------------------
# 13. Flat layout (no src/): resolver works with explicit package_root
# ---------------------------------------------------------------------------
class TestFlatLayout:
    def test_flat_layout_resolves(self, tmp_path):
        project_root, pkg = _make_flat_layout(tmp_path)
        a = _write(os.path.join(pkg, "a.py"), "from mypkg.b import x\nimport requests\n")
        b = _write(os.path.join(pkg, "b.py"), "x = 1\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        internal, external = resolver.resolve([a])

        assert a in internal
        assert b in internal
        assert "requests" in external


# ---------------------------------------------------------------------------
# 14. Init re-exports: __init__.py added by ensure_init_files re-exports sibling
# ---------------------------------------------------------------------------
class TestInitReexportsSibling:
    def test_init_reexport_pulls_in_sibling(self, tmp_path):
        """Reproduces the trend_intensity bug: __init__.py re-exports a sibling
        that wasn't directly imported by any resolved file."""
        project_root, pkg = _make_src_layout(tmp_path)
        indicators = os.path.join(pkg, "indicators", "momentum")
        os.makedirs(indicators, exist_ok=True)
        _write(os.path.join(pkg, "indicators", "__init__.py"))
        # __init__.py re-exports from a sibling module
        momentum_init = _write(
            os.path.join(indicators, "__init__.py"),
            "from .trend_intensity import trend_intensity_index\n",
        )
        trend = _write(
            os.path.join(indicators, "trend_intensity.py"),
            "def trend_intensity_index(): pass\n",
        )
        nimble_mod = _write(
            os.path.join(indicators, "nimble.py"),
            "from mypkg.indicators.momentum.nimble import NimbleTrend\nclass NimbleTrend: pass\n",
        )
        # Strategy imports nimble from the momentum subpackage
        strategy = _write(
            os.path.join(pkg, "strategy.py"),
            "from mypkg.indicators.momentum.nimble import NimbleTrend\n",
        )

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        internal, _external = resolver.resolve([strategy])

        # The key assertion: trend_intensity.py must be included because
        # momentum/__init__.py (added as a parent init) re-exports from it
        assert trend in internal
        assert momentum_init in internal
        assert nimble_mod in internal
        assert strategy in internal


# ---------------------------------------------------------------------------
# resolve_module_to_file edge cases
# ---------------------------------------------------------------------------
class TestResolveModuleToFile:
    def test_external_package_returns_none(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        resolver = ModuleResolver(pkg, project_root, "mypkg")
        assert resolver.resolve_module_to_file(["numpy", "core"]) is None

    def test_empty_parts_returns_none(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        resolver = ModuleResolver(pkg, project_root, "mypkg")
        assert resolver.resolve_module_to_file([]) is None

    def test_package_root_init(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        resolver = ModuleResolver(pkg, project_root, "mypkg")
        init_path = os.path.join(pkg, "__init__.py")
        assert resolver.resolve_module_to_file(["mypkg"]) == init_path

    def test_py_before_pyx(self, tmp_path):
        """When both .py and .pyx exist, .py is preferred."""
        project_root, pkg = _make_src_layout(tmp_path)
        py = _write(os.path.join(pkg, "mod.py"), "")
        _write(os.path.join(pkg, "mod.pyx"), "")
        resolver = ModuleResolver(pkg, project_root, "mypkg")
        assert resolver.resolve_module_to_file(["mypkg", "mod"]) == py


# ---------------------------------------------------------------------------
# Full resolve() convenience method
# ---------------------------------------------------------------------------
class TestResolveConvenience:
    def test_resolve_returns_tuple(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        a = _write(os.path.join(pkg, "a.py"), "import os\nfrom mypkg.b import x\n")
        b = _write(os.path.join(pkg, "b.py"), "import json\nx = 1\n")

        resolver = ModuleResolver(pkg, project_root, "mypkg")
        internal, external = resolver.resolve([a])

        # Internal should include a, b, and __init__.py
        assert a in internal
        assert b in internal
        init = os.path.join(pkg, "__init__.py")
        assert init in internal

        # External should include os and json
        assert "os" in external
        assert "json" in external

    def test_resolve_empty_entries(self, tmp_path):
        project_root, pkg = _make_src_layout(tmp_path)
        resolver = ModuleResolver(pkg, project_root, "mypkg")
        internal, external = resolver.resolve([])
        assert internal == set()
        assert external == set()
