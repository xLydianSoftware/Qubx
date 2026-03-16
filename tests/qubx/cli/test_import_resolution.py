"""
Tests for import resolution functionality in the release command.

This module tests the critical import detection and dependency resolution logic
that ensures all necessary files are included in release packages.
"""

import os
import tempfile

import pytest

from qubx.cli.release import get_imports, resolve_relative_import, ImportResolutionError, Import


class TestRelativeImportResolution:
    """Test the resolve_relative_import function."""
    
    def test_single_dot_relative_import(self):
        """Test resolving single dot relative imports."""
        # Test from mypackage/submodule/file.py: from .utils import helper
        result = resolve_relative_import(
            ".utils", 
            "/project/mypackage/submodule/file.py", 
            "/project"
        )
        assert result == "mypackage.submodule.utils"
        
    def test_double_dot_relative_import(self):
        """Test resolving double dot relative imports."""
        # Test from mypackage/submodule/file.py: from ..utils import helper  
        result = resolve_relative_import(
            "..utils",
            "/project/mypackage/submodule/file.py",
            "/project"
        )
        assert result == "mypackage.utils"
        
    def test_triple_dot_relative_import(self):
        """Test resolving triple dot relative imports."""
        # Test from mypackage/sub1/sub2/file.py: from ...utils import helper
        result = resolve_relative_import(
            "...utils", 
            "/project/mypackage/sub1/sub2/file.py",
            "/project"
        )
        assert result == "mypackage.utils"
        
    def test_relative_import_from_root(self):
        """Test relative import from package root."""
        # Test from mypackage/file.py: from .utils import helper
        result = resolve_relative_import(
            ".utils",
            "/project/mypackage/file.py", 
            "/project"
        )
        assert result == "mypackage.utils"
        
    def test_relative_import_beyond_root_raises_error(self):
        """Test that relative imports beyond project root raise errors."""
        with pytest.raises(ImportResolutionError):
            resolve_relative_import(
                "...utils",  # Too many levels up
                "/project/mypackage/file.py",
                "/project"
            )
            
    def test_plain_import_no_dots(self):
        """Test that non-relative imports are returned as-is."""
        result = resolve_relative_import(
            "utils.helper",
            "/project/mypackage/file.py",
            "/project" 
        )
        assert result == "utils.helper"


class TestGetImports:
    """Test the get_imports function that parses AST for import statements."""

    def test_direct_import_detection(self):
        """Test that direct imports like 'import module' are detected correctly."""
        code = """
import xincubator.models
import xincubator.utils.helper
import other_module  # Should be filtered out
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            try:
                imports = list(get_imports(f.name, ["xincubator"]))
                
                # Should detect both xincubator imports
                assert len(imports) == 2
                
                # Check first import
                assert imports[0].module == ["xincubator", "models"]
                assert imports[0].name == ["models"]
                assert imports[0].alias is None
                
                # Check second import  
                assert imports[1].module == ["xincubator", "utils", "helper"]
                assert imports[1].name == ["helper"]
                assert imports[1].alias is None
                
            finally:
                os.unlink(f.name)

    def test_from_import_detection(self):
        """Test that 'from module import name' statements are detected correctly."""
        code = """
from xincubator.models import Portfolio
from xincubator.utils import helper, calculator
from other_module import something  # Should be filtered out
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            try:
                imports = list(get_imports(f.name, ["xincubator"]))
                
                # Should detect 3 imports (Portfolio, helper, calculator)
                assert len(imports) == 3
                
                # Check Portfolio import
                portfolio_import = next(imp for imp in imports if imp.name == ["Portfolio"])
                assert portfolio_import.module == ["xincubator", "models"]
                
                # Check helper import
                helper_import = next(imp for imp in imports if imp.name == ["helper"])
                assert helper_import.module == ["xincubator", "utils"]
                
                # Check calculator import
                calc_import = next(imp for imp in imports if imp.name == ["calculator"])
                assert calc_import.module == ["xincubator", "utils"]
                
            finally:
                os.unlink(f.name)

    def test_relative_import_detection_current_behavior(self):
        """Test current behavior with relative imports (they are skipped)."""
        code = """
from .utils import helper
from ..models import Portfolio
from .subpackage.module import function
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            try:
                # Current implementation skips relative imports
                imports = list(get_imports(f.name, ["xincubator"]))
                
                # Currently returns empty - relative imports are skipped
                assert len(imports) == 0  # Current behavior - skips relative imports
                
            finally:
                os.unlink(f.name)

    def test_relative_import_resolution_with_context(self):
        """Test relative import resolution when file path context is provided."""
        # Create a temporary project structure
        temp_dir = tempfile.mkdtemp()
        try:
            # Create package structure: mypackage/submodule/file.py
            package_dir = os.path.join(temp_dir, "mypackage")
            submodule_dir = os.path.join(package_dir, "submodule")
            os.makedirs(submodule_dir)
            
            # Create __init__.py files
            with open(os.path.join(package_dir, "__init__.py"), 'w') as f:
                f.write("")
            with open(os.path.join(submodule_dir, "__init__.py"), 'w') as f:
                f.write("")
                
            # Create utils.py in package root
            with open(os.path.join(package_dir, "utils.py"), 'w') as f:
                f.write("def helper(): pass")
                
            # Create file with relative imports
            file_with_relatives = os.path.join(submodule_dir, "strategy.py")
            with open(file_with_relatives, 'w') as f:
                f.write("""
from ..utils import helper  # Should resolve to mypackage.utils
from .local_module import something  # Should resolve to mypackage.submodule.local_module
""")
            
            # Test with project root context - should resolve relative imports
            imports = list(get_imports(file_with_relatives, ["mypackage"], project_root=temp_dir))
            
            # Should now resolve the relative imports
            assert len(imports) == 2
            
            # Check that ..utils resolved to mypackage.utils
            utils_import = next(imp for imp in imports if "utils" in imp.module)
            assert utils_import.module == ["mypackage", "utils"]
            assert utils_import.name == ["helper"]
            
            # Check that .local_module resolved to mypackage.submodule.local_module
            local_import = next(imp for imp in imports if "local_module" in imp.module)
            assert local_import.module == ["mypackage", "submodule", "local_module"] 
            assert local_import.name == ["something"]
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_import_with_alias(self):
        """Test that imports with aliases are handled correctly."""
        code = """
import xincubator.models as models
from xincubator.utils import helper as h
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            try:
                imports = list(get_imports(f.name, ["xincubator"]))
                
                # Should detect both imports
                assert len(imports) == 2
                
                # Check aliased direct import
                models_import = next(imp for imp in imports if imp.alias == "models")
                assert models_import.module == ["xincubator", "models"]
                assert models_import.name == ["models"]
                
                # Check aliased from import
                helper_import = next(imp for imp in imports if imp.alias == "h")
                assert helper_import.module == ["xincubator", "utils"]
                assert helper_import.name == ["helper"]
                
            finally:
                os.unlink(f.name)

    def test_star_import_detection(self):
        """Test that star imports are detected."""
        code = """
from xincubator.models import *
from other_module import *  # Should be filtered out
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            try:
                imports = list(get_imports(f.name, ["xincubator"]))
                
                # Should detect the star import
                assert len(imports) == 1
                assert imports[0].module == ["xincubator", "models"]
                assert imports[0].name == ["*"]
                assert imports[0].alias is None
                
            finally:
                os.unlink(f.name)

    def test_no_filter_returns_all_imports(self):
        """Test that when no filter is provided, all imports are returned."""
        code = """
import xincubator.models
import other_module
from third_module import something
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            try:
                # Pass empty filter to get all imports
                imports = list(get_imports(f.name, []))
                
                # Should detect all imports when no filter applied
                assert len(imports) == 3
                
            finally:
                os.unlink(f.name)

    def test_malformed_python_file(self):
        """Test that malformed Python files are handled gracefully."""
        code = """
import xincubator.models
def incomplete_function(
    # Missing closing parenthesis and function body
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            try:
                # Should raise SyntaxError, not return empty list
                with pytest.raises(SyntaxError):
                    list(get_imports(f.name, ["xincubator"]))
                
            finally:
                os.unlink(f.name)


class TestCollectAllImports:
    """Test the _collect_all_imports function that scans strategy files for external imports."""

    def test_collects_top_level_modules(self):
        """Test that top-level module names are collected from strategy files."""
        from qubx.cli.release import _collect_all_imports

        code = """
import numpy as np
from scipy.stats import norm
from qubx.core import IStrategy
from xincubator.utils import helper
import cachetools
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            try:
                imports = _collect_all_imports([f.name], os.path.dirname(f.name))
                assert "numpy" in imports
                assert "scipy" in imports
                assert "qubx" in imports
                assert "xincubator" in imports
                assert "cachetools" in imports
            finally:
                os.unlink(f.name)

    def test_handles_empty_file_list(self):
        """Test that empty file list returns empty set."""
        from qubx.cli.release import _collect_all_imports

        imports = _collect_all_imports([], "/tmp")
        assert imports == set()


class TestParseUvLock:
    """Test the _parse_uv_lock function."""

    def test_parses_lock_file(self):
        """Test parsing a minimal uv.lock file."""
        from qubx.cli.release import _parse_uv_lock

        lock_content = """
version = 1

[[package]]
name = "numpy"
version = "1.26.4"

[[package]]
name = "scipy"
version = "1.12.0"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lock', delete=False) as f:
            f.write(lock_content)
            f.flush()

            try:
                versions = _parse_uv_lock(f.name)
                assert versions["numpy"] == "1.26.4"
                assert versions["scipy"] == "1.12.0"
            finally:
                os.unlink(f.name)

    def test_returns_empty_for_missing_file(self):
        """Test that missing lock file returns empty dict."""
        from qubx.cli.release import _parse_uv_lock

        versions = _parse_uv_lock("/nonexistent/uv.lock")
        assert versions == {}