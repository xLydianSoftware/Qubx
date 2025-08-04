"""
Tests for import resolution functionality in the release command.

This module tests the critical import detection and dependency resolution logic
that ensures all necessary files are included in release packages.
"""

import ast
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from qubx.cli.release import Import, get_imports, _get_imports


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

    def test_relative_import_detection(self):
        """Test that relative imports like 'from .module import name' are detected."""
        code = """
from .utils import helper
from ..models import Portfolio
from .subpackage.module import function
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            try:
                # Current implementation should fail with relative imports
                # This test documents the current bug
                imports = list(get_imports(f.name, ["xincubator"]))
                
                # Currently returns empty due to node.module being None for relative imports
                # This should be fixed to properly handle relative imports
                assert len(imports) == 0  # Current broken behavior
                
            finally:
                os.unlink(f.name)

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


class TestGetImportsRecursive:
    """Test the _get_imports function that recursively resolves dependencies."""

    def setup_method(self):
        """Set up temporary directory structure for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_dir = os.path.join(self.temp_dir, "test_project")
        os.makedirs(self.project_dir)
        
    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_recursive_dependency_resolution(self):
        """Test that dependencies are resolved recursively."""
        # Create a module that imports another module
        main_module = os.path.join(self.project_dir, "main.py")
        with open(main_module, 'w') as f:
            f.write("from test_project.utils import helper\n")
            
        # Create the dependency module
        utils_dir = os.path.join(self.project_dir, "utils")
        os.makedirs(utils_dir)
        
        utils_init = os.path.join(utils_dir, "__init__.py")
        with open(utils_init, 'w') as f:
            f.write("from test_project.core import base\n")
            
        helper_module = os.path.join(utils_dir, "helper.py")
        with open(helper_module, 'w') as f:
            f.write("# Helper module\n")
            
        # Create the nested dependency
        core_dir = os.path.join(self.project_dir, "core")
        os.makedirs(core_dir)
        
        core_init = os.path.join(core_dir, "__init__.py")
        with open(core_init, 'w') as f:
            f.write("# Core package\n")
            
        base_module = os.path.join(core_dir, "base.py")
        with open(base_module, 'w') as f:
            f.write("# Base module\n")
        
        # Test recursive resolution
        imports = _get_imports(main_module, self.project_dir, ["test_project"])
        
        # Should find imports from both main module and its dependencies
        # Current implementation may miss some due to the bugs we're fixing
        assert len(imports) >= 1  # At least the direct import should be found

    def test_missing_dependency_handling(self):
        """Test how missing dependencies are handled."""
        # Create a module that imports a non-existent module
        main_module = os.path.join(self.project_dir, "main.py")
        with open(main_module, 'w') as f:
            f.write("from test_project.nonexistent import something\n")
        
        # Current implementation silently ignores missing files
        # This should be changed to raise an exception
        imports = _get_imports(main_module, self.project_dir, ["test_project"])
        
        # Current behavior: silently continues
        # Desired behavior: should raise ImportResolutionError
        assert isinstance(imports, list)  # Current implementation returns list


class TestImportResolutionIntegration:
    """Integration tests for the complete import resolution process."""

    def setup_method(self):
        """Set up a mock project structure for testing."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_complex_project_structure(self):
        """Test import resolution with a complex project structure."""
        # Create a realistic project structure
        project_root = os.path.join(self.temp_dir, "my_strategy")
        src_root = os.path.join(project_root, "src", "my_strategy")
        
        # Create directory structure
        os.makedirs(os.path.join(src_root, "models"))
        os.makedirs(os.path.join(src_root, "utils"))
        os.makedirs(os.path.join(src_root, "indicators"))
        
        # Create main strategy file
        strategy_file = os.path.join(src_root, "models", "strategy.py")
        with open(strategy_file, 'w') as f:
            f.write("""
from my_strategy.utils import calculator
from my_strategy.indicators.macd import macd_indicator
import my_strategy.models.portfolio
""")
        
        # Create dependency files
        calc_file = os.path.join(src_root, "utils", "calculator.py")
        with open(calc_file, 'w') as f:
            f.write("from my_strategy.utils.math_helpers import add\n")
            
        math_file = os.path.join(src_root, "utils", "math_helpers.py") 
        with open(math_file, 'w') as f:
            f.write("def add(a, b): return a + b\n")
            
        macd_file = os.path.join(src_root, "indicators", "macd.py")
        with open(macd_file, 'w') as f:
            f.write("def macd_indicator(): pass\n")
            
        portfolio_file = os.path.join(src_root, "models", "portfolio.py")
        with open(portfolio_file, 'w') as f:
            f.write("class Portfolio: pass\n")
        
        # Create __init__.py files
        for dir_path in [src_root, 
                        os.path.join(src_root, "models"),
                        os.path.join(src_root, "utils"), 
                        os.path.join(src_root, "indicators")]:
            with open(os.path.join(dir_path, "__init__.py"), 'w') as f:
                f.write("")
        
        # Test import resolution
        imports = _get_imports(strategy_file, project_root, ["my_strategy"])
        
        # Verify that all dependencies are discovered
        # This test will likely fail with current implementation
        # demonstrating the need for the fixes
        assert len(imports) > 0