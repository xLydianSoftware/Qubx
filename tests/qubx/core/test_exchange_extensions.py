"""Test exchange extensions pattern."""

from qubx.core.interfaces import EmptyExchangeExtensions, IExchangeExtensions


def test_empty_extensions_returns_empty_dict():
    """Test that EmptyExchangeExtensions returns empty dict for list_methods."""
    extensions = EmptyExchangeExtensions("TEST_EXCHANGE")
    assert extensions.list_methods() == {}


def test_empty_extensions_raises_not_implemented():
    """Test that EmptyExchangeExtensions raises NotImplementedError for call_method."""
    extensions = EmptyExchangeExtensions("TEST_EXCHANGE")

    try:
        extensions.call_method("some_method")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "TEST_EXCHANGE" in str(e)
        assert "does not have extension methods" in str(e)


def test_extensions_list_methods_excludes_internal():
    """Test that list_methods excludes internal reflection methods."""

    class TestExtensions(IExchangeExtensions):
        def test_method_one(self):
            """Test method one."""
            pass

        def test_method_two(self):
            """Test method two."""
            pass

        def _private_method(self):
            """Private method."""
            pass

    extensions = TestExtensions()
    methods = extensions.list_methods()

    # Should include our test methods
    assert "test_method_one" in methods
    assert "test_method_two" in methods

    # Should exclude internal methods
    assert "list_methods" not in methods
    assert "call_method" not in methods
    assert "get_method_signature" not in methods
    assert "help" not in methods
    assert "_private_method" not in methods


def test_extensions_call_method_works():
    """Test that call_method can invoke methods dynamically."""

    class TestExtensions(IExchangeExtensions):
        def add(self, a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        def multiply(self, a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

    extensions = TestExtensions()

    # Test positional args
    result = extensions.call_method("add", 5, 3)
    assert result == 8

    # Test keyword args
    result = extensions.call_method("multiply", a=4, b=7)
    assert result == 28


def test_extensions_call_method_raises_for_nonexistent():
    """Test that call_method raises AttributeError for nonexistent methods."""

    class TestExtensions(IExchangeExtensions):
        def real_method(self):
            """Real method."""
            pass

    extensions = TestExtensions()

    try:
        extensions.call_method("nonexistent_method")
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "nonexistent_method" in str(e)
        assert "not found" in str(e)


def test_extensions_get_method_signature():
    """Test that get_method_signature returns correct information."""

    class TestExtensions(IExchangeExtensions):
        def sample_method(self, param1: str, param2: int = 42) -> str:
            """
            Sample method for testing.

            This is a multi-line docstring.
            """
            return f"{param1}:{param2}"

    extensions = TestExtensions()
    sig_info = extensions.get_method_signature("sample_method")

    assert "signature" in sig_info
    assert "description" in sig_info
    assert "parameters" in sig_info
    assert "return_type" in sig_info

    # Check parameters
    params = sig_info["parameters"]
    assert len(params) == 2

    # First parameter
    assert params[0]["name"] == "param1"
    assert "str" in params[0]["annotation"]
    assert params[0]["default"] is None

    # Second parameter
    assert params[1]["name"] == "param2"
    assert "int" in params[1]["annotation"]
    assert params[1]["default"] == "42"

    # Check return type
    assert "str" in sig_info["return_type"]


def test_extensions_help_all_methods():
    """Test that help() without argument lists all methods."""

    class TestExtensions(IExchangeExtensions):
        def method_a(self):
            """Method A description."""
            pass

        def method_b(self):
            """Method B description."""
            pass

    extensions = TestExtensions()
    help_text = extensions.help()

    assert "Available extension methods:" in help_text
    assert "method_a" in help_text
    assert "method_b" in help_text
    assert "Method A description" in help_text
    assert "Method B description" in help_text


def test_extensions_help_specific_method():
    """Test that help(method_name) returns detailed help for that method."""

    class TestExtensions(IExchangeExtensions):
        def test_method(self, arg1: str, arg2: int = 10) -> bool:
            """
            Test method for demonstration.

            This has a longer description.
            """
            return True

    extensions = TestExtensions()
    help_text = extensions.help("test_method")

    assert "Method: test_method" in help_text
    assert "Test method for demonstration" in help_text


def test_extensions_help_nonexistent_method():
    """Test that help() for nonexistent method returns error message."""

    class TestExtensions(IExchangeExtensions):
        pass

    extensions = TestExtensions()
    help_text = extensions.help("nonexistent")

    assert "not found" in help_text


def test_empty_extensions_help():
    """Test that EmptyExchangeExtensions returns appropriate help."""
    extensions = EmptyExchangeExtensions("TEST_EXCHANGE")
    help_text = extensions.help()

    assert "No extension methods available" in help_text
