from typing import TYPE_CHECKING, Any, Callable, Dict, Type, TypeVar

from qubx.utils.misc import class_import

if TYPE_CHECKING:
    from qubx.data.readers import DataReader

    T = TypeVar("T", bound="DataReader")
else:
    # Use Any as a placeholder during runtime
    T = TypeVar("T", bound=Any)


class ReaderRegistry:
    """
    A registry for data readers that allows registering and retrieving reader classes by name.

    This registry is used to map reader names (like 'mqdb', 'csv', etc.) to their respective
    reader classes, allowing for dynamic lookup and instantiation of readers.
    """

    _readers: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a reader class with the registry.

        Args:
            name: The name to register the reader under

        Returns:
            A decorator function that registers the class
        """

        def decorator(reader_cls: Type[T]) -> Type[T]:
            cls._readers[name] = reader_cls
            return reader_cls

        return decorator

    @classmethod
    def get(cls, reader_name: str, **kwargs) -> "DataReader":
        """
        Get a reader instance by name.

        This method handles both URI-style names (e.g., 'mqdb::nebula') and
        fully qualified class names (e.g., 'sty.data.readers.MyCustomDataReader').

        Args:
            reader_name: The name of the reader to retrieve, can be a URI or class name
            **kwargs: Additional arguments to pass to the reader constructor

        Returns:
            An instance of the reader

        Raises:
            ValueError: If the reader cannot be found or instantiated
        """
        _is_uri = "::" in reader_name

        if _is_uri:
            # Handle URI-style names like 'mqdb::nebula' or 'csv::/data/rawdata/'
            db_conn, db_name = reader_name.split("::")

            # Try to get the reader from the registry
            reader_cls = cls._readers.get(db_conn)
            if reader_cls is not None:
                # Different readers have different parameter names for the connection string
                # For CSV readers, it's 'path', for database readers, it's 'host'
                return reader_cls(db_name, **kwargs)  # type: ignore  # noqa
        else:
            # Check if it's a simple name registered in the registry
            reader_cls = cls._readers.get(reader_name)
            if reader_cls is not None:
                return reader_cls(**kwargs)

            # Try to import the class directly
            try:
                reader_cls = class_import(reader_name)
                return reader_cls(**kwargs)
            except (ImportError, AttributeError, ValueError):
                pass

        # If we get here, we couldn't find or instantiate the reader
        raise ValueError(f"Unknown reader type: {reader_name}")

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a reader is registered with the given name.

        Args:
            name: The name to check

        Returns:
            True if the reader is registered, False otherwise
        """
        return name in cls._readers

    @classmethod
    def get_all_readers(cls) -> Dict[str, Type[Any]]:
        """
        Get all registered readers.

        Returns:
            A dictionary mapping reader names to reader classes
        """
        return cls._readers.copy()


def reader(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for registering a reader class with the registry.

    This is a convenience function that calls ReaderRegistry.register.

    Args:
        name: The name to register the reader under

    Returns:
        A decorator function that registers the class
    """
    return ReaderRegistry.register(name)
