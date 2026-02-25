from typing import TYPE_CHECKING, Any, Callable, Type, TypeVar

from qubx.utils.misc import class_import

if TYPE_CHECKING:
    from qubx.data.storage import IStorage

    S = TypeVar("S", bound="IStorage")
else:
    # Use Any as a placeholder during runtime
    T = TypeVar("T", bound=Any)
    S = TypeVar("S", bound=Any)


class StorageRegistry:
    """
    A registry for data storages that allows registering and retrieving storage classes by name.

    This registry is used to map reader names (like 'mqdb', 'csv', etc.) to their respective
    storage classes, allowing for dynamic lookup and instantiation of storages.
    """

    _storages: dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[S]], Type[S]]:
        """
        Decorator to register a storage class with the registry.

        Args:
            name: The name to register the storage under

        Returns:
            A decorator function that registers the class
        """

        def decorator(storage_cls: Type[S]) -> Type[S]:
            cls._storages[name] = storage_cls
            return storage_cls

        return decorator

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a storage is already registered with the given name.

        Args:
            name: The name to check

        Returns:
            True if the storage is registered, False otherwise
        """
        return name in cls._storages

    @classmethod
    def get(cls, storage_name: str, **kwargs) -> "IStorage":
        """
        Get a storage instance by name.

        This method handles both URI-style names (e.g., 'mqdb::nebula') and
        fully qualified class names (e.g., 'sty.data.storages.questdb.QuestDBStorage').

        Args:
            storage_name: The name of the storage to retrieve, can be a URI or class name
            **kwargs: Additional arguments to pass to the storage constructor

        Returns:
            An instance of the storage

        Raises:
            ValueError: If the storage cannot be found or instantiated
        """
        _is_uri = "::" in storage_name

        if _is_uri:
            # Handle URI-style names like 'mqdb::nebula' or 'csv::/data/rawdata/'
            db_conn, db_name = storage_name.split("::")

            # Try to get the reader from the registry
            storage_cls = cls._storages.get(db_conn)
            if storage_cls is not None:
                # Different storages have different parameter names for the connection string
                # For CSV storage, it's 'path', for database readers, it's 'host'
                return storage_cls(db_name, **kwargs)  # type: ignore  # noqa
        else:
            # Check if it's a simple name registered in the registry
            storage_cls = cls._storages.get(storage_name)
            if storage_cls is not None:
                return storage_cls(**kwargs)

            # Try to import the class directly
            try:
                storage_cls = class_import(storage_name)
                return storage_cls(**kwargs)
            except (ImportError, AttributeError, ValueError):
                pass

        # If we get here, we couldn't find or instantiate the storage
        raise ValueError(f"Unknown storage type: {storage_name}")

    @classmethod
    def get_all_storages(cls) -> dict[str, Type[Any]]:
        """
        Get all registered storages.

        Returns:
            A dictionary mapping storage names to it's class
        """
        return cls._storages.copy()

    @classmethod
    def get_class(cls, name: str) -> Type[Any]:
        """
        Get the storage class by name without instantiating.

        Handles both simple names (like 'csv') and URI-style names (like 'csv::path').

        Args:
            name: The name of the storage to retrieve

        Returns:
            The storage class

        Raises:
            ValueError: If the storage cannot be found
        """
        from qubx.utils.misc import class_import

        # Handle URI-style names like 'csv::tests/data/csv_1h/'
        if "::" in name:
            db_conn = name.split("::")[0]
            _stor_clazz = cls._storages.get(db_conn)
            if _stor_clazz is not None:
                return _stor_clazz

        # Check if it's a simple name registered in the registry
        _stor_clazz = cls._storages.get(name)
        if _stor_clazz is not None:
            return _stor_clazz

        # Try to import the class directly
        try:
            return class_import(name)
        except (ImportError, AttributeError, ValueError):
            pass

        raise ValueError(f"Storage '{name}' not found. Available: {list(cls._storages.keys())}")


def storage(name: str) -> Callable[[Type[S]], Type[S]]:
    """
    Decorator for registering a storage class with the registry.

    This is a convenience function that calls StorageRegistry.register.

    Args:
        name: The name to register the storage under

    Returns:
        A decorator function that registers the class
    """
    return StorageRegistry.register(name)
