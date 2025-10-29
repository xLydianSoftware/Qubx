from typing import Any, Iterator, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

_T = TypeVar("_T", bound=np.generic)

class RingBuffer:
    """
    A circular buffer implementation using numpy arrays.

    Provides efficient O(1) append/pop operations at both ends of the buffer.
    When the buffer is full, new elements overwrite the oldest elements.
    """

    def __init__(self, capacity: int, dtype: DTypeLike = ...) -> None:
        """
        Initializes the RingBuffer with a specified capacity and data type.

        Args:
            capacity: The maximum number of elements the buffer can hold.
            dtype: Desired data-type for the array. Default is np.float64.

        Raises:
            ValueError: If the capacity is not a positive integer.
        """
        ...

    @property
    def is_full(self) -> bool:
        """
        Checks if the buffer is full.

        Returns:
            True if the buffer is full, False otherwise.
        """
        ...

    @property
    def dtype(self) -> np.dtype[Any]:
        """
        Returns the data type of the buffer elements.

        Returns:
            The data type of the buffer elements.
        """
        ...

    @property
    def shape(self) -> tuple[int]:
        """
        Returns the shape of the buffer.

        Returns:
            The shape of the buffer as a single-element tuple.
        """
        ...

    @property
    def maxlen(self) -> int:
        """
        Returns the maximum length of the buffer.

        Returns:
            The maximum number of elements the buffer can hold.
        """
        ...

    def append(self, value: Any) -> None:
        """
        Appends a value to the right end of the buffer.

        If the buffer is full, the leftmost element is removed.

        Args:
            value: The value to append.
        """
        ...

    def appendleft(self, value: Any) -> None:
        """
        Appends a value to the left end of the buffer.

        If the buffer is full, the rightmost element is removed.

        Args:
            value: The value to append.
        """
        ...

    def clear(self) -> None:
        """
        Clears the buffer, removing all elements.
        """
        ...

    def peek(self) -> Any:
        """
        Returns the rightmost element without removing it.

        Returns:
            The rightmost element.

        Raises:
            IndexError: If the buffer is empty.
        """
        ...

    def peekleft(self) -> Any:
        """
        Returns the leftmost element without removing it.

        Returns:
            The leftmost element.

        Raises:
            IndexError: If the buffer is empty.
        """
        ...

    def pop(self) -> Any:
        """
        Removes and returns the rightmost element.

        Returns:
            The rightmost element.

        Raises:
            IndexError: If the buffer is empty.
        """
        ...

    def popleft(self) -> Any:
        """
        Removes and returns the leftmost element.

        Returns:
            The leftmost element.

        Raises:
            IndexError: If the buffer is empty.
        """
        ...

    def __array__(self) -> NDArray[Any]:
        """
        Returns the buffer as a numpy array.

        Returns:
            The buffer as a contiguous numpy array.
        """
        ...

    def __len__(self) -> int:
        """
        Returns the number of elements in the buffer.

        Returns:
            The number of elements currently in the buffer.
        """
        ...

    @overload
    def __getitem__(self, item: int) -> Any: ...
    @overload
    def __getitem__(self, item: slice) -> NDArray[Any]: ...
    @overload
    def __getitem__(self, item: ArrayLike) -> NDArray[Any]: ...
    def __getitem__(self, item: int | slice | ArrayLike) -> Any | NDArray[Any]:
        """
        Returns the element(s) at the specified index or slice.

        Args:
            item: The index, slice, or array of indices to retrieve.

        Returns:
            The element(s) at the specified index or slice.
        """
        ...

    def __iter__(self) -> Iterator[Any]:
        """
        Returns an iterator over the buffer.

        Returns:
            An iterator over the buffer elements.
        """
        ...

    def __repr__(self) -> str:
        """
        Returns a string representation of the buffer.

        Returns:
            A string representation of the buffer.
        """
        ...
