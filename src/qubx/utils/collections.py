"""
Fast, fixed-size circular buffer implementation using Numba for performance.
When the deque is full, pushing new elements overwrites the oldest elements.
"""

from typing import Any, Protocol, Type, TypeVar, cast, runtime_checkable

import numpy as np
from numba import from_dtype, int32, types
from numba.experimental import jitclass


@runtime_checkable
class DequeProtocol(Protocol):
    """Protocol defining the interface that all Deque instances implement."""

    def __init__(self, capacity: int) -> None: ...

    def push_back(self, record: Any) -> None: ...

    def push_front(self, record: Any) -> None: ...

    def pop_front(self) -> Any: ...

    def pop_back(self) -> Any: ...

    def is_empty(self) -> bool: ...

    def is_full(self) -> bool: ...

    def get_size(self) -> int: ...

    def __len__(self) -> int: ...

    def front(self) -> Any: ...

    def back(self) -> Any: ...

    def __getitem__(self, idx: int) -> Any: ...

    def to_array(self) -> np.ndarray: ...

    def push_back_fields(self, *args: Any) -> None: ...


T = TypeVar("T", bound=DequeProtocol)


def create_deque_class(element_dtype: np.dtype) -> Type[DequeProtocol]:
    """
    Dynamically create and return a Deque jitclass that stores elements of the given NumPy dtype.

    This is a fast, fixed-size circular buffer implementation using Numba for performance.
    When the deque is full, pushing new elements overwrites the oldest elements.

    Args:
        element_dtype (np.dtype): The NumPy dtype for elements to store

    Returns:
        Type[DequeProtocol]: A compiled Numba class for the deque that implements the DequeProtocol

    Raises:
        ValueError: If the element_dtype is not supported
    """
    # Determine whether it's a structured dtype or a scalar dtype
    if element_dtype.fields is not None:
        # It's a structured dtype. Convert to a Numba 'record' type
        element_type = from_dtype(element_dtype)
        data_type = element_type[:]  # 1D array of that struct
    else:
        # It's a scalar dtype (e.g., float32)
        if element_dtype == np.float32:
            data_type = types.float32[:]
        elif element_dtype == np.float64:
            data_type = types.float64[:]
        elif element_dtype == np.int32:
            data_type = types.int32[:]
        elif element_dtype == np.int64:
            data_type = types.int64[:]
        else:
            raise ValueError(f"Unsupported scalar dtype: {element_dtype}")

    # Build the class spec
    spec = [
        ("data", data_type),
        ("capacity", int32),
        ("head", int32),
        ("tail", int32),
        ("size", int32),
    ]

    class Deque:  # type: ignore
        def __init__(self, capacity: int) -> None:
            self.data = np.empty(capacity, dtype=element_dtype)
            self.capacity = capacity
            self.head = 0
            self.tail = 0
            self.size = 0

        def push_back(self, record: Any) -> None:
            if self.size == self.capacity:
                # Overwrite oldest from the front
                self.head = (self.head + 1) % self.capacity
                self.size -= 1

            self.data[self.tail] = record
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1

        def push_front(self, record: Any) -> None:
            if self.size == self.capacity:
                # Overwrite oldest from the back
                self.tail = (self.tail - 1) % self.capacity
                self.size -= 1

            self.head = (self.head - 1) % self.capacity
            self.data[self.head] = record
            self.size += 1

        def pop_front(self) -> Any:
            if self.size == 0:
                raise IndexError("Deque is empty")
            record = self.data[self.head]
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
            return record

        def pop_back(self) -> Any:
            if self.size == 0:
                raise IndexError("Deque is empty")
            self.tail = (self.tail - 1) % self.capacity
            record = self.data[self.tail]
            self.size -= 1
            return record

        def is_empty(self) -> bool:
            return self.size == 0

        def is_full(self) -> bool:
            return self.size == self.capacity

        def get_size(self) -> int:
            return self.size

        def __len__(self) -> int:
            return self.size

        def front(self) -> Any:
            if self.size == 0:
                raise IndexError("Deque is empty")
            return self.data[self.head]

        def back(self) -> Any:
            if self.size == 0:
                raise IndexError("Deque is empty")
            return self.data[(self.tail - 1) % self.capacity]

        def __getitem__(self, idx: int) -> Any:
            if idx < 0 or idx >= self.size:
                raise IndexError("Index out of bounds")
            return self.data[(self.tail - idx - 1) % self.capacity]

        def to_array(self) -> np.ndarray:
            """Return a NumPy array of the current elements in the Deque, from oldest to newest."""
            out = np.empty(self.size, dtype=self.data.dtype)
            for i in range(self.size):
                idx = (self.head + i) % self.capacity
                out[i] = self.data[idx]
            return out

        def push_back_fields(self, *args: Any) -> None: ...

    # Generate push_back_fields(...) for structured dtypes
    if element_dtype.fields is not None:
        field_names = list(element_dtype.fields.keys())
        arg_list = ", ".join(field_names)
        lines = []
        lines.append(f"def push_back_fields(self, {arg_list}):")
        lines.append("    if self.size == self.capacity:")
        lines.append("        self.head = (self.head + 1) % self.capacity")
        lines.append("        self.size -= 1")
        lines.append("")
        for f in field_names:
            lines.append(f"    self.data[self.tail]['{f}'] = {f}")
        lines.append("    self.tail = (self.tail + 1) % self.capacity")
        lines.append("    self.size += 1")

        method_src = "\n".join(lines)
        tmp_ns = {}
        exec(method_src, {}, tmp_ns)
        push_back_fields_func = tmp_ns["push_back_fields"]
        setattr(Deque, "push_back_fields", push_back_fields_func)

    _Deque = jitclass(spec)(Deque)  # type: ignore
    return cast(Type[DequeProtocol], _Deque)  # Cast the jitclass to our protocol type


# Pre-compiled deque types for common use cases
DequeFloat64 = create_deque_class(np.dtype(np.float64))
DequeFloat32 = create_deque_class(np.dtype(np.float32))
DequeInt64 = create_deque_class(np.dtype(np.int64))
DequeInt32 = create_deque_class(np.dtype(np.int32))

# Deque type for storing indicator values with timestamps
DequeIndicator = create_deque_class(
    np.dtype(
        [
            ("timestamp", np.int64),
            ("value", np.float64),
        ],
        align=True,
    )
)

# Instance types for use in other jitclasses
DequeIndicator_instance = DequeIndicator.class_type.instance_type  # type: ignore
DequeFloat64_instance = DequeFloat64.class_type.instance_type  # type: ignore
DequeFloat32_instance = DequeFloat32.class_type.instance_type  # type: ignore
DequeInt64_instance = DequeInt64.class_type.instance_type  # type: ignore
DequeInt32_instance = DequeInt32.class_type.instance_type  # type: ignore
