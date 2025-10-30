"""Nonce provider for generating unique request identifiers."""

import time

import numpy as np

from .ringbuffer import RingBuffer


class NonceProvider:
    """
    Provides unique nonce values for exchange requests.

    Uses a ring buffer to track recently generated nonces and ensures
    no duplicates or overlaps occur between consecutive requests.
    """

    def __init__(self, buffer_size: int = 50):
        """
        Initialize the NonceProvider.

        Args:
            buffer_size: Size of the ring buffer for tracking recent nonces.
                        Default is 50, which should be sufficient for most use cases.
        """
        self.nonce_buffer = RingBuffer(buffer_size, dtype=np.int64)

    def get_nonce(self) -> int:
        """
        Get a unique nonce value using a nonce buffer to ensure no overlaps
        between exchange requests.

        The nonce is based on the current time in nanoseconds. If the generated
        nonce already exists in the buffer, it uses the maximum value in the
        buffer plus one to ensure uniqueness.

        Returns:
            A unique nonce value.
        """
        nonce = time.time_ns()
        if nonce in self.nonce_buffer:
            nonce = int(np.max(np.asarray(self.nonce_buffer))) + 1
        nonce = int(nonce)
        self.nonce_buffer.append(nonce)
        return nonce

    def clear(self) -> None:
        """
        Clear the nonce buffer.

        This can be useful for testing or resetting the provider state.
        """
        self.nonce_buffer.clear()
