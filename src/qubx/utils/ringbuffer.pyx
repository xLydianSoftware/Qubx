import numpy as np
cimport numpy as cnp

cimport cython

cdef class RingBuffer:
    """
    A circular buffer implementation using numpy arrays.
    """
    
    def __init__(self, int capacity, dtype=np.float64):
        """
        Initializes the RingBuffer with a specified capacity and data type.
        
        Args:
            capacity (int): The maximum number of elements the buffer can hold.
            dtype (data-type, optional): Desired data-type for the array. Default is np.float64.
        
        Raises:
            ValueError: If the capacity is not a positive integer.
        """
        if capacity <= 0:
            raise ValueError("RingBuffer capacity must be positive integral value.")
        self._arr = np.zeros(capacity, dtype)
        self._left_index = 0
        self._right_index = 0
        self._capacity = capacity

    cdef cnp.ndarray _unwrap(self):
        """
        Unwraps the buffer into a contiguous numpy array.
        
        Returns:
            cnp.ndarray: The unwrapped buffer.
        """
        if self._right_index <= self._capacity:
            return self._arr[self._left_index:self._right_index].copy()
        else:
            return np.concatenate((
                self._arr[self._left_index:],
                self._arr[:self._right_index % self._capacity]
            ))

    cdef void _wraparound(self):
        """
        Adjusts the indices to ensure they remain within the buffer's capacity.
        """
        if self._left_index >= self._capacity:
            self._left_index -= self._capacity
            self._right_index -= self._capacity
        elif self._left_index < 0:
            self._left_index += self._capacity
            self._right_index += self._capacity

    property is_full:
        def __get__(self):
            """
            Checks if the buffer is full.
            
            Returns:
                bool: True if the buffer is full, False otherwise.
            """
            return self.check_full()

    property dtype:
        def __get__(self):
            """
            Returns the data type of the buffer elements.
            
            Returns:
                data-type: The data type of the buffer elements.
            """
            return self._arr.dtype

    property shape:
        def __get__(self):
            """
            Returns the shape of the buffer.
            
            Returns:
                tuple: The shape of the buffer.
            """
            return tuple([self._arr.shape[0]])

    property maxlen:
        def __get__(self):
            """
            Returns the maximum length of the buffer.
            
            Returns:
                int: The maximum number of elements the buffer can hold.
            """
            return self._capacity

    def __array__(self):
        """
        Returns the buffer as a numpy array.
        
        Returns:
            cnp.ndarray: The buffer as a numpy array.
        """
        return self._unwrap()

    cdef bint check_full(self):
        """
        Checks if the buffer is full.
        
        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return (self._right_index - self._left_index) == self._capacity

    cpdef void append(self, value):
        """
        Appends a value to the right end of the buffer.
        
        Args:
            value: The value to append.
        """
        if self.check_full():
            self._left_index += 1

        self._arr[self._right_index % self._capacity] = value
        self._right_index += 1
        self._wraparound()

    cpdef void appendleft(self, value):
        """
        Appends a value to the left end of the buffer.
        
        Args:
            value: The value to append.
        """
        if self.check_full():
            self._right_index -= 1

        self._left_index -= 1
        self._wraparound()
        self._arr[self._left_index] = value

    cpdef void clear(self):
        """
        Clears the buffer.
        """
        self._arr.fill(0)
        self._left_index = 0
        self._right_index = 0

    def peek(self):
        """
        Returns the rightmost element without removing it.
        
        Returns:
            The rightmost element.
        
        Raises:
            IndexError: If the buffer is empty.
        """
        if len(self) == 0:
            raise IndexError("peeked empty buffer")
        return self._arr[(self._right_index - 1) % self._capacity]

    def peekleft(self):
        """
        Returns the leftmost element without removing it.
        
        Returns:
            The leftmost element.
        
        Raises:
            IndexError: If the buffer is empty.
        """
        if len(self) == 0:
            raise IndexError("peeked empty buffer")
        return self._arr[self._left_index]

    def pop(self):
        """
        Removes and returns the rightmost element.
        
        Returns:
            The rightmost element.
        
        Raises:
            IndexError: If the buffer is empty.
        """
        if len(self) == 0:
            raise IndexError("popped empty buffer")
        self._right_index -= 1
        res = self._arr[self._right_index % self._capacity]
        return res

    def popleft(self):
        """
        Removes and returns the leftmost element.
        
        Returns:
            The leftmost element.
        
        Raises:
            IndexError: If the buffer is empty.
        """
        if len(self) == 0:
            raise IndexError("popped empty buffer")
        res = self._arr[self._left_index]
        self._left_index += 1
        self._wraparound()
        return res

    def __len__(self):
        """
        Returns the number of elements in the buffer.
        
        Returns:
            int: The number of elements in the buffer.
        """
        return self._right_index - self._left_index

    def __getitem__(self, item):
        """
        Returns the element(s) at the specified index or slice.
        
        Args:
            item (int or slice): The index or slice to retrieve.
        
        Returns:
            The element(s) at the specified index or slice.
        """
        indexer = np.asarray(item)
        with cython.boundscheck(True), cython.wraparound(True), cython.cdivision(False):
            if issubclass(indexer.dtype.type, np.integer):
                indexer = (indexer + self._left_index) % self._capacity
                return self._arr[indexer]

            return self._unwrap()[item]

    def __iter__(self):
        """
        Returns an iterator over the buffer.
        
        Returns:
            iterator: An iterator over the buffer.
        """
        return iter(self._unwrap())

    def __repr__(self):
        """
        Returns a string representation of the buffer.
        
        Returns:
            str: A string representation of the buffer.
        """
        return 'RingBuffer({!r})'.format(np.asarray(self))