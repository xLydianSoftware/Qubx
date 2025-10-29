import numpy as np
cimport numpy as cnp

cdef class RingBuffer:
    cdef:
        cnp.ndarray _arr
        int _left_index
        int _right_index
        int _capacity

    cdef cnp.ndarray _unwrap(self)
    cdef void _wraparound(self)
    cdef bint check_full(self)
    
    cpdef void append(self, value)
    cpdef void appendleft(self, value)
    cpdef void clear(self)