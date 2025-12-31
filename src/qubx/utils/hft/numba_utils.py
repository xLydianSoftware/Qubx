import numpy as np
from numba import njit


@njit
def argsort(arr):
    return np.argsort(arr)


@njit
def argmin(arr):
    return np.argmin(arr)
