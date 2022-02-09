import numpy as np
from numba import njit


@njit
def np_apply_along_axis(func1d, axis, arr):
    """Since the argument 'axis' is not available for many numpy functions in
    numba, we can use this workaround found in [1]. Works only for 2d-arrays.

    [1] https://github.com/numba/numba/issues/1269#issuecomment-472574352"""
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@njit
def np_max(array, axis=0):
    return np_apply_along_axis(np.max, axis, array)


@njit
def np_mean(array, axis=0):
    return np_apply_along_axis(np.mean, axis, array)
