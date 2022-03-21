import numpy as np
from operator import add
import scipy.sparse as sparse

from numba import njit, int32, float64
from numba.experimental import jitclass
from numba.typed import Dict
from numba.core.types import int32

spec = [
    ('data', float64[::1]),
    ('indices', int32[::1]),
    ('indptr', int32[::1]),
    ('shape', int32[::1]),
    ('nnz', int32)
]


@jitclass(spec)
class CSRMatrix():

    def __init__(self, data, indices, indptr, shape=None):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        if shape is None:
            self.shape = np.array(
                [indptr.shape[0], indptr[-1]], dtype=np.int32)
        else:
            self.shape = np.array([shape[0], shape[1]], dtype=np.int32)
        self.nnz = np.int32(data.shape[0])

    def toarray(self):
        res = np.zeros((self.shape[0], self.shape[1]), dtype=np.float64)
        n_rows = len(self.indptr)
        for i in range(n_rows):
            row_start = self.indptr[i]
            row_end = len(self.data) if i == n_rows-1 else self.indptr[i+1]
            for j in range(row_start, row_end):
                res[i, self.indices[j]] = self.data[j]
        return res

    def dot(self, v):
        """v is a numpy array"""
        assert v.shape[0] == self.shape[1]
        n_rows = self.shape[0]

        res = np.zeros(n_rows, dtype=np.float64)
        for i in range(n_rows):
            row_start = self.indptr[i]
            row_end = self.indptr[i+1]

            cols = self.indices[row_start:row_end]
            res[i] = np.sum(self.data[row_start:row_end] * v[cols])

        return res

    # def __add__(self, other):
    #     return _coefwise_csr_matrix_operation(add, self, other)


    # def __getitem__(self, idx):
    #     if isinstance(idx, tuple):
    #         return X.getrow(idx[0]).getcol(idx[1])


def scipy_to_csrmatrix(X):
    data = X.data
    indices = X.indices
    indptr = X.indptr
    csr_matrix = CSRMatrix(data, indices, indptr, shape=X.shape)
    return csr_matrix


@njit
def _coefwise_csr_matrix_operation(op, x, y):
    """
    NOTE: Doesn't work because isinstance is not available in nopython mode.
    """
    assert x.shape == y.shape

    if (isinstance(x, CSRMatrix) or isinstance(y, CSRMatrix)):
        n_rows, n_cols = x.shape
        if isinstance(x, CSRMatrix) and isinstance(y, CSRMatrix):
            data = []
            indices = []
            indptr = np.zeros(n_rows + 1, dtype=np.int32)
            for r in range(n_rows):
                row_start_x = x.indptr[r]
                row_end_x = len(x.data) if r == n_rows-1 else x.indptr[r+1]
                row_start_y = y.indptr[r]
                row_end_y = len(y.data) if r == n_rows-1 else y.indptr[r+1]


                x_col = _build_numba_dict(
                    x.indices[row_start_x:row_end_x], len(data)
                )
                y_col = _build_numba_dict(
                    y.indices[row_start_y:row_end_y], len(data)
                )
                # x_col = dict(
                #     (k, i + len(data))
                #     for i, k in enumerate(x.indices[row_start_x:row_end_x])
                # )
                # y_col = dict(
                #         (k, i + len(data))
                #         for i, k in enumerate(y.indices[row_start_y:row_end_y])
                #     )

                cols_set = set(x_col).union(set(y_col))
                for col in cols_set:
                    if col not in x_col:
                        if col not in y_col:
                            s = op(0, 0)
                        else:
                            s = op(0, y.data[y_col[col]])
                    else:
                        if col not in y_col:
                            s = op(x.data[x_col[col]], 0)
                        else:
                            s = op(x.data[x_col[col]], y.data[y_col[col]])
                    if s != 0:
                        data.append(s)
                        indices.append(col)

                indptr[r + 1] = len(data)
            return CSRMatrix(np.array(data), np.array(indices), indptr,
                             shape=x.shape)
        elif not isinstance(y, CSRMatrix):
            return op(x.toarray(), y)
        else:
            return op(x, y.toarray())
    else:
        return op(x, y)


@njit
def _build_numba_dict(tab, offset):
    d = Dict.empty(
        key_type=int32,
        value_type=int32,
    )
    for i, k in enumerate(tab):
        d[k] = i + offset
    return d

# def _extract_csr_submatrix(X, row_idx=None, col_idx=None):
#     if row_idx is None:
#         rows_start = 0
#         rows_end = len(X.indptr)
#     else:
#         rows_start = X.indptr[row_idx]
#         rows_end = X.indptr[row_idx + 1]

#     if col_idx is None:
#         col_idx = np.arange(0, X.indices.max())

#     row_slice = np.r_[
#         slice(rows_start[i], rows_end[i]) for i in range(len(row_idx))
#     ]
#     values = X.data[row_slice]
#     cols = X.indices[row_slice]
#     cols = cols[cols == col_idx]



#         sub_matrix = CSRMatrix(X.data[row_start:row_end], col_idx, np.array([0]))

