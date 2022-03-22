import numpy as np

from numba import njit, int32, float64
from numba.experimental import jitclass

spec = [
    ('data', float64[:]),
    ('indices', int32[:]),
    ('indptr', int32[:]),
    ('shape', int32[:]),
    ('nnz', int32)
]


@jitclass(spec)
class CSRMatrix():

    def __init__(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = np.array([shape[0], shape[1]], dtype=np.int32)

        self.nnz = np.int32(data.shape[0])

    def toarray(self):
        res = np.zeros((self.shape[0], self.shape[1]), dtype=np.float64)
        n_rows = len(self.indptr)
        for i in range(n_rows):
            row_start = self.indptr[i]
            row_end = self.indptr[i+1]
            for j in range(row_start, row_end):
                res[i, self.indices[j]] = self.data[j]
        return res

    def dot(self, v):
        """v is a numpy array"""
        assert v.shape[0] == self.shape[1]
        n_rows = self.shape[0]

        if v.ndim == 1:
            res = np.zeros(n_rows, dtype=np.float64)
            for i in range(n_rows):
                row_start = self.indptr[i]
                row_end = self.indptr[i+1]

                cols = self.indices[row_start:row_end]
                res[i] = np.sum(self.data[row_start:row_end] * v[cols])
        elif v.ndim == 2:
            res = np.zeros((n_rows, v.shape[1]), dtype=np.float64)
            for j in range(v.shape[1]):
                for i in range(n_rows):
                    row_start = self.indptr[i]
                    row_end = self.indptr[i+1]

                    cols = self.indices[row_start:row_end]
                    res[i, j] = np.sum(
                        self.data[row_start:row_end] * v[cols, j]
                    )

        return res

    @property
    def T(self):
        return CSCMatrix(
            self.data, self.indices, self.indptr,
            (self.shape[1], self.shape[0])
        )

    def __getitem__(self, row_slice):
        return _extract_csr_rows(self, row_slice)


@jitclass(spec)
class CSCMatrix():

    def __init__(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = np.array([shape[0], shape[1]], dtype=np.int32)

        self.nnz = np.int32(data.shape[0])

    def toarray(self):
        res = np.zeros((self.shape[0], self.shape[1]), dtype=np.float64)
        n_cols = len(self.indptr)
        for i in range(n_cols):
            col_start = self.indptr[i]
            col_end = self.indptr[i+1]
            for j in range(col_start, col_end):
                res[self.indices[j], i] = self.data[j]
        return res

    def dot(self, v):
        """v is a numpy array"""
        assert v.shape[0] == self.shape[1]
        n_rows = self.shape[0]
        n_cols = self.shape[1]

        if v.ndim == 1:
            res = np.zeros(n_rows, dtype=np.float64)
            for i in range(n_cols):
                col_start = self.indptr[i]
                col_end = self.indptr[i+1]

                rows = self.indices[col_start:col_end]
                res[rows] += self.data[col_start:col_end] * v[i]
        elif v.ndim == 2:
            res = np.zeros((n_rows, v.shape[1]), dtype=np.float64)
            for j in range(v.shape[1]):
                for i in range(n_cols):
                    col_start = self.indptr[i]
                    col_end = self.indptr[i+1]

                    rows = self.indices[col_start:col_end]
                    res[rows, j] += self.data[col_start:col_end] * v[i, j]

        return res

    @property
    def T(self):
        return CSRMatrix(
            self.data, self.indices, self.indptr,
            (self.shape[1], self.shape[0])
        )


def scipy_to_csrmatrix(X):
    data = X.data
    indices = X.indices
    indptr = X.indptr
    csr_matrix = CSRMatrix(data, indices, indptr, X.shape)
    return csr_matrix


def scipy_to_cscmatrix(X):
    data = X.data
    indices = X.indices
    indptr = X.indptr
    csc_matrix = CSCMatrix(data, indices, indptr, X.shape)
    return csc_matrix


@njit
def _extract_csr_rows(X, row_slice):
    rows = np.arange(X.shape[0])[row_slice]
    n_rows = rows.shape[0]
    indptr = np.zeros(n_rows + 1, dtype=np.int32)

    rows_start = X.indptr[rows[0]]
    rows_end = X.indptr[rows[-1] + 1]

    data = X.data[rows_start:rows_end]
    indices = X.indices[rows_start:rows_end]
    for i, j in zip(range(1, rows.shape[0]), rows[1:]):
        indptr[i] = indptr[i-1] + X.indptr[j] - X.indptr[j-1]
    indptr[-1] = data.shape[0]
    return CSRMatrix(data, indices, indptr, (n_rows, X.shape[1]))
