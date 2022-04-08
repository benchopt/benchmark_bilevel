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
    """Numba class implementing Compressed Sparse Row Matrices.

    Parameters
    ----------
    data: vector, shape (nnz, ), dtype float64
          CSR format data array of the matrix

    indices: vector, shape (nnz, ), dtype int32
             CSR format index array of the matrix

    ipdptr: vector, shape (n_row, ), dtype int32
            CSR format index pointer array of the matrix


    Attributes
    ----------
    data: vector, shape (nnz, )
        CSR format data array of the matrix

    indices: vector, shape (nnz, )
             CSR format index array of the matrix

    ipdptr: vector, shape (n_row, )
            CSR format index pointer array of the matrix

    shape: vector, shape (2, )
           Shape of the matrix

    nnz: int
         Number of stored values, including explicit zeros
    """

    def __init__(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = np.array([shape[0], shape[1]], dtype=np.int32)

        self.nnz = np.int32(data.shape[0])

    def toarray(self):
        """Returns a copy of the matrix as a numpy array
        """
        res = np.zeros((self.shape[0], self.shape[1]), dtype=np.float64)
        n_rows = len(self.indptr) - 1
        for i in range(n_rows):
            row_start = self.indptr[i]
            row_end = self.indptr[i+1]
            for j in range(row_start, row_end):
                res[i, self.indices[j]] = self.data[j]
        return res

    def dot(self, v):
        """Returns the product of the matrix with v."""
        assert v.shape[0] == self.shape[1]
        n_rows = self.shape[0]

        if v.ndim == 1:
            res = np.zeros(n_rows, dtype=np.float64)
            for i in range(n_rows):
                row_start = self.indptr[i]
                row_end = self.indptr[i+1]
                for col in range(row_start, row_end):
                    res[i] += self.data[col] * v[self.indices[col]]
        elif v.ndim == 2:
            res = np.zeros((n_rows, v.shape[1]), dtype=np.float64)
            for j in range(v.shape[1]):
                for i in range(n_rows):
                    row_start = self.indptr[i]
                    row_end = self.indptr[i+1]
                    for col in range(row_start, row_end):
                        res[i, j] += self.data[col] * v[self.indices[col], j]
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
    """Numba class implementing Compressed Sparse Columns Matrices.

    Parameters
    ----------
    data: vector, shape (nnz, ), dtype float64
          CSR format data array of the matrix

    indices: vector, shape (nnz, ), dtype int32
             CSR format index array of the matrix

    ipdptr: vector, shape (n_row, ), dtype int32
            CSR format index pointer array of the matrix


    Attributes
    ----------
    data: vector, shape (nnz, )
        CSR format data array of the matrix

    indices: vector, shape (nnz, )
             CSR format index array of the matrix

    ipdptr: vector, shape (n_row, )
            CSR format index pointer array of the matrix

    shape: vector, shape (2, )
           Shape of the matrix

    nnz: int
         Number of stored values, including explicit zeros
    """

    def __init__(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = np.array([shape[0], shape[1]], dtype=np.int32)

        self.nnz = np.int32(data.shape[0])

    def toarray(self):
        """Returns a copy of the matrix as a numpy array
        """
        res = np.zeros((self.shape[0], self.shape[1]), dtype=np.float64)
        n_cols = len(self.indptr) - 1
        for i in range(n_cols):
            col_start = self.indptr[i]
            col_end = self.indptr[i+1]
            for j in range(col_start, col_end):
                res[self.indices[j], i] = self.data[j]
        return res

    def dot(self, v):
        """Returns the product of the matrix with v."""
        assert v.shape[0] == self.shape[1]
        n_rows, n_cols = self.shape

        if v.ndim == 1:
            res = np.zeros(n_rows, dtype=np.float64)
            for i in range(n_cols):
                col_start = self.indptr[i]
                col_end = self.indptr[i+1]
                for row in range(col_start, col_end):
                    res[self.indices[row]] += self.data[row] * v[i]
        elif v.ndim == 2:
            res = np.zeros((n_rows, v.shape[1]), dtype=np.float64)
            for j in range(v.shape[1]):
                for i in range(n_cols):
                    col_start = self.indptr[i]
                    col_end = self.indptr[i+1]

                    for row in range(col_start, col_end):
                        res[self.indices[row], j] += self.data[row] * v[i, j]

        return res

    @property
    def T(self):
        return CSRMatrix(
            self.data, self.indices, self.indptr,
            (self.shape[1], self.shape[0])
        )


def scipy_to_csrmatrix(X):
    """Converts a CSR scipy matrix to a CSR numba matrix."""
    data = X.data
    indices = X.indices
    indptr = X.indptr
    csr_matrix = CSRMatrix(data, indices, indptr, X.shape)
    return csr_matrix


def scipy_to_cscmatrix(X):
    """Converts a CSC scipy matrix to a CSC numba matrix."""
    data = X.data
    indices = X.indices
    indptr = X.indptr
    csc_matrix = CSCMatrix(data, indices, indptr, X.shape)
    return csc_matrix


@njit
def _extract_csr_rows(X, row_slice):
    """Selects the rows of X with the slice row_slice."""
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
