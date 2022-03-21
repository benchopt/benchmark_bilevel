import numpy as np
from scipy.sparse import csr_array

from benchopt.utils.safe_import import set_benchmark
set_benchmark('.')

from objective import scipy_to_csrmatrix  # noqa: E402


def test_csr_matrix():
    n_rows, n_cols = 1000, 300
    matrix = np.random.randn(n_rows, n_cols)

    scipy_matrix = csr_array(matrix)
    csr_matrix = scipy_to_csrmatrix(scipy_matrix)

    assert np.allclose(csr_matrix.toarray(), matrix)


def test_dot():
    n_rows, n_cols = 1000, 300
    matrix = np.random.randn(n_rows, n_cols)
    v = np.random.randn(n_cols)

    scipy_matrix = csr_array(matrix)
    csr_matrix = scipy_to_csrmatrix(scipy_matrix)

    assert np.allclose(matrix.dot(v), csr_matrix.dot(v))
