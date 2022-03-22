import numpy as np
from scipy.sparse import csr_array, csc_array

from benchopt.utils.safe_import import set_benchmark
set_benchmark('.')

from objective import scipy_to_csrmatrix, scipy_to_cscmatrix  # noqa: E402


def test_csr_matrix():
    n_rows, n_cols = 1000, 300
    matrix = np.random.randn(n_rows, n_cols)

    scipy_matrix = csr_array(matrix)
    csr_matrix = scipy_to_csrmatrix(scipy_matrix)

    assert np.allclose(csr_matrix.toarray(), matrix)


def test_csr_dot():
    n_rows, n_cols = 1000, 300
    matrix = np.random.randn(n_rows, n_cols)
    v = np.random.randn(n_cols)

    scipy_matrix = csr_array(matrix)
    csr_matrix = scipy_to_csrmatrix(scipy_matrix)

    assert np.allclose(matrix.dot(v), csr_matrix.dot(v))

    v = np.random.randn(n_cols, 20)
    assert np.allclose(matrix.dot(v), csr_matrix.dot(v))


def test_csr_transpose():
    n_rows, n_cols = 1000, 300
    matrix = np.random.randn(n_rows, n_cols)

    scipy_matrix = csr_array(matrix)
    csr_matrix = scipy_to_csrmatrix(scipy_matrix)

    assert np.allclose(matrix.T, csr_matrix.T.toarray())


def test_csr_getitem():
    n_rows, n_cols = 1000, 300
    matrix = np.random.randn(n_rows, n_cols)

    scipy_matrix = csr_array(matrix)
    csr_matrix = scipy_to_csrmatrix(scipy_matrix)

    a, b = np.random.randint(0, n_rows, 2)
    a, b = min(a, b), max(a, b)

    assert np.allclose(matrix[a:b], csr_matrix[a:b].toarray())


def test_csc_matrix():
    n_rows, n_cols = 1000, 300
    matrix = np.random.randn(n_rows, n_cols)

    scipy_matrix = csc_array(matrix)
    csc_matrix = scipy_to_cscmatrix(scipy_matrix)

    assert np.allclose(csc_matrix.toarray(), matrix)


def test_csc_dot():
    n_rows, n_cols = 1000, 300
    matrix = np.random.randn(n_rows, n_cols)
    v = np.random.randn(n_cols)

    scipy_matrix = csc_array(matrix)
    csc_matrix = scipy_to_cscmatrix(scipy_matrix)

    assert np.allclose(matrix.dot(v), csc_matrix.dot(v))

    v = np.random.randn(n_cols, 20)
    assert np.allclose(matrix.dot(v), csc_matrix.dot(v))


def test_csc_transpose():
    n_rows, n_cols = 1000, 300
    matrix = np.random.randn(n_rows, n_cols)

    scipy_matrix = csc_array(matrix)
    csc_matrix = scipy_to_cscmatrix(scipy_matrix)

    assert np.allclose(matrix.T, csc_matrix.T.toarray())
