import numpy as np
from scipy.optimize import check_grad

from benchopt.utils.safe_import import set_benchmark
set_benchmark('.')

from objective import oracles  # noqa: E402


def _make_oracle(n_samples, n_features, n_classes):
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    f = oracles.MulticlassLogisticRegressionOracle(X, y, reg=True)

    outer_var = 2 * np.random.rand(n_features)
    outer_var = np.log(outer_var)

    inner_var = np.random.randn(n_features * n_classes)
    v = np.random.randn(n_features * n_classes)

    return f, inner_var, outer_var, v


def test_oracle_grad_inner():
    n_samples, n_features, n_classes = 1000, 10, 5

    f, inner_var, outer_var, v = _make_oracle(
        n_samples, n_features, n_classes
    )

    res = check_grad(f.get_value, f.get_grad_inner_var, inner_var, outer_var)
    assert res < 1e-6

    # check that the oracle is correct
    g_inner_var = f.get_grad_inner_var(inner_var, outer_var)
    _, g_inner_var_oracle, _, _ = f.get_oracles(
        inner_var, outer_var, v
    )
    assert np.allclose(g_inner_var, g_inner_var_oracle)


def test_oracle_grad_outer():
    n_samples, n_features, n_classes = 1000, 10, 5

    f, inner_var, outer_var, v = _make_oracle(
        n_samples, n_features, n_classes
    )

    def func(x):
        return f.get_value(inner_var, x)

    def grad(x):
        return f.get_grad_outer_var(inner_var, x)

    res = check_grad(func, grad, outer_var)
    assert res < 1e-6

    # check that the oracle is correct
    g_outer_var = f.get_grad_outer_var(inner_var, outer_var)
    _, g_outer_var_oracle = f.get_grad(inner_var, outer_var)
    assert np.allclose(g_outer_var, g_outer_var_oracle)


def test_oracle_cross():
    n_samples, n_features, n_classes = 1000, 10, 5

    f, inner_var, outer_var, v = _make_oracle(
        n_samples, n_features, n_classes
    )

    def func(x):
        return f.get_grad_inner_var(inner_var, x) @ v

    def grad(x):
        return f.get_cross(inner_var, x, v)

    res = check_grad(func, grad, outer_var)
    assert res < 1e-6

    # check that the oracle is correct
    cross_v = f.get_cross(inner_var, outer_var, v)
    _, _, _, cross_v_oracle = f.get_oracles(
        inner_var, outer_var, v
    )
    assert np.allclose(cross_v, cross_v_oracle)


def test_oracle_hvp():
    n_samples, n_features, n_classes = 1000, 10, 5

    f, inner_var, outer_var, v = _make_oracle(
        n_samples, n_features, n_classes
    )

    def func(x):
        return f.get_grad_inner_var(x, outer_var) @ v

    def grad(x):
        return f.get_hvp(x, outer_var, v)

    res = check_grad(func, grad, inner_var)
    assert res < 1e-6

    # check that the oracle is correct
    hvp = f.get_hvp(inner_var, outer_var, v)
    _, _, hvp_oracle, _ = f.get_oracles(
        inner_var, outer_var, v
    )
    assert np.allclose(hvp, hvp_oracle)


def test_oracle_inner_var_star():
    n_samples, n_features, n_classes = 1000, 10, 5

    f, inner_var, outer_var, v = _make_oracle(
        n_samples, n_features, n_classes
    )

    inner_var = f.get_inner_var_star(outer_var)
    g_inner_var_star = f.get_grad_inner_var(inner_var, outer_var)
    # XXX: Investigate why the norm is not small
    assert np.allclose(g_inner_var_star, 0)
