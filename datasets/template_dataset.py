from benchopt import BaseDataset
from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    import jax
    import jax.numpy as jnp
    from functools import partial

    from jaxopt import LBFGS


def generate_matrices(dim_inner, dim_outer, key=jax.random.PRNGKey(0)):
    """Generates the different matrices of the inner and outer quadratic
    functions."""
    keys = jax.random.split(key, 4)
    eig_inner = jnp.logpsace(-1, 0, dim_inner)
    eig_outer = jnp.logpsace(-1, 0, dim_inner)
    eig_cross = jnp.logpsace(-1, 0, min(dim_inner, dim_outer))

    # Matrix generation for the inner function
    # Generate a PSD matrix with eigenvalues `eig_inner`
    hess_inner_inner = jax.random.normal(keys[0], (dim_inner, dim_inner))
    U, _, _ = jnp.linalg.svd(hess_inner_inner)
    hess_inner_inner = U @ jnp.diag(eig_inner) @ U.T

    # Generate a PSD matrix with eigenvalues `eig_outer`
    hess_outer_inner = jax.random.normal(keys[1], (dim_outer, dim_outer))
    U, _, _ = jnp.linalg.svd(hess_outer_inner)
    hess_outer_inner = U @ jnp.diag(eig_outer) @ U.T

    # Generate a PSD matrix with eigenvalues `eig_outer`
    cross_inner = jax.random.normal(keys[2], (dim_outer, dim_inner))
    D = jnp.zeros((dim_outer, dim_inner))
    D = D.at[:min(dim_outer, dim_inner), :min(dim_outer, dim_inner)].set(
        jnp.diag(eig_cross)
    )
    U, _, V = jnp.linalg.svd(cross_inner)
    cross_inner = U @ D @ V.T

    hess_inner_outer = jax.random.normal(keys[3], (dim_inner, dim_inner))
    U, _, _ = jnp.linalg.svd(hess_inner_outer)
    hess_inner_outer = U @ jnp.diag(eig_inner) @ U.T

    return hess_inner_inner, hess_outer_inner, cross_inner, hess_inner_outer


def quadratic(inner_var, outer_var, hess_inner, hess_outer, cross):
    res = .5 * inner_var @ (hess_inner @ inner_var)
    res += .5 * outer_var @ (hess_outer @ outer_var)
    res += outer_var @ cross @ inner_var
    return res


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    """How to add a new problem to the benchmark?

    This template dataset is an adaptation of the dataset from the benchopt
    template benchmark (https://github.com/benchopt/template_benchmark/) to
    the bilevel setting.
    """
    # Name to select the dataset in the CLI and to display the results.
    name = "Template dataset"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'dim_inner': [10],
        'dim_outer': [10],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data.

        hess_inner_inner, hess_outer_inner, cross, hess_inner_outer = (
            generate_matrices(
                self.dim_inner, self.dim_outer
            )
        )

        @partial(jax.jit, static_argnames=('batch_size'))
        def f_inner(inner_var, outer_var, start=0, batch_size=1):
            return quadratic(inner_var, outer_var,
                             hess_inner_inner, hess_outer_inner,
                             cross)

        @partial(jax.jit, static_argnames=('batch_size'))
        def f_outer(inner_var, outer_var, start=0, batch_size=1):
            return quadratic(inner_var, outer_var, hess_inner_outer,
                             jnp.zeros_like(hess_outer_inner),
                             jnp.zeros_like(cross))

        f_inner_fb = partial(
            f_inner, batch_size=X_train.shape[0], start=0
        )
        f_outer_fb = partial(
            f_outer, batch_size=X_val.shape[0], start=0
        )

        solver_inner = LBFGS(fun=f_inner_fb)

        def value_function(outer_var):
            inner_var_star = solver_inner.run(
                jnp.zeros(X_train.shape[1]), outer_var
            ).params

            return f_outer_fb(inner_var_star, outer_var), inner_var_star

        value_and_grad = jax.jit(
            jax.value_and_grad(value_function, has_aux=True)
        )

        def metrics(inner_var, outer_var):
            # Defines the metrics that are computed when calling the method
            # Objective.evaluating_results(inner_var, outer_var) and saved
            # in the result file. The output is a dictionary that contains at
            # least the key `value`. The keyword arguments of this function are
            # the keys of the dictionary returned by `Solver.get_result`.
            (value_fun, inner_star), grad_value = value_and_grad(outer_var)
            return dict(
                value_func=float(value_fun),
                value=float(jnp.linalg.norm(grad_value)**2),
                inner_distance=float(jnp.linalg.norm(inner_star-inner_var)**2),
                norm_outer_var=float(jnp.linalg.norm(outer_var)**2),
                norm_regul=float(jnp.linalg.norm(np.exp(outer_var))**2),
            )

        def init_var(key):
            # Provides an initialization of inner_var and outer_var.
            keys = jax.random.split(key, 2)
            inner_var0 = jax.random.normal(keys[0], (self.dim_inner,))
            outer_var0 = jax.random.uniform(keys[1], (self.dim_outer,))
            if self.reg_parametrization == 'exp':
                outer_var0 = jnp.log(outer_var0)
            return inner_var0, outer_var0

        data = dict(
            pb_inner=(f_inner, self.n_samples_inner, self.dim_inner,
                      f_inner_fb),
            pb_outer=(f_outer, self.n_samples_outer, self.dim_outer,
                      f_outer_fb),
            metrics=metrics,
            init_var=init_var,
        )

        # The output should be a dict that contains the keys `pb_inner`,
        # `pb_outer`, `metrics`, and optionnally `init_var`.
        # `pb_inner`` is a tuple that contains the inner function, the number
        # of inner samples, the dimension of the inner variable and the full
        # batch version of the inner version.
        # `pb_outer` in analogous.
        # The key `metrics` contains the function `metrics`.
        # The key `init_var` contains the function `init_var` when applicable.
        return data
