from benchopt import BaseDataset
from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from libsvmdata import fetch_libsvm

    import jax
    import jax.numpy as jnp
    from functools import partial

    from jaxopt import LBFGS


def loss_sample(inner_var, outer_var, x, y):
    return -jax.nn.log_sigmoid(y*jnp.dot(inner_var, x))


def loss(inner_var, outer_var, X, y):
    batched_loss = jax.vmap(loss_sample, in_axes=(None, None, 0, 0))
    return jnp.mean(batched_loss(inner_var, outer_var, X, y), axis=0)


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    """Hyperparameter optimization with IJCNN1 dataset."""
    # Name to select the dataset in the CLI and to display the results.
    name = "ijcnn1"
    """How to add a new problem to the benchmark?

    This template dataset is an adaptation of the dataset from the benchopt
    template benchmark (https://github.com/benchopt/template_benchmark/) to
    the bilevel setting.
    """

    install_cmd = 'conda'
    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = ['pip:libsvmdata', 'scikit-learn']

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'reg_parametrization': ['exp'],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data.

        X_train, y_train = fetch_libsvm('ijcnn1')
        X_val, y_val = fetch_libsvm('ijcnn1_test')

        X_train, y_train = jnp.array(X_train), jnp.array(y_train)
        X_val, y_val = jnp.array(X_val), jnp.array(y_val)

        self.n_samples_inner = X_train.shape[0]
        self.dim_inner = X_train.shape[1]
        self.n_samples_outer = X_val.shape[0]
        self.dim_outer = X_val.shape[1]

        @partial(jax.jit, static_argnames=('batch_size'))
        def f_inner(inner_var, outer_var, start=0, batch_size=1):
            x = jax.lax.dynamic_slice(
                X_train, (start, 0), (batch_size, X_train.shape[1])
            )
            y = jax.lax.dynamic_slice(
                y_train, (start, ), (batch_size, )
            )
            res = loss(inner_var, outer_var, x, y)

            if self.reg_parametrization == 'exp':
                res += jnp.dot(jnp.exp(outer_var) * inner_var, inner_var)/2
            elif self.reg_parametrization == 'lin':
                res += jnp.dot(outer_var * inner_var, inner_var)/2
            return res

        @partial(jax.jit, static_argnames=('batch_size'))
        def f_outer(inner_var, outer_var, start=0, batch_size=1):
            x = jax.lax.dynamic_slice(
                X_val, (start, 0), (batch_size, X_val.shape[1])
            )
            y = jax.lax.dynamic_slice(
                y_val, (start, ), (batch_size, )
            )
            res = loss(inner_var, outer_var, x, y)
            return res

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
