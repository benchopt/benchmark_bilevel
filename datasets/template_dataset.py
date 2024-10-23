from benchopt import BaseDataset
from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import jax
    import jax.numpy as jnp
    from functools import partial  # useful for just-in-time compilation

    from jaxopt import LBFGS  # useful to define the value function


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
        """This method retrieves/simulates the data, defines the inner and
        outer objectives and the metrics to evaluate the results. It is
        mandatory for each dataset. The return arguments of this function are
        passed as keyword arguments to `Objective.set_data`.

        Returns
        -------
        data: dict
            A dictionary containing the keys `pb_inner`, `pb_outer`, `metrics`
            and optionnally `init_var`.

        The entries of the dictionary are:
        - `pb_inner`: tuple
            Contains the inner function, the number of inner samples, the
            dimension of the inner variable and the full batch version of the
            inner objective.

        - `pb_outer`: tuple
            Contains the outer function, the number of outer samples, the
            dimension of the outer variable and the full batch version of the
            outer objective.

        - `metrics`: function
            Function that computes the metrics of the problem.

        - `init_var`: function, optional
            Function that initializes the inner and outer variables.
        """

        # This decorator is used to jit the inner and the outer objective.
        # static_argnames=('batch_size') means that the function is recompiled
        # each time it is used with a new batch size.
        @partial(jax.jit, static_argnames=('batch_size'))
        def f_inner(inner_var, outer_var, start=0, batch_size=1):
            """Defines the inner objective function. It should be a pure jax
            function so that it can be jitted.

            Parameters
            ----------
            inner_var: pytree
                Inner variable.

            outer_var: pytree
                Outer variable.

            start: int, default=0
                For stochastic problems, index of the first sample of the
                batch.

            batch_size: int, default=1
                For stochastic problems, size of the batch.

            Returns
            -------
            float
                Value of the inner objective function.
            """
            return inner_var ** 2 + 2 * inner_var * outer_var

        # This is similar to f_inner
        @partial(jax.jit, static_argnames=('batch_size'))
        def f_outer(inner_var, outer_var, start=0, batch_size=1):
            return inner_var ** 2

        # For stochastic problems, it is useful to define the full batch
        # version of f_inner and f_outer, for instance to compute metrics
        # or to be used in some solvers (e.g. SRBA). For non-stochastic
        # problems, just define f_inner_fb = f_inner and f_outer_fb = f_outer.
        f_inner_fb = f_inner
        f_outer_fb = f_outer

        solver_inner = LBFGS(fun=f_inner_fb)

        # The value function is useful for the metrics. Note that it is not
        # mandatory to define it. In particular, for large scale problems,
        # evaluating it can be cumbersome.
        def value_function(outer_var):
            inner_var_star = solver_inner.run(
                jnp.zeros(self.dim_inner), outer_var
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

        return data
