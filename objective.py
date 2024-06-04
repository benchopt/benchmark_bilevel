from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import operator
    import jax
    import jax.numpy as jnp


class Objective(BaseObjective):
    name = "Bilevel Optimization"
    url = "https://github.com/benchopt/benchmark_bilevel"

    requirements = ["scikit-learn", "jax", "jaxlib"]
    min_benchopt_version = "1.5"

    parameters = {
        'random_state': [2442]
    }

    def __init__(self, random_state=2442):
        self.random_state = random_state

    def get_one_result(self):
        inner_shape, outer_shape = self.dim_inner, self.dim_outer
        return dict(
            inner_var=jax.tree_util.tree_map(jnp.zeros, inner_shape, is_leaf=lambda x: isinstance(x, tuple)),  # can be a pytree
            outer_var=jax.tree_util.tree_map(jnp.zeros, outer_shape, is_leaf=lambda x: isinstance(x, tuple)),
        )

    def set_data(self, pb_inner, pb_outer, metrics, init_var=None):

        (self.f_inner, self.n_samples_inner, self.dim_inner,
         self.f_inner_fb) = pb_inner
        (self.f_outer, self.n_samples_outer, self.dim_outer,
         self.f_outer_fb) = pb_outer
        self.metrics = metrics

        key = jax.random.PRNGKey(self.random_state)
        if init_var is not None:
            # Define random inits per datasets
            self.inner_var0, self.outer_var0 = init_var(key)
        else:
            self.inner_var0 = jax.tree_util.tree_map(jnp.zeros, self.dim_inner, is_leaf=lambda x: isinstance(x, tuple))
            self.outer_var0 = jax.tree_util.tree_map(lambda x: - 2 * jnp.ones(x), self.dim_outer, is_leaf=lambda x: isinstance(x, tuple))

    def evaluate_result(self, inner_var, outer_var):
        if jax.tree_util.tree_reduce(operator.or_, jax.tree_util.tree_map(lambda x: jnp.isnan(x).any(), outer_var)):
            raise ValueError

        metrics = self.metrics(inner_var, outer_var)
        return metrics

    def get_objective(self):
        return dict(
            f_inner=self.f_inner,
            f_outer=self.f_outer,
            n_inner_samples=self.n_samples_inner,
            n_outer_samples=self.n_samples_outer,
            inner_var0=self.inner_var0,
            outer_var0=self.outer_var0
        )
