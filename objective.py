from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
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
            inner_var=jnp.zeros(inner_shape),
            outer_var=jnp.zeros(outer_shape)
        )

    def set_data(self, pb_inner, pb_outer, metrics, n_reg, reg=None,
                 oracle=None):

        (self.f_inner, self.n_samples_inner, self.dim_inner,
         self.f_inner_fb) = pb_inner
        (self.f_outer, self.n_samples_outer, self.dim_outer,
         self.f_outer_fb) = pb_outer
        self.metrics = metrics

        key = jax.random.PRNGKey(self.random_state)
        if oracle == "logreg":
            keys = jax.random.split(key, 2)
            self.inner_var0 = jax.random.normal(keys[0], (self.dim_inner, ))
            self.outer_var0 = jax.random.uniform(keys[1], (self.dim_outer, ))
            if reg == 'exp':
                self.outer_var0 = jnp.log(self.outer_var0)
            if n_reg == 1:
                self.outer_var0 = self.outer_var0[:1]
        else:
            self.inner_var0 = jnp.zeros(self.dim_inner)
            self.outer_var0 = -2 * jnp.ones(self.dim_outer)
        # XXX: Try random inits

    def evaluate_result(self, inner_var, outer_var):
        if jnp.isnan(outer_var).any():
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
