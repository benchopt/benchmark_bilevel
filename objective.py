from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.utils import check_random_state


class Objective(BaseObjective):
    name = "Bilevel Optimization"
    url = "https://github.com/benchopt/benchmark_bilevel"

    requirements = ["scikit-learn", "numba", "jax"]
    min_benchopt_version = "1.5"

    parameters = {
        'random_state': [2442]
    }

    def __init__(self, random_state=2442):
        self.random_state = random_state

    def get_one_result(self):
        inner_shape, outer_shape = self.get_inner_oracle().variables_shape
        return dict(inner_var=np.zeros(*inner_shape),
                    outer_var=np.zeros(*outer_shape),
                    memory=0.)

    def set_data(self, get_inner_oracle, get_outer_oracle, oracle, metrics,
                 n_reg):

        self.get_inner_oracle = get_inner_oracle
        self.get_outer_oracle = get_outer_oracle
        self.metrics = metrics

        rng = check_random_state(self.random_state)
        inner_shape, outer_shape = self.get_inner_oracle().variables_shape
        if oracle == "logreg":
            self.inner_var0 = rng.randn(*inner_shape)
            self.outer_var0 = rng.rand(*outer_shape)
            if self.get_inner_oracle().reg == 'exp':
                self.outer_var0 = np.log(self.outer_var0)
            if n_reg == 1:
                self.outer_var0 = self.outer_var0[:1]
        else:
            self.inner_var0 = np.zeros(*inner_shape)
            self.outer_var0 = -2 * np.ones(*outer_shape)
            # XXX: Try random inits

    def evaluate_result(self, inner_var, outer_var, memory):
        if np.isnan(outer_var).any():
            raise ValueError

        metrics = self.metrics(inner_var, outer_var)
        metrics.update({'memory': memory})
        return metrics

    def get_objective(self):
        return dict(
            f_train=self.get_inner_oracle,
            f_val=self.get_outer_oracle,
            n_inner_samples=self.get_inner_oracle().n_samples,
            n_outer_samples=self.get_outer_oracle().n_samples,
            inner_var0=self.inner_var0,
            outer_var0=self.outer_var0,
        )
