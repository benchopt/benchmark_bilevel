
from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Single loop."""
    name = 'single-loop'

    stop_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {'step_size': [1e-4, 1e-2, 1, 1e2, 'auto']}

    def set_objective(self, f_train, f_test, inner_var0, outer_var0):
        self.f_inner = f_train
        self.f_outer = f_test
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

    def run(self, callback):
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        v = np.zeros_like(inner_var)
        while callback((inner_var, outer_var)):
            _, grad_inner_fun, cross_inner_fun_v, hvp = (
                self.f_inner.get_oracles(inner_var, outer_var, v)
            )
            outer_grad_inner_var, outer_grad_outer_var = self.f_outer.get_grad(
                inner_var,
                outer_var
            )

            if self.step_size == 'auto':
                inner_step_size = 1 / self.f_inner.lipschitz_inner(
                    inner_var, outer_var
                )
                outer_step_size = 1
            else:
                inner_step_size = outer_step_size = self.step_size

            inner_var -= inner_step_size * grad_inner_fun
            v -= inner_step_size * (hvp - outer_grad_inner_var)
            outer_var -= outer_step_size * (
                outer_grad_outer_var - cross_inner_fun_v
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta
