
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Single loop."""
    name = 'single-loop'

    stopping_criterion = SufficientProgressCriterion(
        patience=7, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [1e-3, 1e-2, 1e-1, 1],
        'batch_size, vr': [
            (1, 'saga'), (32, 'none'), (64, 'none'), (128, 'none')
        ]
    }

    def set_objective(self, f_train, f_test, inner_var0, outer_var0):
        self.f_inner = f_train
        self.f_outer = f_test
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

    def run(self, callback):
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        v = np.zeros_like(inner_var)
        if self.step_size == 'auto':
            inner_step_size = 1 / self.f_inner.lipschitz_inner(
                inner_var, outer_var
            )
            outer_step_size = 1
        else:
            inner_step_size = outer_step_size = self.step_size
        while callback((inner_var, outer_var)):
            for i in range(max(1, 1024 // self.batch_size)):
                grad_in, grad_out = self.f_outer.get_batch_grad(
                    inner_var, outer_var,
                    batch_size=self.batch_size, vr=self.vr
                )
                _, grad_inner_fun, hvp, implicit_grad = (
                    self.f_inner.get_batch_oracles(
                        inner_var, outer_var, v,
                        batch_size=self.batch_size, vr=self.vr
                    )
                )

                inner_var -= inner_step_size * grad_inner_fun
                v -= inner_step_size * (hvp - grad_in)
                outer_var -= outer_step_size * (
                    grad_out - implicit_grad
                )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta
