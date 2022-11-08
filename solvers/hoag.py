from benchopt import BaseSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from hoag import hoag_lbfgs
    constants = import_ctx.import_from('constants')

class Solver(BaseSolver):
    """Hyperparameter Selection with Approximate Gradient."""
    name = 'HOAG'
    stopping_strategy = 'iteration'

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_train, f_test, inner_var0, outer_var0, numba):
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        self.f_inner = f_train
        self.f_outer = f_test
        self.numba = numba

    def run(self, n_iter):
        if n_iter == 0:
            inner_var = self.inner_var0.copy()
            outer_var = self.outer_var0.copy()
        else:
            def h_func_grad(inner_var, outer_var):
                return (
                    self.f_inner.get_value(inner_var, outer_var),
                    self.f_inner.get_grad_inner_var(inner_var, outer_var)
                )

            def h_hessian(inner_var, outer_var):
                def f(v):
                    return self.f_inner.get_hvp(inner_var, outer_var, v)
                return f

            def g_func_grad(inner_var, outer_var):
                return (
                    self.f_inner.get_value(inner_var, outer_var),
                    self.f_inner.get_grad_inner_var(inner_var, outer_var)
                )

            inner_var, outer_var, _ = hoag_lbfgs(
                h_func_grad,
                h_hessian,
                self.f_inner.cross_mat,
                g_func_grad,
                self.inner_var0,
                lambda0=self.outer_var0
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta
