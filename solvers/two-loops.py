
from benchopt import BaseSolver


class Solver(BaseSolver):
    """Two loops solver."""
    name = 'two-loop'

    stop_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {'n_inner_step': [10, 100, 1000]}

    def set_objective(self, f_train, f_test, inner_var0, outer_var0):
        self.f_inner = f_train
        self.f_outer = f_test
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

    def solver_inner(self, inner_var, outer_var):
        L = self.f_inner.lipschitz_inner(inner_var, outer_var)
        for _ in range(self.n_inner_step):
            grad_inner = self.f_inner.get_grad_inner_var(inner_var, outer_var)
            inner_var -= 1/L * grad_inner

        return inner_var

    def run(self, callback):
        outer_step_size = 1
        outer_var = self.outer_var0.copy()
        inner_var = self.inner_var0.copy()
        inner_var = self.solver_inner(inner_var, outer_var)
        while callback((inner_var, outer_var)):
            outer_grad_inner_var, outer_grad_outer_var = self.f_outer.get_grad(
                inner_var, outer_var
            )

            inverse_hessian = self.f_inner.get_inverse_hessian_vector_prod(
                inner_var, outer_var, outer_grad_inner_var
            )
            cross_inverse_hessian = self.f_inner.get_cross(
                inner_var, outer_var, inverse_hessian
            )
            grad_outer_var = outer_grad_outer_var - cross_inverse_hessian

            outer_var -= outer_step_size * grad_outer_var
            inner_var = self.solver_inner(inner_var, outer_var)

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta
