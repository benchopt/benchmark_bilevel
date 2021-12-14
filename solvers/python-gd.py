
from benchopt import BaseSolver


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'GD'

    stop_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    # parameters = {'use_acceleration': [False, True]}

    def set_objective(self, f_train, f_test, inner_var0, outer_var0):
        self.f_train = f_train
        self.f_test = f_test
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

    def run(self, callback):
        lr = 1e2
        outer_var = self.outer_var0.copy()
        inner_var = self.f_train.get_inner_var_star(outer_var)
        while callback((inner_var, outer_var)):
            outer_grad_inner_var, outer_grad_outer_var = self.f_test.get_grad(
                inner_var,
                outer_var
            )
            cross_inner_fun_hvp = self.f_train.get_cross(
                inner_var,
                outer_var,
                self.f_train.get_inverse_hessian_vector_prod(
                    inner_var,
                    outer_var,
                    outer_grad_inner_var
                )
            )
            grad = outer_grad_outer_var - cross_inner_fun_hvp
            outer_var -= lr * grad
            inner_var = self.f_train.get_inner_var_star(outer_var)

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta
