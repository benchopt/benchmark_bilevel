
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion


class Solver(BaseSolver):
    """Two loops solver."""
    name = 'two-loop'

    stopping_criterion = SufficientProgressCriterion(
        patience=7, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'n_inner_step': [10, 100, 1000],
        'batch_size': [64, 128, 'all']
    }

    def set_objective(self, f_train, f_test, inner_var0, outer_var0):
        self.f_inner = f_train
        self.f_outer = f_test
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

    def solver_inner(self, inner_var, outer_var):
        L = self.f_inner.lipschitz_inner(inner_var, outer_var)
        for _ in range(self.n_inner_step):
            grad_inner = self.f_inner.get_batch_grad_inner_var(
                inner_var, outer_var, batch_size=self.batch_size
            )
            inner_var -= 1/L * grad_inner

        return inner_var

    def run(self, callback):
        outer_step_size = 1
        outer_var = self.outer_var0.copy()
        inner_var = self.inner_var0.copy()
        callback((inner_var, outer_var))
        inner_var = self.solver_inner(inner_var, outer_var)
        while callback((inner_var, outer_var)):
            grad_in, grad_out = self.f_outer.get_batch_grad(
                inner_var, outer_var, batch_size=self.batch_size
            )

            *_, implicit_grad = self.f_inner.get_batch_oracles(
                inner_var, outer_var, grad_in, batch_size=self.batch_size,
                inverse='cg'
            )
            grad_outer_var = grad_out - implicit_grad

            outer_var -= outer_step_size * grad_outer_var
            inner_var = self.solver_inner(inner_var, outer_var)

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta

    def line_search(self, outer_var, grad):
        pass
