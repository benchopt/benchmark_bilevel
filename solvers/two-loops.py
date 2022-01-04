
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from numba import njit
    from oracles.minibatch_sampler import MinibatchSampler


class Solver(BaseSolver):
    """Two loops solver."""
    name = 'two-loops'

    stopping_criterion = SufficientProgressCriterion(
        patience=15, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'n_inner_step': [10, 100, 1000],
        'batch_size': [32, 64],
        'step_size': [1e-2],
        'outer_ratio': [5, 20],
    }

    @staticmethod
    def get_next(stop_val):
        return max(1, min(stop_val * 2, stop_val + 50))

    def set_objective(self, f_train, f_test, inner_var0, outer_var0):
        self.f_inner = f_train
        self.f_outer = f_test
        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0

        if self.batch_size == 'all':
            self.inner_batch_size = self.f_inner.n_samples
            self.outer_batch_size = self.f_outer.n_samples
        else:
            self.inner_batch_size = self.batch_size
            self.outer_batch_size = self.batch_size

    def run(self, callback):
        n_eval_freq = max(1, 1_024 // self.n_inner_step)
        inner_step_size = self.step_size
        outer_step_size = self.step_size / self.outer_ratio
        outer_var = self.outer_var0.copy()
        inner_var = self.inner_var0.copy()
        inner_sampler = MinibatchSampler(
            self.f_inner.numba_oracle, self.inner_batch_size
        )
        outer_sampler = MinibatchSampler(
            self.f_outer.numba_oracle, self.outer_batch_size
        )

        callback((inner_var, outer_var))
        # L = self.f_inner.lipschitz_inner(inner_var, outer_var)
        inner_var = sgd_inner(
            self.f_inner.numba_oracle, inner_var, outer_var,
            step_size=inner_step_size,
            inner_sampler=inner_sampler, n_inner_step=self.n_inner_step
        )
        while callback((inner_var, outer_var)):
            inner_var, outer_var = two_loops(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, n_eval_freq, self.n_inner_step,
                inner_sampler, outer_sampler, inner_step_size, outer_step_size
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta

    def line_search(self, outer_var, grad):
        pass


@njit
def sgd_inner(inner_oracle, inner_var, outer_var,
              step_size, inner_sampler, n_inner_step):
    for _ in range(n_inner_step):
        inner_slice, _ = inner_sampler.get_batch(inner_oracle)
        grad_inner = inner_oracle.grad_inner_var(
            inner_var, outer_var, inner_slice
        )
        inner_var -= step_size * grad_inner

    return inner_var


@njit
def two_loops(inner_oracle, outer_oracle, inner_var, outer_var,
              max_iter, n_inner_step, inner_sampler, outer_sampler,
              inner_step_size, outer_step_size):

    for i in range(max_iter):
        outer_slice, _ = outer_sampler.get_batch(outer_oracle)
        grad_in, grad_out = outer_oracle.grad(
            inner_var, outer_var, outer_slice
        )

        inner_slice, _ = inner_sampler.get_batch(inner_oracle)
        _, _, _, implicit_grad = inner_oracle.oracles(
            inner_var, outer_var, grad_in, inner_slice, inverse='cg'
        )
        grad_outer_var = grad_out - implicit_grad

        outer_var -= outer_step_size * grad_outer_var
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        inner_var = sgd_inner(
            inner_oracle, inner_var, outer_var, step_size=inner_step_size,
            inner_sampler=inner_sampler, n_inner_step=n_inner_step
        )
    return inner_var, outer_var
