
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from oracles.minibatch_sampler import MinibatchSampler


class Solver(BaseSolver):
    """Two loops solver."""
    name = 'BSA'

    stopping_criterion = SufficientProgressCriterion(
        patience=100, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [1e-2],
        'n_inner_step': [10],
        'batch_size': [32, 100],
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
            inner_var, outer_var = bsa(
                self.f_inner.numba_oracle, self.f_outer.numba_oracle,
                inner_var, outer_var, n_eval_freq, outer_step_size,
                self.n_inner_step, inner_step_size,
                n_hia_step=self.n_inner_step, hia_step_size=inner_step_size,
                inner_sampler=inner_sampler, outer_sampler=outer_sampler,
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
def hia(inner_oracle, inner_var, outer_var, v, inner_sampler,
        n_step, step_size):
    """Hessian Inverse Approximation subroutine from [Ghadimi2018].

    This implement Algorithm.3
    """
    p = np.random.randint(n_step)
    for i in range(p):
        inner_slice, _ = inner_sampler.get_batch(inner_oracle)
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * hvp
    return p * step_size * v


@njit
def bsa(inner_oracle, outer_oracle, inner_var, outer_var,
        max_iter, outer_step_size, n_inner_step, inner_step_size,
        n_hia_step, hia_step_size, inner_sampler, outer_sampler
        ):
    """Numba compatible BSA algorithm.

    Parameters
    ----------
    inner_oracle, outer_oracle: NumbaOracle
        Inner and outer problem oracles used to compute gradients, etc...
    inner_var, outer_var: ndarray
        Current estimates of the inner and outer variables of the bi-level
        problem.
    max_iter: int
        Maximal number of iteration for the outer problem.
    outer_step_size: float
        Step size to update the outer variable.
    n_inner_step: int
        Maximal number of iteration for the inner problem.
    inner_step_size: float
        Step size to update the inner variable.
    n_hia_step: int
        Maximal number of iteration for the HIA problem.
    hia_step_size: float
        Step size for the HIA sub-routine.
    inner_sampler, outer_sampler: MinibatchSampler
        Sampler to get minibatch in a fast and efficient way for the inner and
        outer problems.
    """

    for i in range(max_iter):
        outer_slice, _ = outer_sampler.get_batch(outer_oracle)
        grad_in, grad_out = outer_oracle.grad(
            inner_var, outer_var, outer_slice
        )

        implicit_grad = hia(
            inner_oracle, inner_var, outer_var, grad_in,
            inner_sampler, n_hia_step, hia_step_size
        )
        inner_slice, _ = inner_sampler.get_batch(inner_oracle)
        implicit_grad = inner_oracle.cross(
            inner_var, outer_var, implicit_grad, inner_slice
        )
        grad_outer_var = grad_out - implicit_grad

        outer_var -= outer_step_size * grad_outer_var
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        inner_var = sgd_inner(
            inner_oracle, inner_var, outer_var, step_size=inner_step_size,
            inner_sampler=inner_sampler, n_inner_step=n_inner_step
        )
    return inner_var, outer_var
