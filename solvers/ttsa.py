
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.hessian_approximation import hia
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec


class Solver(BaseSolver):
    """Single loop Bi-level optimization algorithm."""
    name = 'TTSA'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_hia_step': [10],
        'batch_size': [64],
        'eval_freq': [128],
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def skip(self, f_train, f_test, inner_var0, outer_var0, numba):
        if self.batch_size == 'full' and numba:
            return True, "numba is not useful for full bach resolution."

        return False, None

    def set_objective(self, f_train, f_test, inner_var0, outer_var0, numba):

        if numba:
            self.f_inner = f_train.numba_oracle
            self.f_outer = f_test.numba_oracle

            # JIT necessary functions and classes
            njit_hia = njit(hia)
            njit_ttsa = njit(_ttsa)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

            def ttsa(*args, **kwargs):
                return njit_ttsa(njit_hia, *args, **kwargs)
            self.ttsa = ttsa

        else:
            self.f_inner = f_train
            self.f_outer = f_test

            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

            def ttsa(*args, **kwargs):
                return _ttsa(hia, *args, **kwargs)
            self.ttsa = ttsa

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.numba = numba

    def run(self, callback):
        eval_freq = self.eval_freq
        rng = np.random.RandomState(constants.RANDOM_STATE)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()

        # Init sampler and lr scheduler
        if self.batch_size == 'full':
            batch_size_inner = self.f_inner.n_samples
            batch_size_outer = self.f_outer.n_samples
        else:
            batch_size_inner = self.batch_size
            batch_size_outer = self.batch_size
        inner_sampler = self.MinibatchSampler(
            self.f_inner.n_samples, batch_size=batch_size_inner
        )
        outer_sampler = self.MinibatchSampler(
            self.f_outer.n_samples, batch_size=batch_size_outer
        )
        step_sizes = np.array(
            [self.step_size, self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = np.array([.4, 0., .6])
        lr_scheduler = self.LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        while callback((inner_var, outer_var)):
            inner_var, outer_var, = self.ttsa(
                self.f_inner, self.f_outer,
                inner_var, outer_var,
                eval_freq, inner_sampler, outer_sampler, lr_scheduler,
                self.n_hia_step, seed=rng.randint(constants.MAX_SEED)
            )
        self.beta = (inner_var, outer_var)
        if self.numba:
            self.run_once(2)

    def get_result(self):
        return self.beta


def _ttsa(
    hia, inner_oracle, outer_oracle, inner_var, outer_var, max_iter,
    inner_sampler, outer_sampler, lr_scheduler, n_hia_step, seed=None
):
    """Numba compatible TTSA algorithm.

    Parameters
    ----------
    inner_oracle, outer_oracle: NumbaOracle
        Inner and outer problem oracles used to compute gradients, etc...
    inner_var, outer_var: ndarray
        Current estimates of the inner and outer variables of the bi-level
        problem.
    max_iter: int
        Maximal number of iterations for the outer problem.
    outer_step_size: float
        Step size to update the outer variable.
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

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_lr, hia_lr, outer_lr = lr_scheduler.get_lr()

        # Step.1 - Update direction for z with momentum
        slice_inner, _ = inner_sampler.get_batch()
        grad_inner_var = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice_inner
        )

        # Step.2 - Update the inner variable
        inner_var -= inner_lr * grad_inner_var

        # Step.3 - Compute implicit grad approximation with HIA
        slice_outer, _ = outer_sampler.get_batch()
        grad_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        ihvp = hia(
            inner_oracle, inner_var, outer_var, grad_outer,
            inner_sampler, n_hia_step, hia_lr
        )
        impl_grad -= inner_oracle.cross(
            inner_var, outer_var, ihvp, slice_inner
        )

        # Step.4 - update the outer variables
        outer_var -= outer_lr * impl_grad

        # Step.6 - project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)
    return inner_var, outer_var
