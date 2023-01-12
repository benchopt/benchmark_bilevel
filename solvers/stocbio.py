
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.sgd_inner import sgd_inner
    from benchmark_utils.hessian_approximation import shia
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec


class Solver(BaseSolver):
    """StocBio solver from [Ji2021]

    :cat:`two-loops`
    """
    name = 'StocBiO'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_inner_step': [10],
        'batch_size': [64],
        'n_shia_steps': [10],
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
            self.shia = njit(shia)
            njit_stocbio = njit(_stocbio)
            self.sgd_inner = njit(sgd_inner)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

            def stocbio(*args, **kwargs):
                return njit_stocbio(
                    self.sgd_inner, self.shia, *args, **kwargs
                )
            self.stocbio = stocbio
        else:
            self.f_inner = f_train
            self.f_outer = f_test
            self.sgd_inner = sgd_inner
            self.shia = shia
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

            def stocbio(*args, **kwargs):
                return _stocbio(sgd_inner, shia, *args, **kwargs)

            self.stocbio = stocbio

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.numba = numba
        if self.numba:
            self.run_once(2)

    def run(self, callback):
        eval_freq = self.eval_freq  # // self.batch_size
        rng = np.random.RandomState(constants.RANDOM_STATE)

        # Init variables
        outer_var = self.outer_var0.copy()
        inner_var = self.inner_var0.copy()

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
        exponents = np.zeros(3)
        lr_scheduler = self.LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        # Start algorithm
        inner_var = self.sgd_inner(
            self.f_inner, inner_var, outer_var,
            step_size=self.step_size,
            inner_sampler=inner_sampler, n_inner_step=self.n_inner_step
        )
        while callback((inner_var, outer_var)):
            inner_var, outer_var = self.stocbio(
                self.f_inner, self.f_outer,
                inner_var, outer_var, eval_freq, lr_scheduler,
                self.n_inner_step, self.n_shia_steps,
                inner_sampler, outer_sampler,
                seed=rng.randint(constants.MAX_SEED)
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta

    def line_search(self, outer_var, grad):
        pass


def _stocbio(sgd_inner, shia, inner_oracle, outer_oracle, inner_var, outer_var,
             max_iter, lr_scheduler, n_inner_step, n_shia_step,
             inner_sampler, outer_sampler, seed=None):
    """Numba compatible stocBiO algorithm.

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
    n_inner_step: int
        Maximal number of iterations for the inner problem.
    inner_step_size: float
        Step size to update the inner variable.
    n_shia_step: int
        Maximal number of iterations for the shia problem.
    shia_step_size: float
        Step size for the shia sub-routine.
    inner_sampler, outer_sampler: MinibatchSampler
        Sampler to get minibatch in a fast and efficient way for the inner and
        outer problems.
    """

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_lr, shia_lr, outer_lr = lr_scheduler.get_lr()
        outer_slice, _ = outer_sampler.get_batch()
        grad_in, grad_out = outer_oracle.grad(
            inner_var, outer_var, outer_slice
        )

        implicit_grad = shia(
            inner_oracle, inner_var, outer_var, grad_in,
            inner_sampler, n_shia_step, shia_lr
        )
        inner_slice, _ = inner_sampler.get_batch()
        implicit_grad = inner_oracle.cross(
            inner_var, outer_var, implicit_grad, inner_slice
        )
        grad_outer_var = grad_out - implicit_grad

        outer_var -= outer_lr * grad_outer_var
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        inner_var = sgd_inner(
            inner_oracle, inner_var, outer_var, step_size=inner_lr,
            inner_sampler=inner_sampler, n_inner_step=n_inner_step
        )
    return inner_var, outer_var
