from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit
    from numba.experimental import jitclass

    from benchmark_utils import constants
    from benchmark_utils.minibatch_sampler import MinibatchSampler
    from benchmark_utils.minibatch_sampler import spec as mbs_spec
    from benchmark_utils.learning_rate_scheduler import LearningRateScheduler
    from benchmark_utils.learning_rate_scheduler import spec as sched_spec


class Solver(BaseSolver):
    """Stochastic Bi-level Algorithm (SOBA)."""
    name = 'SOBA'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
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
            self.soba = njit(soba)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

        else:
            self.f_inner = f_train
            self.f_outer = f_test

            self.soba = soba
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.numba = numba
        if self.numba:
            self.run_once(2)

    def run(self, callback):
        eval_freq = self.eval_freq  # // self.batch_size
        rng = np.random.RandomState(constants.RANDOM_STATE)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        v = np.zeros_like(inner_var)

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
            [self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = np.array(
            [.5, .5]
        )
        lr_scheduler = self.LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )
        # Start algorithm
        while callback((inner_var, outer_var)):
            inner_var, outer_var, v = self.soba(
                self.f_inner, self.f_outer,
                inner_var, outer_var, v, eval_freq,
                inner_sampler, outer_sampler, lr_scheduler,
                seed=rng.randint(constants.MAX_SEED)
            )

        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


def soba(inner_oracle, outer_oracle, inner_var, outer_var, v, max_iter,
         inner_sampler, outer_sampler, lr_scheduler, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_step_size, outer_step_size = lr_scheduler.get_lr()

        # Step.1 - get all gradients and compute the implicit gradient.
        slice_inner, _ = inner_sampler.get_batch()
        _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
            inner_var, outer_var, v, slice_inner, inverse='id'
        )

        slice_outer, _ = outer_sampler.get_batch()
        grad_in_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        impl_grad -= cross_v

        # Step.2 - update inner variable with SGD.
        inner_var -= inner_step_size * grad_inner_var

        # Step.3 - update auxillary variable v with SGD
        v -= inner_step_size * (hvp - grad_in_outer)

        # Step.4 - update outer_variable with SGD
        outer_var -= outer_step_size * impl_grad

        # Use prox to make sure we do not diverge
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

    return inner_var, outer_var, v
