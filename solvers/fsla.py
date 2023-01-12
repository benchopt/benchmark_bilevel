
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
    """Fully Single Loop Algorithm (FSLA) for Bi-level optimization."""
    name = 'FSLA'

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
            self.fsla = njit(fsla)
            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

        else:
            self.f_inner = f_train
            self.f_outer = f_test

            self.fsla = fsla
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
        memory_outer = np.zeros((2, *outer_var.shape))

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
        # Use 1 / sqrt(t) for the learning rates
        exponents = 0.5 * np.ones(len(step_sizes))
        lr_scheduler = self.LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        # Start algorithm
        while callback((inner_var, outer_var)):
            inner_var, outer_var, v = self.fsla(
                self.f_inner, self.f_outer,
                inner_var, outer_var, v, memory_outer,
                eval_freq, inner_sampler, outer_sampler, lr_scheduler,
                seed=rng.randint(constants.MAX_SEED)
            )
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


def fsla(inner_oracle, outer_oracle, inner_var, outer_var, v,
         memory_outer, max_iter, inner_sampler, outer_sampler,
         lr_scheduler, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_lr, eta, outer_lr = lr_scheduler.get_lr()

        # Step.1 - SGD step on the inner problem
        slice_inner, _ = inner_sampler.get_batch()
        grad_inner_var = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice_inner
        )
        inner_var_old = inner_var.copy()
        inner_var -= inner_lr * grad_inner_var

        # Step.2 - SGD step on the auxillary variable v
        slice_inner2, _ = inner_sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, slice_inner2)
        slice_outer, _ = outer_sampler.get_batch()
        grad_in_outer = outer_oracle.grad_inner_var(
            inner_var, outer_var, slice_outer
        )
        v_old = v.copy()
        v -= inner_lr * (hvp - grad_in_outer)

        # Step.3 - compute the implicit gradient estimates, for the old
        # and new variables
        slice_outer2, _ = outer_sampler.get_batch()
        impl_grad = outer_oracle.grad_outer_var(
            inner_var, outer_var, slice_outer2
        )
        impl_grad_old = outer_oracle.grad_outer_var(
            inner_var_old, memory_outer[0], slice_outer2
        )
        slice_inner3, _ = inner_sampler.get_batch()
        impl_grad -= inner_oracle.cross(inner_var, outer_var, v, slice_inner3)
        impl_grad_old -= inner_oracle.cross(
            inner_var_old, memory_outer[0], v_old, slice_inner3
        )

        # Step.4 - update direction with momentum
        memory_outer[1] = (
            impl_grad + (1-eta) * (memory_outer[1] - impl_grad_old)
        )

        # Step.5 - update the outer variable
        memory_outer[0] = outer_var
        outer_var -= outer_lr * memory_outer[1]

        # Step.6 - project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

    return inner_var, outer_var, v
