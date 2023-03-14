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

    from benchmark_utils.sgd_inner import sgd_inner_vrbo
    from benchmark_utils.hessian_approximation import shia, joint_shia


class Solver(BaseSolver):
    """Bi-level optimization algorithm."""
    name = 'VRBO'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.001],
        'outer_ratio': [1.],
        'n_hia_step': [10],
        'batch_size': [64],
        'period_frac': [128],
        'eval_freq': [1],
        'n_inner_step': [10],
        'random_state': [1]
    }

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def set_objective(self, f_train, f_test, inner_var0, outer_var0, numba):

        if numba:
            self.f_inner = f_train.numba_oracle
            self.f_outer = f_test.numba_oracle

            njit_vrbo = njit(_vrbo)
            njit_shia = njit(shia)
            njit_joint_shia = njit(joint_shia)
            _sgd_inner_vrbo = njit(sgd_inner_vrbo)

            @njit
            def njit_sgd_inner_vrbo(inner_oracle, outer_oracle,  inner_var, 
                outer_var, inner_lr, inner_sampler, outer_sampler, n_inner_step,
                memory_inner,memory_outer, n_hia_step, hia_lr
            ):
                return _sgd_inner_vrbo(njit_joint_shia, inner_oracle, 
                    outer_oracle, inner_var, outer_var, inner_lr, inner_sampler,
                    outer_sampler, n_inner_step, memory_inner, memory_outer, 
                    n_hia_step, hia_lr
                )
        

            self.MinibatchSampler = jitclass(MinibatchSampler, mbs_spec)
            self.LearningRateScheduler = jitclass(
                LearningRateScheduler, sched_spec
            )

            def vrbo(*args, **kwargs):
                return njit_vrbo(njit_sgd_inner_vrbo, njit_shia, *args, **kwargs)
            self.vrbo = vrbo

        else:
            self.f_inner = f_train
            self.f_outer = f_test

            def _sgd_inner_vrbo(*args, **kwargs):
                return sgd_inner_vrbo(joint_shia, *args, *kwargs)
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler
            def vrbo(*args, **kwargs):
                return _vrbo(_sgd_inner_vrbo, shia, *args, **kwargs)
            self.vrbo = vrbo


        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.numba = numba
        if self.numba:
            self.run_once(2)

    def run(self, callback):
        eval_freq = self.eval_freq  # // self.batch_size
        rng = np.random.RandomState(self.random_state)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        memory_inner = np.zeros((2, *inner_var.shape), inner_var.dtype)
        memory_outer = np.zeros((2, *outer_var.shape), outer_var.dtype)

        period = self.f_inner.n_samples + self.f_outer.n_samples
        period *= self.period_frac
        period /= self.batch_size
        period = int(period)


        # Init sampler and lr scheduler
        inner_sampler = self.MinibatchSampler(
            self.f_inner.n_samples, batch_size=self.batch_size
        )
        outer_sampler = self.MinibatchSampler(
            self.f_outer.n_samples, batch_size=self.batch_size
        )
        step_sizes = np.array(  # (inner_ss, hia_lr, outer_ss)
            [
                self.step_size,
                self.step_size,
                self.step_size / self.outer_ratio,
            ]
        )
        exponents = np.zeros(3)
        lr_scheduler = self.LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        # Start algorithm
        while callback((inner_var, outer_var)):
            inner_var, outer_var, memory_inner, memory_outer = self.vrbo(
                self.f_inner, self.f_outer,
                inner_var, outer_var, memory_inner, memory_outer,
                eval_freq, inner_sampler, outer_sampler,
                lr_scheduler, self.n_hia_step, self.n_inner_step, self.period,
                seed=rng.randint(constants.MAX_SEED)
            )
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


def _vrbo(sgd_inner_vrbo, shia, inner_oracle, outer_oracle, inner_var, outer_var,
         memory_inner, memory_outer, max_iter, inner_sampler, outer_sampler,
         lr_scheduler, n_hia_step, n_inner_steps, period, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(max_iter):
        inner_lr, hia_lr, outer_lr = lr_scheduler.get_lr()

        # Step.1 - (Re)initialize directions for z and x
        if i % period == 0:
            slice_inner, _ = inner_sampler.get_batch()
            grad_inner_var = inner_oracle.grad_inner_var(
                inner_var, outer_var, slice_inner
            )
            memory_inner[1] = grad_inner_var

            slice_outer, _ = outer_sampler.get_batch()
            grad_outer, impl_grad = outer_oracle.grad(
                inner_var, outer_var, slice_outer
            )
            ihvp = shia(
                inner_oracle, inner_var, outer_var, grad_outer, inner_sampler,
                n_hia_step, hia_lr
            )
            impl_grad -= inner_oracle.cross(
                inner_var, outer_var, ihvp, slice_inner
            )
            memory_outer[1] = impl_grad

        # Step.2 - Update outer variable and memory
        memory_outer[0] = outer_var
        outer_var -= outer_lr * memory_outer[1]

        # Step.3 - Project back to the constraint set
        inner_var, outer_var = inner_oracle.prox(inner_var, outer_var)

        inner_var, outer_var, memory_inner, memory_outer = sgd_inner_vrbo(
            inner_oracle, outer_oracle, inner_var, outer_var, inner_lr,
            inner_sampler, outer_sampler, n_inner_steps, memory_inner,
            memory_outer, n_hia_step, hia_lr
        )

    return inner_var, outer_var, memory_inner, memory_outer
