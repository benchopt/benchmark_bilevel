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
    """Adaptation of SVRG for bilevel optimization"""
    name = 'BiO-SVRG'

    stopping_criterion = SufficientProgressCriterion(
        patience=constants.PATIENCE, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'batch_size': [64],
        'period_frac': [1],
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
            self.bio_svrg = njit(bio_svrg)
            self.MinibatchSampler = jitclass(MinibatchSampler,
                                             mbs_spec)

            self.LearningRateScheduler = jitclass(LearningRateScheduler,
                                                  sched_spec)

        else:
            self.f_inner = f_train
            self.f_outer = f_test

            self.bio_svrg = bio_svrg
            self.MinibatchSampler = MinibatchSampler
            self.LearningRateScheduler = LearningRateScheduler

        self.inner_var0 = inner_var0
        self.outer_var0 = outer_var0
        self.numba = numba

    def run(self, callback):
        eval_freq = self.eval_freq  # // self.batch_size
        rng = np.random.RandomState(constants.RANDOM_STATE)

        # Init variables
        inner_var = self.inner_var0.copy()
        outer_var = self.outer_var0.copy()
        v = np.zeros_like(inner_var)

        period = self.f_inner.n_samples + self.f_outer.n_samples
        period *= self.period_frac

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
                self.step_size / self.outer_ratio,
            ]
        )
        exponents = np.zeros(2)
        lr_scheduler = self.LearningRateScheduler(
            np.array(step_sizes, dtype=float), exponents
        )

        inner_var_ref = inner_var.copy()
        outer_var_ref = outer_var.copy()
        v_ref = v.copy()
        d_inner_ref = np.zeros_like(inner_var)
        d_v_ref = np.zeros_like(inner_var)
        d_outer_ref = np.zeros_like(outer_var)
        i_min = 0
        # Start algorithm
        while callback((inner_var, outer_var)):
            inner_var, outer_var, inner_var_ref, v_ref, outer_var_ref,\
                d_inner_ref, d_v_ref, d_outer_ref, i_min = self.bio_svrg(
                    self.f_inner, self.f_outer,
                    inner_var, outer_var, v,
                    eval_freq, inner_sampler, outer_sampler,
                    lr_scheduler, inner_var_ref=inner_var_ref, v_ref=v_ref,
                    outer_var_ref=outer_var_ref, d_inner_ref=d_inner_ref,
                    d_v_ref=d_v_ref, d_outer_ref=d_outer_ref, i_min=i_min,
                    period=period, seed=rng.randint(constants.MAX_SEED)
                )
        self.beta = (inner_var, outer_var)

    def get_result(self):
        return self.beta


def bio_svrg(inner_oracle, outer_oracle, inner_var, outer_var, v, max_iter,
             inner_sampler, outer_sampler, lr_scheduler, inner_var_ref,
             v_ref, outer_var_ref, d_inner_ref, d_v_ref, d_outer_ref,
             i_min=0, period=100, seed=None):

    # Set seed for randomness
    if seed is not None:
        np.random.seed(seed)

    for i in range(i_min, i_min+max_iter):
        inner_lr, outer_lr = lr_scheduler.get_lr()
        # Computation of the directions
        if i % period == 0:  # Full batch computations
            inner_var_ref = inner_var.copy()
            v_ref = v.copy()
            outer_var_ref = outer_var.copy()
            slice_inner = slice(0, inner_oracle.n_samples)
            _, d_inner_ref, hvp, cross_v = inner_oracle.oracles(
                inner_var,
                outer_var,
                v,
                slice_inner,
                inverse='id'
            )

            slice_outer = slice(0, outer_oracle.n_samples)
            grad_in, d_outer = outer_oracle.grad(
                inner_var,
                outer_var,
                slice_outer
            )

            d_v_ref = hvp - grad_in
            d_outer_ref -= cross_v

            d_inner = d_inner_ref.copy()
            d_v = d_v_ref.copy()
            d_outer = d_outer_ref.copy()

        else:  # Stochastic computations
            slice_inner, _ = inner_sampler.get_batch()
            _, grad_inner_var, hvp, cross_v = inner_oracle.oracles(
                inner_var, outer_var, v, slice_inner, inverse='id'
            )
            _, grad_inner_var_ref, hvp_ref, cross_v_ref = inner_oracle.oracles(
                inner_var_ref, outer_var_ref, v_ref, slice_inner, inverse='id'
            )

            slice_outer, _ = outer_sampler.get_batch()
            grad_in_outer, grad_out_outer = outer_oracle.grad(
                inner_var, outer_var, slice_outer
            )
            grad_in_outer_ref, grad_out_outer_ref = outer_oracle.grad(
                inner_var_ref, outer_var_ref, slice_outer
            )

            d_inner = grad_inner_var - grad_inner_var_ref + d_inner_ref
            d_v = (hvp - hvp_ref) - (grad_in_outer - grad_in_outer_ref)
            d_v += d_v_ref
            d_outer = (grad_out_outer - grad_out_outer_ref) + d_outer_ref
            d_outer -= (cross_v - cross_v_ref)

        # Update of the variables
        inner_var -= inner_lr * d_inner
        v -= inner_lr * d_v
        outer_var -= outer_lr * d_outer

    return (
        inner_var, outer_var, inner_var_ref, v_ref, outer_var_ref, d_inner_ref,
        d_v_ref, d_outer_ref, i_min+max_iter
    )
